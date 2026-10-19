from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from neuoct.config.schema import DataConfig

from .camera import CameraIntrinsics
from .io import PointCloudCache, load_exr_tensor, load_image_tensor, load_json
from .manifest import ManifestEntry, read_manifest
from .occupancy import OccupancyCache
from .sample import RawSample


class ShapeNetMultiViewDataset(Dataset[RawSample]):
    """CPU-only reader for the NeuOct ShapeNet multi-view schema."""

    def __init__(
        self, config: DataConfig, *, entries: list[ManifestEntry] | None = None
    ):
        config.validate(require_paths=True)
        self.config = config
        self.entries = (
            list(entries)
            if entries is not None
            else read_manifest(config.manifest, take=config.take)
        )
        if not self.entries:
            raise ValueError("Dataset contains no samples")
        self.point_cache = PointCloudCache(config.object_cache_size)
        self.occupancy_cache = OccupancyCache(config.object_cache_size)
        self._metadata_cache: OrderedDict[str, tuple[Path, list[object]]] = OrderedDict()
        self._view_cache: OrderedDict[str, dict[int, dict[str, object]]] = OrderedDict()

    def _load_object_metadata(self, object_key: str) -> tuple[Path, list[object]]:
        if object_key in self._metadata_cache:
            self._metadata_cache.move_to_end(object_key)
            return self._metadata_cache[object_key]

        category, object_id = object_key.split("/", 1)
        object_dir = self.config.image_root / category / object_id
        transform_path = object_dir / "transforms.json"
        metadata = load_json(transform_path)
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"No frames found in {transform_path}")
        result = (object_dir, frames)
        if self.config.sample_cache_size > 0:
            self._metadata_cache[object_key] = result
            self._metadata_cache.move_to_end(object_key)
            while len(self._metadata_cache) > self.config.sample_cache_size:
                evicted_key, _ = self._metadata_cache.popitem(last=False)
                self._view_cache.pop(evicted_key, None)
        return result

    def _load_view(self, entry: ManifestEntry) -> dict[str, object]:
        cached_views = self._view_cache.get(entry.object_key)
        if cached_views is not None and entry.frame_id in cached_views:
            self._view_cache.move_to_end(entry.object_key)
            self._metadata_cache.move_to_end(entry.object_key)
            return cached_views[entry.frame_id]

        object_dir, frames = self._load_object_metadata(entry.object_key)
        if entry.frame_id >= len(frames):
            raise IndexError(f"Frame {entry.frame_id} is unavailable for {entry.object_key}")
        frame = frames[entry.frame_id]
        if not isinstance(frame, dict):
            raise ValueError(
                f"Frame {entry.frame_id} for {entry.object_key} is not a JSON object"
            )

        expected = (self.config.image_height, self.config.image_width)
        stem = Path(str(frame["file_path"])).stem
        color = load_image_tensor(object_dir / f"{stem}.png")[..., :3]
        depth_raw = load_image_tensor(object_dir / f"{stem}_depth.png")
        normal_png_path = object_dir / f"{stem}_normal.png"
        if self.config.normal_source == "exr":
            normal_path = object_dir / f"{stem}_normal.exr"
            if not normal_path.exists() and normal_png_path.is_symlink():
                normal_path = normal_png_path.resolve().with_suffix(".exr")
            normal = load_exr_tensor(normal_path)
        else:
            normal = load_image_tensor(normal_png_path)[..., :3]
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., 0]
        actual_height, actual_width = depth_raw.shape
        if (actual_height, actual_width) != expected:
            raise ValueError(
                f"Image shape {(actual_height, actual_width)} for {entry.relative_name} "
                f"does not match configured {expected}."
            )
        if color.shape[:2] != expected or normal.shape[:2] != expected:
            raise ValueError(
                f"RGB/depth/normal dimensions differ for {entry.relative_name}"
            )
        depth_info = frame.get("depth", {})
        depth_min = float(depth_info["min"])
        depth_max = float(depth_info["max"])
        valid_depth = depth_raw != 65535
        depth = depth_raw.float() * (depth_max - depth_min) / 65535.0 + depth_min
        view = {
            "intrinsics": CameraIntrinsics.from_horizontal_fov(
                float(frame["camera_angle_x"]), actual_width, actual_height
            ),
            "camera_to_world": torch.tensor(
                frame["transform_matrix"], dtype=torch.float32
            ),
            "color_image": color.contiguous(),
            "depth_image": depth.contiguous(),
            "normal_image": normal.contiguous(),
            "valid_depth": valid_depth.contiguous(),
            "depth_min": depth_min,
            "depth_max": depth_max,
        }
        if self.config.sample_cache_size > 0:
            cached_views = self._view_cache.setdefault(entry.object_key, {})
            cached_views[entry.frame_id] = view
            self._view_cache.move_to_end(entry.object_key)
        return view

    @classmethod
    def for_object(
        cls, config: DataConfig, object_key: str, *, take: int = -1
    ) -> "ShapeNetMultiViewDataset":
        """Creates a view dataset for every frame in one ShapeNet object.

        The local optimization branch trains one scene against all available views,
        whereas the global manifest generally contains one or a few views per object.
        Frame ids are read from the object's transforms file so no temporary manifest
        is needed.
        """
        parts = [part for part in object_key.replace("\\", "/").split("/") if part]
        if len(parts) != 2:
            raise ValueError("object_key must be '<category>/<object_id>'")
        category, object_id = parts
        transform_path = config.image_root / category / object_id / "transforms.json"
        metadata = load_json(transform_path)
        frames = metadata.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"No frames found in {transform_path}")
        entries = [ManifestEntry(category, object_id, frame_id) for frame_id in range(len(frames))]
        if take > 0:
            entries = entries[:take]
        return cls(config, entries=entries)

    def __len__(self) -> int:
        return len(self.entries)

    def _occupancy_points(self, entry: ManifestEntry) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Mesh-derived surface cell representatives for one object, or ``None``.

        These are appended to the back-projected point cloud so that every
        query-depth cell the real surface passes through has an octree node, which
        removes the ``MISSING`` holes the 8-view point cloud leaves behind.
        """
        root = self.config.occupancy_root
        if root is None:
            return None
        object_id = entry.object_key.split("/", 1)[1]
        path = root / f"{object_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"DATA.occupancy_root is set but {path} is missing; "
                f"build it with scripts/build_mesh_occupancy.py"
            )
        return self.occupancy_cache.load(
            path,
            query_depth=self.config.query_depth,
            ray_scale=self.config.ray_scale,
            point_scale=self.config.point_scale,
            max_cells=self.config.occupancy_max_cells,
        )

    def __getitem__(self, index: int) -> RawSample:
        entry = self.entries[index]
        view = self._load_view(entry)
        point_path = (
            self.config.pointcloud_root / entry.category / entry.object_id / "pointcloud.pth"
        )
        point_positions, point_colors = self.point_cache.load(point_path)
        occupancy = self._occupancy_points(entry)
        if occupancy is not None:
            extra_positions, extra_colors = occupancy
            # the occupancy file stores the *world* frame, matching ``pointcloud.pth``
            point_positions = torch.cat([point_positions, extra_positions], dim=0)
            point_colors = torch.cat([point_colors, extra_colors], dim=0)
        return RawSample(
            entry=entry,
            point_positions=point_positions,
            point_colors=point_colors,
            **view,
        )


def collate_samples(samples: list[RawSample]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    shapes = {sample.image_shape for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"All images in a batch must have the same shape, got {sorted(shapes)}")
    return {"samples": samples}
