from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _value(node: Any, name: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def feature_channels(feature_spec: str) -> int:
    """Returns the number of octree input channels represented by a feature spec."""
    spec = feature_spec.upper()
    unknown = set(spec) - set("NFLDP")
    if unknown:
        raise ValueError(f"Unsupported MODEL.feature codes: {sorted(unknown)}")
    return (
        (3 if "N" in spec else 0)
        + (3 if "F" in spec else 0)
        + (3 if "L" in spec else 0)
        + (1 if "D" in spec else 0)
        + (3 if "P" in spec else 0)
    )


@dataclass(frozen=True)
class DataConfig:
    image_root: Path
    pointcloud_root: Path
    manifest: Path
    occupancy_root: Path | None = None
    occupancy_max_cells: int = 0
    octree_depth: int = 9
    full_depth: int = 4
    query_depth: int = 7
    image_width: int = 512
    image_height: int = 512
    point_scale: float = 0.99
    ray_scale: float = 0.5
    camera_convention: str = "neuoct"
    orbit_up_axis: str = "y"
    normal_source: str = "png"
    object_cache_size: int = 8
    sample_cache_size: int = 1
    octree_cache_size: int = 2
    take: int = -1

    @classmethod
    def from_flags(cls, flags: Any) -> "DataConfig":
        take = int(_value(flags, "take", -1))
        legacy_take = int(_value(flags, "takes", -1))
        if take < 1 and legacy_take > 0:
            take = legacy_take
        pointcloud_root = _value(flags, "pointcloud_root", "")
        if not str(pointcloud_root).strip():
            raise ValueError(
                "DATA.pointcloud_root is required; legacy hard-coded roots are unsupported"
            )
        occupancy_root = str(_value(flags, "occupancy_root", "") or "").strip()
        return cls(
            image_root=Path(str(_value(flags, "location", ""))).expanduser(),
            pointcloud_root=Path(str(pointcloud_root)).expanduser(),
            occupancy_root=Path(occupancy_root).expanduser() if occupancy_root else None,
            occupancy_max_cells=int(_value(flags, "occupancy_max_cells", 0) or 0),
            manifest=Path(str(_value(flags, "filelist", ""))).expanduser(),
            octree_depth=int(_value(flags, "depth", 9)),
            full_depth=int(_value(flags, "full_depth", 4)),
            query_depth=int(_value(flags, "q_depth", 7)),
            image_width=int(_value(flags, "image_width", 512)),
            image_height=int(_value(flags, "image_height", 512)),
            point_scale=float(_value(flags, "point_scale", 0.99)),
            ray_scale=float(_value(flags, "ray_scale", 0.5)),
            camera_convention=str(_value(flags, "camera_convention", "neuoct")),
            orbit_up_axis=str(_value(flags, "orbit_up_axis", "y")),
            normal_source=str(_value(flags, "normal_source", "png")),
            object_cache_size=int(_value(flags, "object_cache_size", 8)),
            sample_cache_size=int(_value(flags, "sample_cache_size", 1)),
            octree_cache_size=int(_value(flags, "octree_cache_size", 2)),
            take=take,
        )

    def validate(self, *, require_paths: bool = True) -> None:
        if self.octree_depth <= 0:
            raise ValueError(f"octree depth must be positive, got {self.octree_depth}")
        if self.query_depth not in (5, 6, 7):   # 5 added for the fixed-size table sweep
            raise ValueError(f"query depth must be 5, 6 or 7, got {self.query_depth}")
        if not 0 < self.full_depth <= self.query_depth <= self.octree_depth:
            raise ValueError(
                "Expected 0 < full_depth <= query_depth <= octree_depth, got "
                f"{self.full_depth}, {self.query_depth}, {self.octree_depth}."
            )
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if not 0.0 < self.point_scale <= 1.0:
            raise ValueError("point_scale must be in (0, 1]")
        if self.ray_scale <= 0.0:
            raise ValueError("ray_scale must be positive")
        if self.camera_convention not in ("neuoct", "legacy_blender"):
            raise ValueError(
                "camera_convention must be 'neuoct' or 'legacy_blender', got "
                f"{self.camera_convention!r}"
            )
        if self.orbit_up_axis not in ("x", "y", "z"):
            raise ValueError(
                "orbit_up_axis must be 'x', 'y' or 'z', got "
                f"{self.orbit_up_axis!r}"
            )
        if self.normal_source not in ("png", "exr"):
            raise ValueError(
                "normal_source must be 'png' or 'exr', got "
                f"{self.normal_source!r}"
            )
        if self.object_cache_size < 0 or self.sample_cache_size < 0 or self.octree_cache_size < 0:
            raise ValueError(
                "object_cache_size, sample_cache_size and octree_cache_size must be non-negative"
            )
        if require_paths:
            for label, path in (
                ("image root", self.image_root),
                ("point-cloud root", self.pointcloud_root),
                ("manifest", self.manifest),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Configured {label} does not exist: {path}")
            if self.occupancy_root is not None and not self.occupancy_root.exists():
                raise FileNotFoundError(
                    "Configured mesh-occupancy root does not exist: "
                    f"{self.occupancy_root}"
                )


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int
    feature_spec: str
    query_depth: int
    use_vae: bool = False
    renderer_hidden_dim: int = 128
    renderer_iterations: int = 3
    segment_position: bool = False
    voxel_feature_dim: int = 0
    color_skip_weight: float = 0.0
    activation_checkpointing: bool = False
    material_fp32: bool = False
    norm_type: str = "batch"
    norm_groups: int = 8
    # Intersection sampler of the split-voxel renderer.  "recurrent" is the
    # original 4-step chained sampler; "recurrent_pool" is the pooled-fusion
    # adaptive sampler that won the single-object sampling ablation
    # (SAMPLING_ABLATION_ARMADILLO_20260914.md) and is now exposed to the
    # feedforward branch as well.
    sampling_mode: str = "recurrent"

    @classmethod
    def from_flags(cls, flags: Any) -> "ModelConfig":
        return cls(
            input_channels=int(_value(flags, "channel", 0)),
            feature_spec=str(_value(flags, "feature", "")),
            query_depth=int(_value(flags, "q_depth", 7)),
            use_vae=bool(_value(flags, "use_vae", False)),
            renderer_hidden_dim=int(_value(flags, "renderer_hidden_dim", 128)),
            renderer_iterations=int(_value(flags, "renderer_iterations", 3)),
            segment_position=bool(_value(flags, "segment_position", False)),
            voxel_feature_dim=int(_value(flags, "voxel_feature_dim", 0)),
            color_skip_weight=float(_value(flags, "color_skip_weight", 0.0)),
            activation_checkpointing=bool(
                _value(flags, "activation_checkpointing", False)
            ),
            material_fp32=bool(_value(flags, "material_fp32", False)),
            norm_type=str(_value(flags, "norm_type", "batch")),
            norm_groups=int(_value(flags, "norm_groups", 8)),
            sampling_mode=str(_value(flags, "sampling_mode", "recurrent")),
        )

    def validate(self) -> None:
        if self.use_vae:
            raise NotImplementedError(
                "VAE variants are intentionally retained outside the refactored core for this phase."
            )
        if self.query_depth not in (5, 6, 7):   # 5 added for the fixed-size table sweep
            raise ValueError(f"MODEL.q_depth must be 5, 6 or 7, got {self.query_depth}")
        if self.renderer_hidden_dim <= 0 or self.renderer_iterations <= 0:
            raise ValueError("renderer_hidden_dim and renderer_iterations must be positive")
        if self.voxel_feature_dim < 0:
            raise ValueError("MODEL.voxel_feature_dim must be non-negative")
        if self.norm_type.lower().replace("_", "-") not in {
            "batch", "bn", "batchnorm", "rms", "rmsnorm", "layer", "ln",
            "layernorm", "group", "gn", "groupnorm",
        }:
            raise ValueError(
                "MODEL.norm_type must be batch, rms, layer, or group"
            )
        if self.norm_groups <= 0:
            raise ValueError("MODEL.norm_groups must be positive")
        if not 0.0 <= self.color_skip_weight <= 1.0:
            raise ValueError("MODEL.color_skip_weight must be in [0, 1]")
        if self.sampling_mode not in (
            "recurrent",
            "uniform",
            "single",
            "recurrent_endpoints",
            "uniform_endpoints",
            "uniform_concat",
            "uniform_concat_endpoints",
            "recurrent_pool",
        ):
            raise ValueError(f"MODEL.sampling_mode {self.sampling_mode!r} is not supported")
        expected_channels = feature_channels(self.feature_spec)
        if expected_channels != self.input_channels:
            raise ValueError(
                f"MODEL.channel={self.input_channels} does not match "
                f"MODEL.feature={self.feature_spec!r} ({expected_channels} channels)."
            )


def validate_solver_config(flags: Any, *, require_paths: bool = True) -> None:
    """Validates the THSolver/YACS configuration before allocating model memory."""
    model = ModelConfig.from_flags(flags.MODEL)
    train = DataConfig.from_flags(flags.DATA.train)
    test = DataConfig.from_flags(flags.DATA.test)
    model.validate()
    train.validate(require_paths=require_paths)
    test.validate(require_paths=require_paths)
    if train.octree_depth != 9 or test.octree_depth != 9:
        raise ValueError(
            "The checkpoint-compatible global feature U-Net requires octree depth 9; "
            f"got train={train.octree_depth}, test={test.octree_depth}."
        )
    if train.query_depth != model.query_depth or test.query_depth != model.query_depth:
        raise ValueError(
            "MODEL.q_depth and DATA.{train,test}.q_depth must be identical: "
            f"model={model.query_depth}, train={train.query_depth}, test={test.query_depth}."
        )
    if (train.image_width, train.image_height) != (test.image_width, test.image_height):
        raise ValueError("Train and test image dimensions must match in one solver run")
