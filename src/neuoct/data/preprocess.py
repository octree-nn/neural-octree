from __future__ import annotations

from collections import OrderedDict

import torch
from ocnn.octree import Points

from neuoct.config.schema import DataConfig
from neuoct.geometry.octree_adapter import NeuOctOctree, merge_octree_batch
from neuoct.geometry.ray_traversal import RayVoxelPairs, query_supervised_pairs

from .camera import unproject_rgbd
from .sample import PreparedBatch, RawSample


class DeviceBatchPreprocessor:
    """Builds octrees and ray supervision on the selected accelerator."""

    def __init__(self, config: DataConfig, device: torch.device | str):
        self.config = config
        self.device = torch.device(device)
        self._octree_cache: OrderedDict[str, NeuOctOctree] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def _build_octree(self, sample: RawSample) -> NeuOctOctree:
        cache_key = sample.entry.object_key
        if cache_key in self._octree_cache:
            self.cache_hits += 1
            self._octree_cache.move_to_end(cache_key)
            return self._octree_cache[cache_key]
        self.cache_misses += 1
        positions = sample.point_positions.to(self.device, non_blocking=True)
        colors = sample.point_colors.to(self.device, non_blocking=True)
        points = Points(positions, features=colors)
        points.points = points.points * self.config.point_scale
        octree = NeuOctOctree(
            depth=self.config.octree_depth,
            full_depth=self.config.full_depth,
            device=self.device,
        )
        octree.build_octree(points)
        octree.construct_all_neigh()
        if self.config.octree_cache_size > 0:
            self._octree_cache[cache_key] = octree
            self._octree_cache.move_to_end(cache_key)
            while len(self._octree_cache) > self.config.octree_cache_size:
                self._octree_cache.popitem(last=False)
        return octree

    def _supervision(self, sample: RawSample, octree: NeuOctOctree, batch_id: int) -> RayVoxelPairs:
        camera_to_world = sample.camera_to_world.to(self.device, non_blocking=True)
        depth = sample.depth_image.to(self.device, non_blocking=True)
        normal = sample.normal_image.to(self.device, non_blocking=True)
        color = sample.color_image.to(self.device, non_blocking=True)
        valid = sample.valid_depth.to(self.device, non_blocking=True).reshape(-1)
        origins, directions, points, normals, colors = unproject_rgbd(
            depth,
            normal,
            color,
            sample.intrinsics,
            camera_to_world,
            convention=self.config.camera_convention,
        )
        world_scale = self.config.point_scale * self.config.ray_scale
        scale = 1 << self.config.query_depth
        half_scale = scale >> 1
        points = points * world_scale * scale + half_scale
        origins = origins * world_scale * scale + half_scale
        intersections = torch.cat([valid.float().unsqueeze(1), points, normals, colors], dim=1)
        label_offset = batch_id * self.config.image_height * self.config.image_width
        return query_supervised_pairs(
            origins,
            directions,
            intersections,
            octree,
            scale=scale,
            depth=self.config.query_depth,
            batch_id=batch_id,
            ray_label_offset=label_offset,
        )

    def build_octree(self, sample: RawSample) -> NeuOctOctree:
        """Build (or retrieve) the octree used by an object optimization run."""
        return self._build_octree(sample)

    def prepare_sample(
        self, sample: RawSample, octree: NeuOctOctree, *, batch_id: int = 0
    ) -> RayVoxelPairs:
        """Build supervised ray/voxel pairs against an already fixed octree."""
        return self._supervision(sample, octree, batch_id)

    def __call__(self, samples: list[RawSample]) -> PreparedBatch:
        if not samples:
            raise ValueError("At least one raw sample is required")
        trees = [self._build_octree(sample) for sample in samples]
        octree = merge_octree_batch(trees) if len(trees) > 1 else trees[0]
        if len(trees) > 1:
            octree.construct_all_neigh()
        pair_batches = [
            self._supervision(sample, octree, batch_id) for batch_id, sample in enumerate(samples)
        ]
        pairs = RayVoxelPairs.concatenate(pair_batches)
        return PreparedBatch(
            octree=octree,
            pins=pairs.pins,
            pouts=pairs.pouts,
            voxel_indices=pairs.voxel_indices,
            hits=pairs.hits,
            positions=pairs.positions,
            normals=pairs.normals,
            colors=pairs.colors,
            ray_labels=pairs.ray_labels,
            batch_size=len(samples),
            image_height=self.config.image_height,
            image_width=self.config.image_width,
            entries=tuple(sample.entry for sample in samples),
        )
