from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from neuoct.data.camera import CameraIntrinsics, generate_rays
from neuoct.geometry.ray_traversal import enumerate_ray_voxel_pairs, grouped_interleave_indices
from neuoct.models.intersection_renderer import SplitVoxelRendererModel
from neuoct.models.vertex_features import VertexFeatureField

from .output import RenderOutput


class SparseRenderer:
    def __init__(
        self,
        renderer_model: SplitVoxelRendererModel,
        feature_field: VertexFeatureField,
        octree: object,
        *,
        depth: int,
        point_scale: float = 0.99,
        ray_scale: float = 0.5,
        camera_convention: str = "neuoct",
        max_batch_size: int = 262144,
        hit_threshold: float = 0.5,
        orient_normals_to_camera: bool = False,
        supersample_factor: int = 1,
        supersample_hit_fraction: float = 0.5,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        # 0.0 is allowed as the inclusive lower boundary: with
        # `hit = hit_probability > hit_threshold` it accepts every traced ray,
        # which is used for hit-classifier boundary diagnostics.
        if not 0.0 <= hit_threshold < 1.0:
            raise ValueError("hit_threshold must be in [0, 1)")
        if supersample_factor <= 0:
            raise ValueError("supersample_factor must be positive")
        if not 0.0 < supersample_hit_fraction <= 1.0:
            raise ValueError("supersample_hit_fraction must be in (0, 1]")
        self.renderer_model = renderer_model
        self.feature_field = feature_field
        self.octree = octree
        self.depth = depth
        self.scale = 1 << depth
        self.point_scale = point_scale
        self.ray_scale = ray_scale
        self.camera_convention = camera_convention
        self.max_batch_size = max_batch_size
        self.hit_threshold = hit_threshold
        self.orient_normals_to_camera = bool(orient_normals_to_camera)
        self.supersample_factor = int(supersample_factor)
        self.supersample_hit_fraction = float(supersample_hit_fraction)
        self.background_color = background_color

    @staticmethod
    def _first_per_ray(labels: Tensor) -> Tensor:
        if labels.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=labels.device)
        order = torch.argsort(labels, stable=True)
        sorted_labels = labels[order]
        first = torch.cat(
            [
                torch.ones(1, dtype=torch.bool, device=labels.device),
                sorted_labels[1:] != sorted_labels[:-1],
            ]
        )
        return order[first]

    @torch.no_grad()
    def render_frame(
        self, camera_to_world: Tensor, intrinsics: CameraIntrinsics, *, batch_id: int = 0
    ) -> RenderOutput:
        if self.supersample_factor == 1:
            return self._render_frame(camera_to_world, intrinsics, batch_id=batch_id)
        factor = self.supersample_factor
        high_resolution = CameraIntrinsics(
            width=intrinsics.width * factor,
            height=intrinsics.height * factor,
            fx=intrinsics.fx * factor,
            fy=intrinsics.fy * factor,
            cx=intrinsics.cx * factor,
            cy=intrinsics.cy * factor,
        )
        rendered = self._render_frame(camera_to_world, high_resolution, batch_id=batch_id)
        return self._downsample(rendered, factor, self.supersample_hit_fraction)

    @staticmethod
    def _downsample(
        rendered: RenderOutput, factor: int, hit_fraction: float = 0.5
    ) -> RenderOutput:
        if factor == 1:
            return rendered
        height, width = rendered.hit.shape
        if height % factor or width % factor:
            raise ValueError("Supersampled output dimensions must be divisible by factor")
        output_height = height // factor
        output_width = width // factor

        def blocks(value: Tensor) -> Tensor:
            trailing = value.shape[2:]
            return value.reshape(
                output_height,
                factor,
                output_width,
                factor,
                *trailing,
            )

        hit_blocks = blocks(rendered.hit).float()
        hit_count = hit_blocks.sum(dim=(1, 3))
        hit = hit_count >= (factor * factor * hit_fraction)
        weights = hit_blocks.unsqueeze(-1)
        denominator = hit_count.clamp_min(1.0).unsqueeze(-1)
        position = (blocks(rendered.position) * weights).sum(dim=(1, 3)) / denominator
        normal = (blocks(rendered.normal) * weights).sum(dim=(1, 3)) / denominator
        normal = F.normalize(normal, dim=-1, eps=1.0e-8)
        position = position * hit.unsqueeze(-1)
        normal = normal * hit.unsqueeze(-1)
        color = blocks(rendered.color).mean(dim=(1, 3))
        return RenderOutput(position, normal, color, hit, rendered.scale)

    @torch.no_grad()
    def _render_frame(
        self, camera_to_world: Tensor, intrinsics: CameraIntrinsics, *, batch_id: int = 0
    ) -> RenderOutput:
        device = self.feature_field.features.device
        camera_to_world = camera_to_world.to(device)
        origins, directions = generate_rays(
            intrinsics, camera_to_world, convention=self.camera_convention
        )
        origins = (
            origins * (self.point_scale * self.ray_scale) * self.scale + self.scale / 2
        )
        pairs = enumerate_ray_voxel_pairs(
            origins,
            directions,
            self.octree,
            scale=self.scale,
            depth=self.depth,
            batch_id=batch_id,
        )
        ray_count = intrinsics.height * intrinsics.width
        position_image = torch.zeros((ray_count, 3), dtype=torch.float32, device=device)
        normal_image = torch.zeros((ray_count, 3), dtype=torch.float32, device=device)
        color_image = (
            torch.tensor(self.background_color, device=device).expand(ray_count, 3).clone()
        )
        hit_image = torch.zeros(ray_count, dtype=torch.bool, device=device)
        if pairs.ray_labels.numel() == 0:
            return RenderOutput(
                position_image.reshape(intrinsics.height, intrinsics.width, 3),
                normal_image.reshape(intrinsics.height, intrinsics.width, 3),
                color_image.reshape(intrinsics.height, intrinsics.width, 3),
                hit_image.reshape(intrinsics.height, intrinsics.width),
                self.scale,
            )
        order = grouped_interleave_indices(pairs.ray_labels)
        pins = pairs.pins[order]
        pouts = pairs.pouts[order]
        voxel_indices = pairs.voxel_indices[order]
        labels = pairs.ray_labels[order]
        voxel_coordinates = pairs.voxel_coordinates[order]
        self.renderer_model.eval()
        while labels.numel() > 0:
            count = min(self.max_batch_size, labels.numel())
            selected = slice(0, count)
            query_features = self.feature_field.gather(voxel_indices[selected])
            voxel_features = None
            if int(getattr(self.renderer_model, "voxel_feature_dim", 0)):
                gather_voxels = getattr(self.feature_field, "gather_voxels", None)
                if gather_voxels is None:
                    raise RuntimeError(
                        "Renderer requires voxel-local features, but the feature field "
                        "does not provide gather_voxels()"
                    )
                voxel_features = gather_voxels(voxel_indices[selected])
            base_colors = None
            if float(getattr(self.renderer_model, "color_skip_weight", 0.0)) > 0:
                gather_colors = getattr(self.feature_field, "gather_colors", None)
                if gather_colors is None:
                    raise RuntimeError(
                        "Renderer requires voxel colors, but the feature field does not "
                        "provide gather_colors()"
                    )
                base_colors = gather_colors(voxel_indices[selected])
            query_directions = F.normalize(pouts[selected] - pins[selected], dim=1)
            prediction = self.renderer_model(
                pins[selected],
                query_directions,
                query_features,
                pouts[selected],
                voxel_features=voxel_features,
                base_colors=base_colors,
            ).final
            hit = prediction.hit_probability > self.hit_threshold
            if bool(hit.any()):
                hit_labels = labels[selected][hit]
                first = self._first_per_ray(hit_labels)
                accepted_labels = hit_labels[first]
                accepted_position = (
                    prediction.position[hit][first] + voxel_coordinates[selected][hit][first]
                )
                accepted_normal = prediction.normal[hit][first]
                if self.orient_normals_to_camera:
                    accepted_direction = query_directions[hit][first]
                    reverse = (accepted_normal * accepted_direction).sum(dim=1) > 0
                    accepted_normal = torch.where(
                        reverse.unsqueeze(-1), -accepted_normal, accepted_normal
                    )
                position_image[accepted_labels] = accepted_position
                normal_image[accepted_labels] = accepted_normal
                color_image[accepted_labels] = prediction.color[hit][first]
                hit_image[accepted_labels] = True
            pins = pins[count:]
            pouts = pouts[count:]
            voxel_indices = voxel_indices[count:]
            labels = labels[count:]
            voxel_coordinates = voxel_coordinates[count:]
            if bool(hit.any()) and labels.numel() > 0:
                keep = ~torch.isin(labels, accepted_labels)
                pins = pins[keep]
                pouts = pouts[keep]
                voxel_indices = voxel_indices[keep]
                labels = labels[keep]
                voxel_coordinates = voxel_coordinates[keep]
        return RenderOutput(
            position_image.reshape(intrinsics.height, intrinsics.width, 3),
            normal_image.reshape(intrinsics.height, intrinsics.width, 3),
            color_image.reshape(intrinsics.height, intrinsics.width, 3),
            hit_image.reshape(intrinsics.height, intrinsics.width),
            self.scale,
        )
