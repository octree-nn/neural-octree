from __future__ import annotations

import ocnn
import torch
from torch import nn

from .feature_unet import FeatureUNet
from .intersection_renderer import SplitVoxelRendererModel
from .outputs import IntersectionPrediction, IntersectionStage
from .vertex_features import VertexFeatureAggregator, VertexFeatureField


class NeuralIntersectionModel(nn.Module):
    """Octree feature extractor plus shared-vertex iterative ray intersection model."""

    def __init__(
        self,
        in_channels: int,
        q_depth: int = 7,
        *,
        renderer_hidden_dim: int = 128,
        renderer_iterations: int = 3,
        segment_position: bool = False,
        voxel_feature_dim: int = 0,
        color_skip_weight: float = 0.0,
        activation_checkpointing: bool = False,
        material_fp32: bool = False,
        norm_type: str = "batch",
        norm_groups: int = 8,
        sampling_mode: str = "recurrent",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.q_depth = q_depth
        self.feature_model = FeatureUNet(
            in_channels,
            query_depth=q_depth,
            norm_type=norm_type,
            norm_groups=norm_groups,
            activation_checkpointing=activation_checkpointing,
        )
        self.voxel_feature_dim = int(voxel_feature_dim)
        self.color_skip_weight = float(color_skip_weight)
        self.sampling_mode = str(sampling_mode)
        self.voxel_feature_head = (
            nn.Linear(self.feature_model.output_channels, self.voxel_feature_dim)
            if self.voxel_feature_dim
            else None
        )
        self.renderer_model = SplitVoxelRendererModel(
            f_dim=self.feature_model.output_channels // 8,
            hidden_dim=renderer_hidden_dim,
            n_iters=renderer_iterations,
            segment_position=segment_position,
            voxel_feature_dim=self.voxel_feature_dim,
            color_skip_weight=self.color_skip_weight,
            activation_checkpointing=activation_checkpointing,
            material_fp32=material_fp32,
            sampling_mode=self.sampling_mode,
        )

    def _query_depth_colors(self, octree: object) -> torch.Tensor | None:
        if self.color_skip_weight <= 0:
            return None
        colors = octree.features[octree.depth]
        if colors is None:
            raise ValueError("Color skip requires point colors in the input octree")
        colors = colors.float()
        for depth in range(octree.depth, self.q_depth, -1):
            colors = ocnn.nn.octree_avg_pool(
                colors, octree, depth, "222", stride=2, nempty=True
            )
        return colors.clamp(0.0, 1.0)

    def extract_scene_features(
        self, data: torch.Tensor, octree: object, depth: int
    ) -> tuple[torch.Tensor, VertexFeatureField]:
        # OCNN 2.3 sparse operations do not consistently support autocast. Keep
        # the complete octree path in FP32 and allow AMP for the dense renderer.
        with torch.autocast(device_type=data.device.type, enabled=False):
            voxel_features = self.feature_model(data.float(), octree, depth)
            voxel_local_features = (
                None
                if self.voxel_feature_head is None
                else self.voxel_feature_head(voxel_features)
            )
            field = VertexFeatureAggregator.from_voxel_corner_features(
                octree,
                self.q_depth,
                voxel_features,
                voxel_local_features=voxel_local_features,
                voxel_colors=self._query_depth_colors(octree),
            )
        return voxel_features, field

    def _render_chunk(
        self,
        field: VertexFeatureField,
        pin: torch.Tensor,
        direction: torch.Tensor,
        voxel_indices: torch.Tensor,
        pout: torch.Tensor | None,
    ) -> IntersectionPrediction:
        query_features = field.gather(voxel_indices)
        voxel_features = (
            None
            if field.voxel_features is None
            else field.gather_voxels(voxel_indices)
        )
        base_colors = (
            None
            if field.voxel_colors is None
            else field.gather_colors(voxel_indices)
        )
        return self.renderer_model(
            pin,
            direction,
            query_features,
            pout,
            voxel_features=voxel_features,
            base_colors=base_colors,
        )

    def forward(
        self,
        data: torch.Tensor,
        octree: object,
        depth: int,
        pin: torch.Tensor,
        direction: torch.Tensor,
        voxel_indices: torch.Tensor,
        pout: torch.Tensor | None = None,
        chunk_size: int = 0,
    ) -> IntersectionPrediction:
        """Runs the scene U-Net once, then the per-candidate renderer.

        ``chunk_size`` splits the *candidate* dimension only.  Every renderer output
        row depends solely on its own candidate segment (the recurrent refinement
        uses only that segment's pin/pout/features), so the result is identical to
        an unchunked pass while the activation peak no longer scales with the whole
        object.  Coverage-complete mesh octrees carry ~1.6x more supervised
        candidates than the 8-view point clouds, which is what pushed the renderer
        over 24 GB on the heaviest Objaverse objects.
        """
        _, field = self.extract_scene_features(data, octree, depth)
        total = int(pin.shape[0])
        if chunk_size <= 0 or total <= chunk_size:
            return self._render_chunk(field, pin, direction, voxel_indices, pout)
        pieces: list[IntersectionPrediction] = []
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            pieces.append(
                self._render_chunk(
                    field,
                    pin[start:stop],
                    direction[start:stop],
                    voxel_indices[start:stop],
                    None if pout is None else pout[start:stop],
                )
            )
        stages = []
        for index in range(len(pieces[0].stages)):
            stages.append(
                IntersectionStage(
                    *[
                        torch.cat([piece.stages[index].as_tuple()[slot] for piece in pieces], dim=0)
                        for slot in range(4)
                    ]
                )
            )
        return IntersectionPrediction(tuple(stages))
