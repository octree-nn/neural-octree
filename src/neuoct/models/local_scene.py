from __future__ import annotations

import torch
from torch import Tensor, nn

from neuoct.geometry.vertex_index import VertexIndex

from .intersection_renderer import SplitVoxelRendererModel
from .outputs import IntersectionPrediction


class LocalVertexFeatureField(nn.Module):
    """A learnable feature at each unique vertex of one fixed scene octree.

    The global model predicts voxel-corner features with an OCNN U-Net.  Local
    optimization instead keeps the octree fixed and optimizes this compact table
    directly, matching the representation used by ``NeuOct_local``.
    """

    def __init__(
        self,
        vertex_index: VertexIndex,
        feature_dim: int = 32,
        *,
        init_std: float = 0.01,
        voxel_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        if vertex_index.coordinates.ndim != 2 or vertex_index.coordinates.shape[1] != 4:
            raise ValueError("Local vertex coordinates must have shape (N, 4) = batch,x,y,z")
        if vertex_index.voxel_vertices.ndim != 2 or vertex_index.voxel_vertices.shape[1] != 8:
            raise ValueError("voxel_vertices must have shape (num_voxels, 8)")
        if feature_dim <= 0 or voxel_feature_dim < 0:
            raise ValueError("feature_dim must be positive and voxel_feature_dim non-negative")
        self.register_buffer("vertex_coordinates", vertex_index.coordinates.to(torch.int32))
        self.register_buffer("voxel_vertices", vertex_index.voxel_vertices.to(torch.long))
        self.features = nn.Parameter(
            torch.randn(vertex_index.num_vertices, feature_dim) * float(init_std)
        )
        if voxel_feature_dim:
            self.voxel_features = nn.Parameter(
                torch.randn(vertex_index.voxel_vertices.shape[0], voxel_feature_dim)
                * float(init_std)
            )
        else:
            self.register_parameter("voxel_features", None)

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_vertices(self) -> int:
        return int(self.features.shape[0])

    @property
    def voxel_feature_dim(self) -> int:
        return 0 if self.voxel_features is None else int(self.voxel_features.shape[1])

    def gather(self, voxel_indices: Tensor) -> Tensor:
        indices = self.voxel_vertices[voxel_indices.long()]
        return self.features[indices.reshape(-1, 2, 2, 2)]

    def gather_voxels(self, voxel_indices: Tensor) -> Tensor:
        if self.voxel_features is None:
            raise RuntimeError("This feature field has no voxel-local features")
        return self.voxel_features[voxel_indices.long()]

    def vertex_index(self) -> VertexIndex:
        return VertexIndex(self.vertex_coordinates, self.voxel_vertices)


class LocalSceneModel(nn.Module):
    """Renderer plus a learnable vertex field for one ShapeNet object."""

    def __init__(
        self,
        vertex_index: VertexIndex,
        *,
        feature_dim: int = 32,
        hidden_dim: int = 128,
        n_iters: int = 8,
        init_std: float = 0.01,
        geometry_only: bool = False,
        segment_position: bool = False,
        voxel_feature_dim: int = 0,
        position_encoding_frequencies: int = 0,
        sampling_mode: str = "recurrent",
    ) -> None:
        super().__init__()
        self.feature_field = LocalVertexFeatureField(
            vertex_index,
            feature_dim=feature_dim,
            init_std=init_std,
            voxel_feature_dim=voxel_feature_dim,
        )
        self.renderer_model = SplitVoxelRendererModel(
            f_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_iters=n_iters,
            geometry_only=geometry_only,
            segment_position=segment_position,
            voxel_feature_dim=voxel_feature_dim,
            position_encoding_frequencies=position_encoding_frequencies,
            sampling_mode=sampling_mode,
        )
        # Set when the renderer is compiled with CUDA graphs (``reduce-overhead``):
        # a compiled module that is invoked more than once per optimizer step must
        # be told where each step begins, otherwise the backward pass reads a
        # graph output that a later invocation has already overwritten.
        self.cudagraph_step_begin = False

    def forward(
        self,
        pins: Tensor,
        directions: Tensor,
        voxel_indices: Tensor,
        pouts: Tensor | None = None,
    ) -> IntersectionPrediction:
        if self.cudagraph_step_begin:
            torch.compiler.cudagraph_mark_step_begin()
        query_features = self.feature_field.gather(voxel_indices)
        voxel_features = (
            self.feature_field.gather_voxels(voxel_indices)
            if self.feature_field.voxel_feature_dim
            else None
        )
        return self.renderer_model(
            pins,
            directions,
            query_features,
            pouts,
            voxel_features=voxel_features,
        )
