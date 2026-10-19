from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .outputs import IntersectionPrediction, IntersectionStage


def trilinear_interpolation(features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    dx, dy, dz = positions.unbind(dim=1)
    wx = torch.stack([1 - dx, dx], dim=1)
    wy = torch.stack([1 - dy, dy], dim=1)
    wz = torch.stack([1 - dz, dz], dim=1)
    weights = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    return (features * weights.unsqueeze(-1)).sum(dim=(1, 2, 3))


class ResidualFCLayer3(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.norm3 = nn.RMSNorm(dim)
        self.relu = nn.ReLU()
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.relu(self.norm1(self.fc1(inputs)))
        output = self.relu(self.norm2(self.fc2(output)))
        output = self.fc3(output) + inputs
        return self.relu(self.norm3(output))


class SplitVoxelRendererModel(nn.Module):
    def __init__(
        self,
        f_dim: int = 32,
        hidden_dim: int = 128,
        out_dim: int = 10,
        n_iters: int = 4,
        *,
        geometry_only: bool = False,
        segment_position: bool = False,
        voxel_feature_dim: int = 0,
        position_encoding_frequencies: int = 0,
        color_skip_weight: float = 0.0,
        activation_checkpointing: bool = False,
        material_fp32: bool = False,
        sampling_mode: str = "recurrent",
    ):
        super().__init__()
        del out_dim
        if voxel_feature_dim < 0 or position_encoding_frequencies < 0:
            raise ValueError("voxel feature dimension and position frequencies must be non-negative")
        if sampling_mode not in (
            "recurrent",
            "uniform",
            "single",
            "recurrent_endpoints",
            "uniform_endpoints",
            "uniform_concat",
            "uniform_concat_endpoints",
            "recurrent_pool",
            "single_endpoints",
        ):
            raise ValueError(
                "sampling_mode must be 'recurrent', 'uniform', 'single', "
                "'recurrent_endpoints', 'uniform_endpoints', 'uniform_concat', "
                "'uniform_concat_endpoints', 'recurrent_pool' or 'single_endpoints'"
            )
        self.n_iters = n_iters
        self.sampling_mode = sampling_mode
        self.geometry_only = bool(geometry_only)
        self.segment_position = bool(segment_position)
        self.voxel_feature_dim = int(voxel_feature_dim)
        self.position_encoding_frequencies = int(position_encoding_frequencies)
        if not 0.0 <= color_skip_weight <= 1.0:
            raise ValueError("color_skip_weight must be in [0, 1]")
        self.color_skip_weight = float(color_skip_weight)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.material_fp32 = bool(material_fp32)
        position_encoding_dim = 6 * self.position_encoding_frequencies
        scene_feature_dim = f_dim + self.voxel_feature_dim
        if self.concat_input:
            # Every sampled point contributes [feature, position encoding, position]
            # to one flat input; direction and the previous normal/hit are global.
            per_point_dim = scene_feature_dim + position_encoding_dim + 3
            if self.single_endpoints:
                point_blocks = 2
            else:
                point_blocks = self.n_iters + (2 if self.endpoint_context else 0)
            geometry_input_dim = point_blocks * per_point_dim + 3 + 3 + 1
            material_input_dim = point_blocks * per_point_dim + 3 + 3
        elif self.endpoint_context:
            # Entry feature/point, exit feature/point, query feature/point,
            # direction, previous normal and previous hit probability.
            geometry_input_dim = 3 * scene_feature_dim + 3 * position_encoding_dim + 9 + 3 + 3 + 1
            material_input_dim = 3 * scene_feature_dim + 3 * position_encoding_dim + 9 + 3 + 3
        else:
            geometry_input_dim = scene_feature_dim + position_encoding_dim + 3 + 3 + 3 + 1
            material_input_dim = scene_feature_dim + position_encoding_dim + 3 + 3 + 3
        self.geometry_embedding = nn.Linear(geometry_input_dim, hidden_dim)
        self.geometry_hidden2hidden = ResidualFCLayer3(hidden_dim)
        self.geometry_output_layer = nn.Linear(hidden_dim, 7)
        self.init_normal = nn.Parameter(torch.zeros((1, 3), dtype=torch.float))
        self.init_hit_prob = nn.Parameter(torch.zeros((1,), dtype=torch.float))
        self.init_geo_hidden = nn.Parameter(torch.zeros((1, hidden_dim), dtype=torch.float))
        if self.geometry_only:
            self.material_embedding = None
            self.material_hidden2hidden = None
            self.material_output_layer = None
            self.register_parameter("init_color", None)
            self.register_parameter("init_mat_hidden", None)
        else:
            self.material_embedding = nn.Linear(material_input_dim, hidden_dim)
            self.material_hidden2hidden = ResidualFCLayer3(hidden_dim)
            self.material_output_layer = nn.Linear(hidden_dim, 3)
            self.init_color = nn.Parameter(torch.zeros((1, 3), dtype=torch.float))
            self.init_mat_hidden = nn.Parameter(
                torch.zeros((1, hidden_dim), dtype=torch.float)
            )

    def get_scene_feature_for_points(
        self, vertex_features: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        return trilinear_interpolation(vertex_features, positions)

    def encode_position(self, positions: torch.Tensor) -> torch.Tensor:
        if self.position_encoding_frequencies == 0:
            return positions.new_empty((positions.shape[0], 0))
        frequencies = 2.0 ** torch.arange(
            self.position_encoding_frequencies,
            device=positions.device,
            dtype=positions.dtype,
        )
        phase = positions.unsqueeze(-1) * frequencies.view(1, 1, -1) * torch.pi
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(1)

    @property
    def recurrent(self) -> bool:
        return self.sampling_mode in ("recurrent", "recurrent_endpoints")

    @property
    def endpoint_context(self) -> bool:
        """Whether the entry and exit point are concatenated into every query."""
        return self.sampling_mode in (
            "recurrent_endpoints",
            "uniform_endpoints",
            "uniform_concat_endpoints",
            "single_endpoints",
        )

    @property
    def single_endpoints(self) -> bool:
        """Entry + exit only: one wide input, one network pass, no interior query."""
        return self.sampling_mode == "single_endpoints"

    @property
    def concat_input(self) -> bool:
        """Whether all sampled points are flattened into a single network input."""
        return self.sampling_mode in (
            "uniform_concat",
            "uniform_concat_endpoints",
            "single_endpoints",
        )

    @property
    def pool_recurrent(self) -> bool:
        """Adaptive query points fused by pooling instead of concatenation."""
        return self.sampling_mode == "recurrent_pool"

    def endpoint_query_positions(
        self, pin: torch.Tensor, pout: torch.Tensor | None
    ) -> torch.Tensor:
        """Fixed interior schedule for ``uniform_endpoints`` (t = k/(N+1))."""
        if pout is None:
            raise ValueError("pout is required when sampling_mode uses endpoints")
        steps = torch.arange(1, self.n_iters + 1, device=pin.device, dtype=pin.dtype)
        steps = steps / (self.n_iters + 1)
        offsets = (pout - pin).unsqueeze(1) * steps.view(1, -1, 1)
        return pin.unsqueeze(1) + offsets

    def sampled_query_positions(
        self, pin: torch.Tensor, pout: torch.Tensor | None
    ) -> torch.Tensor:
        """Query points for the fixed-resolution (non-recurrent) samplers.

        ``single`` uses only the voxel entry point.  ``uniform`` uses
        ``n_iters`` equidistant points between the voxel entry and exit,
        endpoints included, which gives the same sampling resolution as the
        recurrent renderer without adapting the query position.
        """
        if self.sampling_mode == "single":
            return pin.unsqueeze(1)
        if pout is None:
            raise ValueError("pout is required when sampling_mode is 'uniform'")
        steps = torch.linspace(0.0, 1.0, self.n_iters, device=pin.device, dtype=pin.dtype)
        offsets = (pout - pin).unsqueeze(1) * steps.view(1, -1, 1)
        return pin.unsqueeze(1) + offsets

    def _forward_sampled(
        self,
        pin: torch.Tensor,
        direction: torch.Tensor,
        vertex_features: torch.Tensor,
        pout: torch.Tensor | None,
        voxel_features: torch.Tensor | None,
        base_colors: torch.Tensor | None,
        batch_size: int,
    ) -> "IntersectionPrediction":
        """One-shot prediction pooled over a fixed set of segment query points.

        Every query is encoded independently with the same modules the
        recurrent renderer uses, the per-query hidden increments are averaged,
        and a single output head produces one stage.  No prediction feeds back
        into the query location, so this is the fixed-sampling baseline for the
        recurrent sampler at matched resolution and parameter count.
        """
        queries = self.sampled_query_positions(pin, pout)
        count = int(queries.shape[1])
        initial_normal = self.init_normal.expand(batch_size, -1)
        initial_hit = self.init_hit_prob.expand(batch_size)
        initial_color = (
            None if self.geometry_only else self.init_color.expand(batch_size, -1)
        )
        geometry_increment: torch.Tensor | None = None
        material_increment: torch.Tensor | None = None
        for index in range(count):
            position = queries[:, index]
            feature = self.get_scene_feature_for_points(vertex_features, position)
            if self.voxel_feature_dim:
                if voxel_features is None:
                    raise ValueError("voxel_features are required by this renderer")
                if voxel_features.shape != (batch_size, self.voxel_feature_dim):
                    raise ValueError(
                        "voxel_features shape mismatch: expected "
                        f"({batch_size}, {self.voxel_feature_dim}), got "
                        f"{tuple(voxel_features.shape)}"
                    )
                feature = torch.cat([feature, voxel_features], dim=1)
            encoded_position = self.encode_position(position)
            with torch.autocast(device_type=feature.device.type, enabled=False):
                geometry_input = torch.cat(
                    [
                        feature.float(),
                        encoded_position.float(),
                        direction.float(),
                        position.float(),
                        initial_normal.float(),
                        initial_hit.float().unsqueeze(-1),
                    ],
                    dim=1,
                )
                increment = self.geometry_hidden2hidden(
                    self.geometry_embedding(geometry_input)
                )
                geometry_increment = (
                    increment if geometry_increment is None else geometry_increment + increment
                )
            if initial_color is not None:
                if (
                    self.material_embedding is None
                    or self.material_hidden2hidden is None
                    or self.material_output_layer is None
                ):
                    raise RuntimeError("Material branch is unavailable in color mode")
                precision_context = (
                    torch.autocast(device_type=feature.device.type, enabled=False)
                    if self.material_fp32
                    else nullcontext()
                )
                with precision_context:
                    material_input = torch.cat(
                        [feature, encoded_position, direction, position, initial_color],
                        dim=1,
                    )
                    material_values = self.material_hidden2hidden(
                        self.material_embedding(material_input)
                    )
                    material_increment = (
                        material_values
                        if material_increment is None
                        else material_increment + material_values
                    )
        if geometry_increment is None:
            raise RuntimeError("Fixed sampling produced no query points")
        with torch.autocast(device_type=pin.device.type, enabled=False):
            geometry_hidden = (
                self.init_geo_hidden.expand(batch_size, -1)
                + geometry_increment.float() / count
            )
            geometry_output = self.geometry_output_layer(geometry_hidden)
            raw_position = torch.sigmoid(geometry_output[:, :3])
            normal = F.normalize(geometry_output[:, 3:6], dim=1, eps=1.0e-6)
            hit_probability = torch.sigmoid(geometry_output[:, 6])
        if self.segment_position:
            if pout is None:
                raise RuntimeError("pout unexpectedly unavailable")
            position = pin + raw_position[:, :1] * (pout - pin)
        else:
            position = pin + direction * (
                (raw_position - pin) * direction
            ).sum(dim=1, keepdim=True)
        if self.geometry_only:
            color = pin.new_zeros((batch_size, 3))
        else:
            if material_increment is None:
                raise RuntimeError("Material branch produced no query points")
            material_hidden = (
                self.init_mat_hidden.expand(batch_size, -1)
                + material_increment / count
            )
            precision_context = (
                torch.autocast(device_type=feature.device.type, enabled=False)
                if self.material_fp32
                else nullcontext()
            )
            with precision_context:
                if self.material_fp32:
                    material_hidden = material_hidden.float()
                learned_color = torch.sigmoid(
                    self.material_output_layer(material_hidden)
                )
                color = (
                    learned_color
                    if self.color_skip_weight <= 0
                    else learned_color * (1.0 - self.color_skip_weight)
                    + base_colors.to(learned_color) * self.color_skip_weight
                )
        return IntersectionPrediction(
            (IntersectionStage(position, normal, color, hit_probability),)
        )

    def _forward_endpoint_chain(
        self,
        pin: torch.Tensor,
        direction: torch.Tensor,
        vertex_features: torch.Tensor,
        pout: torch.Tensor | None,
        voxel_features: torch.Tensor | None,
        base_colors: torch.Tensor | None,
        batch_size: int,
    ) -> IntersectionPrediction:
        """Chain renderer whose every query also sees the voxel entry and exit.

        ``recurrent_endpoints`` starts the intermediate query at the segment
        midpoint and feeds each stage's predicted position back as the next
        query.  ``uniform_endpoints`` keeps the identical input layout and
        hidden-state chain but fixes the intermediate queries to
        ``t = k/(N+1)``, so it isolates adaptivity from the endpoint context.
        """
        if pout is None:
            raise ValueError("pout is required when sampling_mode uses endpoints")
        entry_feature = self.get_scene_feature_for_points(vertex_features, pin)
        exit_feature = self.get_scene_feature_for_points(vertex_features, pout)
        if self.voxel_feature_dim:
            if voxel_features is None:
                raise ValueError("voxel_features are required by this renderer")
            if voxel_features.shape != (batch_size, self.voxel_feature_dim):
                raise ValueError(
                    "voxel_features shape mismatch: expected "
                    f"({batch_size}, {self.voxel_feature_dim}), got "
                    f"{tuple(voxel_features.shape)}"
                )
            entry_feature = torch.cat([entry_feature, voxel_features], dim=1)
            exit_feature = torch.cat([exit_feature, voxel_features], dim=1)
        encoded_entry = self.encode_position(pin)
        encoded_exit = self.encode_position(pout)
        fixed_queries = None if self.recurrent else self.endpoint_query_positions(pin, pout)
        position = (pin + pout) * 0.5 if self.recurrent else fixed_queries[:, 0]
        normal = self.init_normal.expand(batch_size, -1)
        color = (
            pin.new_zeros((batch_size, 3))
            if self.geometry_only
            else self.init_color.expand(batch_size, -1)
        )
        hit_probability = self.init_hit_prob.expand(batch_size)
        geometry_hidden = self.init_geo_hidden.expand(batch_size, -1)
        material_hidden = (
            None
            if self.geometry_only
            else self.init_mat_hidden.expand(batch_size, -1)
        )
        stages: list[IntersectionStage] = []
        for index in range(self.n_iters):
            if fixed_queries is not None:
                position = fixed_queries[:, index]
            query_feature = self.get_scene_feature_for_points(vertex_features, position)
            if self.voxel_feature_dim:
                query_feature = torch.cat([query_feature, voxel_features], dim=1)
            encoded_position = self.encode_position(position)

            def iteration_step(
                position: torch.Tensor,
                normal: torch.Tensor,
                color: torch.Tensor,
                hit_probability: torch.Tensor,
                geometry_hidden: torch.Tensor,
                material_hidden: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                with torch.autocast(device_type=query_feature.device.type, enabled=False):
                    geometry_input = torch.cat(
                        [
                            entry_feature.float(),
                            exit_feature.float(),
                            query_feature.float(),
                            encoded_entry.float(),
                            encoded_exit.float(),
                            encoded_position.float(),
                            pin.float(),
                            pout.float(),
                            position.float(),
                            direction.float(),
                            normal.float(),
                            hit_probability.float().unsqueeze(-1),
                        ],
                        dim=1,
                    )
                    geometry_hidden = geometry_hidden.float() + self.geometry_hidden2hidden(
                        self.geometry_embedding(geometry_input)
                    )
                    geometry_output = self.geometry_output_layer(geometry_hidden)
                    raw_position = torch.sigmoid(geometry_output[:, :3])
                    normal = F.normalize(geometry_output[:, 3:6], dim=1, eps=1.0e-6)
                    hit_probability = torch.sigmoid(geometry_output[:, 6])
                if self.segment_position:
                    position = pin + raw_position[:, :1] * (pout - pin)
                else:
                    position = pin + direction * (
                        (raw_position - pin) * direction
                    ).sum(dim=1, keepdim=True)
                if not self.geometry_only:
                    if (
                        self.material_embedding is None
                        or self.material_hidden2hidden is None
                        or self.material_output_layer is None
                    ):
                        raise RuntimeError("Material branch is unavailable in color mode")
                    precision_context = (
                        torch.autocast(device_type=query_feature.device.type, enabled=False)
                        if self.material_fp32
                        else nullcontext()
                    )
                    with precision_context:
                        material_values = [
                            entry_feature,
                            exit_feature,
                            query_feature,
                            encoded_entry,
                            encoded_exit,
                            encoded_position,
                            pin,
                            pout,
                            position,
                            direction,
                            color,
                        ]
                        if self.material_fp32:
                            material_values = [value.float() for value in material_values]
                            material_hidden = material_hidden.float()
                        material_input = torch.cat(material_values, dim=1)
                        material_hidden = material_hidden + self.material_hidden2hidden(
                            self.material_embedding(material_input)
                        )
                        learned_color = torch.sigmoid(
                            self.material_output_layer(material_hidden)
                        )
                        color = (
                            learned_color
                            if self.color_skip_weight <= 0
                            else learned_color * (1.0 - self.color_skip_weight)
                            + base_colors.to(learned_color) * self.color_skip_weight
                        )
                return (
                    position,
                    normal,
                    color,
                    hit_probability,
                    geometry_hidden,
                    material_hidden,
                )

            if self.activation_checkpointing and material_hidden is None:
                raise RuntimeError("Activation checkpointing requires the material branch")
            inputs = (
                position,
                normal,
                color,
                hit_probability,
                geometry_hidden,
                material_hidden,
            )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                outputs = checkpoint(iteration_step, *inputs, use_reentrant=False)
            else:
                outputs = iteration_step(*inputs)
            (
                position,
                normal,
                color,
                hit_probability,
                geometry_hidden,
                material_hidden,
            ) = outputs
            stages.append(IntersectionStage(position, normal, color, hit_probability))
        return IntersectionPrediction(tuple(stages))

    def _forward_uniform_concat(
        self,
        pin: torch.Tensor,
        direction: torch.Tensor,
        vertex_features: torch.Tensor,
        pout: torch.Tensor | None,
        voxel_features: torch.Tensor | None,
        base_colors: torch.Tensor | None,
        batch_size: int,
    ) -> IntersectionPrediction:
        """Flatten every sampled point into one input and run the MLP once.

        Unlike ``uniform`` (each point encoded independently, hidden states
        averaged, one readout) this concatenates all point features and
        coordinates into a single wide vector, so the very first linear layer
        already mixes information across the sampled points.  ``n_iters``
        points are used; with ``uniform_concat_endpoints`` the entry and exit
        blocks are prepended and the remaining points sit on the interior grid
        ``t = k/(N+1)``.  ``single_endpoints`` is the degenerate two-point case:
        the entry and exit blocks only, one network pass, no interior query.
        """
        if self.single_endpoints:
            queries = None
            count = 0
        elif self.endpoint_context:
            queries = self.endpoint_query_positions(pin, pout)
            count = int(queries.shape[1])
        else:
            queries = self.sampled_query_positions(pin, pout)
            count = int(queries.shape[1])
        blocks: list[torch.Tensor] = []

        def point_features(position: torch.Tensor) -> torch.Tensor:
            feature = self.get_scene_feature_for_points(vertex_features, position)
            if self.voxel_feature_dim:
                if voxel_features is None:
                    raise ValueError("voxel_features are required by this renderer")
                if voxel_features.shape != (batch_size, self.voxel_feature_dim):
                    raise ValueError(
                        "voxel_features shape mismatch: expected "
                        f"({batch_size}, {self.voxel_feature_dim}), got "
                        f"{tuple(voxel_features.shape)}"
                    )
                feature = torch.cat([feature, voxel_features], dim=1)
            return feature

        if self.endpoint_context:
            if pout is None:
                raise ValueError("pout is required when sampling_mode uses endpoints")
            blocks.extend([point_features(pin), self.encode_position(pin), pin])
            blocks.extend([point_features(pout), self.encode_position(pout), pout])
        for index in range(count):
            position = queries[:, index]
            blocks.extend(
                [point_features(position), self.encode_position(position), position]
            )
        normal = self.init_normal.expand(batch_size, -1)
        hit_probability = self.init_hit_prob.expand(batch_size)
        with torch.autocast(device_type=pin.device.type, enabled=False):
            geometry_input = torch.cat(
                [value.float() for value in blocks]
                + [
                    direction.float(),
                    normal.float(),
                    hit_probability.float().unsqueeze(-1),
                ],
                dim=1,
            )
            geometry_hidden = (
                self.init_geo_hidden.expand(batch_size, -1)
                + self.geometry_hidden2hidden(self.geometry_embedding(geometry_input))
            )
            geometry_output = self.geometry_output_layer(geometry_hidden)
            raw_position = torch.sigmoid(geometry_output[:, :3])
            normal = F.normalize(geometry_output[:, 3:6], dim=1, eps=1.0e-6)
            hit_probability = torch.sigmoid(geometry_output[:, 6])
        if self.segment_position:
            if pout is None:
                raise RuntimeError("pout unexpectedly unavailable")
            position = pin + raw_position[:, :1] * (pout - pin)
        else:
            position = pin + direction * (
                (raw_position - pin) * direction
            ).sum(dim=1, keepdim=True)
        if self.geometry_only:
            color = pin.new_zeros((batch_size, 3))
        else:
            if (
                self.material_embedding is None
                or self.material_hidden2hidden is None
                or self.material_output_layer is None
            ):
                raise RuntimeError("Material branch is unavailable in color mode")
            color = self.init_color.expand(batch_size, -1)
            precision_context = (
                torch.autocast(device_type=pin.device.type, enabled=False)
                if self.material_fp32
                else nullcontext()
            )
            with precision_context:
                material_values = list(blocks) + [direction, color]
                if self.material_fp32:
                    material_values = [value.float() for value in material_values]
                material_input = torch.cat(material_values, dim=1)
                material_hidden = self.init_mat_hidden.expand(
                    batch_size, -1
                ) + self.material_hidden2hidden(self.material_embedding(material_input))
                learned_color = torch.sigmoid(
                    self.material_output_layer(material_hidden)
                )
                color = (
                    learned_color
                    if self.color_skip_weight <= 0
                    else learned_color * (1.0 - self.color_skip_weight)
                    + base_colors.to(learned_color) * self.color_skip_weight
                )
        return IntersectionPrediction(
            (IntersectionStage(position, normal, color, hit_probability),)
        )

    def _forward_recurrent_pool(
        self,
        pin: torch.Tensor,
        direction: torch.Tensor,
        vertex_features: torch.Tensor,
        pout: torch.Tensor | None,
        voxel_features: torch.Tensor | None,
        base_colors: torch.Tensor | None,
        batch_size: int,
    ) -> IntersectionPrediction:
        """Pooled fusion with adaptive query points.

        The entry and exit are ordinary members of a pooled set, encoded
        independently with the *same* 26-dim per-point input that the
        ``uniform`` sampler uses.  ``n_iters`` further members are placed
        adaptively: the first at the segment midpoint, each following one at
        the position read out from the running pool.  All members are averaged
        in hidden space and a single output head produces one stage, so this is
        architecturally identical to ``uniform`` and differs only in whether
        the interior query points are fixed or adaptive.
        """
        if pout is None:
            raise ValueError("pout is required when sampling_mode is 'recurrent_pool'")
        initial_normal = self.init_normal.expand(batch_size, -1)
        initial_hit = self.init_hit_prob.expand(batch_size)
        initial_color = (
            None if self.geometry_only else self.init_color.expand(batch_size, -1)
        )

        def encode_point(
            position: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            feature = self.get_scene_feature_for_points(vertex_features, position)
            if self.voxel_feature_dim:
                if voxel_features is None:
                    raise ValueError("voxel_features are required by this renderer")
                if voxel_features.shape != (batch_size, self.voxel_feature_dim):
                    raise ValueError(
                        "voxel_features shape mismatch: expected "
                        f"({batch_size}, {self.voxel_feature_dim}), got "
                        f"{tuple(voxel_features.shape)}"
                    )
                feature = torch.cat([feature, voxel_features], dim=1)
            encoded_position = self.encode_position(position)
            with torch.autocast(device_type=feature.device.type, enabled=False):
                geometry_input = torch.cat(
                    [
                        feature.float(),
                        encoded_position.float(),
                        direction.float(),
                        position.float(),
                        initial_normal.float(),
                        initial_hit.float().unsqueeze(-1),
                    ],
                    dim=1,
                )
                geometry_increment = self.geometry_hidden2hidden(
                    self.geometry_embedding(geometry_input)
                )
            material_increment = None
            if initial_color is not None:
                if (
                    self.material_embedding is None
                    or self.material_hidden2hidden is None
                ):
                    raise RuntimeError("Material branch is unavailable in color mode")
                precision_context = (
                    torch.autocast(device_type=feature.device.type, enabled=False)
                    if self.material_fp32
                    else nullcontext()
                )
                with precision_context:
                    material_values = [
                        feature,
                        encoded_position,
                        direction,
                        position,
                        initial_color,
                    ]
                    if self.material_fp32:
                        material_values = [value.float() for value in material_values]
                    material_input = torch.cat(material_values, dim=1)
                    material_increment = self.material_hidden2hidden(
                        self.material_embedding(material_input)
                    )
            return geometry_increment, material_increment

        def encoded(position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            """``encode_point`` under activation checkpointing when it is enabled.

            The pooled sampler keeps every query's activations alive until the pooled
            readout, so on the coverage-complete Objaverse octrees its peak exceeded
            24 GB while the chained sampler fit.  Recomputing each per-point encoder in
            the backward pass is numerically identical and bounds that peak, matching
            what the other sampler paths already do.
            """
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                if self.geometry_only:
                    raise RuntimeError(
                        "Activation checkpointing requires the material branch"
                    )
                return checkpoint(encode_point, position, use_reentrant=False)
            return encode_point(position)

        def readout(
            geometry_sum: torch.Tensor,
            material_sum: torch.Tensor | None,
            count: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            with torch.autocast(device_type=pin.device.type, enabled=False):
                geometry_hidden = (
                    self.init_geo_hidden.expand(batch_size, -1)
                    + geometry_sum.float() / count
                )
                geometry_output = self.geometry_output_layer(geometry_hidden)
                raw_position = torch.sigmoid(geometry_output[:, :3])
                normal = F.normalize(geometry_output[:, 3:6], dim=1, eps=1.0e-6)
                hit_probability = torch.sigmoid(geometry_output[:, 6])
            if self.segment_position:
                position = pin + raw_position[:, :1] * (pout - pin)
            else:
                position = pin + direction * (
                    (raw_position - pin) * direction
                ).sum(dim=1, keepdim=True)
            if material_sum is None:
                color = pin.new_zeros((batch_size, 3))
                return position, normal, hit_probability, color
            if self.material_output_layer is None:
                raise RuntimeError("Material branch is unavailable in color mode")
            precision_context = (
                torch.autocast(device_type=pin.device.type, enabled=False)
                if self.material_fp32
                else nullcontext()
            )
            with precision_context:
                material_hidden = (
                    self.init_mat_hidden.expand(batch_size, -1) + material_sum / count
                )
                if self.material_fp32:
                    material_hidden = material_hidden.float()
                learned_color = torch.sigmoid(self.material_output_layer(material_hidden))
                color = (
                    learned_color
                    if self.color_skip_weight <= 0
                    else learned_color * (1.0 - self.color_skip_weight)
                    + base_colors.to(learned_color) * self.color_skip_weight
                )
            return position, normal, hit_probability, color

        geometry_sum, material_sum = encoded(pin)
        exit_geometry, exit_material = encoded(pout)
        geometry_sum = geometry_sum + exit_geometry
        if material_sum is not None and exit_material is not None:
            material_sum = material_sum + exit_material
        count = 2
        # First adaptive query: segment midpoint.  Each later query is placed at
        # the position read out from the pool accumulated so far.
        position = (pin + pout) * 0.5
        normal = self.init_normal.expand(batch_size, -1)
        hit_probability = self.init_hit_prob.expand(batch_size)
        color = (
            pin.new_zeros((batch_size, 3))
            if self.geometry_only
            else self.init_color.expand(batch_size, -1)
        )
        for _ in range(self.n_iters):
            query_geometry, query_material = encoded(position)
            geometry_sum = geometry_sum + query_geometry
            if material_sum is not None and query_material is not None:
                material_sum = material_sum + query_material
            count += 1
            position, normal, hit_probability, color = readout(
                geometry_sum, material_sum, count
            )
        return IntersectionPrediction(
            (IntersectionStage(position, normal, color, hit_probability),)
        )

    def forward(
        self,
        pin: torch.Tensor,
        direction: torch.Tensor,
        vertex_features: torch.Tensor,
        pout: torch.Tensor | None = None,
        voxel_features: torch.Tensor | None = None,
        base_colors: torch.Tensor | None = None,
    ) -> IntersectionPrediction:
        if self.segment_position and pout is None:
            raise ValueError("pout is required when segment_position is enabled")
        batch_size = pin.shape[0]
        if self.color_skip_weight > 0:
            if base_colors is None or base_colors.shape != (batch_size, 3):
                raise ValueError(
                    "base_colors with shape (batch_size, 3) are required by color skip"
                )
        if self.pool_recurrent:
            return self._forward_recurrent_pool(
                pin,
                direction,
                vertex_features,
                pout,
                voxel_features,
                base_colors,
                batch_size,
            )
        if self.concat_input:
            return self._forward_uniform_concat(
                pin,
                direction,
                vertex_features,
                pout,
                voxel_features,
                base_colors,
                batch_size,
            )
        if self.endpoint_context:
            return self._forward_endpoint_chain(
                pin,
                direction,
                vertex_features,
                pout,
                voxel_features,
                base_colors,
                batch_size,
            )
        if not self.recurrent:
            return self._forward_sampled(
                pin,
                direction,
                vertex_features,
                pout,
                voxel_features,
                base_colors,
                batch_size,
            )
        position = pin
        normal = self.init_normal.expand(batch_size, -1)
        color = (
            pin.new_zeros((batch_size, 3))
            if self.geometry_only
            else self.init_color.expand(batch_size, -1)
        )
        hit_probability = self.init_hit_prob.expand(batch_size)
        geometry_hidden = self.init_geo_hidden.expand(batch_size, -1)
        material_hidden = (
            None
            if self.geometry_only
            else self.init_mat_hidden.expand(batch_size, -1)
        )
        stages: list[IntersectionStage] = []
        for _ in range(self.n_iters):
            def iteration_step(
                position: torch.Tensor,
                normal: torch.Tensor,
                color: torch.Tensor,
                hit_probability: torch.Tensor,
                geometry_hidden: torch.Tensor,
                material_hidden: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                feature = self.get_scene_feature_for_points(vertex_features, position)
                if self.voxel_feature_dim:
                    if voxel_features is None:
                        raise ValueError("voxel_features are required by this renderer")
                    if voxel_features.shape != (batch_size, self.voxel_feature_dim):
                        raise ValueError(
                            "voxel_features shape mismatch: expected "
                            f"({batch_size}, {self.voxel_feature_dim}), got "
                            f"{tuple(voxel_features.shape)}"
                        )
                    feature = torch.cat([feature, voxel_features], dim=1)
                encoded_position = self.encode_position(position)
                # The geometry head is small but numerically sensitive.  Keep
                # it in FP32 even when the outer solver enables FP16 AMP:
                # half-precision linear overflow and zero-norm normalization
                # can otherwise create NaN normals and hit probabilities.
                with torch.autocast(device_type=feature.device.type, enabled=False):
                    geometry_input = torch.cat(
                        [
                            feature.float(),
                            encoded_position.float(),
                            direction.float(),
                            position.float(),
                            normal.float(),
                            hit_probability.float().unsqueeze(-1),
                        ],
                        dim=1,
                    )
                    geometry_hidden = geometry_hidden.float() + self.geometry_hidden2hidden(
                        self.geometry_embedding(geometry_input)
                    )
                    geometry_output = self.geometry_output_layer(geometry_hidden)
                    raw_position = torch.sigmoid(geometry_output[:, :3])
                    normal = F.normalize(
                        geometry_output[:, 3:6], dim=1, eps=1.0e-6
                    )
                    hit_probability = torch.sigmoid(geometry_output[:, 6])
                if self.segment_position:
                    if pout is None:
                        raise RuntimeError("pout unexpectedly unavailable")
                    position = pin + raw_position[:, :1] * (pout - pin)
                else:
                    position = pin + direction * (
                        (raw_position - pin) * direction
                    ).sum(dim=1, keepdim=True)
                if not self.geometry_only:
                    if (
                        self.material_embedding is None
                        or self.material_hidden2hidden is None
                        or self.material_output_layer is None
                    ):
                        raise RuntimeError("Material branch is unavailable in color mode")
                    precision_context = (
                        torch.autocast(device_type=feature.device.type, enabled=False)
                        if self.material_fp32
                        else nullcontext()
                    )
                    with precision_context:
                        material_values = [
                            feature,
                            encoded_position,
                            direction,
                            position,
                            color,
                        ]
                        if self.material_fp32:
                            material_values = [value.float() for value in material_values]
                            material_hidden = material_hidden.float()
                        material_input = torch.cat(material_values, dim=1)
                        material_hidden = material_hidden + self.material_hidden2hidden(
                            self.material_embedding(material_input)
                        )
                        learned_color = torch.sigmoid(
                            self.material_output_layer(material_hidden)
                        )
                        color = (
                            learned_color
                            if self.color_skip_weight <= 0
                            else learned_color * (1.0 - self.color_skip_weight)
                            + base_colors.to(learned_color) * self.color_skip_weight
                        )
                return (
                    position,
                    normal,
                    color,
                    hit_probability,
                    geometry_hidden,
                    material_hidden,
                )

            if self.activation_checkpointing and material_hidden is None:
                raise RuntimeError("Activation checkpointing requires the material branch")
            inputs = (
                position,
                normal,
                color,
                hit_probability,
                geometry_hidden,
                material_hidden,
            )
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                outputs = checkpoint(iteration_step, *inputs, use_reentrant=False)
            else:
                outputs = iteration_step(*inputs)
            (
                position,
                normal,
                color,
                hit_probability,
                geometry_hidden,
                material_hidden,
            ) = outputs
            stages.append(IntersectionStage(position, normal, color, hit_probability))
        return IntersectionPrediction(tuple(stages))
