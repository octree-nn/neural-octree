from __future__ import annotations

from functools import partial

import ocnn
import torch
from torch.utils.checkpoint import checkpoint

from .octree_blocks import OctreeBnResBlock2, OctreeConvBnRelu, OctreeDeconvBnRelu


class FeatureUNet(torch.nn.Module):
    """Depth-9 checkpoint-compatible U-Net with a query-depth-derived decoder."""

    def __init__(
        self,
        in_channels: int,
        query_depth: int = 7,
        input_depth: int = 9,
        nempty: bool = True,
        norm_type: str = "batch",
        norm_groups: int = 8,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        if input_depth != 9 or query_depth not in (6, 7):
            raise ValueError(
                "The legacy-compatible U-Net supports input depth 9 and query depth 6/7"
            )
        self.in_channels = in_channels
        self.nempty = nempty
        self.input_depth = input_depth
        self.query_depth = query_depth
        self.norm_type = str(norm_type)
        self.norm_groups = int(norm_groups)
        self.encoder_channel = [32, 32, 64, 128, 256]
        self.encoder_blocks = [2, 3, 4, 6]
        self.encoder_stages = len(self.encoder_blocks)
        bottleneck_depth = input_depth - self.encoder_stages
        self.decoder_stages = query_depth - bottleneck_depth
        self.decoder_channel = [256, 256, 128][: self.decoder_stages + 1]
        self.decoder_blocks = [2] * self.decoder_stages
        self.bottleneck = 1
        self.resblk = partial(
            OctreeBnResBlock2,
            norm_type=self.norm_type,
            norm_groups=self.norm_groups,
        )

        self.conv1 = OctreeConvBnRelu(
            in_channels,
            self.encoder_channel[0],
            nempty=nempty,
            norm_type=self.norm_type,
            norm_groups=self.norm_groups,
        )
        self.downsample = torch.nn.ModuleList(
            [
                OctreeConvBnRelu(
                    self.encoder_channel[index],
                    self.encoder_channel[index + 1],
                    kernel_size=[2],
                    stride=2,
                    nempty=nempty,
                    norm_type=self.norm_type,
                    norm_groups=self.norm_groups,
                )
                for index in range(self.encoder_stages)
            ]
        )
        self.encoder = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeResBlocks(
                    self.encoder_channel[index + 1],
                    self.encoder_channel[index + 1],
                    self.encoder_blocks[index],
                    self.bottleneck,
                    nempty,
                    self.resblk,
                )
                for index in range(self.encoder_stages)
            ]
        )
        decoder_inputs = [
            self.decoder_channel[index + 1] + self.encoder_channel[-index - 2]
            for index in range(self.decoder_stages)
        ]
        self.upsample = torch.nn.ModuleList(
            [
                OctreeDeconvBnRelu(
                    self.decoder_channel[index],
                    self.decoder_channel[index + 1],
                    kernel_size=[2],
                    stride=2,
                    nempty=nempty,
                    norm_type=self.norm_type,
                    norm_groups=self.norm_groups,
                )
                for index in range(self.decoder_stages)
            ]
        )
        self.decoder = torch.nn.ModuleList(
            [
                ocnn.modules.OctreeResBlocks(
                    decoder_inputs[index],
                    self.decoder_channel[index + 1],
                    self.decoder_blocks[index],
                    self.bottleneck,
                    nempty,
                    self.resblk,
                )
                for index in range(self.decoder_stages)
            ]
        )

    @property
    def output_channels(self) -> int:
        return self.decoder_channel[-1]

    def _stage(self, module, *args):
        """Run one U-Net stage, optionally under activation checkpointing.

        The scene U-Net dominates the retained-activation footprint for the heaviest
        meshes (coverage-complete octrees reach ~8M depth-9 nodes, 43x the point-cloud
        octree).  ``MODEL.activation_checkpointing`` used to be forwarded only to the
        renderer, so every U-Net activation stayed resident and the peak landed in the
        loss; recomputing each stage in the backward pass trades ~20-30 % iteration
        time for that memory back.
        """
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def unet_encoder(
        self, data: torch.Tensor, octree: object, depth: int
    ) -> dict[int, torch.Tensor]:
        encoded: dict[int, torch.Tensor] = {depth: self._stage(self.conv1, data, octree, depth)}
        for index in range(self.encoder_stages):
            current_depth = depth - index
            down = self._stage(self.downsample[index], encoded[current_depth], octree, current_depth)
            encoded[current_depth - 1] = self._stage(
                self.encoder[index], down, octree, current_depth - 1
            )
        return encoded

    def unet_decoder(
        self, encoded: dict[int, torch.Tensor], octree: object, depth: int
    ) -> torch.Tensor:
        decoded = encoded[depth]
        for index in range(self.decoder_stages):
            current_depth = depth + index
            decoded = self._stage(self.upsample[index], decoded, octree, current_depth)
            decoded = torch.cat([encoded[current_depth + 1], decoded], dim=1)
            decoded = self._stage(self.decoder[index], decoded, octree, current_depth + 1)
        return decoded

    def forward(self, data: torch.Tensor, octree: object, depth: int) -> torch.Tensor:
        encoded = self.unet_encoder(data, octree, depth)
        return self.unet_decoder(encoded, octree, depth - self.encoder_stages)


class UNet(FeatureUNet):
    def __init__(self, in_channels: int, interp: str = "linear", nempty: bool = True):
        del interp
        super().__init__(in_channels, query_depth=7, nempty=nempty)


class UNet6(FeatureUNet):
    def __init__(self, in_channels: int, interp: str = "linear", nempty: bool = True):
        del interp
        super().__init__(in_channels, query_depth=6, nempty=nempty)
