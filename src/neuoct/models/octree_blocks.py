from __future__ import annotations

import torch
from ocnn.nn import OctreeConv, OctreeDeconv, OctreeMaxPool


class RMSNorm1d(torch.nn.Module):
    """RMS normalization for sparse feature matrices shaped ``[N, C]``."""

    def __init__(self, channels: int, eps: float = 1.0e-6):
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(self.channels))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Keep the complete calculation in FP32.  Casting the variance back to
        # FP16 before adding eps can underflow eps (1e-8 -> 0) and produce
        # 0 * rsqrt(0) = NaN for empty/near-zero sparse features.
        data_fp32 = data.float()
        variance = data_fp32.square().mean(dim=-1, keepdim=True)
        normalized = data_fp32 * torch.rsqrt(variance + self.eps)
        output = normalized * self.weight.float()
        return output.to(dtype=data.dtype)


def make_norm(channels: int, norm_type: str = "batch", norm_groups: int = 8) -> torch.nn.Module:
    """Create a normalization layer for OCNN's ``[N, C]`` sparse features.

    ``batch`` retains the historical BatchNorm1d behavior.  The alternatives
    avoid batch statistics and are consequently compatible with tiny per-rank
    object batches.  GroupNorm is applied to the feature matrix directly;
    PyTorch treats the second dimension as channels for a 2-D input.
    """
    kind = str(norm_type).lower().replace("_", "-")
    if kind in ("batch", "bn", "batchnorm"):
        return torch.nn.BatchNorm1d(channels)
    if kind in ("rms", "rmsnorm"):
        return RMSNorm1d(channels)
    if kind in ("layer", "ln", "layernorm"):
        return torch.nn.LayerNorm(channels)
    if kind in ("group", "gn", "groupnorm"):
        groups = max(1, min(int(norm_groups), int(channels)))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return torch.nn.GroupNorm(groups, channels)
    raise ValueError(
        f"Unsupported normalization type {norm_type!r}; expected batch, rms, layer, or group"
    )


class OctreeConvBnRelu(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int] | None = None,
        stride: int = 1,
        nempty: bool = False,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        kernel_size = [3] if kernel_size is None else kernel_size
        self.conv = OctreeConv(in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = make_norm(out_channels, norm_type, norm_groups)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: object, depth: int) -> torch.Tensor:
        return self.relu(self.bn(self.conv(data, octree, depth)))


class OctreeDeconvBnRelu(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int] | None = None,
        stride: int = 1,
        nempty: bool = False,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        kernel_size = [3] if kernel_size is None else kernel_size
        self.deconv = OctreeDeconv(in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = make_norm(out_channels, norm_type, norm_groups)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: object, depth: int) -> torch.Tensor:
        return self.relu(self.bn(self.deconv(data, octree, depth)))


class OctreeConvBn(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int] | None = None,
        stride: int = 1,
        nempty: bool = False,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        kernel_size = [3] if kernel_size is None else kernel_size
        self.conv = OctreeConv(in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = make_norm(out_channels, norm_type, norm_groups)

    def forward(self, data: torch.Tensor, octree: object, depth: int) -> torch.Tensor:
        return self.bn(self.conv(data, octree, depth))


class Conv1x1Bn(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        import ocnn

        self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
        self.bn = make_norm(out_channels, norm_type, norm_groups)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(data))


class OctreeBnResBlock2(torch.nn.Module):
    """Checkpoint-compatible residual block used by the legacy NeuOct U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        bottleneck: int = 1,
        nempty: bool = False,
        norm_type: str = "batch",
        norm_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        channelb = int(out_channels / bottleneck)
        if stride == 2:
            self.maxpool = OctreeMaxPool()
        self.conv3x3a = OctreeConvBnRelu(
            in_channels,
            channelb,
            nempty=nempty,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        self.conv3x3b = OctreeConvBn(
            channelb,
            out_channels,
            nempty=nempty,
            norm_type=norm_type,
            norm_groups=norm_groups,
        )
        if in_channels != out_channels:
            self.conv1x1 = Conv1x1Bn(
                in_channels,
                out_channels,
                norm_type=norm_type,
                norm_groups=norm_groups,
            )
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: object, depth: int) -> torch.Tensor:
        if self.stride == 2:
            data = self.maxpool(data, octree, depth)
            depth -= 1
        output = self.conv3x3b(self.conv3x3a(data, octree, depth), octree, depth)
        if self.in_channels != self.out_channels:
            data = self.conv1x1(data)
        return self.relu(output + data)
