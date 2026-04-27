import torch
import numpy as np
import ocnn
from octree import Octree
from ocnn.octree import Octree
from ocnn.nn import OctreeMaxPool
from ocnn.nn import OctreeConv, OctreeDeconv
from typing import Dict, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import copy


def trilinear_interpolation(F: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    dx = P[:, 0]
    dy = P[:, 1]
    dz = P[:, 2]
    
    weights_x = torch.stack([1 - dx, dx], dim=1)  # (B, 2)
    weights_y = torch.stack([1 - dy, dy], dim=1)  # (B, 2)
    weights_z = torch.stack([1 - dz, dz], dim=1)  # (B, 2)
    
    weights_xy = weights_x.unsqueeze(2) * weights_y.unsqueeze(1)  # (B, 2, 2)
    weights_xyz = weights_xy.unsqueeze(3) * weights_z.unsqueeze(1).unsqueeze(2)  # (B, 2, 2, 2)
    weights_expanded = weights_xyz.unsqueeze(-1)  # (B, 2, 2, 2, 1)
    
    weighted_features = F * weights_expanded  # (B, 2, 2, 2, N)
    result = weighted_features.sum(dim=[1, 2, 3])  # (B, N)
    
    return result


class OctreeConvGnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.gn(out)
        out = self.relu(out)
        return out

class OctreeConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out

class OctreeDeconvGnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.deconv = OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.gn(out)
        out = self.relu(out)
        return out

class OctreeDeconvBnRelu(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.deconv = OctreeDeconv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.deconv(data, octree, depth)
        out = self.bn(out)
        out = self.relu(out)
        return out

class OctreeGnResBlock2(torch.nn.Module):
  r''' Basic Octree-based ResNet block. The block is composed of
  a series of :obj:`Conv3x3` and :obj:`Conv3x3`.

  Refer to :class:`OctreeResBlock` for the details of arguments.
  '''

  def __init__(self, in_channels, out_channels, stride=1, bottleneck=1,
               nempty=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.stride = stride
    channelb = int(out_channels / bottleneck)

    if self.stride == 2:
      self.maxpool = OctreeMaxPool(self.depth)
    self.conv3x3a = OctreeConvGnRelu(in_channels, channelb, nempty=nempty)
    self.conv3x3b = OctreeConvGn(channelb, out_channels, nempty=nempty)
    if self.in_channels != self.out_channels:
      self.conv1x1 = Conv1x1Gn(in_channels, out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if self.stride == 2:
      data = self.maxpool(data, octree, depth)
      depth = depth - 1
    conv1 = self.conv3x3a(data, octree, depth)
    conv2 = self.conv3x3b(conv1, octree, depth)
    if self.in_channels != self.out_channels:
      data = self.conv1x1(data)
    out = self.relu(conv2 + data)
    return out

class OctreeBnResBlock2(torch.nn.Module):
    r''' Basic Octree-based ResNet block. The block is composed of
    a series of :obj:`Conv3x3` and :obj:`Conv3x3`.
    
    Refer to :class:`OctreeResBlock` for the details of arguments.
    '''
    
    def __init__(self, in_channels, out_channels, stride=1, bottleneck=1,
                 nempty=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        channelb = int(out_channels / bottleneck)
    
        if self.stride == 2:
            self.maxpool = OctreeMaxPool(self.depth)
        self.conv3x3a = OctreeConvBnRelu(in_channels, channelb, nempty=nempty)
        self.conv3x3b = OctreeConvBn(channelb, out_channels, nempty=nempty)
        if self.in_channels != self.out_channels:
            self.conv1x1 = Conv1x1Bn(in_channels, out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        r''''''
    
        if self.stride == 2:
            data = self.maxpool(data, octree, depth)
            depth = depth - 1
        conv1 = self.conv3x3a(data, octree, depth)
        conv2 = self.conv3x3b(conv1, octree, depth)
        if self.in_channels != self.out_channels:
            data = self.conv1x1(data)
        out = self.relu(conv2 + data)
        return out

class OctreeConvGn(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv` and :obj:`BatchNorm`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels) #, bn_eps, bn_momentum)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.gn(out)
    return out

class OctreeConvBn(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
        super().__init__()
        self.conv = OctreeConv(
            in_channels, out_channels, kernel_size, stride, nempty)
        self.bn = torch.nn.BatchNorm1d(out_channels)
    
    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        out = self.conv(data, octree, depth)
        out = self.bn(out)
        return out

class Conv1x1Gn(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.gn(out)
    return out

class Conv1x1Bn(torch.nn.Module):
    r''' A sequence of :class:`Conv1x1` and :class:`BatchNorm`.
    '''
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
    
    def forward(self, data: torch.Tensor):
        r''''''
    
        out = self.conv(data)
        out = self.bn(out)
        return out

class Conv1x1GnRelu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = torch.nn.GroupNorm(num_groups=out_channels // 4, num_channels=out_channels)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.gn(out)
    out = self.relu(out)
    return out

class Conv1x1BnRelu(torch.nn.Module):
    r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
    '''
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ocnn.modules.Conv1x1(in_channels, out_channels, use_bias=False)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
    
    def forward(self, data: torch.Tensor):
        r''''''
    
        out = self.conv(data)
        out = self.bn(out)
        out = self.relu(out)
        return out



# Positional encoding (from NeRF)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    if input_dims == -1:
        return torch.nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, interp: str = 'linear', nempty: bool = True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.nempty = nempty
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

    # encoder
        self.conv1 = OctreeConvBnRelu(in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([OctreeConvBnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i+1], self.encoder_channel[i + 1],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])

    # # decoder
        channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
                for i in range(self.decoder_stages)]
        self.upsample = torch.nn.ModuleList([OctreeDeconvBnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.decoder_stages)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channel[i], self.decoder_channel[i+1],
            self.decoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.decoder_stages)])
    #     self.header = ocnn.modules.Conv1x1(self.decoder_channel[-1], 63, use_bias=True)
    #     self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

    def config_network(self):
        self.encoder_channel = [32, 32, 64, 128, 256]
        # self.decoder_channel = [256, 256, 128, 96, 96]
        # single decoder
        self.decoder_channel = [256, 256, 128]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2, 2]
        self.bottleneck = 1
        self.resblk = OctreeBnResBlock2


    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):

        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d-1] = self.encoder[i](conv, octree, d-1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], octree: Octree, depth: int):
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)
            deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, octree, d+1)
        return deconv

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        convd = self.unet_encoder(data, octree, depth)
        deconv = self.unet_decoder(convd, octree, depth - self.encoder_stages)
        # deconv = self.header(deconv) # if mul instead of cat
        # interp_depth = depth - self.encoder_stages + self.decoder_stages
        # feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
        # return convd[depth - self.encoder_stages]
        return deconv


class UNet6(torch.nn.Module):
    def __init__(self, in_channels: int, interp: str = 'linear', nempty: bool = True):
        super(UNet6, self).__init__()
        self.in_channels = in_channels
        self.nempty = nempty
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)

    # encoder
        self.conv1 = OctreeConvBnRelu(in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([OctreeConvBnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i+1], self.encoder_channel[i + 1],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])

    # decoder
        channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
                for i in range(self.decoder_stages)]
        self.upsample = torch.nn.ModuleList([OctreeDeconvBnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.decoder_stages)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channel[i], self.decoder_channel[i+1],
            self.decoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.decoder_stages)])
    #     self.header = ocnn.modules.Conv1x1(self.decoder_channel[-1], 63, use_bias=True)
    #     self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

    def config_network(self):
        self.encoder_channel = [32, 32, 64, 128, 256]
        # self.decoder_channel = [256, 256, 128, 96, 96]
        # single decoder
        self.decoder_channel = [256, 256]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2]
        self.bottleneck = 1
        self.resblk = OctreeBnResBlock2


    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):

        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d-1] = self.encoder[i](conv, octree, d-1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], octree: Octree, depth: int):
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)
            deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, octree, d+1)
        return deconv

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        convd = self.unet_encoder(data, octree, depth)
        deconv = self.unet_decoder(convd, octree, depth - self.encoder_stages)
        # deconv = self.header(deconv) # if mul instead of cat
        # interp_depth = depth - self.encoder_stages + self.decoder_stages
        # feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
        # return convd[depth - self.encoder_stages]
        return deconv

    
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

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x: torch.Tensor):
        out1 = self.relu(self.norm1(self.fc1(x)))
        out2 = self.relu(self.norm2(self.fc2(out1)))
        out3 = self.fc3(out2) + x
        return self.relu(self.norm3(out3))

class NeuralIntersectionModel(nn.Module):
    def __init__(self, in_channels: int, q_depth: int=7):
        super().__init__()
        self.in_channels = in_channels
        if q_depth == 6:
            self.feature_model = UNet6(in_channels=self.in_channels)
            self.renderer_model = SplitVoxelRendererModel(f_dim=32, n_iters=3)
        else:
            self.feature_model = UNet(in_channels=self.in_channels)
        # self.renderer_model = VoxelRendererModel(f_dim=16, n_iters=3)
            self.renderer_model = SplitVoxelRendererModel(f_dim=16, n_iters=3)
        self.q_depth = q_depth

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, pin: torch.Tensor, direction: torch.Tensor, voxel_indices: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 特征提取
        vert_feat = self.feature_model(data, octree, depth)
        
        # 2. 构建顶点特征图
        average_feat, vert_indexes = self._build_vertex_feature_map(vert_feat, octree)
        
        # 3. 查询目标体素顶点特征
        vert_feat = self._query_voxel_vertices(
            average_feat, vert_indexes, octree, voxel_indices
        )
        
        # 4. 渲染交点
        intersection = self.renderer_model(pin, direction, vert_feat)
        return intersection
    
    def _build_vertex_feature_map(
        self,
        vert_feat: torch.Tensor,
        octree: Octree
    ) -> torch.Tensor:
        """
        构建顶点特征图
        1. 计算顶点索引
        2. 扩展特征维度
        3. 计算平均特征
        """
        query_d = self.q_depth
        
        # 计算顶点
        n_vertices, vert_indexes = octree.compute_vertices(query_d)
        assert vert_feat.shape[0] == vert_indexes.shape[0]
        
        # 准备特征存储空间
        feat_channels = vert_feat.shape[1] // 8
        device = vert_feat.device
        dtype = vert_feat.dtype
        
        all_vertices = torch.zeros(
            (n_vertices * 8, feat_channels + 1),
            dtype=dtype,
            device=device
        )
        
        # 扩展特征（添加权重通道）
        vert_indexes_expanded = vert_indexes * 8 + torch.arange(
            8, device=device
        ).unsqueeze(0)
        
        vert_feat_reshaped = vert_feat.view(vert_indexes.shape[0] * 8, -1)
        weights = torch.ones(
            (vert_indexes.shape[0] * 8, 1), device=device
        )
        vert_feat_with_weight = torch.cat([vert_feat_reshaped, weights], dim=1)
        
        # 填充特征
        all_vertices[vert_indexes_expanded.view(-1)] = vert_feat_with_weight
        
        # 计算加权平均特征
        reshaped_features = all_vertices.view(-1, 8, feat_channels + 1)
        feature_sum = reshaped_features[:, :, :-1].sum(dim=1)
        weight_sum = reshaped_features[:, :, -1:].sum(dim=1)
        average_feat = feature_sum / weight_sum
        
        assert average_feat.shape[0] == n_vertices
        return average_feat, vert_indexes

    def _query_voxel_vertices(
        self,
        average_feat: torch.Tensor,
        vert_indexes: torch.Tensor,
        octree: Octree,
        voxel_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        查询指定体素的顶点特征
        """        
        # 获取目标体素的顶点索引
        query_vert_indexes = vert_indexes[voxel_indices].view(-1, 2, 2, 2)
        
        # 查询顶点特征
        vert_feat = average_feat[query_vert_indexes]
        return vert_feat
        

class VoxelRendererModel(nn.Module):
    def __init__(self, f_dim = 32, hidden_dim = 128, out_dim = 10, n_iters=4):
        super().__init__()
        self.n_iters = n_iters
        self.embedding = nn.Linear(f_dim + out_dim + 3, hidden_dim)
        self.hidden2hidden = ResidualFCLayer3(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        self.init_normal = nn.Parameter(torch.zeros((1, 3), dtype=torch.float), requires_grad=True)
        self.init_color = nn.Parameter(torch.zeros((1, 3), dtype=torch.float), requires_grad=True)
        self.init_hit_prob = nn.Parameter(torch.zeros((1,), dtype=torch.float), requires_grad=True)
        self.init_hidden = nn.Parameter(torch.zeros((1, hidden_dim), dtype=torch.float), requires_grad=True)

    def get_scene_feature_for_points(self, vert_feat: torch.Tensor, pos: torch.Tensor):
        feat_interp = trilinear_interpolation(vert_feat, pos)  # (B, f_dim)
        return feat_interp
    
    def forward(self, pin: torch.Tensor, direction: torch.Tensor, vert_feat: torch.Tensor):
        B = pin.shape[0]
        position = pin
        normal = self.init_normal.expand(B, -1)
        color = self.init_color.expand(B, -1)
        hit_prob = self.init_hit_prob.expand(B)
        hidden = self.init_hidden.expand(B, -1)

        if self.training:
            outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(self.n_iters):
            feature = self.get_scene_feature_for_points(vert_feat, position)
            # all_combined = torch.cat([feature, direction, normal, hit_prob.unsqueeze(-1)], dim=1)
            all_combined = torch.cat([feature, direction, position, normal, color, hit_prob.unsqueeze(-1)], dim=1)
            # all_combined = torch.cat([feature, direction, position, normal, hit_prob.unsqueeze(-1)], dim=1)
            all_embedded = self.embedding(all_combined)
            hidden = hidden + self.hidden2hidden(all_embedded)
            output = self.output_layer(hidden)
            position = torch.sigmoid(output[:, :3])
            normal = F.normalize(output[:, 3:6], dim=1)
            color = torch.sigmoid(output[:, 6:9])
            hit_prob = F.sigmoid(output[:, 9])
            # hit_prob = torch.sigmoid(output[:, 6])
            #position must be on the ray:
            position = pin + direction * ((position - pin) * direction).sum(dim=1, keepdim=True)
            if self.training:
                outputs.append((position, normal, color, hit_prob))
                # outputs.append((position, normal, hit_prob))

        if self.training:
            return outputs
        else:
            return position, normal, color, hit_prob
            # return position, normal, hit_prob


class SplitVoxelRendererModel(nn.Module):
    def __init__(self, f_dim=32, hidden_dim=128, out_dim=10, n_iters=4):
        super().__init__()
        self.n_iters = n_iters
        
        # 几何属性网络
        geo_input_dim = f_dim + 3 + 3 + 3 + 1  # feature + direction + position + normal + hit_prob
        self.geometry_embedding = nn.Linear(geo_input_dim, hidden_dim)
        self.geometry_hidden2hidden = ResidualFCLayer3(hidden_dim)
        self.geometry_output_layer = nn.Linear(hidden_dim, 7)  # 3(position) + 3(normal) + 1(hit_prob)
        
        # 材质属性网络
        mat_input_dim = f_dim + 3 + 3 + 3  # feature + direction + position + color
        self.material_embedding = nn.Linear(mat_input_dim, hidden_dim)
        self.material_hidden2hidden = ResidualFCLayer3(hidden_dim)
        self.material_output_layer = nn.Linear(hidden_dim, 3)  # 3(color)
        
        # 初始化参数
        self.init_normal = nn.Parameter(torch.zeros((1, 3), dtype=torch.float), requires_grad=True)
        self.init_color = nn.Parameter(torch.zeros((1, 3), dtype=torch.float), requires_grad=True)
        self.init_hit_prob = nn.Parameter(torch.zeros((1,), dtype=torch.float), requires_grad=True)
        
        # 分别的隐藏状态
        self.init_geo_hidden = nn.Parameter(torch.zeros((1, hidden_dim), dtype=torch.float), requires_grad=True)
        self.init_mat_hidden = nn.Parameter(torch.zeros((1, hidden_dim), dtype=torch.float), requires_grad=True)
    
    def get_scene_feature_for_points(self, vert_feat: torch.Tensor, pos: torch.Tensor):
        feat_interp = trilinear_interpolation(vert_feat, pos)  # (B, f_dim)
        return feat_interp
    
    def forward(self, pin: torch.Tensor, direction: torch.Tensor, vert_feat: torch.Tensor):
        B = pin.shape[0]
        
        # 初始化状态
        position = pin
        normal = self.init_normal.expand(B, -1)
        color = self.init_color.expand(B, -1)
        hit_prob = self.init_hit_prob.expand(B)
        
        # 分别的隐藏状态
        geo_hidden = self.init_geo_hidden.expand(B, -1)
        mat_hidden = self.init_mat_hidden.expand(B, -1)
        
        if self.training:
            outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        for i in range(self.n_iters):
            # 获取场景特征
            feature = self.get_scene_feature_for_points(vert_feat, position)
            
            # --- 几何属性预测 ---
            geo_combined = torch.cat([feature, direction, position, normal, hit_prob.unsqueeze(-1)], dim=1)
            geo_embedded = self.geometry_embedding(geo_combined)
            geo_hidden = geo_hidden + self.geometry_hidden2hidden(geo_embedded)
            geo_output = self.geometry_output_layer(geo_hidden)
            
            # 解析几何输出
            position_raw = torch.sigmoid(geo_output[:, :3])
            normal = F.normalize(geo_output[:, 3:6], dim=1)
            hit_prob = F.sigmoid(geo_output[:, 6])
            
            # 确保位置在射线上
            position = pin + direction * ((position_raw - pin) * direction).sum(dim=1, keepdim=True)
            
            # --- 材质属性预测 ---
            mat_combined = torch.cat([feature, direction, position, color], dim=1)
            mat_embedded = self.material_embedding(mat_combined)
            mat_hidden = mat_hidden + self.material_hidden2hidden(mat_embedded)
            mat_output = self.material_output_layer(mat_hidden)
            
            # 解析材质输出
            color = torch.sigmoid(mat_output)
            
            if self.training:
                outputs.append((position, normal, color, hit_prob))
        
        if self.training:
            return outputs
        else:
            return position, normal, color, hit_prob


class LocalPositionEncoder:
    def __init__(self, depth: int):
        ...
    
    def encode(self, global_positions: torch.Tensor):
        return global_positions - global_positions.floor()
    
    def decode(self, local_positions: torch.Tensor, voxel_coords: torch.LongTensor):
        return voxel_coords.float() + local_positions

class PNCHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_pos, pred_norm, pred_col, pred_hit = pred
        tgt_pos, tgt_norm, tgt_col, tgt_hit = target
        
        mask = tgt_hit.float()
        mask_unsqueezed = mask.unsqueeze(-1)
        
        pos_loss = F.l1_loss(pred_pos * mask_unsqueezed, tgt_pos * mask_unsqueezed)
        norm_loss = F.l1_loss(pred_norm * mask_unsqueezed, tgt_norm * mask_unsqueezed)
        # relative l2 loss for color
        relative_col_loss = F.mse_loss(pred_col * mask_unsqueezed, tgt_col * mask_unsqueezed) / (F.mse_loss(tgt_col * mask_unsqueezed, torch.zeros_like(tgt_col)) + 1e-8)        
        # col_loss = F.l1_loss(pred_col * mask_unsqueezed, tgt_col * mask_unsqueezed)

        hit_loss = F.binary_cross_entropy(pred_hit, mask)
        
        # return pos_loss + norm_loss + 2*col_loss + hit_loss
        return pos_loss + norm_loss + 2*relative_col_loss + hit_loss


class PNHLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_pos, pred_norm, pred_hit = pred
        tgt_pos, tgt_norm, tgt_hit = target
        
        mask = tgt_hit.float()
        mask_unsqueezed = mask.unsqueeze(-1)
        
        pos_loss = F.l1_loss(pred_pos * mask_unsqueezed, tgt_pos * mask_unsqueezed)
        norm_loss = F.l1_loss(pred_norm * mask_unsqueezed, tgt_norm * mask_unsqueezed)
        hit_loss = F.binary_cross_entropy(pred_hit, mask)
        
        return 2*pos_loss + norm_loss + 2*hit_loss



class VAEEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, latent_dim: int = 32, interp: str = 'linear', nempty: bool = False):
        super(VAEEncoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.nempty = nempty
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
    # encoder
        self.conv1 = OctreeConvBnRelu(in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([OctreeConvBnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages - 1)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i], self.encoder_channel[i],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])
        self.encoder_final_ln = torch.nn.LayerNorm(self.encoder_channel[-1])
        self.encoder_fc = nn.Linear(self.encoder_channel[-1], latent_dim * 2)  # 输出均值和对数方差

    def config_network(self):
        self.encoder_channel = [32, 64, 128, 128, 256, 256]
        self.encoder_blocks = [2, 2, 2, 2, 2, 2]
        self.bottleneck = 1
        self.resblk = OctreeBnResBlock2

    def sample_from_posterior(self, h: torch.Tensor, sample_posterior: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从后验分布采样"""
        # h的形状: [N, latent_dim*2]
        mean, logvar = h.chunk(2, dim=-1)  # 每个都是[N, latent_dim]
        
        if sample_posterior and self.training:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
            
        return z, mean, logvar

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, sample_posterior: bool = True):
        convd = dict()
        data = ocnn.nn.octree_pad(data, octree, depth)
        conv = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            convd[d] = self.encoder[i](conv, octree, d)
            if i < self.encoder_stages - 1:
                conv = self.downsample[i](convd[d], octree, d)
            # conv = self.downsample[i](convd[d], octree, d)
            # convd[d-1] = self.encoder[i](conv, octree, d-1)
        encoded_feature = convd[depth - self.encoder_stages + 1]
        encoded_feature = self.encoder_final_ln(encoded_feature)
        h = self.encoder_fc(encoded_feature)

        z, mean, logvar = self.sample_from_posterior(h, sample_posterior)
        return z, mean, logvar


class VAEDecoder(torch.nn.Module):
    def __init__(self, out_channels: int, latent_dim: int = 32, interp: str = 'linear', nempty: bool = True):
        super(VAEDecoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.nempty = nempty
        self.config_network()
        self.decoder_stages = len(self.decoder_blocks)

    # # decoder
        self.upsample = torch.nn.ModuleList([OctreeDeconvBnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=False) for i in range(self.decoder_stages - 1)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.decoder_channel[i], self.decoder_channel[i],
            self.decoder_blocks[i], self.bottleneck, False, self.resblk)
            for i in range(self.decoder_stages)])
        self.decoder_input_fc = nn.Linear(latent_dim, self.decoder_channel[0])
        self.decoder_final_ln = torch.nn.LayerNorm(self.decoder_channel[-1])
        self.decoder_output_fc = nn.Linear(self.decoder_channel[-1], out_channels)
        # self.split_heads = nn.ModuleList([ocnn.modules.Conv1x1(self.decoder_channel[i], 1, use_bias=True) for i in range(self.decoder_stages)])
        self.split_heads = nn.ModuleList([self._make_predict_module(self.decoder_channel[i], 2) for i in range(self.decoder_stages)])
    #     self.header = ocnn.modules.Conv1x1(self.decoder_channel[-1], 63, use_bias=True)
    #     self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
        return torch.nn.Sequential(
            ocnn.modules.Conv1x1BnRelu(channel_in, num_hidden),
            ocnn.modules.Conv1x1(num_hidden, channel_out, use_bias=True))

    def config_network(self):
        # self.decoder_channel = [256, 256, 128, 96, 96]
        # single decoder
        self.decoder_channel = [256, 256, 128, 128]
        self.decoder_blocks = [2, 2, 2, 2]
        self.bottleneck = 1
        self.resblk = OctreeBnResBlock2

    def forward(self, z: torch.Tensor, octree: Octree, depth: int, update_octree: bool = False):
        deconv = self.decoder_input_fc(z)
        subs = []
        subs_gt = []
        # octree = copy.deepcopy(octree)
        
        for i in range(self.decoder_stages):
            d = depth + i

            deconv = self.decoder[i](deconv, octree, d)
            sub_logit = self.split_heads[i](deconv)
            subs.append(sub_logit)
            if not update_octree:
                subs_gt.append(octree.get_split(d))
            else:
                sub = sub_logit.argmax(dim=1).int()
                # count sub.squeeze(1) > 0.5
                print(f"Stage {i}, depth {d}, split ratio: {(sub > 0.5).float().mean().item():.4f}")
                # octree.octree_split(sub.squeeze(1) > 0.5, d)
                octree.octree_split(sub, d)
            if i < self.decoder_stages - 1:
                if update_octree:
                    octree.octree_grow(d+1)
                deconv = self.upsample[i](deconv, octree, d)
        deconv = self.decoder_final_ln(deconv)
        deconv = self.decoder_output_fc(deconv)
        deconv = ocnn.nn.octree_depad(deconv, octree, depth + self.decoder_stages - 1)
        if not update_octree:
            return deconv, subs, subs_gt
        else:
            return deconv, octree


class VAE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_dim: int = 32, interp: str = 'linear', nempty: bool = True):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim, interp, False)
        self.decoder = VAEDecoder(out_channels, latent_dim, interp, nempty)

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, sample_posterior: bool = True, update_octree: bool = False):
        z, mean, logvar = self.encoder(data, octree, depth, sample_posterior)
        if not update_octree:
            recon_data, subs, subs_gt = self.decoder(z, octree, depth - self.encoder.encoder_stages + 1, update_octree)
            return recon_data, subs, subs_gt, mean, logvar
        else:
            recon_data, output_oct = self.decoder(z, octree, depth - self.encoder.encoder_stages + 1, update_octree)
            return recon_data, output_oct



class NeuralVAEModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, q_depth: int=7):
        super().__init__()
        self.in_channels = in_channels
        self.feature_model = VAE(in_channels=self.in_channels, out_channels=out_channels)
        # self.renderer_model = VoxelRendererModel(f_dim=16, n_iters=3)
        self.renderer_model = SplitVoxelRendererModel(f_dim=16, n_iters=3)
        self.q_depth = q_depth

    def forward(self, data: torch.Tensor, octree: Octree, depth: int, pin: torch.Tensor, direction: torch.Tensor, voxel_indices: torch.Tensor, sample_posterior: bool = True, update_octree: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 特征提取
        if not update_octree:
            vert_feat, subs, subs_gt, mean, logvar = self.feature_model(data, octree, depth, sample_posterior, update_octree)
        else:
            vert_feat, octree = self.feature_model(data, octree, depth, sample_posterior, update_octree)
        # vert_feat = self.feature_model(data, octree, depth)
        
        # 2. 构建顶点特征图
        average_feat, vert_indexes = self._build_vertex_feature_map(vert_feat, octree)
        
        # 3. 查询目标体素顶点特征
        vert_feat = self._query_voxel_vertices(
            average_feat, vert_indexes, octree, voxel_indices
        )
        
        # 4. 渲染交点
        intersection = self.renderer_model(pin, direction, vert_feat)
        if not update_octree:
            return intersection, subs, subs_gt, mean, logvar
        else:
            return intersection, octree
        # return intersection
    
    def _build_vertex_feature_map(
        self,
        vert_feat: torch.Tensor,
        octree: Octree
    ) -> torch.Tensor:
        """
        构建顶点特征图
        1. 计算顶点索引
        2. 扩展特征维度
        3. 计算平均特征
        """
        query_d = self.q_depth
        
        # 计算顶点
        n_vertices, vert_indexes = octree.compute_vertices(query_d)

        assert vert_feat.shape[0] == vert_indexes.shape[0]
        
        # 准备特征存储空间
        feat_channels = vert_feat.shape[1] // 8
        device = vert_feat.device
        dtype = vert_feat.dtype
        
        all_vertices = torch.zeros(
            (n_vertices * 8, feat_channels + 1),
            dtype=dtype,
            device=device
        )
        
        # 扩展特征（添加权重通道）
        vert_indexes_expanded = vert_indexes * 8 + torch.arange(
            8, device=device
        ).unsqueeze(0)
        
        vert_feat_reshaped = vert_feat.view(vert_indexes.shape[0] * 8, -1)
        weights = torch.ones(
            (vert_indexes.shape[0] * 8, 1), device=device
        )
        vert_feat_with_weight = torch.cat([vert_feat_reshaped, weights], dim=1)
        
        # 填充特征
        all_vertices[vert_indexes_expanded.view(-1)] = vert_feat_with_weight
        
        # 计算加权平均特征
        reshaped_features = all_vertices.view(-1, 8, feat_channels + 1)
        feature_sum = reshaped_features[:, :, :-1].sum(dim=1)
        weight_sum = reshaped_features[:, :, -1:].sum(dim=1)
        average_feat = feature_sum / weight_sum
        
        assert average_feat.shape[0] == n_vertices
        return average_feat, vert_indexes

    def _query_voxel_vertices(
        self,
        average_feat: torch.Tensor,
        vert_indexes: torch.Tensor,
        octree: Octree,
        voxel_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        查询指定体素的顶点特征
        """        
        # 获取目标体素的顶点索引
        query_vert_indexes = vert_indexes[voxel_indices].view(-1, 2, 2, 2)
        
        # 查询顶点特征
        vert_feat = average_feat[query_vert_indexes]
        return vert_feat
        
