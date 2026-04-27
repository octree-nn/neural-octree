import torch
import ocnn
import ognn
import torch.nn.functional as F
import math
import copy
from typing import List, Optional
from ocnn.octree import Octree
from ognn.octreed import OctreeD
from ognn import mpu
from octbvh.octbvh import OctBVH, get_input_feature, init_octbvh
from octbvh.transformer.octformer import MLP
from utils import loss_utils, bvh_utils


class Encoder(torch.nn.Module):
  r''' An encoder takes an octree as input and outputs latent codes on a
  downsampled octree.
  '''

  def __init__(self, in_channels: int,
               channels: List[int],
               resblk_nums: List[int],
               bottleneck: int = 2, 
               nempty: bool = True, **kwargs):
    super().__init__()
    groups = 32
    self.stage_num = len(channels)
    self.delta_depth = self.stage_num - 1

    self.input_conv = torch.nn.ModuleList([ocnn.modules.OctreeConvGnRelu(
        in_channels, channels[i], groups, nempty=nempty) for i in range(self.stage_num)])
    self.blocks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
        channels[i], channels[i], resblk_nums[i], bottleneck, nempty=nempty,
        resblk=ocnn.modules.OctreeResBlockGn, use_checkpoint=True)
        for i in range(self.stage_num)])
    self.mlp = torch.nn.ModuleList(
        [MLP(channels[i], 4 * channels[i], channels[i]) for i in range(self.stage_num)])
    self.downsample = torch.nn.ModuleList([ocnn.modules.OctreeConvGnRelu(
        channels[i], channels[i+1], groups, kernel_size=[2], stride=2, nempty=nempty)
        for i in range(self.stage_num - 1)])  # Note: self.stage_num - 1

  def forward(self, data: dict, octree: Octree, depth: int):
    output = {}
    x = 0.0
    for i in range(self.stage_num):
      di = depth - i
      if di in data:
        x = x + self.input_conv[i](data[di], octree, di)
      x = self.blocks[i](x, octree, di)
      x = self.mlp[i](x)
      output[di] = x
      if i < self.stage_num - 1:
        x = self.downsample[i](x, octree, di)
    return output


class Decoder(torch.nn.Module):
  r''' A decoder takes a downsampled octree and latent codes as input and
  outputs the upsampled octree. This decoder is designed to take dual octrees
  as input and output. The output octree is converted to a continuous surface
  via MPU. It contains a tiny U-Net to increase the ability of the decoder.
  '''

  def __init__(self, n_node_type: int,
               channels: List[int] = [256, 128, 64, 32, 32, 32],
               resblk_nums: List[int] = [2, 4, 2, 1, 1, 1],
               bbox_stage_nums: int = 3, split_stage_nums: int = 3,
               bottleneck: int = 2, nempty: bool = True, **kwargs):
    super().__init__()
    groups = 32
    self.head_channel = 64
    self.use_checkpoint = True
    self.act_type = 'relu'
    self.resblk_type = 'basic'
    self.norm_type = 'group_norm'
    self.n_node_type = n_node_type
    self.resblk_nums = resblk_nums
    self.channels = channels
    self.nempty = nempty
    self.stage_num = len(self.resblk_nums)

    # tiny decoder
    self.upsample = torch.nn.ModuleList([ocnn.modules.OctreeDeconvGnRelu(
        self.channels[i - 1], self.channels[i], groups, kernel_size=[2], stride=2,
        nempty=nempty) for i in range(1, self.stage_num)])
    self.blocks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
        self.channels[i], self.channels[i],
        self.resblk_nums[i], bottleneck, nempty=nempty,
        resblk=ocnn.modules.OctreeResBlockGn, use_checkpoint=True)
        for i in range(self.stage_num)])
    self.mlp = torch.nn.ModuleList(
        [MLP(channels[i], 4 * channels[i], channels[i]) for i in range(self.stage_num)])

    # header
    self.start_pred = self.stage_num - split_stage_nums
    self.pred_split = torch.nn.ModuleList([
      torch.nn.Linear(self.channels[i], 3) for i in range(self.start_pred, self.stage_num)])
    self.start_bbox = self.stage_num - bbox_stage_nums
    self.pred_bbox = torch.nn.ModuleList([
      torch.nn.Linear(self.channels[i], 6) for i in range(self.start_bbox, self.stage_num)])
    self.pred_tri_vert = torch.nn.ModuleList([
      torch.nn.Linear(self.channels[i], 9) for i in range(self.start_bbox, self.stage_num)])

  def decode(self, convs: dict, octbvh_in: OctBVH, octbvh_out: OctBVH,
                     depth: int, update_octree: bool = False):
    splits, bboxes, tri_verts = dict(), dict(), dict()
    deconv = convs[depth]
    for i in range(self.stage_num):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i-1](deconv, octbvh_out, d-1)
        if d in convs:
          if i >= self.start_pred:
            with torch.autocast('cuda', enabled=False):
              skip = ocnn.nn.octree_align(convs[d], octbvh_in, octbvh_out, d)
          else:
            skip = convs[d]
          deconv = deconv + skip  # skip connections
      deconv = self.blocks[i](deconv, octbvh_out, d)
      deconv = self.mlp[i](deconv)

      # predict the splitting label and signal
      if i >= self.start_pred:
        j = i - self.start_pred
        split_d = self.pred_split[j](deconv)
        nnum = octbvh_out.nnum[d]
        splits[d] = split_d[-nnum:]

      # regress signals and pad zeros to non-leaf nodes
      if i >= self.start_bbox:
        j = i - self.start_bbox
        bbox_d = self.pred_bbox[j](deconv)
        tri_vert_d = self.pred_tri_vert[j](deconv)
        if self.nempty:
          bbox_d = ocnn.nn.octree_pad(bbox_d, octbvh_out, d)
          tri_vert_d = ocnn.nn.octree_pad(tri_vert_d, octbvh_out, d)
          
        bbox_d = torch.cat(bvh_utils.box_xyzxyz_to_cxcywhd(octbvh_in.bboxes_min[d], octbvh_in.bboxes_max[d]), dim=1)
        # tri_vert_d = octbvh_in.global_bboxes_verts[d].reshape(-1, 9)
        bboxes[d] = bbox_d
        tri_verts[d] = tri_vert_d

      # update the octree according to predicted labels
      if update_octree:
        if i >= self.start_pred:
          split_d = split_d.argmax(1).long()
        else:
          split_d = octbvh_in.bboxes_type[d]
        bbox_min, bbox_max = bvh_utils.box_cxcyczwhd_to_xyzxyz(bboxes[d][:, :3], bboxes[d][:, 3:])
        octbvh_out.bvh_grow(d, None, bbox_min.detach(), bbox_max.detach(), tri_verts[d].detach())

    return {'splits': splits, 'bboxes': bboxes, 'tri_verts': tri_verts, 'octbvh_out': octbvh_out}

  def forward(self, code: torch.Tensor, depth: int, octbvh_in: OctreeD,
              octbvh_out: OctreeD, pos: torch.Tensor = None,
              update_octree: bool = False):
    # run encoder and decoder
    output = self.decode(code, octbvh_in, octbvh_out, depth, update_octree)
    
    return output

class DiagonalGaussian(object):

  def __init__(self, parameters: torch.Tensor, norm: bool = True, noise_strength: float = 0.5):
    super().__init__()
    self.parameters = parameters
    self.noise_strength = noise_strength
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    if norm:
      self.mean = F.layer_norm(self.mean, (self.mean.shape[1],))
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self):
    z = self.mean + self.std * torch.randn_like(self.mean)
    if self.noise_strength > 0:
      p = torch.distributions.Uniform(0, self.noise_strength)
      z = z + p.sample((z.shape[0], 1)).to(z.device) * torch.randn_like(z)
    return z

  def kl(self, other: Optional['DiagonalGaussian'] = None):
    if other is None:
      out = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
    else:
      out = 0.5 * (torch.pow(self.mean - other.mean, 2) / other.var +
                   self.var / other.var - 1.0 - self.logvar + other.logvar)
    return out


def l2_norm(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
  return x / torch.clamp(x.norm(dim=-1, keepdim=True), min=eps)


class PowerSphericalDistribution:
  def __init__(self, parameters: torch.Tensor, eps: float = 1e-7):
    self.eps = eps
    mu = parameters[..., :-1]
    kappa = parameters[..., -1]
    kappa = F.softplus(kappa) + 1.0
    self.mu = l2_norm(mu, eps)  # [..., m]
    self.kappa = torch.clamp(kappa, min=0.0)

    self.m = self.mu.shape[-1]
    self.d = self.m - 1
    beta_const = 0.5 * self.d
    self.alpha = self.kappa + beta_const  # [...,]
    self.beta = torch.as_tensor(
        beta_const, dtype=self.kappa.dtype, device=self.kappa.device
    ).expand_as(self.kappa)

  def _log_normalizer(self) -> torch.Tensor:
    # log N_X(κ,d) = -[ (α+β)log 2 + β log π + lgamma(α) - lgamma(α+β) ]
    return (
        -(self.alpha + self.beta) * math.log(2.0)
        - self.beta * math.log(math.pi)
        - torch.lgamma(self.alpha)
        + torch.lgamma(self.alpha + self.beta)
    )

  def log_prob(self, x: torch.Tensor) -> torch.Tensor:
    dot = (self.mu * x).sum(dim=-1).clamp(-1.0, 1.0)
    return self._log_normalizer() + self.kappa * torch.log1p(dot)

  def entropy(self) -> torch.Tensor:
    # H = -[ log N_X + κ ( log 2 + ψ(α) - ψ(α+β) ) ]
    return -(
        self._log_normalizer()
        + self.kappa
        * (
            math.log(2.0)
            + (torch.digamma(self.alpha) - torch.digamma(self.alpha + self.beta))
        )
    )

  def kl(self) -> torch.Tensor:
    # KL(q || U(S^{d})) = -H(q) + log |S^{d}|
    d = torch.as_tensor(self.d, dtype=self.kappa.dtype,
                        device=self.kappa.device)
    log_area = (
        math.log(2.0)
        + 0.5 * (d + 1.0) * math.log(math.pi)
        - torch.lgamma(0.5 * (d + 1.0))
    )
    return -self.entropy() + log_area

  def sample(self):
    Z = torch.distributions.Beta(self.alpha, self.beta).rsample()  # [*S, *B]
    t = (2.0 * Z - 1.0).unsqueeze(-1)  # [*S, *B, 1]

    # 2) v ~ U(S^{m-2})
    v = torch.randn(
        *self.mu.shape[:-1],
        self.m - 1,
        device=self.mu.device,
        dtype=self.mu.dtype,
    )  # [*S, *B, m-1]
    v = l2_norm(v, self.eps)

    y = torch.cat(
        [t, torch.sqrt(torch.clamp(1 - t**2, min=0.0)) * v], dim=-1
    )  # [*S, *B, m]

    e1 = torch.zeros_like(self.mu)
    e1[..., 0] = 1.0
    u = l2_norm(e1 - self.mu, self.eps)
    if u.dim() < y.dim():
      u = u.view((1,) * (y.dim() - u.dim()) + u.shape)
    z = y - 2.0 * (y * u).sum(dim=-1, keepdim=True) * u

    parallel = (self.mu - e1).abs().sum(dim=-1, keepdim=True) < 1e-6
    if parallel.any():
      p = parallel
      if p.dim() < y.dim() - 1:
        p = p.view((1,) * (y.dim() - 1 - p.dim()) + p.shape)
      z = torch.where(p, y, z)
    # normalize z by the dimension
    return z

class BVHVAE(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.in_channels = config.in_channels
    self.n_node_type = getattr(config, "n_node_type", 7)
    self.quantizer_type = config.quantizer_type
    self.noise_strength = getattr(config, "noise_strength", 0.0)
    self.nempty = getattr(config, "nempty", True)
    self.config_network()

    self.encoder = Encoder(
        self.in_channels, self.enc_channels, self.enc_resblk_nums, self.bottleneck, self.nempty)
    self.decoder = Decoder(
        self.n_node_type, self.dec_channels, self.dec_resblk_nums, self.bvh_stage_num,
        self.split_stage_nums, self.bottleneck, self.nempty)
    
    if self.quantizer_type == 'None':
      enc_channels = config.embedding_channels
    elif self.quantizer_type == 'Gaussian':
      enc_channels = config.embedding_channels * 2
    elif self.quantizer_type == 'Sphere':
      enc_channels = config.embedding_channels + 1
    dec_channels = config.embedding_channels

    self.pre_proj = torch.nn.ModuleDict()
    self.post_proj = torch.nn.ModuleDict()
    for d in range(self.full_depth, self.code_depth + 1):
      self.pre_proj[str(d)] = torch.nn.Linear(
          self.enc_channels[self.depth - d], enc_channels, bias=True)
      self.post_proj[str(d)] = torch.nn.Linear(
          dec_channels, self.dec_channels[::-1][self.depth - d], bias=True)
  
  def config_network(self):
    self.bottleneck = 1
    self.bvh_stage_num = 7
    self.split_stage_nums = 0
    self.full_depth = 3
    self.depth = 9
    self.code_depth = 9

    self.enc_channels = [16, 32, 64, 128, 256, 256, 512]
    self.enc_resblk_nums = [2, 2, 2, 2, 2, 2, 2]

    self.dec_channels = [512, 256, 256, 128, 64, 32, 16]
    self.dec_resblk_nums = [2, 2, 2, 2, 2, 2, 2]

  def extract_code(self, octbvh_in, batch):
    depth = octbvh_in.depth
    data = get_input_feature(octbvh_in, batch, self.nempty)
    code = self.encoder(data, octbvh_in, depth)
    z = {}
    vae_loss = {}
    for d in range(self.full_depth, self.code_depth + 1):
      # code[d] = self.pre_proj[str(d)](code[d])
      z[d] = code[d]
      # posterior_d = self.get_posterior(code[d])
      # z[d] = posterior_d.sample()
      # vae_loss[d] = posterior_d.kl()

    return z, vae_loss

  def decode_code(self, code: torch.Tensor, code_depth: int, octbvh_in: OctreeD,
                  octbvh_out: OctreeD, pos: torch.Tensor = None,
                  update_octree: bool = False):
    # for d in range(self.full_depth, self.code_depth + 1):
    #   code[d] = self.post_proj[str(d)](code[d])
    #   code[d] = self.normalize(code[d])
    output = self.decoder(code, self.full_depth, octbvh_in, octbvh_out,
                          pos, update_octree)
    return output
  
  def normalize(self, x: torch.Tensor):
    if self.quantizer_type == 'Gaussian':
      # return F.layer_norm(x, (x.shape[1],))
      return x
    elif self.quantizer_type == 'Sphere':
      return l2_norm(x) * (x.shape[-1] ** 0.5)
    else:
      return x
  
  def get_posterior(self, code: torch.Tensor):
    if self.quantizer_type == 'Gaussian':
      posterior = DiagonalGaussian(
        code, norm=True, noise_strength=self.noise_strength if self.training else 0.0)
    elif self.quantizer_type == 'Sphere':
      posterior = PowerSphericalDistribution(code)
    else:
      raise NotImplementedError
    
    return posterior

  def forward_model(self, batch, training: bool = True):
    octbvh_in = batch['octbvh']
    octbvh_out = copy.deepcopy(octbvh_in) 
    #if training else init_octbvh(self.depth, self.full_depth).cuda()
    
    z, vae_loss = self.extract_code(octbvh_in, batch)
    
    output = self.decode_code(z, self.full_depth, octbvh_in,
                              octbvh_out, update_octree=not training)
    # output['vae_loss'] = torch.cat(list(vae_loss.values())).mean()
    # output['code_max'] = z.max()
    # output['code_min'] = z.min()
    return output
  
  def forward(self, batch):
    octbvh_in = batch['octbvh']
    output = self.forward_model(batch, training=True)
    loss_dict = {}
    for d in range(self.full_depth, self.code_depth + 1):
      # if d in output['splits']:
      #   loss_dict[f'split_loss_{d}'] = F.cross_entropy(output['splits'][d], batch['splits'][d])
      # if d in output['bboxes']:
      #   mask_nonbvh = batch['splits'][d] > 0
      #   if mask_nonbvh.any():
      #     loss_dict[f'bbox_loss_{d}'] = F.l1_loss(output['bboxes'][d][mask_nonbvh], batch['bboxes'][d][mask_nonbvh])
      #     loss_dict[f'iou_loss_{d}'] = loss_utils.diou_3d_loss(output['bboxes'][d][mask_nonbvh], batch['bboxes'][d][mask_nonbvh])
      if d in output['tri_verts']:
        mask_leaf = batch['splits'][d] > 0
        if mask_leaf.any():
          loss_dict[f'tri_vert_loss_{d}'] = F.l1_loss(output['tri_verts'][d][mask_leaf], batch['tri_verts'][d][mask_leaf])
    return loss_dict
  def generate(self, batch):
    output = self.forward_model(batch, training=False)
    return output['octbvh_out']