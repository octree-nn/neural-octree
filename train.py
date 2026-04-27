import os
import torch
from torch.nn import functional as F
import ocnn
from tqdm import tqdm
from thsolver import Solver
import numpy as np
from octree import Octree, merge_octrees
from datascts import get_shapenet_dataset
from models import NeuralIntersectionModel, get_embedder, LocalPositionEncoder, NeuralVAEModel
from models import PNCHLoss as IntersectionLoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from thsolver.tracker import AverageTracker
from datetime import timedelta
from utils_render import Renderer
from utils_train import read_transforms
import math

# 常量定义
_DEFAULT_H = 512
_DEFAULT_W = 512
_RAY_MASK_INITIAL_VALUE = 1
_SSIM_DATA_RANGE = 1.0
_GAUSSIAN_SIGMA = 1
_GAUSSIAN_STEEPNESS = 6


class VisSolver(Solver):
    def __init__(self, flags, is_master=True):
        super().__init__(flags, is_master)
        self.embedder, vd_ch = get_embedder(flags.MODEL.multires)
        self.local_position_encoder = LocalPositionEncoder(depth=5)
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=_SSIM_DATA_RANGE).cuda()
        self._setup_constants()

    def _setup_constants(self):
        """初始化训练相关常量"""
        self.loss_weights = {
            'ssim_norm': 0.3,
            'ssim_color': 1.0,
            'test_ssim_norm': 0.3,
            'test_ssim_color': 1.0, 
            'smoothness': 0.2, 
            'constancy': 0.1,
            'kl': 0.001,
            'sub': 0.1
        }

    def get_model(self, flags):
        if flags.use_vae:
            return NeuralVAEModel(flags.channel, flags.out_channel, flags.q_depth)
        else:
            return NeuralIntersectionModel(flags.channel, flags.q_depth)

    def get_dataset(self, flags):
        return get_shapenet_dataset(flags)

    def get_input_feature(self, octree, flags):
        """提取八叉树特征"""
        depth = octree.depth
        feature_config = flags.feature.upper()
        features = []

        self._add_normal_feature(features, octree, depth, feature_config)
        self._add_feature_data(features, octree, depth, feature_config)
        self._add_local_features(features, octree, depth, feature_config)
        self._add_global_points(features, octree, depth, feature_config)

        return torch.cat(features, dim=1) if features else torch.empty(0)

    def _add_normal_feature(self, features, octree, depth, feature_config):
        """添加法线特征"""
        if 'N' in feature_config:
            features.append(octree.normals[depth])

    def _add_feature_data(self, features, octree, depth, feature_config):
        """添加特征数据"""
        if 'F' in feature_config:
            features.append(octree.features[depth])

    def _add_local_features(self, features, octree, depth, feature_config):
        """添加局部特征（局部点坐标和距离）"""
        if 'L' in feature_config or 'D' in feature_config:
            local_points = octree.points[depth].frac() - 0.5
            
        if 'L' in feature_config:
            features.append(local_points)
            
        if 'D' in feature_config:
            distance = torch.sum(local_points * octree.normals[depth], dim=1, keepdim=True)
            features.append(distance)

    def _add_global_points(self, features, octree, depth, feature_config):
        """添加全局坐标特征"""
        if 'P' in feature_config:
            scale = 2 ** (1 - depth)  # 归一化 [0, 2^depth] -> [-1, 1]
            global_points = octree.points[depth] * scale - 1.0
            features.append(global_points)

    def _build_single_octree(self, points, flags):
        """构建单个八叉树"""
        octree = Octree(flags.depth, flags.full_depth)
        octree.build_octree(points)
        return octree

    def process_batch_multi(self, batch, flags):
        """处理多个样本的批次数据"""
        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [self._build_single_octree(pts, flags) for pts in points]
        octree = merge_octrees(octrees)
        octree.construct_all_neigh()
        
        batch['points'] = ocnn.octree.merge_points(points)
        batch['octree'] = octree
        
        # 批量处理张量并转移到GPU
        tensor_keys = ['pins', 'pouts', 'indices', 'hits', 'positions', 
                      'normals', 'colors', 'labels']
        for key in tensor_keys:
            if key in batch:
                tensors = [t.cuda(non_blocking=True) for t in batch[key]]
                batch[key] = torch.cat(tensors, dim=0)
                
        return batch

    def process_batch(self, batch, flags):
        return batch  # 单样本处理保持原样

    def remapping_image(self, labels, colors, H=_DEFAULT_H, W=_DEFAULT_W, fill=0.0):
        """将标签和颜色重新映射为图像"""
        C = colors.shape[-1]
        assert colors.device != torch.device('cpu'), "Colors should be on GPU"
        
        image = torch.zeros((H * W, C), device=colors.device, dtype=colors.dtype)
        image[labels] = colors
        return image.view(H, W, C)

    def _prepare_ray_data(self, batch):
        """准备光线数据用于前向传播"""
        data_keys = ['octree', 'points', 'pins', 'pouts', 'indices', 
                    'hits', 'positions', 'normals', 'colors', 'labels']
        return (batch[key] for key in data_keys)

    def _prepare_target_maps(self, normals, colors, labels):
        """准备目标图像映射"""
        tgt_normal_map = self.remapping_image(labels, normals, fill=0)
        tgt_color_map = self.remapping_image(labels, colors, fill=0)
        return tgt_normal_map, tgt_color_map

    def model_forward(self, batch):
        """模型前向传播"""
        octree, points, pins, pouts, indices, hits, positions, normals, colors, labels = \
            self._prepare_ray_data(batch)
            
        input_feature = self.get_input_feature(octree, self.FLAGS.MODEL)
        tgt_normal_map, tgt_color_map = self._prepare_target_maps(normals, colors, labels)
        tgt_local_pos = self.local_position_encoder.encode(positions)
        directions = F.normalize(pouts - pins, dim=-1)
        tgt_intersects = (tgt_local_pos, normals, colors, hits)
        tgt_maps = (tgt_normal_map, tgt_color_map)
        if self.FLAGS.MODEL.use_vae:
            # if self.model.training:
            pred_intersects, subs, subs_gt, mean, logvar = self.model(input_feature, octree, octree.depth, pins, directions, indices)
            return pred_intersects, tgt_intersects, tgt_maps, labels, subs, subs_gt, mean, logvar
            # else:
            #     pred_intersects, output_oct = self.model(input_feature, octree, octree.depth, pins, directions, indices)
            #     return pred_intersects, tgt_intersects, tgt_maps, labels, output_oct
        else:
            pred_intersects = self.model(input_feature, octree, octree.depth, pins, directions, indices)        
            return pred_intersects, tgt_intersects, tgt_maps, labels

    def _extract_hit_predictions(self, pred_data, labels, fill_tensor_fn):
        """从预测数据中提取命中点的预测结果"""
        pred_pos, pred_norm, pred_color, pred_hit = pred_data
        pred_hit_mask = pred_hit > 0.5
        
        if not pred_hit_mask.any():
            return fill_tensor_fn()
            
        pred_hit_norm = pred_norm[pred_hit_mask]
        pred_hit_color = pred_color[pred_hit_mask]
        pred_hit_rays = labels[pred_hit_mask]
        
        # 为每个光线保留第一个命中点
        unique_ray_mask = self._get_unique_ray_mask(pred_hit_rays)
        unique_pred_hit_norm = pred_hit_norm[unique_ray_mask]
        unique_pred_hit_color = pred_hit_color[unique_ray_mask]
        unique_pred_hit_rays = pred_hit_rays[unique_ray_mask]
        
        pred_normal_map = self.remapping_image(unique_pred_hit_rays, unique_pred_hit_norm, fill=0)
        pred_color_map = self.remapping_image(unique_pred_hit_rays, unique_pred_hit_color, fill=0)
        
        image_mask = torch.zeros((pred_normal_map.shape[0] * pred_normal_map.shape[1]), dtype=torch.bool, device=pred_normal_map.device)
        image_mask[unique_pred_hit_rays] = True
        
        return pred_normal_map, pred_color_map, image_mask.view(pred_normal_map.shape[0], pred_normal_map.shape[1])

    def _get_unique_ray_mask(self, rays):
        """获取每个光线的唯一命中点掩码"""
        return torch.cat([
            torch.tensor([_RAY_MASK_INITIAL_VALUE], dtype=torch.bool, device=rays.device),
            rays[1:] != rays[:-1]
        ], dim=0)

    def _calculate_ssim_loss(self, pred_map, tgt_map):
        """计算SSIM损失"""
        pred_permuted = pred_map.permute(2, 0, 1).unsqueeze(0)
        tgt_permuted = tgt_map.permute(2, 0, 1).unsqueeze(0)
        return 1.0 - self.ssim_loss(pred_permuted, tgt_permuted)

    def _color_smoothness_loss(self, color_map, mask=None):
        """
        计算颜色平滑度损失
        
        参数:
            color_map: 预测的颜色图，形状为 [H, W, 3]
            mask: 可选掩码，形状为 [H, W]，True表示有效像素
            
        返回:
            smoothness_loss: 颜色平滑度损失
        """
        H, W, C = color_map.shape
        
        # 将颜色图转换为 [C, H, W] 格式
        color_map_permuted = color_map.permute(2, 0, 1)
        
        # 计算水平和垂直方向的梯度
        sobel_x_kernel = torch.tensor([[-1, 0, 1], 
                                       [-2, 0, 2], 
                                       [-1, 0, 1]], dtype=torch.float32, device=color_map.device).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], 
                                       [0, 0, 0], 
                                       [1, 2, 1]], dtype=torch.float32, device=color_map.device).view(1, 1, 3, 3)
        
        # 对每个颜色通道计算梯度
        grad_x_list = []
        grad_y_list = []
        
        for c in range(C):
            channel = color_map_permuted[c:c+1, :, :].unsqueeze(0)  # 形状: [1, 1, H, W]
            grad_x = F.conv2d(channel, sobel_x_kernel, padding=1)
            grad_y = F.conv2d(channel, sobel_y_kernel, padding=1)
            grad_x_list.append(grad_x)
            grad_y_list.append(grad_y)
        
        grad_x = torch.cat(grad_x_list, dim=1)  # 形状: [1, C, H, W]
        grad_y = torch.cat(grad_y_list, dim=1)  # 形状: [1, C, H, W]
        
        # 计算梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        if mask is not None:
            # 扩展掩码以匹配梯度图的形状
            mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # 形状: [1, 1, H, W]
            mask_expanded = mask_expanded.expand(1, C, H, W)
            
            # 只计算有效像素的梯度
            valid_gradient = gradient_magnitude * mask_expanded
            valid_count = mask_expanded.sum()
            
            if valid_count > 0:
                loss = valid_gradient.sum() / valid_count
            else:
                loss = torch.tensor(0.0, device=color_map.device)
        else:
            # 计算所有像素的平均梯度
            loss = gradient_magnitude.mean()
        
        return loss

    def _color_constancy_loss(self, pred_color, tgt_color):
        """
        计算颜色一致性损失
        
        参数:
            pred_color: 预测的颜色图，形状为 [H, W, 3]
            tgt_color: 目标颜色图，形状为 [H, W, 3]
            
        返回:
            constancy_loss: 颜色一致性损失
        """
        # 计算pred_color和tgt_color之间的平均颜色
        pred_mean_color = pred_color.mean(dim=(0, 1))  # 形状: [3]
        tgt_mean_color = tgt_color.mean(dim=(0, 1))  # 形状: [3]
        # 计算平均颜色之间的L2距离
        constancy_loss = F.mse_loss(pred_mean_color, tgt_mean_color)
        return constancy_loss

    def test_loss_function(self, pred_intersects, tgt_intersect, tgt_maps, labels):
        """测试阶段的损失函数"""
        i_loss_fn = IntersectionLoss()
        tgt_normal_map, tgt_color_map = tgt_maps
        
        # 创建填充张量的函数
        def create_fill_tensors():
            return (torch.full_like(tgt_normal_map, fill_value=0),
                   torch.full_like(tgt_color_map, fill_value=0))
        
        # 提取预测结果
        pred_normal_map, pred_color_map, pred_hit_mask = self._extract_hit_predictions(
            pred_intersects, labels, create_fill_tensors
        )
        
        # 计算各项损失
        ssim_norm = self._calculate_ssim_loss(pred_normal_map, tgt_normal_map)
        ssim_color = self._calculate_ssim_loss(pred_color_map, tgt_color_map)
        iloss = i_loss_fn(pred_intersects, tgt_intersect)
        
        # 总损失
        total_loss = (iloss + 
                     ssim_norm * self.loss_weights['test_ssim_norm'] + 
                     ssim_color * self.loss_weights['test_ssim_color'])
        
        return total_loss, pred_normal_map, pred_color_map

    def loss_function(self, pred_intersects, tgt_intersect, tgt_maps, labels, epoch):
        """训练阶段的损失函数"""
        i_loss_fn = IntersectionLoss()
        loss_weights = self._get_gaussian_loss_weights(
            epoch, self.FLAGS.SOLVER.max_epoch, len(pred_intersects)
        )
        tgt_normal_map, tgt_color_map = tgt_maps
        
        total_loss = 0.0
        for loss_weight, intersect in zip(loss_weights, pred_intersects):
            # 创建填充张量的函数
            def create_fill_tensors():
                return (torch.full_like(tgt_normal_map, fill_value=0),
                       torch.full_like(tgt_color_map, fill_value=0),
                          torch.zeros((tgt_normal_map.shape[0], tgt_normal_map.shape[1]), dtype=torch.bool, device=tgt_normal_map.device))
            
            # 提取预测结果
            pred_normal_map, pred_color_map, pred_hit_mask = self._extract_hit_predictions(
                intersect, labels, create_fill_tensors
            )
            
            # 计算各项损失
            ssim_norm = self._calculate_ssim_loss(pred_normal_map, tgt_normal_map)
            ssim_color = self._calculate_ssim_loss(pred_color_map, tgt_color_map)
            iloss = i_loss_fn(intersect, tgt_intersect)
            smoothness_loss = self._color_smoothness_loss(pred_color_map, mask=pred_hit_mask)
            constancy_loss = self._color_constancy_loss(pred_color_map, tgt_color_map)
            
            # 加权损失
            weighted_loss = (iloss + 
                            ssim_norm * self.loss_weights['ssim_norm'] + 
                            ssim_color * self.loss_weights['ssim_color'] + 
                            smoothness_loss * self.loss_weights['smoothness'] + 
                            constancy_loss * self.loss_weights['constancy']) * loss_weight
            total_loss += weighted_loss
            
        return total_loss, pred_normal_map, pred_color_map

    def accuracy(self, pred_maps, tgt_maps):
        """计算精度指标（PSNR）"""
        pred_norm_map, pred_color_map = pred_maps
        tgt_norm_map, tgt_color_map = tgt_maps
        
        # 计算颜色PSNR
        mse_color = torch.mean((pred_color_map - tgt_color_map) ** 2)
        psnr_color = -10.0 * torch.log10(mse_color + 1e-8)
        
        # 计算法线PSNR
        mse_norm = torch.mean((pred_norm_map - tgt_norm_map) ** 2)
        psnr_norm = -10.0 * torch.log10(mse_norm + 1e-8)
        
        return psnr_color, psnr_norm

    def _sigmoid(self, x, steepness):
        """Sigmoid函数"""
        return 1 / (1 + math.exp(-steepness * (x - 0.5)))

    def _get_gaussian_loss_weights(self, epoch, total_epochs, num_groups):
        """获取高斯分布损失权重"""
        progress = (epoch / total_epochs - 0.5) * 2 + 0.5
        mu = (self._sigmoid(progress, _GAUSSIAN_STEEPNESS) - 0.5) / (2 * self._sigmoid(1, _GAUSSIAN_STEEPNESS) - 1) + 0.5
        
        x = torch.linspace(0, 1, num_groups)
        group_weights = torch.exp(-torch.square((x - mu) / _GAUSSIAN_SIGMA) / 2) + 0.5
        return group_weights / torch.sum(group_weights)

    def train_step(self, batch, epoch):
        """训练步骤"""
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        if self.FLAGS.MODEL.use_vae:
            pred_intersects, tgt_intersects, tgt_maps, labels, subs, subs_gt, mean, logvar = self.model_forward(batch)
            recon_loss, pred_norm_map, pred_color_map = self.loss_function(
                pred_intersects, tgt_intersects, tgt_maps, labels, epoch
            )
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            total_loss = recon_loss + kl_loss * self.loss_weights['kl']
            subs_loss = 0.0
            for i, (sub, sub_gt) in enumerate(zip(subs, subs_gt)):
                sub_ = sub.argmax(dim=1).int()
                # if epoch % 5 == 0 and self.is_master:
                    # print(f"Sub {i} - GT Split ratio: {(sub_gt > 0.5).float().mean().item():.4f}")
                    # print(f"Sub {i} - Predicted Split ratio: {(sub_ > 0.5).float().mean().item():.4f}")
                if i == 0:
                    gt_split_ratio = (sub_gt > 0.5).float().mean()
                    pred_split_ratio = (sub_ > 0.5).float().mean()
                # sub_loss = F.binary_cross_entropy_with_logits(sub, sub_gt.float())
                sub_criterion = torch.nn.CrossEntropyLoss()
                sub_loss = sub_criterion(sub, sub_gt.long())
                subs_loss += sub_loss
            total_loss += subs_loss * self.loss_weights['sub']
            return {
                'train/loss': total_loss,
                'train/recon_loss': recon_loss,
                'train/kl_loss': kl_loss,
                'train/sub_loss': subs_loss,
                'train/psnr_c': self.accuracy((pred_norm_map, pred_color_map), tgt_maps)[0],
                'train/psnr_n': self.accuracy((pred_norm_map, pred_color_map), tgt_maps)[1],
                'train/gt_split_ratio': gt_split_ratio,
                'train/pred_split_ratio': pred_split_ratio
            }
        pred_intersects, tgt_intersects, tgt_maps, labels = self.model_forward(batch)
        loss, pred_norm_map, pred_color_map = self.loss_function(
            pred_intersects, tgt_intersects, tgt_maps, labels, epoch
        )
        psnr_color, psnr_norm = self.accuracy((pred_norm_map, pred_color_map), tgt_maps)
        
        return {
            'train/loss': loss,
            'train/psnr_c': psnr_color,
            'train/psnr_n': psnr_norm
        }

    def test_step(self, batch, iteration):
        """测试步骤"""
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        
        with torch.no_grad():
            if self.FLAGS.MODEL.use_vae:
                pred_intersects, tgt_intersects, tgt_maps, labels, subs, subs_gt, mean, logvar = self.model_forward(batch)
            else:
                pred_intersects, tgt_intersects, tgt_maps, labels = self.model_forward(batch)

        loss, pred_norm_map, pred_color_map = self.test_loss_function(
            pred_intersects, tgt_intersects, tgt_maps, labels
        )
        psnr_color, psnr_norm = self.accuracy((pred_norm_map, pred_color_map), tgt_maps)
        
        return {
            'test/loss': loss,
            'test/psnr_c': psnr_color,
            'test/psnr_n': psnr_norm
        }

    def _train_backward_step(self, loss, flags):
        """训练反向传播步骤"""
        clip_grad = flags.clip_grad
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        flags = self.FLAGS.SOLVER
        train_tracker = AverageTracker()
        log_per_iter = flags.log_per_iter
        
        for it in tqdm(range(len(self.train_loader)), ncols=80, leave=False, disable=self.disable_tqdm):
            # 定期清理缓存
            if flags.empty_cache > 0 and it % flags.empty_cache == 0:
                torch.cuda.empty_cache()

            # 加载数据
            batch = next(self.train_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch

            # 前向和反向传播
            self.optimizer.zero_grad(flags.zero_grad_to_none)
            
            with torch.autocast('cuda', enabled=self.use_amp):
                output = self.train_step(batch, epoch)
                loss = output['train/loss']

            self._train_backward_step(loss, flags)
            train_tracker.update(output)

            # 输出中间日志
            if self.is_master and log_per_iter > 0 and it % log_per_iter == 0:
                notes = f'iter: {it}'
                train_tracker.log(epoch, msg_tag='- ', notes=notes, print_time=False)

        # 保存日志
        if self.world_size > 1:
            train_tracker.average_all_gather()
        if self.is_master:
            train_tracker.log(epoch, self.summary_writer)

    def test_epoch(self, epoch):
        """测试一个epoch"""
        self.model.eval()
        test_tracker = AverageTracker()
        
        for it in tqdm(range(len(self.test_loader)), ncols=80, leave=False, disable=self.disable_tqdm):
            # 定期清理缓存
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            # 前向传播
            batch = next(self.test_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch
            output = self.test_step(batch, it)

            test_tracker.update(output, record_time=False)

        if self.world_size > 1:
            test_tracker.average_all_gather()
        if self.is_master:
            self.result_callback(test_tracker, epoch)
            self.save_best_checkpoint(test_tracker, epoch)
            test_tracker.log(epoch, self.summary_writer, self.log_file, msg_tag='=>')

    def _prepare_ground_truth_maps(self, batch, device):
        """准备真实值图像映射"""
        hits = batch['hits']
        labels = batch['labels']
        positions = batch['positions']
        normals = batch['normals']
        colors = batch['colors']
        
        hit_mask = hits > 0.1
        rays = labels[hit_mask]
        q_depth = self.FLAGS.DATA.test.q_depth
        scale = 2 ** q_depth
        pos = (positions[hit_mask] * (255 / scale)).byte()
        # pos = (positions[hit_mask] * (255 / 128)).byte()
        norms = ((normals[hit_mask] + 1.0) / 2.0 * 255).byte()
        cols = (colors[hit_mask] * 255).byte()
        
        H = W = _DEFAULT_H
        gt_position_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
        gt_normal_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
        gt_col_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
        gt_hit_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)

        gt_position_map[rays] = pos
        gt_normal_map[rays] = norms
        gt_col_map[rays] = cols
        gt_hit_map[rays] = 255

        return torch.cat([
            gt_position_map.view(H, W, 3),
            gt_normal_map.view(H, W, 3),
            gt_col_map.view(H, W, 3),
            gt_hit_map.view(H, W, 3)
        ], dim=1)

    def _prepare_vertex_features(self, octree, vert_feat):
        """准备顶点特征用于渲染"""
        q_depth = self.FLAGS.DATA.test.q_depth
        n_vertices, vert_indexes = octree.compute_vertices(q_depth)
        assert vert_feat.shape[0] == vert_indexes.shape[0]
        
        all_vertices = torch.zeros(
            (n_vertices * 8, vert_feat.shape[1] // 8 + 1),
            dtype=vert_feat.dtype, device=vert_feat.device
        )
        
        vert_indexes_expanded = vert_indexes * 8 + torch.arange(8, device=vert_indexes.device).unsqueeze(0)
        vert_feat_with_i = torch.cat([
            vert_feat.view(vert_indexes.shape[0] * 8, -1),
            torch.ones((vert_indexes.shape[0] * 8, 1), device=vert_feat.device)
        ], dim=1)
        
        all_vertices[vert_indexes_expanded.view(-1)] = vert_feat_with_i.view(vert_indexes.shape[0] * 8, -1)
        
        # 计算平均特征
        sum_features = all_vertices.view(-1, 8, vert_feat.shape[1] // 8 + 1)[:, :, :-1].sum(dim=1)
        count = all_vertices.view(-1, 8, vert_feat.shape[1] // 8 + 1)[:, :, -1:].sum(dim=1)
        return sum_features / count

    def render(self):
        """渲染场景"""
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()
        
        for it in tqdm(range(len(self.test_loader)), ncols=80, leave=False, disable=self.disable_tqdm):
            filename = self.test_loader.dataset.filenames[it]
            object_name, i = filename.rsplit('/', 1)
            # i = filename.rsplit('/', 1)[1]
            # i = filename.rsplit('/', 1)[1].split('.')[0]
            # sub_data_path = data_path.rsplit('/', 1)[1]
            # object_name, frame_name = filename.split('/')
            
            # 定期清理缓存
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            # 准备相机矩阵
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # data_path = os.path.join(
            #     "/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results/02958343/render_16views",
            #     object_name
            # )
            data_path = os.path.join(
                self.FLAGS.DATA.test.location, object_name
            )
            
            frames, camera_matrix = read_transforms(
                os.path.join(data_path, 'transforms.json'), device
            )
            
            frame_id = int(i)
            frame = frames[frame_id]
            transform_matrix = torch.tensor(
                frame['transform_matrix'], device=device
            )
            transform_matrices = [transform_matrix]

            camera_angle_x = frame['camera_angle_x']

            # 前向传播
            batch = next(self.test_iter)
            batch = self.process_batch(batch, self.FLAGS.DATA.test)
            
            # 准备真实值图像
            gt_img = self._prepare_ground_truth_maps(batch, device)
            
            # 提取特征
            octree = batch['octree']
            labels = batch['labels']
            normals = batch['normals']
            colors = batch['colors']
            positions = batch['positions']
            pins = batch['pins']
            pouts = batch['pouts']
            indices = batch['indices']
            
            tgt_normal_map = self.remapping_image(labels, normals, fill=0)
            tgt_color_map = self.remapping_image(labels, colors, fill=0)
            tgt_local_pos = self.local_position_encoder.encode(positions)
            directions = F.normalize(pouts - pins, dim=-1)
            
            input_feature = self.get_input_feature(octree, self.FLAGS.MODEL)
            n_vertices, _ = octree.compute_vertices(7)
            print("original vertex num:", n_vertices)
            vert_feat, octree = self.model.feature_model(input_feature, octree, octree.depth, update_octree=True)
            n_vertices_updated, _ = octree.compute_vertices(7)
            print("updated vertex num:", n_vertices_updated)
            # 准备顶点特征
            average_feat = self._prepare_vertex_features(octree, vert_feat)
            q_depth = self.FLAGS.DATA.test.q_depth
            # 渲染
            renderer = Renderer(
                self.model.renderer_model, average_feat, 
                octree.compute_vertices(q_depth)[1], octree, transform_matrices, depth=q_depth, camera_angle_x=camera_angle_x
            )
            
            output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_views')
            os.makedirs(output_dir, exist_ok=True)
            
            renderer.render(
                save_images=True, save_video=False, output_dir=output_dir,
                object_name=object_name, gt_img=[gt_img], frame_ids=[frame_id]
            )
            
            if self.FLAGS.SOLVER.save_video:
                #save video
                video_transform = Renderer.generate_views(120)
                video_renderer = Renderer(
                    self.model.renderer_model, average_feat,
                    octree.compute_vertices(q_depth)[1], octree, video_transform, depth=q_depth, camera_angle_x=camera_angle_x
                )

                video_output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_videos')
                os.makedirs(video_output_dir, exist_ok=True)
                video_renderer.render(
                    save_images=False, save_video=True, output_dir=video_output_dir,
                    object_name=object_name, gt_img=None, frame_ids=None
                )
    

    
    def gen_sp(self):
        """渲染场景"""
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()
        
        for it in tqdm(range(len(self.test_loader)), ncols=80, leave=False, disable=self.disable_tqdm):
            filename = self.test_loader.dataset.filenames[it]
            object_name, i = filename.rsplit('/', 1)
            # i = filename.rsplit('/', 1)[1]
            # i = filename.rsplit('/', 1)[1].split('.')[0]
            # sub_data_path = data_path.rsplit('/', 1)[1]
            # object_name, frame_name = filename.split('/')
            
            # 定期清理缓存
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            # 准备相机矩阵
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # data_path = os.path.join(
            #     "/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results/02958343/render_16views",
            #     object_name
            # )
            data_path = os.path.join(
                self.FLAGS.DATA.test.location, object_name
            )
            
            frames, camera_matrix = read_transforms(
                os.path.join(data_path, 'transforms.json'), device
            )
            
            frame_id = int(i)
            frame = frames[frame_id]
            transform_matrix = torch.tensor(
                frame['transform_matrix'], device=device
            )
            transform_matrices = [transform_matrix]

            camera_angle_x = frame['camera_angle_x']

            # 前向传播
            batch = next(self.test_iter)
            batch = self.process_batch(batch, self.FLAGS.DATA.test)
            
            # 准备真实值图像
            gt_img = self._prepare_ground_truth_maps(batch, device)
            
            # 提取特征
            octree = batch['octree']
            labels = batch['labels']
            normals = batch['normals']
            colors = batch['colors']
            positions = batch['positions']
            pins = batch['pins']
            pouts = batch['pouts']
            indices = batch['indices']
            
            tgt_normal_map = self.remapping_image(labels, normals, fill=0)
            tgt_color_map = self.remapping_image(labels, colors, fill=0)
            tgt_local_pos = self.local_position_encoder.encode(positions)
            directions = F.normalize(pouts - pins, dim=-1)
            
            input_feature = self.get_input_feature(octree, self.FLAGS.MODEL)
            vert_feat = self.model.feature_model(input_feature, octree, octree.depth)
            # nx, ny, nz, b = self.xyzb(target_depth, nempty=True)
            nx, ny, nz, b = octree.xyzb(self.FLAGS.DATA.test.q_depth, nempty=True)
            voxel_coords = torch.stack([nx, ny, nz], dim=1)  # (N, 3)

            assert vert_feat.shape[0] == voxel_coords.shape[0], "Vertex features and voxel coordinates must have the same number of vertices"
            # save coord + feat
            sp_output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'sparse_points')
            category = object_name.split('/')[0]
            sp_output_dir_c = os.path.join(sp_output_dir, category)
            os.makedirs(sp_output_dir_c, exist_ok=True)
            # sp_file_path = os.path.join(sp_output_dir, f'{object_name}_frame{_id}_q{self.FLAGS.DATA.test.q_depth}.npz')
            sp_file_path = os.path.join(sp_output_dir, f'{object_name}.pt')

            torch.save((voxel_coords.cpu(), vert_feat.cpu()), sp_file_path)

            # # --- 比较逻辑：从指定的重建目录加载对应 .pt 文件并比较 coords 项是否与 voxel_coords 完全相同 ---
            # recon_dir = '/mnt/sdc/wangjh/neuoct_recons/neuoct_vae_next_dc_f16c32_fp16_step0320000_128'
            # recon_file_path = os.path.join(recon_dir, f'{object_name}.pt')

            # def _load_coords_from_recon(path):
            #     if not os.path.exists(path):
            #         return None, f'Reconstruction file not found: {path}'
            #     try:
            #         data = torch.load(path, map_location='cpu')
            #     except Exception as e:
            #         return None, f'Failed to load {path}: {e}'

            #     # 支持多种可能的存储格式：dict with 'coords', tuple/list (coords, feat), or tensor/ndarray directly
            #     coords = None
            #     if isinstance(data, dict):
            #         if 'coords' in data:
            #             coords = data['coords']
            #         elif 'vertices' in data:
            #             coords = data['vertices']
            #         else:
            #             # try to find a tensor/ndarray entry that looks like coords
            #             for v in data.values():
            #                 if isinstance(v, (torch.Tensor, np.ndarray)) and (v.ndim == 2 and v.shape[1] == 3):
            #                     coords = v
            #                     break
            #     elif isinstance(data, (list, tuple)) and len(data) > 0:
            #         candidate = data[0]
            #         if isinstance(candidate, (torch.Tensor, np.ndarray)) and (candidate.ndim == 2 and candidate.shape[1] == 3):
            #             coords = candidate
            #     elif isinstance(data, (torch.Tensor, np.ndarray)) and (data.ndim == 2 and data.shape[1] == 3):
            #         coords = data

            #     if coords is None:
            #         return None, f'No coords-like entry found in {path}'

            #     # Convert numpy -> torch if needed
            #     if isinstance(coords, np.ndarray):
            #         coords = torch.from_numpy(coords)

            #     return coords, None

            # recon_coords, err = _load_coords_from_recon(recon_file_path)
            # if err is not None:
            #     print(err)
            # else:
            #     # Ensure same device/dtype/shape for fair comparison
            #     target_coords = voxel_coords.cpu()
            #     if not isinstance(recon_coords, torch.Tensor):
            #         recon_coords = torch.tensor(recon_coords)

            #     # If recon coords are floats of integer values, cast to long for comparison
            #     if recon_coords.dtype.is_floating_point and target_coords.dtype != torch.float32:
            #         # try to round then cast
            #         recon_coords_cmp = recon_coords.round().to(target_coords.dtype)
            #     else:
            #         recon_coords_cmp = recon_coords.to(target_coords.dtype)

            #     equal = False
            #     if recon_coords_cmp.shape == target_coords.shape:
            #         try:
            #             equal = torch.equal(recon_coords_cmp, target_coords)
            #         except Exception:
            #             equal = (recon_coords_cmp.size() == target_coords.size()) and torch.all(recon_coords_cmp == target_coords)
            #     else:
            #         equal = False

            #     # 打印比较结果
            #     print(f'Reconstruction file: {recon_file_path}')
            #     print(f'Computed voxel_coords: shape={target_coords.shape} dtype={target_coords.dtype}')
            #     print(f'Recon coords: shape={getattr(recon_coords, "shape", None)} dtype={getattr(recon_coords, "dtype", None)}')
            #     print(f'Coords equal: {equal}')

    def gen_sp_compare_render(self):
        """生成稀疏点并对比，同时分别渲染原始网络预测和重建(recon)的结果

        行为：
        - 计算当前样本的 voxel_coords 和 vert_feat
        - 从 recon 目录加载对应的 .pt（尝试解析 coords 和 feat）
        - 比较 coords 是否完全相同并打印结果
        - 使用原始网络的顶点平均特征进行渲染
        - 如果 recon 中包含可用的特征，则使用 recon 特征进行第二次渲染
        """
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()

        # recon_dir = '/mnt/sdc/wangjh/neuoct_recons/neuoct_vae_next_dc_f16c32_fp16_step0320000_128'
        recon_dir = '/mnt/sdc/wangjh/neuoct_recons/neuoct_sub_all_vae_next_dc_f16c32_fp16_step0520000_1024'

        for it in tqdm(range(len(self.test_loader)), ncols=80, leave=False, disable=self.disable_tqdm):
            filename = self.test_loader.dataset.filenames[it]
            object_name, i = filename.rsplit('/', 1)

            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_path = os.path.join(self.FLAGS.DATA.test.location, object_name)
            frames, camera_matrix = read_transforms(os.path.join(data_path, 'transforms.json'), device)

            frame_id = int(i)
            frame = frames[frame_id]
            transform_matrix = torch.tensor(frame['transform_matrix'], device=device)
            transform_matrices = [transform_matrix]
            camera_angle_x = frame['camera_angle_x']

            # 前向传播，提取必要数据
            batch = next(self.test_iter)
            batch = self.process_batch(batch, self.FLAGS.DATA.test)

            gt_img = self._prepare_ground_truth_maps(batch, device)

            octree = batch['octree']
            labels = batch['labels']
            normals = batch['normals']
            colors = batch['colors']
            positions = batch['positions']
            pins = batch['pins']
            pouts = batch['pouts']
            indices = batch['indices']

            input_feature = self.get_input_feature(octree, self.FLAGS.MODEL)
            vert_feat = self.model.feature_model(input_feature, octree, octree.depth)

            # 计算 voxel coords
            nx, ny, nz, b = octree.xyzb(self.FLAGS.DATA.test.q_depth, nempty=True)
            voxel_coords = torch.stack([nx, ny, nz], dim=1)  # (N, 3)

            # print(f'[gen_sp_compare_render] voxel_coords dtype={voxel_coords.dtype} shape={voxel_coords.shape}')
            # print(f'[gen_sp_compare_render] vert_feat shape={vert_feat.shape} min={vert_feat.min()} max={vert_feat.max()}')
            assert vert_feat.shape[0] == voxel_coords.shape[0], "Vertex features and voxel coordinates must have the same number of vertices"

            # helper: 加载 recon 文件中 coords 和 feats（如果存在）
            def _load_recon_coords_and_feats(path):
                if not os.path.exists(path):
                    return None, None, f'Reconstruction file not found: {path}'
                try:
                    data = torch.load(path, map_location='cpu')
                except Exception as e:
                    return None, None, f'Failed to load {path}: {e}'

                coords = None
                feats = None
                # 支持多种格式
                if isinstance(data, dict):
                    # 常见键名
                    if 'coords' in data:
                        coords = data['coords']
                    elif 'vertices' in data:
                        coords = data['vertices']

                    if 'feat' in data:
                        feats = data['feat']
                    elif 'features' in data:
                        feats = data['features']
                    elif 'vert_feat' in data:
                        feats = data['vert_feat']

                    # 尝试从 dict 值中推断
                    if coords is None or feats is None:
                        for v in data.values():
                            if coords is None and isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 2 and v.shape[1] == 3:
                                coords = v
                            if feats is None and isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 2 and v.shape[1] != 3:
                                # 可能是 feature
                                feats = v
                            if coords is not None and feats is not None:
                                break

                elif isinstance(data, (list, tuple)) and len(data) >= 1:
                    # 常见保存格式 (coords, feat) 或 (coords, feat, ...)
                    cand0 = data[0]
                    cand1 = data[1] if len(data) > 1 else None
                    if isinstance(cand0, (torch.Tensor, np.ndarray)) and cand0.ndim == 2 and cand0.shape[1] == 3:
                        coords = cand0
                    if cand1 is not None and isinstance(cand1, (torch.Tensor, np.ndarray)) and cand1.ndim == 2:
                        feats = cand1

                elif isinstance(data, (torch.Tensor, np.ndarray)) and data.ndim == 2 and data.shape[1] == 3:
                    coords = data

                if coords is None:
                    return None, None, f'No coords-like entry found in {path}'

                # numpy -> torch
                if isinstance(coords, np.ndarray):
                    coords = torch.from_numpy(coords)
                if isinstance(feats, np.ndarray):
                    feats = torch.from_numpy(feats)

                return coords, feats, None
            # for multi class
            object_name = object_name.split('/')[1] if '/' in object_name else object_name
            # recon_file_path = os.path.join(recon_dir, f'{object_name}.pt')
            recon_file_path = os.path.join(recon_dir, f'{object_name}_gen_all.pt')
            recon_coords, recon_feats, err = _load_recon_coords_and_feats(recon_file_path)
            if err is not None:
                print(err)
            else:
                # 比较 coords
                target_coords = voxel_coords.cpu()
                if not isinstance(recon_coords, torch.Tensor):
                    recon_coords = torch.tensor(recon_coords)

                # 对浮点 coords 做近似整形以便比较整数坐标
                if recon_coords.dtype.is_floating_point:
                    recon_cmp = recon_coords.round().to(target_coords.dtype)
                else:
                    recon_cmp = recon_coords.to(target_coords.dtype)

                coords_equal = False
                if recon_cmp.shape == target_coords.shape:
                    coords_equal = torch.equal(recon_cmp, target_coords)
                print(f'[gen_sp_compare_render] Recon file: {recon_file_path}')
                print(f'[gen_sp_compare_render] Computed voxel_coords shape={target_coords.shape} dtype={target_coords.dtype}')
                print(f'[gen_sp_compare_render] Recon coords shape={getattr(recon_coords, "shape", None)} dtype={getattr(recon_coords, "dtype", None)}')
                print(f'[gen_sp_compare_render] Coords equal: {coords_equal}')

            # 渲染：1) 原始网络预测
            average_feat = self._prepare_vertex_features(octree, vert_feat)
            q_depth = self.FLAGS.DATA.test.q_depth
            if self.FLAGS.SOLVER.save_video:
                # render video
                video_transform = Renderer.generate_views(120)
                video_renderer = Renderer(
                    self.model.renderer_model, average_feat,
                    octree.compute_vertices(q_depth)[1], octree, video_transform, depth=q_depth, camera_angle_x=camera_angle_x
                )
                video_output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_videos_orig')
                os.makedirs(video_output_dir, exist_ok=True)
                video_renderer.render(
                    save_images=False, save_video=True, output_dir=video_output_dir,
                    object_name=object_name, gt_img=None, frame_ids=None
                )
            else:
                renderer = Renderer(
                    self.model.renderer_model, average_feat,
                    octree.compute_vertices(q_depth)[1], octree, transform_matrices, depth=q_depth, camera_angle_x=camera_angle_x
                )
                output_dir_orig = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_orig')
                os.makedirs(output_dir_orig, exist_ok=True)
                # print(f'[gen_sp_compare_render] Rendering original prediction to {output_dir_orig} for {object_name}')
                renderer.render(save_images=True, save_video=False, output_dir=output_dir_orig, object_name=object_name, gt_img=[gt_img], frame_ids=[frame_id])

            # 渲染：2) recon 重建（如有可用特征）
            if recon_feats is not None:
                # 处理 recon_feats 使其成为每顶点的平均特征（n_vertices, C)
                n_vertices, vert_indexes = octree.compute_vertices(q_depth)
                recon_avg_feat = None
                # pad recon_feats to vert_feat
                # 从recon_feats中随机丢掉4个点，其他点仍然按从小到大排列，以测试渲染对部分点缺失的鲁棒性
                # if recon_feats.shape[0] > 4:
                #     recon_feats = recon_feats[:-4]
                print(recon_feats.shape, vert_feat.shape, n_vertices)
                if recon_feats.shape[0] != vert_feat.shape[0]:
                    from ocnn.octree import Points
                    recon_p = (recon_cmp + 0.5) / (2 ** (q_depth - 1)) - 1.0
                    points = Points(recon_p.to(device), features=recon_feats.to(device))
                    octree = Octree(depth=7, device=device)
                    octree.build_octree(points)

                    recon_xyzb = torch.cat([recon_cmp, torch.zeros((recon_coords.shape[0], 1), dtype=recon_cmp.dtype)], dim=1).to(device)
                    s_indices = octree.search_xyzb(recon_xyzb, depth = q_depth, nempty=True)
                    print("s_indices shape:", s_indices.shape, "s_indices min:", s_indices.min(), "s_indices max:", s_indices.max())
                    recon_avg_feat = torch.zeros((recon_feats.shape[0], recon_feats.shape[1]), dtype=recon_feats.dtype, device=device)
                    # 去掉s_indices中的-1（表示未找到匹配的点），并将对应的recon_feats填入recon_avg_feat
                    valid_mask = s_indices >= 0
                    valid_s_indices = s_indices[valid_mask]
                    print("valid_s_indices shape:", valid_s_indices.shape, "valid_s_indices min:", valid_s_indices.min(), "valid_s_indices max:", valid_s_indices.max())
                    valid_recon_feats = recon_feats.to(device)[valid_mask]
                    recon_avg_feat[valid_s_indices] = valid_recon_feats
                    # recon_avg_feat[s_indices] = recon_feats

                # recon_avg_feat = self._prepare_vertex_features(octree, recon_avg_feat)    
                # exit(0)
                try:
                    # 如果 recon_feats 与 vert_feat 对应（每体素特征），尝试与 octree 一样的处理
                    if recon_feats.shape[0] == vert_feat.shape[0]:
                        # print("exact match in feature count, applying same vertex preparation for recon features")
                        recon_avg_feat = self._prepare_vertex_features(octree, recon_feats)
                    elif recon_feats.shape[0] == n_vertices:
                        # 已经是每个顶点的特征
                        # print("recon features already at vertex level, using directly")
                        recon_avg_feat = recon_feats
                    else:
                        recon_avg_feat = self._prepare_vertex_features(octree, recon_avg_feat)
                        # 无法直接匹配，尝试填充/聚合：用 zeros 并在匹配 coords 处填入
                        # try:
                        #     recon_avg_feat = torch.zeros((n_vertices, recon_feats.shape[1]), dtype=recon_feats.dtype)
                        #     # 如果 recon_coords 存在，尝试把 recon_feats 映射到最近顶点索引 - 简单策略: 如果 coords 与 voxel_coords 相同数量且顺序相同
                        #     if recon_coords is not None and recon_coords.shape[0] == voxel_coords.shape[0]:
                        #         # 假设对应顺序相同：先 compute mapping from voxel index -> vertex index by using octree.compute_vertices vert_indexes
                        #         # 这里尽力而为：如果 recon_feats 对应每个体素，我们 reuse _prepare_vertex_features
                        #         recon_avg_feat = self._prepare_vertex_features(octree, recon_feats)
                        #         print("recon_avg_feat shape:", recon_avg_feat.shape)
                        # except Exception:
                        #     recon_avg_feat = None
                except Exception as e:
                    print(f'[gen_sp_compare_render] Failed to prepare recon features: {e}')
                    recon_avg_feat = None

                if recon_avg_feat is not None:
                    # ensure device and dtype
                    recon_avg_feat = recon_avg_feat.to(device)
                    if self.FLAGS.SOLVER.save_video:
                        recon_video_transform = Renderer.generate_views(120)
                        recon_video_renderer = Renderer(
                            self.model.renderer_model, recon_avg_feat,
                            octree.compute_vertices(q_depth)[1], octree, recon_video_transform, depth=q_depth, camera_angle_x=camera_angle_x
                        )
                        recon_video_output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_videos_recon')
                        os.makedirs(recon_video_output_dir, exist_ok=True)
                        recon_video_renderer.render(
                            save_images=False, save_video=True, output_dir=recon_video_output_dir,
                            object_name=object_name, gt_img=None, frame_ids=None
                        )
                    else:
                        recon_renderer = Renderer(
                            self.model.renderer_model, recon_avg_feat,
                            octree.compute_vertices(q_depth)[1], octree, transform_matrices, depth=q_depth, camera_angle_x=camera_angle_x
                        )
                        output_dir_recon = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_recon')
                        os.makedirs(output_dir_recon, exist_ok=True)
                        # print(f'[gen_sp_compare_render] Rendering recon prediction to {output_dir_recon} for {object_name}')
                        recon_renderer.render(save_images=True, save_video=False, output_dir=output_dir_recon, object_name=object_name, gt_img=[gt_img], frame_ids=[frame_id])
                else:
                    print(f'[gen_sp_compare_render] Recon features present but could not prepare per-vertex features for rendering.')
            else:
                print(f'[gen_sp_compare_render] No recon features found in {recon_file_path}; skipping recon render.')

    @classmethod
    def worker(cls, rank, FLAGS):
        """多进程工作函数"""
        gpu = FLAGS.SOLVER.gpu
        torch.cuda.set_device(gpu[rank])

        world_size = len(gpu)
        if world_size > 1:
            url = f'tcp://localhost:{FLAGS.SOLVER.port}'
            torch.distributed.init_process_group(
                backend='nccl', init_method=url, world_size=world_size, 
                rank=rank, timeout=timedelta(seconds=7200)
            )

        is_master = rank == 0
        the_solver = cls(FLAGS, is_master)
        the_solver.run()


if __name__ == "__main__":
    VisSolver.main()