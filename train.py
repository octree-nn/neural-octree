import os
import torch
from torch.nn import functional as F
import ocnn
from tqdm import tqdm
from thsolver import Solver
import numpy as np
from octree import Octree, merge_octrees
from datascts import get_shapenet_dataset
from models import NeuralIntersectionModel, get_embedder, LocalPositionEncoder
from models import PNCHLoss as IntersectionLoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from thsolver.tracker import AverageTracker
from datetime import timedelta
from RayDataset import Renderer
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
            'constancy': 0.1
        }

    def get_model(self, flags):
        return NeuralIntersectionModel(flags.channel)

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
        
        pred_intersects = self.model(input_feature, octree, octree.depth, pins, directions, indices)
        tgt_intersects = (tgt_local_pos, normals, colors, hits)
        tgt_maps = (tgt_normal_map, tgt_color_map)
        
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
                       torch.full_like(tgt_color_map, fill_value=0))
            
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
        pos = (positions[hit_mask] * (255 / 128)).byte()
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
        n_vertices, vert_indexes = octree.compute_vertices(7)
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
            object_name, frame_name = filename.split('/')
            
            # 定期清理缓存
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            # 准备相机矩阵
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            data_path = os.path.join(
                "/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results/02958343/render_16views",
                object_name
            )
            
            frames, camera_matrix = read_transforms(
                os.path.join(data_path, 'transforms.json'), device
            )
            
            frame_id = int(frame_name.split('.')[0])
            frame = frames[frame_id]
            transform_matrix = torch.tensor(
                frame['transform_matrix'], dtype=torch.float32, device=device
            )
            transform_matrices = [transform_matrix]

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
            
            # 准备顶点特征
            average_feat = self._prepare_vertex_features(octree, vert_feat)
            
            # 渲染
            renderer = Renderer(
                self.model.renderer_model, average_feat, 
                octree.compute_vertices(7)[1], octree, transform_matrices, depth=7
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
                    octree.compute_vertices(7)[1], octree, video_transform, depth=7
                )

                video_output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_videos')
                os.makedirs(video_output_dir, exist_ok=True)
                video_renderer.render(
                    save_images=False, save_video=True, output_dir=video_output_dir,
                    object_name=object_name, gt_img=None, frame_ids=None
                )

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