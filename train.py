import os
import torch
from torch.nn import functional as F
import ocnn
from tqdm import tqdm
from thsolver import Solver
import numpy as np
from octree import Octree
from octree import merge_octrees

from datascts import get_shapenet_dataset, generate_camera_matrices
from models import NeuralIntersectionModel, get_embedder, LocalPositionEncoder
# from models import PNHLoss as IntersectionLoss 
from models import PNCHLoss as IntersectionLoss
from torchmetrics.image import StructuralSimilarityIndexMeasure
from thsolver.tracker import AverageTracker
from thsolver.sampler import InfSampler, DistributedInfSampler
import math

from datetime import timedelta

from RayDataset import Renderer
from utils_train import read_transforms


class VisSolver(Solver):
    def __init__(self, flags, is_master=True):
        super().__init__(flags, is_master)
        self.embedder, vd_ch = get_embedder(flags.MODEL.multires)
        self.local_position_encoder = LocalPositionEncoder(depth=5)
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    
    def get_model(self, flags):
        model = NeuralIntersectionModel(flags.channel)
        return model
    
    def get_dataset(self, flags):
        return get_shapenet_dataset(flags)
    
    def get_input_feature(self, octree, flags):
        depth = octree.depth
        f = flags.feature
        f = f.upper()
        features = []
        if 'N' in f:
            features.append(octree.normals[depth])
        if 'F' in f:
            features.append(octree.features[depth])
        if 'L' in f or 'D' in f:
            local_points = octree.points[depth].frac() - 0.5
        if 'L' in f:
            features.append(local_points)
        if 'D' in f:
            dis = torch.sum(local_points * octree.normals[depth], dim=1, keepdim=True)
            features.append(dis)
        if 'P' in f:
            scale = 2 ** (1 - depth)   # normalize [0, 2^depth] -> [-1, 1]
            global_points = octree.points[depth] * scale - 1.0
            features.append(global_points)
        out = torch.cat(features, dim=1)
        return out

    def process_batch_multi(self, batch, flags):
        def points2octree(points):
            octree = Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['points'] = ocnn.octree.merge_points(points)
        batch['octree'] = octree
        pins = [pin.cuda(non_blocking=True) for pin in batch['pins']]
        batch['pins'] = torch.cat(pins, dim=0)
        pouts = [pout.cuda(non_blocking=True) for pout in batch['pouts']]
        batch['pouts'] = torch.cat(pouts, dim=0)
        indices = [idx.cuda(non_blocking=True) for idx in batch['indices']]
        batch['indices'] = torch.cat(indices, dim=0)
        hits = [hit.cuda(non_blocking=True) for hit in batch['hits']]
        batch['hits'] = torch.cat(hits, dim=0)
        positions = [pos.cuda(non_blocking=True) for pos in batch['positions']]
        batch['positions'] = torch.cat(positions, dim=0)
        normals = [normal.cuda(non_blocking=True) for normal in batch['normals']]
        batch['normals'] = torch.cat(normals, dim=0)
        colors = [color.cuda(non_blocking=True) for color in batch['colors']]
        batch['colors'] = torch.cat(colors, dim=0)
        labels = [label.cuda(non_blocking=True) for label in batch['labels']]
        batch['labels'] = torch.cat(labels, dim=0)
        # viewdirs = [vd.cuda() for vd in batch['viewdirs']]
        # batch['viewdirs'] = torch.cat(viewdirs, dim=0)
        # labels = [label.cuda() for label in batch['labels']]
        # batch['labels'] = torch.cat(labels, dim=0).reshape(-1,1)
        return batch
    
    def process_batch(self, batch, flags):
        # def points2octree(points):
        #     octree = Octree(flags.depth, flags.full_depth, device=points.device)
        #     octree.build_octree(points)
        #     return octree

        # points = batch['points'].cuda(non_blocking=True)
        # octree = points2octree(points)
        # # octree = merge_octrees(octrees)
        # octree.construct_all_neigh()
        # batch['points'] = points
        # batch['octree'] = octree
        # # pins = [pin.cuda(non_blocking=True) for pin in batch['pins']]
        # batch['pins'] = batch['pins'].cuda(non_blocking=True)
        # # batch['pins'] = torch.cat(pins, dim=0)
        # # pouts = [pout.cuda(non_blocking=True) for pout in batch['pouts']]
        # # batch['pouts'] = torch.cat(pouts, dim=0)
        # batch['pouts'] = batch['pouts'].cuda(non_blocking=True)
        # # indices = [idx.cuda(non_blocking=True) for idx in batch['indices']]
        # # batch['indices'] = torch.cat(indices, dim=0)
        # # hits = [hit.cuda(non_blocking=True) for hit in batch['hits']]
        # # batch['hits'] = torch.cat(hits, dim=0)
        # batch['indices'] = batch['indices'].cuda(non_blocking=True)
        # batch['hits'] = batch['hits'].cuda(non_blocking=True)
        # # positions = [pos.cuda(non_blocking=True) for pos in batch['positions']]
        # # batch['positions'] = torch.cat(positions, dim=0)
        # # normals = [normal.cuda(non_blocking=True) for normal in batch['normals']]
        # # batch['normals'] = torch.cat(normals, dim=0)
        # # colors = [color.cuda(non_blocking=True) for color in batch['colors']]
        # # batch['colors'] = torch.cat(colors, dim=0)
        # # labels = [label.cuda(non_blocking=True) for label in batch['labels']]
        # # batch['labels'] = torch.cat(labels, dim=0)
        # batch['positions'] = batch['positions'].cuda(non_blocking=True)
        # batch['normals'] = batch['normals'].cuda(non_blocking=True)
        # batch['colors'] = batch['colors'].cuda(non_blocking=True)
        # batch['labels'] = batch['labels'].cuda(non_blocking=True)
        # viewdirs = [vd.cuda() for vd in batch['viewdirs']]
        # batch['viewdirs'] = torch.cat(viewdirs, dim=0)
        # labels = [label.cuda() for label in batch['labels']]
        # batch['labels'] = torch.cat(labels, dim=0).reshape(-1,1)
        # batch['octree'].construct_all_neigh()
        return batch


    def remapping_image(self, labels: torch.Tensor, colors: torch.Tensor, H = 512, W = 512, fill: float = 0):
        C = colors.shape[-1]
        # image = torch.full((H * W, C), fill, dtype=colors.dtype, device=colors.device)
        # print((H * W, C))
        assert colors.device != torch.device('cpu')
        image = torch.zeros((H * W, C), device=colors.device, dtype=colors.dtype)

        image[labels] = colors
        return image.view(H, W, C)

    def model_forward(self, batch):
        octree, points, pins, pouts, indices, hits, positions, normals, colors, labels = \
            batch['octree'], batch['points'], batch['pins'], batch['pouts'], batch['indices'], \
            batch['hits'], batch['positions'], batch['normals'], batch['colors'], batch['labels']
        input_feature = self.get_input_feature(batch['octree'], self.FLAGS.MODEL)
        # query_pts = torch.cat([points.points, points.batch_id], dim=1)
        # viewdirs = self.embedder(viewdirs.view(-1, 3))
        tgt_normal_map = self.remapping_image(labels, normals, fill=0)
        tgt_color_map = self.remapping_image(labels, colors, fill=0)
        tgt_local_pos = self.local_position_encoder.encode(positions)
        directions = F.normalize(pouts - pins, dim=-1)
        
        pred_intersects = self.model(input_feature, octree, octree.depth, pins, directions, indices)
        tgt_intersects = (tgt_local_pos, normals, colors, hits)
        # tgt_intersects = (tgt_local_pos, normals, hits)
        tgt_maps = (tgt_normal_map, tgt_color_map)
        return pred_intersects, tgt_intersects, tgt_maps, labels
        # return pred_intersects, tgt_intersects, tgt_normal_map, labels
    
    def train_step(self, batch, epoch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        pred_intersects, tgt_intersects, tgt_maps, labels = self.model_forward(batch)
        loss, pred_norm_map, pred_color_map = self.loss_function(pred_intersects, tgt_intersects, tgt_maps, labels, epoch)
        # loss, pred_norm_map = self.loss_function(pred_intersects, tgt_intersects, tgt_maps, labels, epoch)
        # loss = self.loss_function(pred_intersects, tgt_intersects, tgt_maps, labels, epoch)
        psnr_color, psnr_norm = self.accuracy((pred_norm_map, pred_color_map), tgt_maps)
        # psnr_norm = self.accuracy(pred_norm_map, tgt_maps)
        return {'train/loss': loss, 'train/psnr_c': psnr_color, 'train/psnr_n': psnr_norm}
        # return {'train/loss': loss, 'train/psnr_n': psnr_norm}
        # return {'train/loss': loss}
    
    def test_step(self, batch, iteration):   
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            pred_intersects, tgt_intersects, tgt_maps, labels = self.model_forward(batch)
        loss, pred_norm_map, pred_color_map = self.test_loss_function(pred_intersects, tgt_intersects, tgt_maps, labels)
        # loss, pred_norm_map = self.test_loss_function(pred_intersects, tgt_intersects, tgt_maps, labels)
        psnr_color, psnr_norm = self.accuracy((pred_norm_map, pred_color_map), tgt_maps)
        # psnr_norm = self.accuracy(pred_norm_map, tgt_maps)
        # save pred_norm_map and pred_color_map as images if needed
        # if self.FLAGS.SOLVER.save_pred and self.is_master:
        #     save_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'gt_images')
        #     os.makedirs(save_dir, exist_ok=True)
        #     iter_num = iteration
        #     norm_image = (pred_norm_map + 1.0) / 2.0 * 255
        #     norm_image = norm_image.cpu().numpy().astype(np.uint8)
        #     color_image = (pred_color_map.cpu().numpy() * 255).astype(np.uint8)
        #     from PIL import Image
        #     norm_pil = Image.fromarray(norm_image)
        #     color_pil = Image.fromarray(color_image)
        #     norm_pil.save(os.path.join(save_dir, 'norm_%06d.png' % iter_num))
        #     color_pil.save(os.path.join(save_dir, 'color_%06d.png' % iter_num))
        return {'test/loss': loss, 'test/psnr_c': psnr_color, 'test/psnr_n': psnr_norm}
        # return {'test/loss': loss, 'test/psnr_n': psnr_norm}
    
    def eval_step(self, batch):
        pass

    def test_loss_function(self, pred_intersects, tgt_intersect, tgt_maps, labels):
        i_loss_fn = IntersectionLoss()
        # ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(labels.device)
        tgt_normal_map, tgt_color_map = tgt_maps
        # tgt_normal_map = tgt_maps
        loss = 0.0
        pred_pos, pred_norm, pred_color, pred_hit = pred_intersects
        # pred_pos, pred_norm, pred_hit = pred_intersects
        tgt_pos, tgt_norm, tgt_color, tgt_hit = tgt_intersect
        # tgt_pos, tgt_norm, tgt_hit = tgt_intersect
        pred_hit_mask = pred_hit > 0.5
        # tgt_hit_mask = tgt_hit > 0.5
        # pred_hit_mask = tgt_hit_mask
        if pred_hit_mask.any():
            pred_hit_norm = pred_norm[pred_hit_mask]
            pred_hit_color = pred_color[pred_hit_mask]
            pred_hit_rays = labels[pred_hit_mask]
            unique_ray_mask = torch.cat([torch.tensor([1], dtype=torch.bool, device=labels.device), pred_hit_rays[1:] != pred_hit_rays[:-1]], dim=0)
            unique_pred_hit_norm = pred_hit_norm[unique_ray_mask]
            unique_pred_hit_color = pred_hit_color[unique_ray_mask]
            unique_pred_hit_rays = pred_hit_rays[unique_ray_mask]
            pred_normal_map = self.remapping_image(unique_pred_hit_rays, unique_pred_hit_norm, fill=0)
            pred_color_map = self.remapping_image(unique_pred_hit_rays, unique_pred_hit_color, fill=0)
        else:
            pred_normal_map = torch.full_like(tgt_normal_map, fill_value=0)
            pred_color_map = torch.full_like(tgt_color_map, fill_value=0)
        ssim_norm = 1.0 - self.ssim_loss(pred_normal_map.permute(2,0,1).unsqueeze(0), tgt_normal_map.permute(2,0,1).unsqueeze(0))
        ssim_color = 1.0 - self.ssim_loss(pred_color_map.permute(2,0,1).unsqueeze(0), tgt_color_map.permute(2,0,1).unsqueeze(0))
        iloss = i_loss_fn(pred_intersects, tgt_intersect)
        loss = iloss + ssim_norm * 0.3 + ssim_color * 0.3
        # loss = iloss + ssim_norm * 0.3
        return loss, pred_normal_map, pred_color_map
        # return loss, pred_normal_map

    def loss_function(self, pred_intersects, tgt_intersect, tgt_maps, labels, epoch):
        i_loss_fn = IntersectionLoss()
        # ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(labels.device)
        loss_weights = self.get_gaussian_loss_weights(epoch, total_epochs=self.FLAGS.SOLVER.max_epoch, num_groups=len(pred_intersects), sigma=1, steepness=6)
        tgt_normal_map, tgt_color_map = tgt_maps
        # tgt_normal_map = tgt_maps
        loss = 0.0
        for loss_weight, intersect in list(zip(loss_weights, pred_intersects)):
            pred_pos, pred_norm, pred_color, pred_hit = intersect
            # pred_pos, pred_norm, pred_hit = intersect
            pred_hit_mask = pred_hit > 0.5
            if pred_hit_mask.any():
                # pred_hit_pos = pred_pos[pred_hit_mask]
                pred_hit_norm = pred_norm[pred_hit_mask]
                pred_hit_color = pred_color[pred_hit_mask]
                pred_hit_rays = labels[pred_hit_mask]
                unique_ray_mask = torch.cat([torch.tensor([1], dtype=torch.bool, device=labels.device), pred_hit_rays[1:] != pred_hit_rays[:-1]], dim=0)
                # unique_pred_hit_pos = pred_hit_pos[unique_ray_mask]
                unique_pred_hit_norm = pred_hit_norm[unique_ray_mask]
                unique_pred_hit_color = pred_hit_color[unique_ray_mask]
                unique_pred_hit_rays = pred_hit_rays[unique_ray_mask]
                # pred_position_map = dataset.remapping_image(unique_pred_hit_rays, unique_pred_hit_pos, fill=0)
                pred_normal_map = self.remapping_image(unique_pred_hit_rays, unique_pred_hit_norm, fill=0)
                pred_color_map = self.remapping_image(unique_pred_hit_rays, unique_pred_hit_color, fill=0)
            else:
                # pred_position_map = torch.zeros_like(tgt_position_map)
                pred_normal_map = torch.full_like(tgt_normal_map, fill_value=0)
                pred_color_map = torch.full_like(tgt_color_map, fill_value=0)
            # ssim_pos = 1.0 - ssim_loss(pred_position_map.permute(2,0,1).unsqueeze(0), tgt_position_map.permute(2,0,1).unsqueeze(0))
            ssim_norm = 1.0 - self.ssim_loss(pred_normal_map.permute(2,0,1).unsqueeze(0), tgt_normal_map.permute(2,0,1).unsqueeze(0))
            ssim_color = 1.0 - self.ssim_loss(pred_color_map.permute(2,0,1).unsqueeze(0), tgt_color_map.permute(2,0,1).unsqueeze(0))
            # criterion = torch.nn.CrossEntropyLoss()
            iloss = i_loss_fn(intersect, tgt_intersect)
            loss = (iloss + ssim_norm * 0.3 + ssim_color * 1) * loss_weight + loss
            # loss = (iloss + ssim_norm * 0.3) * loss_weight + loss
            # loss = iloss * loss_weight + loss
        return loss, pred_normal_map, pred_color_map
        # return loss, pred_normal_map
        # return loss
    
    def accuracy(self, pred_maps, tgt_maps):
        pred_norm_map, pred_color_map = pred_maps
        # pred_norm_map = pred_maps
        tgt_norm_map, tgt_color_map = tgt_maps
        # tgt_norm_map = tgt_maps
        # cal psnr
        mse_color = torch.mean((pred_color_map - tgt_color_map) ** 2)
        psnr_color = -10.0 * torch.log10(mse_color + 1e-8)
        mse_norm = torch.mean((pred_norm_map - tgt_norm_map) ** 2)
        psnr_norm = -10.0 * torch.log10(mse_norm + 1e-8)
        return psnr_color, psnr_norm
        # return psnr_norm
        # pred = visibilities.argmax(dim=-1)
        # correct = pred.eq(labels).float().mean()
        # return correct
    
    def get_gaussian_loss_weights(self, epoch, total_epochs = 100, num_groups = 6, sigma = 1, steepness = 6):
        progress = (epoch / total_epochs - 0.5) * 2 + 0.5
        def sigmoid(x):
            return 1 / (1 + math.exp(-steepness * (x - 0.5)))
        mu = (sigmoid(progress) - 0.5) / (2 * sigmoid(1) - 1) + 0.5
        x = torch.linspace(0, 1, num_groups)
        group_weights = torch.exp(-torch.square((x - mu) / sigma) / 2) + 0.5
        group_weights_normalized = group_weights / torch.sum(group_weights)
        return group_weights_normalized

    def train_epoch(self, epoch):
        self.model.train()
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        flags = self.FLAGS.SOLVER
        train_tracker = AverageTracker()
        log_per_iter = flags.log_per_iter
        rng = range(len(self.train_loader))
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            # clear cache every 50 iterations
            if flags.empty_cache > 0 and it % flags.empty_cache == 0:
                torch.cuda.empty_cache()

            # load data
            batch = next(self.train_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch

            # forward and backward
            self.optimizer.zero_grad(flags.zero_grad_to_none)
            with torch.autocast('cuda', enabled=self.use_amp):
                output = self.train_step(batch, epoch)
                # 'train/loss' is a special key reserved for backward
                loss = output['train/loss']

            clip_grad = self.FLAGS.SOLVER.clip_grad
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

            # track the averaged tensors
            train_tracker.update(output)

            # output intermediate logs
            if self.is_master and log_per_iter > 0 and it % log_per_iter == 0:
                notes = 'iter: %d' % it
                train_tracker.log(epoch, msg_tag='- ', notes=notes, print_time=False)

        # save logs
        if self.world_size > 1:
            train_tracker.average_all_gather()
        if self.is_master:
            train_tracker.log(epoch, self.summary_writer)

    def test_epoch(self, epoch):
        self.model.eval()
        test_tracker = AverageTracker()
        rng = range(len(self.test_loader))
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            # clear cache every 50 iterations
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            # forward
            batch = next(self.test_iter)
            batch['iter_num'] = it
            batch['epoch'] = epoch
            # with torch.no_grad():
            output = self.test_step(batch, it)

            # track the averaged tensors
            test_tracker.update(output, record_time=False)

        if self.world_size > 1:
            test_tracker.average_all_gather()
        if self.is_master:
            self.result_callback(test_tracker, epoch)
            self.save_best_checkpoint(test_tracker, epoch)
            test_tracker.log(epoch, self.summary_writer, self.log_file, msg_tag='=>')

    def render(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.config_dataloader(disable_train_data=True)
        self.load_checkpoint()
        self.model.eval()
        rng = range(len(self.test_loader))
        # read camera matrices
        for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
            filename = self.test_loader.dataset.filenames[it]
            # print(filename)
            object_name, frame_name = filename.split('/')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # data_path = os.path.join("/data/wangjh/transforms_extracted", object_name)
            data_path = os.path.join("/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results/02958343/render_16views", object_name)
            # print(object_name)
            # exit(0)
            frames, camera_matrix = read_transforms(os.path.join(data_path, 'transforms.json'), device)
            frame_id = frame_name.split('.')[0]
            frame_id = int(frame_id)
            frame = frames[frame_id]
            transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)
            transform_matrices = [transform_matrix]
            # clear cache every 50 iterations
            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            # forward
            batch = next(self.test_iter)
            batch = self.process_batch(batch, self.FLAGS.DATA.test)
            octree, points, pins, pouts, indices, hits, positions, normals, colors, labels = \
                batch['octree'], batch['points'], batch['pins'], batch['pouts'], batch['indices'], \
                batch['hits'], batch['positions'], batch['normals'], batch['colors'], batch['labels']
            hit_mask = hits > 0.1
            rays = labels[hit_mask]
            pos = (positions[hit_mask] * (255 / 128)).byte()
            norms = ((normals[hit_mask] + 1.0) / 2.0 * 255).byte()
            cols = (colors[hit_mask] * 255).byte()
            H = W = 512
            gt_position_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
            gt_normal_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
            gt_col_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)
            gt_hit_map = torch.zeros((H * W, 3), dtype=torch.uint8, device=device)

            gt_position_map[rays] = pos
            gt_normal_map[rays] = norms
            gt_col_map[rays] = cols
            gt_hit_map[rays] = 255

            gt_img = torch.cat([gt_position_map.view(H, W, 3), gt_normal_map.view(H, W, 3), gt_col_map.view(H, W, 3), gt_hit_map.view(H, W, 3)], dim=1)
            # gt_img = torch.cat([gt_position_map.view(H, W, 3), gt_normal_map.view(H, W, 3), gt_hit_map.view(H, W, 3)], dim=1)
            input_feature = self.get_input_feature(batch['octree'], self.FLAGS.MODEL)
            # print(input_feature.shape)
            # query_pts = torch.cat([points.points, points.batch_id], dim=1)
            # viewdirs = self.embedder(viewdirs.view(-1, 3))
            tgt_normal_map = self.remapping_image(labels, normals, fill=0)
            tgt_color_map = self.remapping_image(labels, colors, fill=0)
            tgt_local_pos = self.local_position_encoder.encode(positions)
            directions = F.normalize(pouts - pins, dim=-1)
            vert_feat = self.model.feature_model(input_feature, octree, octree.depth)
            # print(vert_feat.shape)
            n_vertices, vert_indexes = octree.compute_vertices(7)
            assert vert_feat.shape[0] == vert_indexes.shape[0]
            all_vertieces = torch.zeros((n_vertices * 8, vert_feat.shape[1] // 8 + 1), dtype=vert_feat.dtype, device=vert_feat.device)
            vert_indexes_expanded = vert_indexes * 8 + torch.arange(8, device=vert_indexes.device).unsqueeze(0)
            vert_feat_with_i = torch.cat([vert_feat.view(vert_indexes.shape[0] * 8, -1), torch.ones((vert_indexes.shape[0] * 8, 1), device=vert_feat.device)], dim=1)
            all_vertieces[vert_indexes_expanded.view(-1)] = vert_feat_with_i.view(vert_indexes.shape[0] * 8, -1)
            average_feat = all_vertieces.view(-1,8, vert_feat.shape[1] // 8 + 1)[:, :, :-1].sum(dim=1) / all_vertieces.view(-1,8, vert_feat.shape[1] // 8 + 1)[:, :, -1:].sum(dim=1)
            renderer = Renderer(self.model.renderer_model, average_feat, vert_indexes, octree, transform_matrices, depth=7)
            output_dir = os.path.join(self.FLAGS.SOLVER.logdir, 'rendering_views')
            os.makedirs(output_dir, exist_ok=True)
            renderer.render(save_images=True, save_video=False, output_dir=output_dir, object_name=object_name, gt_img=[gt_img], frame_ids=[frame_id])
            # batch['iter_num'] = it
            # batch['epoch'] = epoch
            # with torch.no_grad():
            # output = self.test_step(batch, it)

    @classmethod
    def worker(cls, rank, FLAGS):
        # Set the GPU to use.
        gpu = FLAGS.SOLVER.gpu
        torch.cuda.set_device(gpu[rank])

        world_size = len(gpu)
        # print('World size: ', world_size)
        if world_size > 1:
            # Initialize the process group. Currently, the code only supports the
            # `single node + multiple GPU` mode.
            url = 'tcp://localhost:%d' % FLAGS.SOLVER.port
            torch.distributed.init_process_group(
                backend='nccl', init_method=url, world_size=world_size, rank=rank, timeout=timedelta(seconds=7200))

        # The master process is responsible for logging, writing and loading
        # checkpoints. In the multi-GPU setting, we assign the master role to
        # the rank 0 process.
        is_master = rank == 0
        the_solver = cls(FLAGS, is_master)
        the_solver.run()


    # def get_dataloader(self, flags):
    #     dataset, collate_fn = self.get_dataset(flags)

    #     if self.world_size > 1:
    #         sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
    #     else:
    #         sampler = InfSampler(dataset, shuffle=flags.shuffle)

    #     data_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
    #         sampler=sampler, collate_fn=collate_fn)
    #     return data_loader
    # thsolver v1.2.1, have bugs, to be fixed
    # def train_epoch(self, epoch):
    #     self.model.train()
    #     if self.world_size > 1:
    #         self.train_loader.sampler.set_epoch(epoch)

    #     flags = self.FLAGS.SOLVER
    #     avg_tracker = AverageTracker()
    #     rng = range(len(self.train_loader))
    #     for it in tqdm(rng, ncols=80, leave=False, disable=self.disable_tqdm):
    #     # clear cache every 50 iterations
    #         if flags.empty_cache > 0 and it % flags.empty_cache == 0:
    #             torch.cuda.empty_cache()

    #         # load data
    #         batch = next(self.train_iter)
    #         batch['iter_num'] = it
    #         batch['epoch'] = epoch

    #         # forward and backward
    #         self.optimizer.zero_grad(flags.zero_grad_to_none)
    #         with torch.autocast('cuda', enabled=self.use_amp):
    #             output = self.train_step(batch, epoch)
    #             # 'train/loss' is a special key reserved for backward
    #             loss = output['train/loss']

    #         clip_grad = self.FLAGS.SOLVER.clip_grad
    #         if self.use_amp:
    #             self.scaler.scale(loss).backward()
    #             if clip_grad > 0:
    #                 self.scaler.unscale_(self.optimizer)
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
    #             self.scaler.step(self.optimizer)
    #             self.scaler.update()
    #         else:
    #             loss.backward()
    #             if clip_grad > 0:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
    #             self.optimizer.step()

    #         # track the averaged tensors
    #         avg_tracker.update(output)
    #         avg_tracker.record_time()

    #         # output intermediate logs
    #         log_per_iter = flags.log_per_iter
    #         if self.is_master and log_per_iter > 0 and it % log_per_iter == 0:
    #             notes = 'iter: %d' % it
    #             avg_tracker.log(epoch, msg_tag='- ', notes=notes, print_time=False)

    #     # save logs
    #     if self.world_size > 1:
    #         avg_tracker.average_all_gather()
    #     if self.is_master:
    #         avg_tracker.log(epoch, self.summary_writer, print_time=True)


if __name__ == "__main__":
   VisSolver.main()