import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# from modules import NeuralIntersectionModel5 as NeuralIntersectionModel
# from models import IdenticalLocalPositionEncoder as LocalPositionEncoder
# from models import ZeroMeanLocalPositionEncoder as LocalPositionEncoder
# from models import SymmetricLocalPositionEncoder as LocalPositionEncoder
from models import LocalPositionEncoder as LocalPositionEncoder
from utils_train import generate_rays, grouped_interleave_indices, read_image, read_transforms, create_point_cloud
# from sparse_voxels import SparseVoxels, create_sparse_voxels_from_points
from models import VoxelRendererModel as NeuralRendererModel
from Arguments import config_name, object_name, use_tqdm, depth
from octree import Octree

import os
import gc
import cv2
from PIL import Image
from tqdm import tqdm
from time import time
from typing import List, Tuple, Dict

local_position_encode = LocalPositionEncoder(depth)

class TrainingRayDataset(Dataset):
    def __init__(self, data_path : str, depth = 5, device = 'cuda', cache_first: bool = True):

        scale = 1 << depth
        half_scale = scale >> 1

        frames, camera_matrix = read_transforms(os.path.join(data_path, 'transforms.json'), device)

        self.pin: List[Tensor] = []
        self.pout: List[Tensor] = []
        self.indices: List[Tensor] = []
        self.hit: List[Tensor] = []
        self.pos: List[Tensor] = []
        self.norm: List[Tensor] = []
        self.labels: List[Tensor] = []
        self.grid_bases: List[Tensor] = []

        cache_dir = f'./cache_grid/depth{depth}/{object_name}'
        os.makedirs(cache_dir, exist_ok=True)

        cache_sparse_voxels = os.path.join(cache_dir, 'sparse_voxels.pth')
        if os.path.isfile(cache_sparse_voxels) and cache_first:
            sparse_voxels_data = torch.load(cache_sparse_voxels, weights_only=False)
            self.sparse_voxels = SparseVoxels(
                sparse_voxels_data.to(device), 
                depth, 
                device
            )
        else:
            sparse_voxels_coords = get_sparse_voxels(data_path, depth, device)
            self.sparse_voxels = SparseVoxels(sparse_voxels_coords, depth, device)
            torch.save(sparse_voxels_coords.cpu(), cache_sparse_voxels)
        
        print(f'voxel counts: {self.sparse_voxels.num_voxels}')
        print(f'vert counts: {self.sparse_voxels.num_vertices}', flush=True)

        cache_transform_matrices = os.path.join(cache_dir, 'transform_matrices.pth')
        if os.path.isfile(cache_transform_matrices) and cache_first:
            self.transform_matrices: torch.Tensor = torch.load(cache_transform_matrices, weights_only=False)
            transform_matrices_loaded = True
        else:
            transform_matrices_loaded = False
            self.transform_matrices: List[torch.Tensor] = []


        if use_tqdm:
            iterator = tqdm(frames, desc='Loading frames')
        else:
            iterator = frames
            start_time = time()

        for i, frame in enumerate(iterator):
            cache_file = os.path.join(cache_dir, f'{i}.pth')

            file_path = frame['file_path'].split('.')[0]
            depth_min, depth_max = frame['depth']['min'], frame['depth']['max']

            if os.path.isfile(cache_file) and cache_first:
                pins, pouts, indices, hits, positions, normals, labels, grid_bases = torch.load(cache_file, weights_only=False)
                if not transform_matrices_loaded:
                    transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)
                    self.transform_matrices.append(transform_matrix)

            else:
                if transform_matrices_loaded:
                    transform_matrix = self.transform_matrices[i]
                else:
                    transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)
                    self.transform_matrices.append(transform_matrix)
                
                normal_image = read_image(os.path.join(data_path, f'{file_path}_normal.png'))[:,:,:3].to(device)
                depth_image = read_image(os.path.join(data_path, f'{file_path}_depth.png')).to(device)
                valid = (depth_image != 65535).reshape(-1)
                depth_image = depth_image.float() * (depth_max - depth_min) / 65535.0 + depth_min

                origins, dirs, points, normals = create_point_cloud(depth_image, normal_image, camera_matrix, transform_matrix)

                points = points * scale + half_scale
                origins = origins * scale + half_scale
                hits = valid.float()

                intersections = torch.cat([hits.unsqueeze(-1), points, normals], dim=-1)
                pins, pouts, indices, hits, positions, normals, labels, grid_bases = query_valid_ray_voxel_pair(origins, dirs, self.sparse_voxels, intersections, scale, depth)
                pins, pouts, indices, hits, positions, normals, labels, grid_bases = pins.cpu(), pouts.cpu(), indices.cpu(), hits.float().cpu(), positions.cpu(), normals.cpu(), labels.cpu(), grid_bases.cpu()

                torch.save((pins, pouts, indices, hits, positions, normals, labels, grid_bases), cache_file)

            self.pin.append(pins)
            self.pout.append(pouts)
            self.indices.append(indices)
            self.hit.append(hits)
            self.pos.append(positions)
            self.norm.append(F.normalize(normals, dim=-1))
            self.labels.append(labels)
            self.grid_bases.append(grid_bases)

        if use_tqdm:
            iterator.close()
        else:
            end_time = time()
            delta_time = end_time - start_time
            print(f'Finish loading {len(iterator)} frames in {delta_time:.2f}s, fps = {len(iterator) / delta_time:.2f}', flush=True)

        if not transform_matrices_loaded:
            self.transform_matrices = torch.stack(self.transform_matrices, dim=0).cpu()
            torch.save(self.transform_matrices, cache_transform_matrices)
        self.sparse_voxels = self.sparse_voxels.cpu()
        self.length = len(self.pin)
        print(self.length)

        print(f'num training items: {sum([len(item) for item in self.pin])} | Hit counts: {sum([torch.count_nonzero(item).item() for item in self.hit])}', flush=True)

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        '''
        一次加载一个视角
        '''
        return self.pin[idx], self.pout[idx], self.indices[idx], self.hit[idx], self.pos[idx], self.norm[idx], self.labels[idx], self.grid_bases[idx]
    
    @staticmethod
    def remapping_image(labels: torch.Tensor, colors: torch.Tensor, H = 512, W = 512, fill: float = 0):
        C = colors.shape[-1]
        image = torch.full((H * W, C), fill, dtype=colors.dtype, device=colors.device)
        image[labels] = colors
        return image.view(H, W, C)


class Renderer:
    def __init__(self, model: NeuralRendererModel, vfeats: torch.Tensor, vidxs: torch.Tensor, sparse_voxels: Octree, transform_matrices: list[torch.Tensor], camera_angle_x = 0.6981317007977318, W = 512, H = 512, depth = 5, max_batch_size = 262144, device = 'cuda'):
        focal_length = 0.5 / np.tan(camera_angle_x / 2) * W

        self.depth = depth
        self.scale = 2 ** depth
        self.W = W
        self.H = H

        self.model = model
        self.sparse_voxels = sparse_voxels.to(device)
        self.v_feats = vfeats.to(device)
        self.v_idxs = vidxs.to(device)
        self.transform_matrices = transform_matrices
        self.camera_matrix = torch.tensor([
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1],
        ], dtype=torch.float, device=device)

        self.device = device
        self.max_batch_size = max_batch_size

    def render_new_frame(self, transform_matrix: torch.Tensor):
        return Renderer.render_frame(transform_matrix, self.camera_matrix, self.sparse_voxels, self.model, self.v_feats, self.v_idxs, self.depth, self.scale, self.W, self.H, self.max_batch_size, self.device)

    def render(self, save_images = True, save_video = False, output_dir = f'./rendered_views/{config_name}', object_name=f'{object_name}', gt_img=None, fps=60):
        if save_images:
            object_dir = os.path.join(output_dir, object_name)
            os.makedirs(object_dir, exist_ok=True)
        elif save_video:
            os.makedirs(output_dir, exist_ok=True)

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(os.path.join(output_dir, f'{object_name}.mp4'), fourcc, fps, (self.W * 3, self.H))
    
        if use_tqdm:
            iterator = tqdm(self.transform_matrices, desc='Rendering')
        else:
            iterator = self.transform_matrices
            start_time = time()

        for i, transform_matrix in enumerate(iterator):
            image = self.render_new_frame(transform_matrix)
            if gt_img is not None:
                image = torch.cat([image, gt_img[i]], dim=0)
            image = image.cpu().numpy()
            if save_images:
                Image.fromarray(image).save(os.path.join(object_dir, f'{i}.png'))
            if save_video:
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if use_tqdm:
            iterator.close()
        else:
            end_time = time()
            delta_time = end_time - start_time
            # print(f'Finish rendering {len(iterator)} frames in {delta_time:.2f}s, fps = {len(iterator) / delta_time:.2f}', flush=True)

        if save_video:
            writer.release()

    @staticmethod
    def render_frame(transform_matrix: torch.Tensor, camera_matrix: torch.Tensor, sparse_voxels: Octree, model: NeuralRendererModel, v_feats, v_idxs, depth: int, scale: int, W: int, H: int, max_batch_size = 262144, device = 'cuda'):
        B = H * W
        origins, dirs = generate_rays(camera_matrix, transform_matrix)
        origins = origins * scale + scale / 2
        invdirs = 1.0 / (torch.where(torch.abs(dirs) < 1e-10, 1e-10, dirs))
        planes = torch.arange(0, scale + 1, dtype=torch.float32, device=device)

        t = (planes.view(1, -1, 1).expand(B, -1, 3) - origins.unsqueeze(1)) * invdirs.unsqueeze(1)
        t = t.reshape(B, -1)
        t = torch.sort(t, dim=-1).values

        xyz = origins[:, torch.newaxis, :] + t.unsqueeze(-1) * dirs[:, torch.newaxis, :]
        pin = xyz[:, :-1, :]
        pout = xyz[:, 1:, :]
        labels = torch.arange(B, dtype=torch.int, device=device).unsqueeze(-1).expand(-1, pin.shape[1])
        in_bound_mask = ((pin >= -0.001) & (pin <= scale + 0.001) & (pout >= -0.001) & (pout <= scale + 0.001)).all(dim=-1)

        pin = pin[in_bound_mask]
        pout = pout[in_bound_mask]
        labels = labels[in_bound_mask]

        p_query = ((pin + pout) / 2).long()
        # torch.cat([p_query, torch.zeros((p_query.shape[0], 1), dtype=torch.long, device=device)], dim=-1)
        p_query_b = torch.cat([p_query, torch.zeros((p_query.shape[0], 1), dtype=torch.long, device=device)], dim=-1)
        all_index = sparse_voxels.search_xyzb(p_query_b, depth=5, nempty=True)
        valid_mask = torch.nonzero(all_index != -1, as_tuple=False).squeeze(-1)
        
        indices = all_index[valid_mask]
        labels = labels[valid_mask]
        cxyz = p_query[valid_mask]
        pin = pin[valid_mask] - cxyz
        pout = pout[valid_mask] - cxyz

        pred_pos = []
        pred_norm = []
        # pred_color = []
        hit_ray_labels = []

        # 光线交错排列
        interleaved_indices = grouped_interleave_indices(labels)
        pin = pin[interleaved_indices]
        pout = pout[interleaved_indices]
        indices = indices[interleaved_indices]
        labels = labels[interleaved_indices]
        cxyz = cxyz[interleaved_indices]

        # Ray Marching
        with torch.no_grad():
            model.eval()
            while True:
                if len(labels) <= max_batch_size:
                    # print(indices.shape)
                    query_v_indexes = v_idxs[indices].view(-1, 2, 2, 2)
                    query_v_feats = v_feats[query_v_indexes]
                    # pos, norm, color, hit = model(pin, F.normalize(pout - pin, dim=1), query_v_feats)
                    pos, norm, hit = model(pin, F.normalize(pout - pin, dim=1), query_v_feats)
                    hit_mask = hit > 0.5
                    if hit_mask.any():
                        pos = local_position_encode.decode(pos[hit_mask], cxyz[hit_mask]) # pos[hit_mask] + cxyz[hit_mask]
                        norm = norm[hit_mask]
                        # color = color[hit_mask]
                        labels = labels[hit_mask]
                        labels_sorted, sort_indices = torch.sort(labels, stable=True)
                        unique_indices = sort_indices[torch.cat([torch.tensor([True], dtype=torch.bool, device=device), labels_sorted[1:] != labels_sorted[:-1]])]
                        pred_pos.append(pos[unique_indices])
                        pred_norm.append(norm[unique_indices])
                        # pred_color.append(color[unique_indices])
                        hit_ray_labels.append(labels[unique_indices])
                    break

                # 分配光线
                chosen_pin = pin[:max_batch_size]
                chosen_pout = pout[:max_batch_size]
                chosen_indices = indices[:max_batch_size]
                chosen_labels = labels[:max_batch_size]
                chosen_cxyz = cxyz[:max_batch_size]
                
                pin = pin[max_batch_size:]
                pout = pout[max_batch_size:]
                indices = indices[max_batch_size:]
                labels = labels[max_batch_size:]
                cxyz = cxyz[max_batch_size:]

                # 输入选择出来的光线-体素
                query_v_indexes = v_idxs[chosen_indices].view(-1, 2, 2, 2)
                query_v_feats = v_feats[query_v_indexes]
                # pos, norm, color, hit = model(chosen_pin, F.normalize(chosen_pout - chosen_pin, dim=1), query_v_feats)
                pos, norm, hit = model(chosen_pin, F.normalize(chosen_pout - chosen_pin, dim=1), query_v_feats)
                hit_mask = hit > 0.5
                # pos_mask = pos[hit_mask]
                # print(f'pos_mask min: {pos_mask.min(dim=0).values}, max: {pos_mask.max(dim=0).values}')
                if hit_mask.any():
                    pos = local_position_encode.decode(pos[hit_mask], chosen_cxyz[hit_mask]) # pos[hit_mask] + chosen_cxyz[hit_mask]
                    # print(f'pos min: {pos.min(dim=0).values}, max: {pos.max(dim=0).values}')
                    norm = norm[hit_mask]
                    # color = color[hit_mask]
                    hit_rays = chosen_labels[hit_mask]
                    hit_rays_sorted, sort_indices = torch.sort(hit_rays, stable=True)
                    unique_indices = sort_indices[torch.cat([torch.tensor([True], dtype=torch.bool, device=device), hit_rays_sorted[1:] != hit_rays_sorted[:-1]])]
                    unique_hit_rays = hit_rays[unique_indices]
                    pred_pos.append(pos[unique_indices])
                    pred_norm.append(norm[unique_indices])
                    # pred_color.append(color[unique_indices])
                    hit_ray_labels.append(unique_hit_rays)

                    # 删除已经相交的光线
                    hit_ray_logic = torch.cat([unique_hit_rays, torch.tensor([B + 1], dtype=unique_hit_rays.dtype, device=device)], dim=0)
                    idx = torch.bucketize(labels, hit_ray_logic)
                    not_hit_rays = labels != hit_ray_logic[idx]
                    if torch.count_nonzero(not_hit_rays).item() == 0:
                        break
                    pin = pin[not_hit_rays]
                    pout = pout[not_hit_rays]
                    indices = indices[not_hit_rays]
                    labels = labels[not_hit_rays]
                    cxyz = cxyz[not_hit_rays]
        
        pos_img = torch.zeros((B, 3), dtype=torch.uint8, device=device)
        norm_img = torch.zeros((B, 3), dtype=torch.uint8, device=device)
        # color_img = torch.zeros((B, 3), dtype=torch.uint8, device=device)
        hit_img = torch.zeros((B,), dtype=torch.uint8, device=device)

        if len(hit_ray_labels) > 0:
            pred_pos = torch.cat(pred_pos, dim=0)
            pred_norm = torch.cat(pred_norm, dim=0)
            # pred_color = torch.cat(pred_color, dim=0)
            hit_ray_labels = torch.cat(hit_ray_labels)

            pos_img[hit_ray_labels] = (pred_pos * (255.0 / scale)).byte()
            # pos_img[hit_ray_labels] = (pred_pos * 255.0).byte()
            norm_img[hit_ray_labels] = ((pred_norm + 1) * 127.5).byte()
            # color_img[hit_ray_labels] = (pred_color * 255.0).byte()
            hit_img[hit_ray_labels] = 255

        pos_img = pos_img.reshape(H, W, 3)
        norm_img = norm_img.reshape(H, W, 3)
        # color_img = color_img.reshape(H, W, 3)
        hit_img = hit_img.reshape(H, W, 1).repeat(1, 1, 3)
        # return torch.cat([pos_img, norm_img, hit_img], dim=1).cpu().numpy()
        # return torch.cat([pos_img, norm_img, color_img, hit_img], dim=1)
        return torch.cat([pos_img, norm_img, hit_img], dim=1)
        # return torch.cat([pos_img, norm_img, color_img, hit_img], dim=1).cpu().numpy()

    @staticmethod
    def generate_views(num_views = 60, device = 'cuda'):
        phi = torch.linspace(0, 2 * torch.pi, num_views, dtype=torch.float, device=device)
        theta = torch.pi * 0.1 * torch.ones_like(phi)
        x = 2 * torch.cos(theta) * torch.cos(phi)
        y = 2 * torch.cos(theta) * torch.sin(phi)
        z = 2 * torch.sin(theta)
        camera_positions = torch.column_stack((x, y, z))

        transform_matrices = []
        for T in camera_positions:
            t0, t1, t2 = T
            norm_T = torch.norm(T, keepdim=True)
            col2 = T / norm_T
            if torch.isclose(t0, torch.zeros_like(t0)) & torch.isclose(t1, torch.zeros_like(t1)):
                col0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=device)
            else:
                col0 = torch.tensor([-t1, t0, 0.0], dtype=torch.float, device=device)
                norm_col0 = torch.norm(col0, keepdim=True)
                col0 /= norm_col0
            col1 = torch.cross(col2, col0, dim=0)
            R = torch.column_stack((col0, col1, col2))
            transform_matrix = torch.eye(4, dtype=torch.float, device=device)
            transform_matrix[:3, :3] = R
            transform_matrix[:3, 3] = T
            transform_matrices.append(transform_matrix)
        transform_matrices = torch.stack(transform_matrices, dim=0)
        return transform_matrices
