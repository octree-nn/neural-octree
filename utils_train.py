import torch
import numpy as np
from PIL import Image
import json
import os
from typing import Tuple, List, Dict

# from sparse_voxels import SparseVoxels, create_sparse_voxels_from_points
from octree import Octree

def read_transforms(json_path: str, device = 'cuda') -> Tuple[List[Dict], torch.Tensor]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    frames = data['frames']
    camera_angle_x = frames[0]['camera_angle_x']
    focal_length = 0.5 / np.tan(camera_angle_x / 2) * 512
    camera_matrix = torch.tensor([
        [focal_length, 0, 256],
        [0, focal_length, 256],
        [0, 0, 1],
    ], dtype=torch.float, device=device)
    return frames, camera_matrix

def read_image(image_path):
    return torch.from_numpy(np.array(Image.open(image_path)))

def generate_directions(W, H, cx, cy, fx, fy, device = 'cuda'):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32, device=device) + 0.5, torch.arange(H, dtype=torch.float32, device=device) + 0.5, indexing='xy')
    # directions = torch.stack([-(i - cx) / fx, -(j - cy) / fy, torch.ones_like(i)], dim=-1) # pytorch3d camera coordinate system
    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], dim=-1)
    return directions

def generate_rays(camera_matrix: torch.Tensor, transform_matrix: torch.Tensor):
    device = camera_matrix.device
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    R, T = transform_matrix[:3, :3], transform_matrix[:3, 3]
    W, H = int(cx * 2), int(cy * 2)

    directions = generate_directions(W, H, cx, cy, fx, fy, device=device)
    dirs = directions @ R.T
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3)
    origins = T.expand(dirs.shape[0], 3)
    return origins, dirs

def create_point_cloud(depth_image: torch.Tensor, normal_image: torch.Tensor, camera_matrix: torch.Tensor, transform_matrix: torch.Tensor):
    device = depth_image.device
    H, W = depth_image.shape
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    R, T = transform_matrix[:3, :3], transform_matrix[:3, 3]

    directions = generate_directions(W, H, cx, cy, fx, fy, device=device)
    dirs = directions @ R.T
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3)
    origins = T.expand(dirs.shape[0], 3)

    points = directions * depth_image[:, :, torch.newaxis]
    points_transformed = points @ R.T + T
    points_transformed = points_transformed.reshape(-1, 3)

    normals = normal_image.reshape(-1, 3).float() / 255.0 * 2.0 - 1.0
    return origins, dirs, points_transformed, normals

def create_point_cloud_color(depth_image: torch.Tensor, normal_image: torch.Tensor, color_image: torch.Tensor, camera_matrix: torch.Tensor, transform_matrix: torch.Tensor):
    device = depth_image.device
    H, W = depth_image.shape
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]
    R, T = transform_matrix[:3, :3], transform_matrix[:3, 3]

    directions = generate_directions(W, H, cx, cy, fx, fy, device=device)
    dirs = directions @ R.T
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3)
    origins = T.expand(dirs.shape[0], 3)

    points = directions * depth_image[:, :, torch.newaxis]
    points_transformed = points @ R.T + T
    points_transformed = points_transformed.reshape(-1, 3)

    normals = normal_image.reshape(-1, 3).float() / 255.0 * 2.0 - 1.0
    #normals to BRG
    normals = normals[:, [2, 0, 1]]
    normals = normals * torch.tensor([1, -1, 1], device=device)

    colors = color_image.reshape(-1, 3).float() / 255.0
    return origins, dirs, points_transformed, normals, colors


def grouped_interleave_indices(sorted_tensor: torch.Tensor, stable: bool = True) -> torch.LongTensor:
    '''
    Args:
        sorted_tensor: Sorted 1-D IntType Tensor.
        stable: If `True`, then fronter element of the same value will get a smaller index in `return_indices`. Default: `True`.

    Returns:
        result_indices: Interleaved arrangement of elements with different values in `sorted_tensor`.
    
    Example:
        
        >>> x = torch.tensor([0, 0, 1, 1, 1, 3, 4, 4])
        >>> indices = grouped_interleave_indices(x)
        >>> indices
        tensor([0, 2, 5, 6, 1, 3, 7, 4])
        >>> y = x[indices]
        >>> y
        tensor([0, 1, 3, 4, 0, 1, 4, 1])
    '''
    n = sorted_tensor.shape[0]
    # 找到每个唯一值的起始位置和计数
    unique_values, counts = torch.unique_consecutive(sorted_tensor, return_counts=True)
    # 计算每个组的起始索引
    start_indices = torch.cat([torch.tensor([0], device=sorted_tensor.device), torch.cumsum(counts, 0)[:-1]])
    # 为每个元素计算在组内的相对位置
    indices = torch.arange(n, device=sorted_tensor.device)
    group_starts = torch.repeat_interleave(start_indices, counts)
    relative_indices = indices - group_starts
    # 按相对位置排序
    result_indices = torch.argsort(relative_indices, stable=stable)
    return result_indices

def query_valid_ray_voxel_pair(origins: torch.Tensor, dirs: torch.Tensor, octree: Octree, intersections: torch.Tensor, scale: int, depth: int):
    '''
    Args:
        origins: Shape (B, 3) . Origins of rays.
        dirs: Shape (B, 3) . Directions of rays.
        octree: The octree of object.
        intersections: Shape (B, 10) . Known intersections of rays and object; hit or not hit in column 0 , positions in column 1~3 and normals in column 4~6.
        scale: Int, equals to 2^depth.
        depth: Int, the depth of octree.

    Returns:
        pin: Shape (N, 3) . Where rays hit into the voxels.
        pout: Shape (N, 3) . Where rays get out of the voxels.
        indices: Shape (N,) . The key of voxels in octree.
        hits: Shape (N,) . Whether or not rays hit the object in the voxels.
        positions: Shape (N, 3) . (If hit) Where the rays hit the voxels.
        normals: Shape (N, 3) . (If hit) The normal of points where the rays hit the voxels.
        tags: Shape (N,) . The index of rays.
    '''
    device = origins.device
    B = origins.shape[0]

    hit_ray_mask = (intersections[:, 0] != 0).unsqueeze(1)
    # print(hit_ray_mask.count_nonzero())

    invdirs = 1.0 / dirs # (torch.where(torch.abs(dirs) < 1e-10, 1e-10, dirs))

    planes = torch.arange(0, scale + 1, step=1, dtype=torch.float, device=device)
    t = (planes.view(1, -1, 1).expand(B, -1, 3) - origins.unsqueeze(1)) * invdirs.unsqueeze(1)
    t = t.view(B, -1)
    t, _ = torch.sort(t, dim=-1)

    points = intersections[:, 1:4]
    # t_hit = (torch.norm(points - origins, dim=1) / torch.norm(dirs, dim=1)).unsqueeze(-1)
    t_hit = ((points[:, 2] - origins[:, 2]) / dirs[:, 2]).unsqueeze(-1)
    before_hit_mask = (t[:, :-1] <= t_hit)
    before_hit_mask_shifted = torch.cat([before_hit_mask[:, 1:], torch.zeros((B, 1), dtype=torch.bool, device=device)], dim=1)
    hit_mask = (before_hit_mask != before_hit_mask_shifted) & hit_ray_mask
    # print(torch.count_nonzero(hit_mask))
    # print(hit_mask.shape)

    xyz = origins[:, torch.newaxis, :] + t.unsqueeze(-1) * dirs[:, torch.newaxis, :]
    pin = xyz[:, :-1, :]
    pout = xyz[:, 1:, :]
    tags = torch.arange(B, dtype=torch.int, device=device).unsqueeze(-1).expand(-1, pin.shape[1])
    hits = torch.zeros_like(tags)
    hits[hit_mask] = 1

    in_bound_mask = ((pin >= -0.001) & (pin <= scale + 0.001) & (pout >= -0.001) & (pout <= scale + 0.001)).all(dim=-1)
    mask1 = before_hit_mask & in_bound_mask

    pin = pin[mask1]
    pout = pout[mask1]
    tags = tags[mask1]
    hits = hits[mask1]

    p_query = ((pin + pout) / 2).long()
    # print(p_query.shape, p_query.min(dim=0).values, p_query.max(dim=0).values)
    p_query_b = torch.cat([p_query, torch.zeros((p_query.shape[0], 1), dtype=torch.long, device=device)], dim=-1)
    all_index = octree.search_xyzb(p_query_b, depth, nempty=True)
    valid_mask = (all_index != -1)
    # print(valid_mask.count_nonzero())

    grid_base = p_query[valid_mask]
    pin = pin[valid_mask] - grid_base
    pout = pout[valid_mask] - grid_base
    tags = tags[valid_mask]
    hits = hits[valid_mask]
    indices = all_index[valid_mask]

    positions = intersections[tags, 1:4]
    # positions[hits < 0.5] = 0
    # positions = positions - torch.floor(positions)
    normals = intersections[tags, 4:7]
    #如果方向点乘法线小于0，说明法线方向错误，取反。方向根据pout - pin
    view_dirs = (pout - pin) / torch.norm(pout - pin, dim=-1, keepdim=True)
    normal_dot_view = (normals * view_dirs).sum(dim=-1)
    wrong_normal_mask = (normal_dot_view > 0)
    normals[wrong_normal_mask] = -normals[wrong_normal_mask]
    
    colors = intersections[tags, 7:10]
    # normals[hits < 0.5] = 0
    # print(hits.count_nonzero().item())

    return pin, pout, indices, hits, positions, normals, colors, tags


def merge_sparse_voxels(voxels0: torch.Tensor, voxels1: torch.Tensor):
    all_voxels = torch.cat([voxels0, voxels1], dim=0)
    return torch.unique(all_voxels, dim=0)

def get_sparse_voxels(data_path: str, depth: int, device: torch.device):
    scale = 1 << depth
    print(f'scale: {scale}')

    frames, camera_matrix = read_transforms(os.path.join(data_path, 'transforms.json'), device)

    all_sparse_voxels = torch.zeros(0, 3, dtype=torch.int, device=device)

    for i, frame in enumerate(frames):

        transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)
        file_path = frame['file_path'].split('.')[0]
        depth_min, depth_max = frame['depth']['min'], frame['depth']['max']

        depth_image = read_image(os.path.join(data_path, f'{file_path}_depth.png')).to(device=device)
        normal_image = read_image(os.path.join(data_path, f'{file_path}_normal.png'))[:,:,:3].to(device=device)

        valid = (depth_image != 65535).reshape(-1)
        depth_image = depth_image.float() * ((depth_max - depth_min) / 65535.0) + depth_min
        origins, dirs, points, normals = create_point_cloud(depth_image, normal_image, camera_matrix, transform_matrix)

        valid_points = points[valid] + 0.5 # (-0.5, 0.5)
        sparse_voxels = torch.unique(torch.clip(torch.floor(valid_points * scale).int(), 0, scale-1), dim=0)
        all_sparse_voxels = merge_sparse_voxels(all_sparse_voxels, sparse_voxels)
        # print(all_sparse_voxels.max(), sparse_voxels.max())
        
    return all_sparse_voxels

