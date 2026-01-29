import numpy as np
import torch

from thsolver import Dataset
from ocnn.octree import Points
from octree import Octree
import os

# import open3d as o3d

from utils_train import query_valid_ray_voxel_pair, read_image, read_transforms, create_point_cloud_color


def collate_batch(batch : list):
    if len(batch) == 1:
        # exit(0)
        return batch[0]
    outputs = {}
    for key in batch[0].keys():
        outputs[key] = [b[key] for b in batch]
    return outputs


class ShapeNetTransform:
    def __init__(self, noise=0.0, take=8192, random=True, missing=False, has_label=True, has_normal=True, has_vp=True):
        self.noise = noise
        self.random = random
        self.take = take
        self.missing = missing
        self.has_label = has_label
        self.has_normal = has_normal
        self.has_vp = has_vp

    def transform(self, sample: dict, idx: int):
        #random sample N points, num_rand is a random number between 2048 and 8192
        # if self.random:
        #     take = torch.randint(2048, self.take, (1,))
        #     take = take if take < sample['points'].shape[0] else sample['points'].shape[0]
        # else:
        #     take = self.take if self.take < sample['points'].shape[0] else sample['points'].shape[0]
        # idx = torch.randperm(sample['points'].shape[0])[:take]
        # if self.missing and self.has_label:
        #     #for fisrt 8 channel of dim 1 of labels, get idx of those at least one channel is 1
        #     idx = torch.where(sample['labels'][idx, :3].sum(dim=1) > 0)[0]
        # xyz = sample['points'][idx]
        # if self.has_normal:
        #     normal = sample['normals'][idx]
        # if self.has_label:
        #     labels = sample['labels'][idx]
        # if self.has_vp:
        #     vp = sample['vp']
        #add noise
        # if self.noise > 0:
        #     noise = torch.randn_like(xyz)
        #     noise = noise / torch.norm(noise, dim=1, keepdim=True)
        #     noise = noise * self.noise * torch.rand((xyz.shape[0], 1))
        #     xyz = xyz + noise
        # vd = xyz.unsqueeze(1) - vp.unsqueeze(0)
        # vd = vd / torch.norm(vd, dim=2, keepdim=True)
        # points = Points(sample['points'], features=sample['rgb'])
        # # !NOTE: Normalize the points into one unit sphere in [-0.8, 0.8]
        # bbmin, bbmax = points.bbox()
        # points.normalize(bbmin, bbmax, scale=1)

        
        # return {'points': points, 'pins': sample['pins'], 'pouts': sample['pouts'], 
        #         'indices': sample['indices'], 'hits': sample['hits'], 
        #         'positions': sample['positions'],
        #         'normals': sample['normals'],
        #         'colors': sample['colors'],
        #         'labels': sample['labels']}
        return sample

    def __call__(self, sample: dict, idx: int):
        return self.transform(sample, idx)


class ReadNpz:
  def __init__(self, has_normal: bool = True, has_label: bool = False):
    self.has_normal = has_normal
    self.has_label = has_label

  def __call__(self, filename: str, q_depth=7):
    #TODO: read point cloud and normal/rgb/pos from .png file
    # raw = np.load(filename)
    # 把filename的文件名替换为pcd.pth, 保留前面的路径不变，然后用torch.load读取
    '''
    pcd_filename = filename.rsplit('/', 1)[0] + '/pcd.pth'
    points, rgb = torch.load(pcd_filename, weights_only=True)
    pins, pouts, indices, hits, positions, normals, colors, labels = torch.load(filename, weights_only=True)


    output = dict()
    output['points'] = points
    output['rgb'] = rgb
    output['positions'] = positions
    output['hits'] = hits
    output['indices'] = indices
    output['pins'] = pins
    output['pouts'] = pouts
    output['normals'] = normals
    output['colors'] = colors
    output['labels'] = labels
    '''
    data_path = filename.rsplit('/', 1)[0]
    i = filename.rsplit('/', 1)[1].split('.')[0]
    sub_data_path = data_path.rsplit('/', 1)[1]
    pcd_path = "/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results_pcd2pth/02958343/pcd/" + sub_data_path + '/pcd.pth'
    # pcd_path = os.path.join(data_path, 'pcd.pth')
    points, colors = torch.load(pcd_path, weights_only=True)
    points_torch = points.cuda()
    colors_torch = colors.cuda()

    frames, camera_matrix = read_transforms(os.path.join(data_path, 'transforms.json'), device='cuda')
    frame = frames[int(i)]
    # ply_dir = os.path.join(data_path, 'pointcloud.ply')
    # pcd = o3d.io.read_point_cloud(ply_dir)
    # points_np = np.asarray(pcd.points).astype(np.float32)
    # colors_np = np.asarray(pcd.colors).astype(np.float32)
    # points_torch = torch.from_numpy(points_np).cuda()
    # colors_torch = torch.from_numpy(colors_np).cuda()
    points = Points(points_torch, features=colors_torch)
    # bbmin, bbmax = points.bbox()
    # points.normalize(bbmin, bbmax, scale=1)
    points.points = points.points * 1.98
    # print(bbmin, bbmax)
    # exit(0)
    octree = Octree(depth=9, device='cuda')
    octree.build_octree(points)

    file_path = frame['file_path'].split('.')[0]
    depth_min, depth_max = frame['depth']['min'], frame['depth']['max']
    transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float, device='cuda')
    normal_image = read_image(os.path.join(data_path, f'{file_path}_normal.png'))[:,:,:3].cuda()
    # depth_image = read_image(os.path.join(data_path, f'{file_path}_depth.jpg')).cuda()
    depth_image = read_image(os.path.join(data_path, f'{file_path}_depth.png')).cuda()
    albedo_image = read_image(os.path.join(data_path, f'{file_path}_albedo.png'))[:,:,:3].cuda()
    # valid = (depth_image < 255.0).reshape(-1)
    valid = (depth_image != 65535).reshape(-1)
    # depth_image = depth_image.float() * (depth_max - depth_min) / 255.0 + depth_min
    depth_image = depth_image.float() * (depth_max - depth_min) / 65535.0 + depth_min
    origins, dirs, points, normals, albedo = create_point_cloud_color(depth_image, normal_image, albedo_image, camera_matrix, transform_matrix)
    points = points * 0.99
    origins = origins * 0.99
    scale = 1 << q_depth
    half_scale = scale >> 1
    points = points * scale + half_scale
    origins = origins * scale + half_scale
    hits = valid.float()
    intersections = torch.cat([hits.unsqueeze(-1), points, normals, albedo], dim=-1)
    pins, pouts, indices, hits, positions, normals, colors, labels = query_valid_ray_voxel_pair(origins, dirs, octree, intersections, scale, q_depth)
    octree.construct_all_neigh()
    output = dict()
    output['points'] = points
    output['positions'] = positions
    output['hits'] = hits
    output['indices'] = indices
    output['pins'] = pins
    output['pouts'] = pouts
    output['normals'] = normals
    output['colors'] = colors   
    output['labels'] = labels
    output['octree'] = octree
    
    # output['points'] = torch.from_numpy(raw['vertices']).float().view(-1, 3)
    # output["vp"] = torch.from_numpy(raw['viewpoints'][:16, :]).float().view(-1,3)

    # if self.has_normal:
    #   output['normals'] = torch.from_numpy(raw['normals']).float().view(-1, 3)
    # if self.has_label:
    #     output['labels'] = torch.from_numpy(raw['visibilities'][:, :16]).long().unsqueeze(2)
    
    return output

    # def cache_dataset(data_path:str, cache_dir:str, depth:int = 5, device: str = 'cpu'):
    #     scale = 1 << depth
    #     half_scale = scale >> 1

    #     frames, camera_matrix = read_transforms(os.path.join(data_path, 'transforms.json'), device)
    #     # cache_dir = f'./depth{depth}/02958343_car/{object_name}'
    #     os.makedirs(cache_dir, exist_ok=True)
    #     ply_dir = os.path.join(data_path, 'pointcloud.ply')
    #     pcd = o3d.io.read_point_cloud(ply_dir)
    #     points_np = np.asarray(pcd.points).astype(np.float32)
    #     colors_np = np.asarray(pcd.colors).astype(np.float32)
    #     points_torch = torch.from_numpy(points_np)
    #     colors_torch = torch.from_numpy(colors_np)
    #     # torch.save((points_torch, colors_torch), os.path.join(cache_dir, 'pcd.pth'))
    #     points = Points(torch.from_numpy(points_np).to(device))
    #     bbmin, bbmax = points.bbox()
    #     points.normalize(bbmin, bbmax, scale=1)
    #     center = (bbmin + bbmax) / 2
    #     box_size = (bbmax - bbmin).max()
    #     # print(f'Point cloud center: {center}, box size: {box_size}', flush=True)
    #     octree = Octree(depth=depth)
    #     octree.build_octree(points)


    #     iterator = frames
    #     # start_time = time()
        
    #     for i, frame in enumerate(iterator):
    #         cache_file = os.path.join(cache_dir, f'{i}.pth')

    #         file_path = frame['file_path'].split('.')[0]
    #         depth_min, depth_max = frame['depth']['min'], frame['depth']['max']

    #         transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float, device=device)
            
    #         normal_image = read_image(os.path.join(data_path, f'{file_path}_normal.png'))[:,:,:3].to(device)
    #         depth_image = read_image(os.path.join(data_path, f'{file_path}_depth.png')).to(device)
    #         albedo_image = read_image(os.path.join(data_path, f'{file_path}_albedo.png'))[:,:,:3].to(device)
    #         valid = (depth_image != 65535).reshape(-1)
    #         depth_image = depth_image.float() * (depth_max - depth_min) / 65535.0 + depth_min

    #         origins, dirs, points, normals, albedo = create_point_cloud_color(depth_image, normal_image, albedo_image, camera_matrix, transform_matrix)

    #         points = points * scale + half_scale
    #         origins = origins * scale + half_scale
    #         hits = valid.float()

    #         intersections = torch.cat([hits.unsqueeze(-1), points, normals, albedo], dim=-1)
    #         pins, pouts, indices, hits, positions, normals, colors, labels = query_valid_ray_voxel_pair(origins, dirs, octree, intersections, scale, depth)
    #         # pins, pouts, indices, hits, positions, normals, colors, labels = pins.cpu(), pouts.cpu(), indices.cpu(), hits.float().cpu(), positions.cpu(), normals.cpu(), colors.cpu(), labels.cpu()

    #         # torch.save((pins, pouts, indices, hits, positions, normals, colors, labels), cache_file)
    #     return 0

def get_shapenet_dataset(flags):
    transform = ShapeNetTransform(noise=flags.noise, take=flags.takes, random=flags.random, missing=flags.missing)
    read_ply = ReadNpz(has_normal=True, has_label=True)
    dataset = Dataset(flags.location, flags.filelist, transform,
                      read_file=read_ply, take=flags.take)
    return dataset, collate_batch

def generate_camera_matrices(vp):
    """
    Generate camera matrices K, R, and T for a camera pointing towards the origin.
    
    Parameters:
        camera_center (tuple): The 3D coordinates of the camera center (x, y, z).
    
    Returns:
        K (numpy.ndarray): The intrinsic camera matrix.
        R (numpy.ndarray): The rotation matrix.
        T (numpy.ndarray): The translation vector.
    """
    x, y, z = vp[0], vp[1], vp[2]
    
    # 1. Intrinsic camera matrix K
    # Assuming unit focal length and principal point at (0, 0)
    # K = np.eye(3)  # Identity matrix for simplicity
    K = np.array([[537, 0, 240], [0, 537, 240], [0, 0, 1]])
    
    # 2. Rotation matrix R
    # Define the z-axis direction vector pointing towards the origin
    z_axis = np.array([-x, -y, -z])
    z_axis_norm = z_axis / np.linalg.norm(z_axis)
    
    # Define an orthogonal vector in the x-y plane
    if x == 0 and y == 0:
        x_axis = np.array([1, 0, 0])  # Avoid division by zero
    else:
        x_axis = np.array([y, -x, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Compute the third orthogonal vector using cross product
    y_axis = np.cross(z_axis_norm, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Construct the rotation matrix R
    R = np.column_stack((x_axis, y_axis, z_axis_norm))
    
    # 3. Translation vector T
    T = -np.dot(R, np.array([x, y, z]))
    
    return K, R, T