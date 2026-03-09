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
        return batch[0]
    outputs = {}
    for key in batch[0].keys():
        outputs[key] = [b[key] for b in batch]
    return outputs


class ShapeNetTransform:
    def __init__(self):
        pass

    def transform(self, sample: dict, idx: int):
        return sample

    def __call__(self, sample: dict, idx: int):
        return self.transform(sample, idx)


class ReadNpz:
  def __init__(self, has_normal: bool = True, has_label: bool = False):
    self.has_normal = has_normal
    self.has_label = has_label

  def __call__(self, filename: str, q_depth=7):
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

    octree = Octree(depth=9, device='cuda')
    octree.build_octree(points)

    file_path = frame['file_path'].split('.')[0]
    depth_min, depth_max = frame['depth']['min'], frame['depth']['max']
    transform_matrix = torch.tensor(frame['transform_matrix'], dtype=torch.float, device='cuda')
    normal_image = read_image(os.path.join(data_path, f'{file_path}_normal.png'))[:,:,:3].cuda()
    depth_image = read_image(os.path.join(data_path, f'{file_path}_depth.png')).cuda()
    albedo_image = read_image(os.path.join(data_path, f'{file_path}_albedo.png'))[:,:,:3].cuda()
    valid = (depth_image != 65535).reshape(-1)
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
    
    return output

def get_shapenet_dataset(flags):
    transform = ShapeNetTransform(noise=flags.noise, take=flags.takes, random=flags.random, missing=flags.missing)
    read_ply = ReadNpz(has_normal=True, has_label=True)
    dataset = Dataset(flags.location, flags.filelist, transform,
                      read_file=read_ply, take=flags.take)
    return dataset, collate_batch