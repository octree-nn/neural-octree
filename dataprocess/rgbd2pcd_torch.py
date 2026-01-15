import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import os

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def read_image(image_path):
    return np.array(Image.open(image_path))

def create_point_cloud(rgb_image, depth_image, camera_matrix, transform_matrix, valid, depth_scale=1.0, device='cuda'):
    height, width = depth_image.shape
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

    # Create meshgrid of pixel coordinates
    # x = torch.arange(width, device=device)
    # y = torch.arange(height, device=device)
    # x, y = torch.meshgrid(x, y, indexing='xy')
    x, y = torch.meshgrid(torch.arange(width, dtype=torch.float32, device=device) + 0.5, torch.arange(height, dtype=torch.float32, device=device) + 0.5, indexing='xy')

    x = (x - cx) / fx
    y = -(y - cy) / fy
    z = -torch.ones_like(x, device=device)

    point_map = torch.stack((x, y, z), dim=-1)  # Shape (H, W, 3)
    points = point_map * depth_image[:, :, None]  # Scale by depth values

    # Apply camera transform
    R, t = transform_matrix[:3, :3], transform_matrix[:3, 3]
    R = R.to(points.dtype)  # Ensure R has the same dtype as points
    t = t.to(points.dtype)
    points = points @ R.T + t  # Transform points to world coordinates

    # Flatten and filter valid points
    points = points.reshape(-1, 3)
    colors = torch.from_numpy(rgb_image).to(device).reshape(-1, 3) / 255.0
    valid = valid.reshape(-1)
    points = points[valid]
    colors = colors[valid]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

    return pcd

# def pcd2pth(pcd):
#     points = np.asarray(pcd.points)
#     colors = np.asarray(pcd.colors)
#     points_tensor = torch.from_numpy(points).float()
#     colors_tensor = torch.from_numpy(colors).float()
#     return points_tensor, colors_tensor

def main(json_path, image_dir, output_file, device='cuda'):
    data = read_json(json_path)
    scale = data['scale']
    offset = torch.tensor(data['offset'], device=device)
    frames = data['frames']
    camera_angle_x = frames[0]['camera_angle_x']
    focal_length = 0.5 / np.tan(camera_angle_x / 2.0) * 512
    camera_matrix = torch.tensor([
        [focal_length, 0, 256],
        [0, focal_length, 256],
        [0, 0, 1]
    ], device=device)

    pcd_list = []

    for frame in frames:
        file_path = frame['file_path'].split('.')[0]
        transform_matrix = torch.tensor(frame['transform_matrix'], device=device)
        depth_min, depth_max = frame['depth']['min'], frame['depth']['max']

        rgb_image = read_image(f"{image_dir}/{file_path}_albedo.png")[:,:,:3]
        depth_image = read_image(f"{image_dir}/{file_path}_depth.png")
        valid = depth_image != 65535
        depth_image = torch.from_numpy(depth_image.astype(np.float32)).to(device) / 65535.0
        depth_image = (depth_min + depth_image * (depth_max - depth_min))

        pcd = create_point_cloud(rgb_image, depth_image, camera_matrix, transform_matrix, valid, device=device)
        pcd_list.append(pcd)

    # Combine all point clouds
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        combined_pcd += pcd

    # Save the combined point cloud to a file
    o3d.io.write_point_cloud(output_file, combined_pcd)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, required=True)
    args = parser.parse_args()
    # base_dir = f'/mnt/sdc/wangjh/DataProcess/dst/shapenet/{args.c}/render_16views'
    base_dir = f'/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results/{args.c}/render_16views'
    image_dirs = os.listdir(base_dir)
    # image_dirs = ["a1a117f7c2308027ee87b00f75d4f7b1"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for im_dir in tqdm(image_dirs, desc="Processing RGB Point Cloud: ", unit="obj"):
        image_dir = os.path.join(base_dir, im_dir)
        json_path = os.path.join(image_dir, "transforms.json")
        output_file = os.path.join(image_dir, "pointcloud.ply")
        main(json_path, image_dir, output_file, device=device)