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
    x = torch.arange(width, device=device)
    y = torch.arange(height, device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')
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

def main(json_path, image_dir, pcd_file, output_file, device='cuda'):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points_tensor = torch.from_numpy(np.asarray(pcd.points)).float()
    colors_tensor = torch.from_numpy(np.asarray(pcd.colors)).float()
    torch.save((points_tensor, colors_tensor), output_file)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, required=True)
    args = parser.parse_args()
    # base_dir = f'/mnt/sdc/wangjh/DataProcess/dst/shapenet/{args.c}/render_16views'
    base_dir = f'/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results/{args.c}/render_16views'
    image_dirs = os.listdir(base_dir)
    output_base_dir = f'/mnt/sdc/wangjh/pytorch3d_render_colmap/render_shapenet_results_pcd2pth/{args.c}/pcd'    # image_dirs = ["a1a117f7c2308027ee87b00f75d4f7b1"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for im_dir in tqdm(image_dirs, desc="Processing RGB Point Cloud: ", unit="obj"):
        if im_dir.endswith('.json'):
            continue
        image_dir = os.path.join(base_dir, im_dir)
        json_path = os.path.join(image_dir, "transforms.json")
        pcd_file = os.path.join(image_dir, "pointcloud.ply")
        # output_file = os.path.join(image_dir, "pcd.pth")
        output_dir = os.path.join(output_base_dir, im_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(output_dir, "pcd.pth")
        main(json_path, image_dir, pcd_file, output_file, device=device)