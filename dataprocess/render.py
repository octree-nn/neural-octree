import argparse
import os
import cv2
import numpy as np
from PIL import Image

import torch
import pytorch3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesAtlas,
)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.shader import ShaderBase

from tqdm import tqdm

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def generate_directions(camera, cam_location, H, W, device='cpu'):    
    i, j = torch.meshgrid(torch.arange(W, device=device) + 0.5, torch.arange(H, device=device) + 0.5)  # (W, H)
    i = 1 - 2 * i / W  # (W, H)
    j = 1 - 2 * j / H  # (W, H)
    xy_d = torch.stack([i, j, torch.ones_like(i, device=device)], -1)  # (W, H, 3)
    unprojected_points = camera.unproject_points(xy_d, world_coordinates=True)  # (W, H, 3)/

    directions = unprojected_points - cam_location[0]  # (W, H, 3)

    directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
    return directions.unsqueeze(0)



class AlbedoShader(ShaderBase):
    """
    一个简单的着色器，直接返回网格顶点的albedo颜色（漫反射颜色），忽略所有光照。
    """
    def __init__(self, device="cpu", blend_params=None):
        if blend_params is None:
            blend_params = BlendParams()
        # print(blend_params)
        super().__init__(device=device, blend_params=blend_params)
    
    def forward(self, fragments, meshes, **kwargs):
        # fragments: 光栅化后的片段信息
        # meshes: 网格对象
        
        # 获取每个像素所对应面的纹理（颜色）信息。
        texels = meshes.sample_textures(fragments)  # 形状: (N, H, W, K, 3)
        
        # 我们只关心最前面面的颜色（K=0）。
        # 将其从 (N, H, W, 1, 3) 压缩为 (N, H, W, 3)
        albedo = texels[..., 0, :].squeeze(-2)
        # bg white
        mask = fragments.pix_to_face[..., 0] < 0  # 无效像素掩码
        albedo[mask] = 1.0  # 设置背景为白色
        
        return albedo  # 直接返回albedo颜色


def np_depth_to_colormap(depth):
    """ depth: [H, W] """
    depth_normalized = np.ones(depth.shape)
    # print('depth stats:', depth.min(), depth.max(), depth.mean())

    valid_mask = depth > -0.9 # valid
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]
        d_min = 1.1339745962155614
        d_max = 2.8660254037844384
        depth_normalized[valid_mask] = (d_valid - d_min) / (d_max - d_min)
        # depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())

        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
    return depth_color, depth_normalized


def center_normalize_verts(verts):
    # print('verts before', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    # """ normalize """
    bbox_min, bbox_max = verts.min(0)[0], verts.max(0)[0]
    center = (bbox_min + bbox_max) / 2
    scale = (bbox_max - bbox_min).max()
    # fit into [-0.5, 0.5]
    verts = (verts - center) / scale
    
    return verts

def load_mesh_multi_texture(obj_filename, normalize_verts=False):
    """load_mesh with multiple texture maps"""

    # Load obj file
    verts, faces, aux = load_obj(obj_filename, device=device,
                                load_textures=True,
                                create_texture_atlas=True,
                                texture_atlas_size=8,
                                texture_wrap=None,
                                )
    atlas = aux.texture_atlas
    
    if normalize_verts:
        verts = center_normalize_verts(verts)

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas]),
    )        
    
    return mesh


def load_mesh_single_texture(obj_filename, normalize_verts=False):
    mesh = load_objs_as_meshes([obj_filename], device=device)

    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    textures = mesh.textures

    if normalize_verts:
        verts = center_normalize_verts(verts)
    mesh = Meshes(verts=[verts], faces=[faces], textures=mesh.textures)
    return mesh




def get_normals_from_fragments(meshes, fragments, directions):
    """ https://github.com/facebookresearch/pytorch3d/issues/865 """
    faces = meshes.faces_packed()  # (F, 3)

    faces_vertices = meshes.verts_packed()[faces]  # (F, 3, 3)
    fragments_faces_vertices = faces_vertices[fragments.pix_to_face[..., 0]]  # (N, H, W, 3, 3)
    fragments_v0 = fragments_faces_vertices[:, :, :, 1] - fragments_faces_vertices[:, :, :, 0]  # (N, H, W, 3)
    fragments_v1 = fragments_faces_vertices[:, :, :, 2] - fragments_faces_vertices[:, :, :, 0]  # (N, H, W, 3)
    fragments_faces_normals = torch.cross(fragments_v0, fragments_v1, dim=-1)
    fragments_faces_normals = torch.nn.functional.normalize(fragments_faces_normals, p=2.0, dim=-1)
    normals_dot_directions = (fragments_faces_normals * directions).sum(-1, keepdim=True)  # (N, H, W, 1)
    normals_sign = -torch.sign(normals_dot_directions)
    pixel_normals = fragments_faces_normals * normals_sign  # (N, H, W, 3)
    # pixel_normals = fragments_faces_normals
    valid_mask = fragments.pix_to_face[..., 0] < 0  # 无效像素掩码
    pixel_normals[valid_mask] = 0.0
    return pixel_normals




# In[8]:


def setup_renderer(args, camera):
    if camera is not None:
        """
        The camera coordinate sysmte in COLMAP is right-down-forward
        Pytorch3D is left-up-forward
        """
        radii = camera['radius']
        yaw = camera['yaw']
        pitch = camera['pitch']
        fov = camera['fov']
        R, T = look_at_view_transform(
            dist=radii,
            elev=pitch * 180.0 / np.pi,  # 转为度
            azim=yaw * 180.0 / np.pi + 90     # 转为度
        )
        cameras = FoVPerspectiveCameras(
            R=R,                   # 旋转矩阵
            T=T,                   # 平移向量  
            fov=fov * 180.0 / np.pi,      # 视场角（度）
            device=device
        )
        image_size = ((args.image_size),) # args.image_size
        camera_x = radii * np.cos(pitch) * np.sin(yaw + np.pi / 2)
        camera_y = radii * np.sin(pitch)
        camera_z = radii * np.cos(pitch) * np.cos(yaw + np.pi / 2)
        cam_location = torch.tensor([[camera_x, camera_y, camera_z]], device=device, dtype=torch.float32)
        # return cameras
    else:
        raise Exception("Undefined camera")
    # print('Camera R T', R, T)

    # Define the settings for rasterization and shading.
    # print(image_size)
    raster_settings = RasterizationSettings(
        image_size=image_size[0],
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )

    Blend_params = BlendParams(background_color=(0, 0, 0))
    shader = AlbedoShader(device=device, blend_params=Blend_params)

    renderer = MeshRendererWithFragments(
        rasterizer = rasterizer,
        shader=shader
    )

    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 
            'rasterizer': rasterizer, 'renderer': renderer, 'cam_location': cam_location}

    return render_setup


PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val


def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]


def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)



def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def generate_views(num_views, offset=(0, 0)):
    yaws = []
    pitchs = []
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)

    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views  # 弧度转度

    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f}
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    return views



parser = argparse.ArgumentParser()
parser.add_argument('--normalize_verts', default=True, action='store_true')
parser.add_argument('--channel', default='02958343', type=str, help='ShapeNet category')
parser.add_argument('--data_path', default='', type=str, help='data path')
parser.add_argument('--filelist', default='', type=str, help='filelist path')
parser.add_argument('--save_path', default='', type=str, help='save path')
parser.add_argument('--num_views', default=16, type=int, help='number of views to render')
parser.add_argument('--image_size', default=512, type=int, help='render image size')
    
args = parser.parse_args()

args.normalize_verts = True

# data_path = f'/home/zhoucz/DataPreprocessing3D/src/shapenet/{args.channel}'
# filelist = '/mnt/sdc/wangjh/DataProcess/train_pt3d.txt'
# filelist = '/mnt/sdc/wangjh/DataProcess/train_1.txt'
# save_path = f'render_shapenet_results/{args.channel}/render_16views'
# save_path = 'test_render3'
data_path = args.data_path
filelist = args.filelist
save_path = args.save_path
num_views = args.num_views

with open(filelist, 'r') as f:
    lines = f.readlines()
filenames = [line.strip() for line in lines]

record = []

all_counts = len(filenames)

for filename in tqdm(filenames, total=all_counts):

    save_dir = os.path.join(save_path, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    obj_filename = os.path.join(data_path, filename, 'model.obj')

    try:
        mesh = load_mesh_multi_texture(obj_filename, normalize_verts=args.normalize_verts)
    except:
        mesh = load_mesh_single_texture(obj_filename, normalize_verts=args.normalize_verts)

    num_views = 16
    offset = (np.random.rand(), np.random.rand())
    # offset = (0, 0)
    views = generate_views(num_views, offset=offset)
    record.append({'filename': filename, 'offset': offset})

    for i, camera in enumerate(views): # only render the first camera
        render_setup = setup_renderer(args, camera)

        save_name = '%03d' % i
        renderer = render_setup['renderer']
        images, fragments = renderer(mesh)
        image_np = images[0, ..., :3].cpu().numpy()[:, :, ::-1] # BGR
        cv2.imwrite(os.path.join(save_dir, '%s_albedo.png' % save_name), (image_np * 255).astype(np.uint8))


        """ obtain depth map """
        depth = fragments.zbuf[0, :, :, 0].cpu().numpy()
        depth_color, depth_normalized = np_depth_to_colormap(depth)

        # save as 16-bit png
        depth_to_save = (depth_normalized * 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(save_dir, '%s_depth.png' % (save_name)), depth_to_save)

        """ obtain normal map """
        directions = generate_directions(
            render_setup['cameras'], 
            render_setup['cam_location'],
            args.image_size,
            args.image_size,
            device=device
        )  # [1,H,W,3]
        # w, h to h, w
        directions = directions.permute(0, 2, 1, 3)
        normal = get_normals_from_fragments(mesh, fragments, directions)[0,:, :, :]

        normal = normal.cpu().numpy()
        cv2.imwrite(os.path.join(save_dir, '%s_normal.png' % (save_name)), ((normal+1)/2*255).astype(np.uint8))



# save record as json
import json
with open(os.path.join(save_path, 'render_record.json'), 'w') as f:
    json.dump(record, f, indent=4)

