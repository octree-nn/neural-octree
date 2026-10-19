"""Exercise training -> fp16 packing -> rendering with a tiny generated plane."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
import yaml
from PIL import Image
from neuoct.data.camera import CameraIntrinsics, generate_camera_directions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/pipeline_smoke'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        parser.error('Choose a new output directory to avoid overwriting an earlier run')
    image_dir = root/'images/synthetic/plane'
    point_dir = root/'points/synthetic/plane'
    image_dir.mkdir(parents=True)
    point_dir.mkdir(parents=True)
    size = 16
    intr = CameraIntrinsics.from_horizontal_fov(float(np.pi/3), size, size)
    directions = generate_camera_directions(intr, device='cpu')
    transform = torch.eye(4)
    transform[2,3] = -1.0
    depth_raw = np.full((size,size), 32768, dtype=np.uint16)
    depth = 2.0 * float(depth_raw[0,0]) / 65535.0
    points = directions * depth + transform[:3,3]
    color = np.zeros((size,size,3), dtype=np.uint8)
    color[...,0] = np.linspace(32,224,size,dtype=np.uint8)[None,:]
    color[...,1] = np.linspace(224,32,size,dtype=np.uint8)[:,None]
    color[...,2] = 96
    normal = np.full((size,size,3), 128, dtype=np.uint8)
    normal[...,2] = 0
    Image.fromarray(color).save(image_dir/'0.png')
    Image.fromarray(depth_raw).save(image_dir/'0_depth.png')
    Image.fromarray(normal).save(image_dir/'0_normal.png')
    metadata = {'frames':[{'file_path':'0','camera_angle_x':float(np.pi/3),
                          'transform_matrix':transform.tolist(),'depth':{'min':0.0,'max':2.0}}]}
    (image_dir/'transforms.json').write_text(json.dumps(metadata))
    torch.save((points.reshape(-1,3),torch.from_numpy(color).reshape(-1,3).float()/255),point_dir/'pointcloud.pth')
    (root/'manifest.txt').write_text('synthetic/plane/0\n')
    data = dict(location=str(root/'images'),pointcloud_root=str(root/'points'),
                filelist=str(root/'manifest.txt'),depth=5,q_depth=5,full_depth=2,
                image_width=size,image_height=size,point_scale=0.99,ray_scale=0.5,
                camera_convention='neuoct',orbit_up_axis='y')
    cfg = root/'config.yaml'
    cfg.write_text(yaml.safe_dump({'DATA':{'train':data,'test':data}}))
    def run(module, *options):
        subprocess.run([sys.executable,'-m','neuoct.cli.'+module,'--config',str(cfg),
                        '--object','synthetic/plane','--device',args.device,*map(str,options)],check=True)
    command = [sys.executable, str(Path(__file__).with_name('train_object.py')),
               '--config', str(cfg), '--object', 'synthetic/plane', '--output', str(root/'train'),
               '--epochs', '2', '--max-steps', '2', '--views', '1', '--ray-batch-size', '256',
               '--feature-dim', '4', '--hidden-dim', '16', '--device', args.device]
    if args.geometry_only:
        command += ['--geometry-only']
    subprocess.run(command, check=True)
    checkpoint = root/'train/synthetic-plane.pt'
    packed = root/'inference_fp16.pt'
    run('pack_local_checkpoint','--checkpoint',checkpoint,'--output',packed,'--storage-dtype','fp16')
    run('evaluate_object','--checkpoint',packed,'--output',root/'evaluation','--views','1',
        '--hit-threshold','0.4','--orient-normals')
    assert packed.is_file()
    payload = torch.load(packed, map_location='cpu', weights_only=False)
    assert payload['model_config']['geometry_only'] == args.geometry_only
    material = {k:v for k,v in payload['renderer_state_dict'].items() if 'material' in k}
    if not args.geometry_only:
        assert material and all(torch.isfinite(v).all() for v in material.values())
    print('Color branch:', 'disabled' if args.geometry_only else 'trained and packed')
    print('Pipeline smoke completed:',root)
    print('This short execution check is not a reconstruction-quality benchmark.')


if __name__=='__main__':
    main()
