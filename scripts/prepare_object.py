"""Render a custom mesh and write train/test data plus a ready-to-use config."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='GLB, glTF, OBJ, PLY, or STL mesh')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object', default='custom/object', help='category/object_id')
    parser.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    parser.add_argument('--views', type=int, default=64)
    parser.add_argument('--test-every', type=int, default=8, help='Hold out every Nth camera')
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--depth', type=int, choices=[5, 6, 7], default=7)
    parser.add_argument('--point-count', type=int, default=200000)
    parser.add_argument('--cycles-device', choices=['CPU', 'GPU'], default='GPU')
    args = parser.parse_args()
    parts = args.object.split('/')
    if len(parts) != 2 or any(p in ('', '.', '..') or '\\' in p for p in parts):
        parser.error('--object must be category/object_id')
    if args.views < 2 or not 2 <= args.test_every <= args.views:
        parser.error('Require 2 <= --test-every <= --views')
    if args.resolution < 16 or args.point_count < 1:
        parser.error('--resolution must be >= 16 and --point-count must be positive')
    mesh = args.input.resolve()
    if not mesh.is_file() or mesh.suffix.lower() not in ('.glb', '.gltf', '.obj', '.ply', '.stl'):
        parser.error('--input must be an existing GLB, glTF, OBJ, PLY, or STL file')
    out = args.output.resolve()
    if out.exists():
        parser.error('Choose a new --output directory')
    out.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    pointcloud = out/'pointcloud'/args.object/'pointcloud.pth'
    config = {'DATA': {}}
    with tempfile.TemporaryDirectory(prefix='render_', dir=out) as temporary:
        rendered = Path(temporary)/'images'
        subprocess.run([args.blender, '-b', '--factory-startup', '-P',
                        str(repo/'scripts/objaverse/blender_render.py'), '--',
                        '--object', str(mesh), '--output', str(rendered),
                        '--views', str(args.views), '--resolution', str(args.resolution),
                        '--auto-frame', '--cycles-device', args.cycles_device], check=True)
        metadata = json.loads((rendered/'transforms.json').read_text())
        for split in ['train', 'test']:
            frames = [f for i, f in enumerate(metadata['frames'])
                      if ((i % args.test_every == 0) == (split == 'test'))]
            images = out/split/args.object
            images.mkdir(parents=True)
            for frame in frames:
                stem = Path(frame['file_path']).stem
                for suffix in ['.png', '_depth.png', '_normal.exr', '_albedo.png']:
                    shutil.copy2(rendered/(stem+suffix), images/(stem+suffix))
            (images/'transforms.json').write_text(json.dumps({**metadata, 'frames': frames}))
            # Only training views contribute to the input point cloud.
            cloud = pointcloud if split == 'train' else Path(temporary)/'test_cloud.pth'
            subprocess.run([sys.executable, str(repo/'scripts/objaverse/postprocess.py'),
                            '--image-dir', str(images), '--pointcloud', str(cloud),
                            '--uid', parts[1], '--point-count', str(args.point_count)], check=True)
            manifest = out/(split+'.txt')
            manifest.write_text(''.join(f'{args.object}/{i}\n' for i in range(len(frames))))
            config['DATA'][split] = dict(
                location=str(out/split), pointcloud_root=str(out/'pointcloud'),
                filelist=str(manifest), depth=args.depth, q_depth=args.depth, full_depth=4,
                image_width=args.resolution, image_height=args.resolution,
                point_scale=0.99, ray_scale=0.5, camera_convention='legacy_blender',
                orbit_up_axis='z', normal_source='png')
    (out/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Config: {out / "config.yaml"}\nObject: {args.object}')


if __name__ == '__main__':
    main()
