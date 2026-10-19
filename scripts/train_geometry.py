"""Run a paper geometry preset. For custom or colored objects use train_object.py."""
from pathlib import Path
import argparse
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/geometry.yaml')
    parser.add_argument('--model', default='Armadillo')
    parser.add_argument('--tier', choices=['81', '122', '244', '823'], default='823')
    parser.add_argument('--gpus', type=int, default=2, help='Number of visible CUDA devices')
    parser.add_argument('--output', type=Path, default=Path('outputs/geometry'))
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    repo = Path(__file__).resolve().parents[1]
    capacities = yaml.safe_load((repo/'configs/geometry_capacities.yaml').read_text())
    if args.model not in capacities[args.tier]:
        parser.error('Unknown model; choose '+', '.join(capacities[args.tier]))
    cell = capacities[args.tier][args.model]
    cfg = yaml.safe_load(Path(args.config).read_text())
    for split in ['train','test']:
        cfg['DATA'][split].update(depth=cell['octree_depth'], q_depth=cell['octree_depth'])
    output = args.output / (args.tier+'KiB') / args.model
    cfg_path = output/'data_config.yaml'
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           f'--nproc_per_node={args.gpus}', '-m', 'neuoct.cli.optimize_object',
           '--config', str(cfg_path), '--object', 'geometry/'+args.model,
           '--output', str(output), '--epochs', str(args.epochs), '--views', '242',
           '--ray-batch-size', '131072', '--feature-dim', str(cell['feature_dim']),
           '--hidden-dim', str(cell['hidden_dim']), '--n-iters', '3',
           '--sampling-mode', 'recurrent_pool', '--learning-rate', '0.002',
           '--weight-decay', '0.001', '--lr-schedule', 'linear', '--lr-min', '0.00001',
           '--checkpoint-every', '10', '--cache-supervision', '--pair-balanced-ddp',
           '--tf32', '--geometry-only']
    if not args.no_compile:
        cmd += ['--compile-renderer', '--compile-mode', 'default']
    print(' '.join(cmd))
    print('Reference fp16 tensor payload:', cell['payload_bytes'], 'bytes')
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
