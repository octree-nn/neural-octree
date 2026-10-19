"""Optimize geometry and albedo for any prepared category/object pair."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Additional options are forwarded to neuoct-optimize-object. '
        'Color optimization is enabled unless --geometry-only is supplied.'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--object', required=True, help='category/object_id; no fixed object list')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--feature-dim', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-iters', type=int, default=3)
    parser.add_argument('--geometry-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if args.gpus < 1:
        parser.error('--gpus must be positive')
    parts = args.object.split('/')
    if len(parts) != 2 or any(p in ('', '.', '..') or '\\' in p for p in parts):
        parser.error('--object must be category/object_id')
    cmd = [sys.executable]
    if args.gpus > 1:
        cmd += ['-m', 'torch.distributed.run', '--standalone',
                f'--nproc_per_node={args.gpus}']
    cmd += ['-m', 'neuoct.cli.optimize_object', '--config', args.config,
            '--object', args.object, '--output', str(args.output),
            '--epochs', str(args.epochs), '--views', '-1',
            '--feature-dim', str(args.feature_dim), '--hidden-dim', str(args.hidden_dim),
            '--n-iters', str(args.n_iters), '--sampling-mode', 'recurrent_pool',
            '--learning-rate', '0.002', '--lr-schedule', 'linear', '--lr-min', '0.00001']
    if args.geometry_only:
        cmd += ['--geometry-only']
    cmd += extra
    print(' '.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
