"""
worker: run ONE trajectory (index --idx) in its own directory.

This is the single unit of work, launched once per trajectory by either:
    - the local scheduler (workflow.runner), or
    - a SLURM job array (workflow.slurm), where --idx = $SLURM_ARRAY_TASK_ID.

It cd's into the trajectory directory (RP_MASH writes hardcoded filenames -- output.dat, nucR.dat,
nucP.dat, mapSx/y/z.dat, memK.dat, info.txt -- into the cwd), derives this trajectory's seed, and
hands control to the user's system.run_trajectory(cfg, seed).

CLI:
    python -m workflow.worker --config /abs/config.py --idx 7
"""

import os
import sys
import argparse

from .runner import load_config, load_system, seed_for_idx, traj_dirs


def _pin_single_thread():
    """Keep each worker to one BLAS/OpenMP thread so N workers x M threads don't oversubscribe cores."""
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(var, '1')


def run_one(config_path, idx):
    cfg    = load_config(config_path)
    system = load_system(os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                       cfg.get('system_file', 'system.py')))
    _, _, dirs = traj_dirs(config_path)
    outdir = dirs[idx]
    os.makedirs(outdir, exist_ok=True)

    seed = seed_for_idx(cfg.get('base_seed'), idx)

    os.chdir(outdir)
    print(f'[worker] traj {idx}  seed={seed}  dir={outdir}', flush=True)
    system.run_trajectory(cfg, seed)
    print(f'[worker] traj {idx} finished', flush=True)


def _main():
    _pin_single_thread()
    ap = argparse.ArgumentParser(description='Run a single RP_MASH trajectory.')
    ap.add_argument('--config', required=True, help='path to the run config .py')
    ap.add_argument('--idx', type=int, required=True, help='trajectory index (0-based)')
    a = ap.parse_args()
    run_one(a.config, a.idx)


if __name__ == '__main__':
    _main()
