"""
runner: orchestrate N independent trajectories locally.

Loads the user's config.py (a CONFIG dict) and system.py (a run_trajectory function), derives a
reproducible-yet-independent seed per trajectory, and fans the worker out across cores. The heavy
lifting per trajectory happens in workflow.worker, launched once per trajectory index.

CLI:
    python -m workflow.runner --config config.py [--ncores N] [--rerun i j ...]
"""

import os
import sys
import argparse
import importlib.util

import numpy as np

from .scheduler import run_grid_local


# ------------------------------------------------------------------ config / system loading
def _load_module(path, modname):
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(modname, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_config(path):
    """Import the user's config .py and return its CONFIG dict."""
    return _load_module(path, 'rpmash_run_config').CONFIG


def load_system(path):
    """Import the user's system .py (must define run_trajectory(cfg, seed))."""
    return _load_module(path, 'rpmash_run_system')


# ------------------------------------------------------------------ seeding
def seed_for_idx(base_seed, idx):
    """
    Deterministic, independent seed for trajectory `idx`.

    base_seed=None -> return None (each trajectory uses OS entropy; NOT reproducible).
    base_seed=int  -> a distinct, well-separated seed per idx via SeedSequence([base_seed, idx]),
                      reproducible across runs and identical for local vs SLURM (same idx).
    """
    if base_seed is None:
        return None
    return int(np.random.SeedSequence([int(base_seed), int(idx)]).generate_state(1, dtype=np.uint32)[0])


# ------------------------------------------------------------------ directory layout
def traj_dirs(config_path):
    """Return (run_dir, grid_dir, [traj0_dir, ...]) for a given config path."""
    cfg      = load_config(config_path)
    run_dir  = os.path.dirname(os.path.abspath(config_path))
    grid_dir = os.path.join(run_dir, cfg.get('grid_dir', 'traj_grid'))
    dirs     = [os.path.join(grid_dir, f'traj{i}') for i in range(int(cfg['n_traj']))]
    return run_dir, grid_dir, dirs


# ------------------------------------------------------------------ orchestration
def run_trajectories(config_path, ncores=None, indices=None):
    """
    Launch the trajectories described by config_path across local cores.

    indices : optional explicit list of trajectory indices to run (default: all 0..n_traj-1).
              Use this to re-run only the ones that failed.
    Returns the list of failed trajectory indices.
    """
    cfg = load_config(config_path)
    if ncores is None:
        ncores = cfg.get('ncores')
    _, grid_dir, dirs = traj_dirs(config_path)
    os.makedirs(grid_dir, exist_ok=True)

    if indices is None:
        indices = list(range(int(cfg['n_traj'])))
    for i in indices:
        os.makedirs(dirs[i], exist_ok=True)

    # each worker is launched as a module so package imports resolve regardless of cwd
    cmd_prefix = [sys.executable, '-m', 'workflow.worker', '--config', os.path.abspath(config_path)]
    args_list  = [['--idx', i] for i in indices]

    failed = run_grid_local(cmd_prefix, args_list, ncores=ncores)
    # map scheduler's positional failures back onto the actual trajectory indices
    return [indices[f] for f in failed]


def _main():
    ap = argparse.ArgumentParser(description='Run RP_MASH trajectories locally.')
    ap.add_argument('--config', required=True, help='path to the run config .py')
    ap.add_argument('--ncores', type=int, default=None, help='cores to use (default: config or all)')
    ap.add_argument('--rerun',  type=int, nargs='+', default=None,
                    help='run only these trajectory indices (e.g. to redo failures)')
    a = ap.parse_args()
    failed = run_trajectories(a.config, ncores=a.ncores, indices=a.rerun)
    if failed:
        print('trajectories still FAILED:', failed)
        print('re-run just those with:  python -m workflow.runner --config',
              a.config, '--rerun', ' '.join(map(str, failed)))
        sys.exit(1)


if __name__ == '__main__':
    _main()
