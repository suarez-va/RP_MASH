"""
worker: run trajectories in their own directories. Two modes:

  - SINGLE  (--idx N):  run trajectory N. Used by the local scheduler (workflow.runner), which
                        launches one such process per trajectory across cores.
  - CHUNK   (--task-id T --chunk C):  run trajectories T*C .. T*C+C-1 SEQUENTIALLY in this one
                        process (blocking, one after another -- no subprocess, no polling). Used by
                        the chunked SLURM array (workflow.slurm) so a <=500-task array can cover
                        >500 trajectories on 1 core per task.

Either way, for each trajectory it cd's into that trajectory's directory (RP_MASH writes hardcoded
filenames -- output.dat, nucR.dat, ... -- into the cwd), derives the trajectory's seed via
seed_for_idx(base_seed, GLOBAL idx), and calls the user's system.run_trajectory(cfg, seed). The seed
depends only on the global index, so trajectory k is identical whether run singly or inside a chunk.

CLI:
    python -m workflow.worker --config /abs/config.py --idx 7                 # single
    python -m workflow.worker --config /abs/config.py --task-id 3 --chunk 5   # traj 15..19
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


def _load(config_path):
    """Load config + user system module once; return (cfg, system, run_dir, dirs)."""
    cfg    = load_config(config_path)
    system = load_system(os.path.join(os.path.dirname(os.path.abspath(config_path)),
                                       cfg.get('system_file', 'system.py')))
    run_dir, _, dirs = traj_dirs(config_path)
    return cfg, system, run_dir, dirs


def _run_idx(cfg, system, run_dir, dirs, idx):
    """Run trajectory `idx` in its own directory, restoring cwd afterwards (safe in a loop)."""
    outdir = dirs[idx]
    os.makedirs(outdir, exist_ok=True)
    seed = seed_for_idx(cfg.get('base_seed'), idx)
    os.chdir(outdir)
    try:
        print(f'[worker] traj {idx}  seed={seed}  dir={outdir}', flush=True)
        system.run_trajectory(cfg, seed)
        print(f'[worker] traj {idx} finished', flush=True)
    finally:
        os.chdir(run_dir)


def run_one(config_path, idx):
    """SINGLE mode: run one trajectory (used by the local scheduler, one process per trajectory)."""
    cfg, system, run_dir, dirs = _load(config_path)
    _run_idx(cfg, system, run_dir, dirs, idx)


def run_chunk(config_path, task_id, chunk):
    """CHUNK mode: run trajectories task_id*chunk .. task_id*chunk+chunk-1 SEQUENTIALLY in-process
    (no subprocess, no polling). A Python error in one trajectory is reported and skipped so the rest
    of the chunk still runs; if any failed, exit non-zero so SLURM marks the task failed."""
    cfg, system, run_dir, dirs = _load(config_path)
    n_traj = len(dirs)
    start  = task_id * chunk
    stop   = min(start + chunk, n_traj)
    if start >= n_traj:
        print(f'[worker] task {task_id}: start idx {start} >= n_traj {n_traj}; nothing to do', flush=True)
        return
    print(f'[worker] task {task_id}: running trajectories {start}..{stop - 1} '
          f'sequentially (chunk={chunk}, n_traj={n_traj})', flush=True)
    failed = []
    for idx in range(start, stop):
        try:
            _run_idx(cfg, system, run_dir, dirs, idx)
        except Exception as e:
            failed.append(idx)
            print(f'[worker] traj {idx} FAILED: {type(e).__name__}: {e}', flush=True)
    if failed:
        print(f'[worker] task {task_id} finished with FAILURES: {failed}', flush=True)
        sys.exit(1)
    print(f'[worker] task {task_id} finished all of {start}..{stop - 1}', flush=True)


def _main():
    _pin_single_thread()
    ap = argparse.ArgumentParser(description='Run RP_MASH trajectories (single, or a sequential chunk).')
    ap.add_argument('--config', required=True, help='path to the run config .py')
    ap.add_argument('--idx',     type=int, default=None, help='SINGLE: run this trajectory index')
    ap.add_argument('--task-id', type=int, default=None, help='CHUNK: SLURM array task id')
    ap.add_argument('--chunk',   type=int, default=None, help='CHUNK: trajectories per task')
    a = ap.parse_args()
    if a.idx is not None:
        run_one(a.config, a.idx)
    elif a.task_id is not None and a.chunk is not None:
        run_chunk(a.config, a.task_id, a.chunk)
    else:
        ap.error('provide either --idx N  (single)  or  --task-id T --chunk C  (sequential chunk)')


if __name__ == '__main__':
    _main()
