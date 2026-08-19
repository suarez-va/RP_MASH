"""
scheduler: run a list of independent jobs across local CPU cores (embarrassingly parallel).

Each job is a SEPARATE process (crash-isolated: a trajectory that segfaults cannot take down the
others), optionally pinned to a core with `taskset`. This is the local backend; on an HPC cluster
the identical worker is launched via a SLURM job array instead (see workflow.slurm).
"""

import os
import sys
import time
import shutil
import subprocess


def run_grid_local(cmd, args_list, ncores=None, cwd_list=None, use_taskset=None):
    """
    Run len(args_list) jobs across ncores cores.

    cmd       : either a path to a python script (str) -> run as `python <script> <args>`, or a
                full command prefix (list)             -> run as `<cmd...> <args>`
                e.g. [sys.executable, '-m', 'workflow.worker', '--config', '/abs/config.py']
    args_list : list of per-job argument lists (each a list of str/convertible), e.g. [['--idx', 0], ...]
    ncores    : cores to use in parallel (default: os.cpu_count(), capped at the number of jobs)
    cwd_list  : optional per-job working directory (default: inherit the caller's cwd)
    use_taskset : pin each job to a core with `taskset -c` (default: auto if taskset is on PATH)

    Returns the sorted list of job indices that FAILED (non-zero exit); empty if all succeeded.
    """
    n = len(args_list)
    if n == 0:
        return []
    if ncores is None:
        ncores = os.cpu_count() or 1
    ncores = max(1, min(int(ncores), n))
    if use_taskset is None:
        use_taskset = shutil.which('taskset') is not None

    prefix = [sys.executable, cmd] if isinstance(cmd, str) else list(cmd)

    queue   = list(range(n))
    slots   = [None] * ncores          # slots[core] = (Popen, job_idx) or None
    results = {}                       # job_idx -> returncode

    print(f'[scheduler] {n} jobs on {ncores} cores'
          f'{" (taskset pinned)" if use_taskset else ""}', flush=True)

    while queue or any(s is not None for s in slots):
        # reap finished jobs
        for core in range(ncores):
            if slots[core] is not None:
                proc, idx = slots[core]
                if proc.poll() is not None:
                    results[idx] = proc.returncode
                    tag = 'done' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
                    print(f'[core {core:>2}] job {idx} {tag}', flush=True)
                    slots[core] = None
        # launch new jobs into free slots
        for core in range(ncores):
            if slots[core] is None and queue:
                idx = queue.pop(0)
                tset = ['taskset', '-c', str(core)] if use_taskset else []
                full = tset + prefix + [str(a) for a in args_list[idx]]
                cwd  = cwd_list[idx] if cwd_list is not None else None
                slots[core] = (subprocess.Popen(full, cwd=cwd), idx)
                print(f'[core {core:>2}] job {idx} started', flush=True)
        time.sleep(0.05)

    failed = sorted(i for i, rc in results.items() if rc != 0)
    print(f'[scheduler] {n - len(failed)}/{n} succeeded'
          + (f'  -  FAILED: {failed}' if failed else ''), flush=True)
    return failed
