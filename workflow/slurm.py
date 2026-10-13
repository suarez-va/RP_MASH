"""
slurm: generate a CHUNKED sbatch job-array script for the trajectory grid.

To respect clusters that cap job-array size (e.g. <=500 tasks, and array indices that must stay
<1000), each array task runs a CHUNK of trajectories -- task t runs global indices
t*chunk .. t*chunk+chunk-1. So a <=500-task array can cover >500 trajectories.

Cores per task: slurm cpus_per_task = c (default 1) both requests c cpus per task AND makes the task
run c of its trajectories at once (workflow.worker --ncores c), one per core -- like workflow.runner
does locally. So max_tasks * c trajectories run in parallel across the array. c = 1 is the plain
sequential in-process loop.

Sizing: chunk = max(cfg['slurm']['chunk'] or 1, ceil(n_traj / max_tasks)); n_tasks = ceil(n_traj /
chunk). With max_tasks<=500 the array indices stay 0..499 (<1000). Seeds derive from
(base_seed, GLOBAL idx), so trajectory k is identical to a local or 1-per-task run.

SHARD mode (config key `shard_size`, see workflow.shards): the array is sized over SHARDS instead --
n_shards = ceil(n_traj / shard_size), and each task owns a contiguous block of
K = ceil(n_shards / max_tasks) whole shards, so n_tasks = ceil(n_shards / K) <= min(max_tasks,
n_shards) and no shard file is ever written by two tasks. Within a task, the c children only compute
and return each trajectory's data; the task's main process alone writes its shards.

slurm config keys (all optional):
    extra_directives : list[str]  -> extra #SBATCH HEADER lines (module policy, gres, ...)
    extra_commands   : list[str]  -> BODY bash lines emitted verbatim after `cd`/PYTHONPATH and
                                     before the worker launch, e.g. env activation:
                                        ['eval "$(conda shell.bash hook)"', 'conda activate map-rpmd']

CLI:
    python -m workflow.slurm --config config.py            # writes submit_grid.sh next to config
    sbatch submit_grid.sh                                  # then submit it
    python -m workflow.average --config config.py --file mapSz.dat   # average when done
"""

import os
import math
import argparse

from .runner import load_config
from .shards import shard_layout, task_layout


_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{last}{throttle}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --output={logdir}/task_%a.out
#SBATCH --error={logdir}/task_%a.err
{extra}
# one BLAS/OpenMP thread per process: a task runs --ncores ({cpus}) trajectories at once, one per core
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTHONPATH={rp_mash_root}:$PYTHONPATH
cd {run_dir}
{extra_cmds}
{launch}
"""

_LAUNCH_CHUNK = """# each task runs trajectories  (SLURM_ARRAY_TASK_ID * {chunk}) .. +{chunk}-1,  {cpus} at a time
{python} -m workflow.worker --config {config} --task-id $SLURM_ARRAY_TASK_ID --chunk {chunk} --ncores {cpus}"""

_LAUNCH_SHARD = """# each task owns shards  (SLURM_ARRAY_TASK_ID * {per}) .. +{per}-1  ({shard_size} trajectories each),
# runs {cpus} of their trajectories at once, and is the only process writing those shards
{python} -m workflow.worker --config {config} --task-id $SLURM_ARRAY_TASK_ID --shards-per-task {per} --ncores {cpus}"""


def write_array_sbatch(config_path, path=None):
    """Write an sbatch array script for this config's trajectory grid. Returns the path written."""
    cfg = load_config(config_path)
    run_dir  = os.path.dirname(os.path.abspath(config_path))
    grid_dir = os.path.join(run_dir, cfg.get('grid_dir', 'traj_grid'))
    n = int(cfg['n_traj'])

    sl        = cfg.get('slurm', {})
    logdir    = os.path.join(grid_dir, 'logs')
    os.makedirs(logdir, exist_ok=True)
    max_tasks = int(sl.get('max_tasks', 500))
    python    = sl.get('python', 'python')
    config    = os.path.abspath(config_path)
    cpus      = int(sl.get('cpus_per_task', 1) or 1)

    if cfg.get('shard_size'):
        # ---- shard sizing: each task owns a contiguous block of WHOLE shards (one writer per shard) ----
        n_shards, width = shard_layout(n, cfg['shard_size'])
        n_tasks, per    = task_layout(n_shards, max_tasks)
        launch  = _LAUNCH_SHARD.format(per=per, shard_size=cfg['shard_size'], python=python,
                                       config=config, cpus=cpus)
        per_task = per * int(cfg['shard_size'])
        summary = (f'{n} trajectories -> {n_shards} shards of {cfg["shard_size"]} '
                   f'(shard_{0:0{width}d}..shard_{n_shards - 1:0{width}d}) -> '
                   f'{n_tasks} array tasks x {per} shards')
    else:
        # ---- chunk sizing: keep the array <= max_tasks (cluster limit), auto-raising chunk if needed ----
        chunk     = max(int(sl.get('chunk', 1) or 1), math.ceil(n / max_tasks))
        n_tasks   = math.ceil(n / chunk)
        if n_tasks > max_tasks:
            raise ValueError(f'{n} trajectories with chunk={chunk} need {n_tasks} array tasks > '
                             f'max_tasks={max_tasks}; raise slurm["chunk"] to >= {math.ceil(n/max_tasks)}')
        launch  = _LAUNCH_CHUNK.format(chunk=chunk, python=python, config=config, cpus=cpus)
        per_task = chunk
        summary = f'{n} trajectories -> {n_tasks} array tasks x chunk {chunk}'

    throttle  = f"%{sl['max_concurrent']}" if sl.get('max_concurrent') else ''
    extra_lines = []
    for key in ('partition', 'account', 'qos'):
        if sl.get(key):
            extra_lines.append(f'#SBATCH --{key}={sl[key]}')
    if sl.get('mem'):
        extra_lines.append(f"#SBATCH --mem={sl['mem']}")
    for raw in sl.get('extra_directives', []):        # escape hatch for anything not covered above
        extra_lines.append(raw if raw.startswith('#SBATCH') else f'#SBATCH {raw}')

    # body bash lines (env activation, module loads, conda activate, ...) emitted verbatim
    # after the PYTHONPATH/cd block and before the worker launch -- distinct from the
    # #SBATCH-header 'extra_directives' above.
    extra_cmds = '\n'.join(sl.get('extra_commands', []))

    rp_mash_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = _TEMPLATE.format(
        job_name = sl.get('job_name', 'rpmash_grid'),
        last     = n_tasks - 1,
        throttle = throttle,
        cpus     = cpus,
        time     = sl.get('time', '01:00:00'),
        logdir   = logdir,
        extra    = '\n'.join(extra_lines),
        rp_mash_root = rp_mash_root,
        run_dir  = run_dir,
        extra_cmds = extra_cmds,
        launch   = launch,
    )

    if path is None:
        path = os.path.join(run_dir, 'submit_grid.sh')
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)
    print(f'[slurm] wrote {path}')
    print(f'[slurm] {summary}  '
          f'(array 0-{n_tasks-1}{throttle}, cpus/task={cpus} -> {n_tasks * cpus} cores, '
          f'{cpus} trajectories at once per task)')
    if per_task < cpus:
        print(f'[slurm] WARNING: a task has at most {per_task} trajectories but {cpus} cpus; '
              f'{cpus - per_task} cores per task will sit idle. Lower cpus_per_task or max_tasks.')
    if sl.get('mem') and cpus > 1:
        print(f'[slurm] note: mem={sl["mem"]} is per TASK, now shared by {cpus} trajectories at once')
    print(f'[slurm] submit with:  sbatch {path}')
    return path


def _main():
    ap = argparse.ArgumentParser(description='Write a SLURM array script for the trajectory grid.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='output path (default: submit_grid.sh next to config)')
    a = ap.parse_args()
    write_array_sbatch(a.config, path=a.out)


if __name__ == '__main__':
    _main()
