"""
slurm: generate a CHUNKED sbatch job-array script for the trajectory grid.

To respect clusters that cap job-array size (e.g. <=500 tasks, and array indices that must stay
<1000), each array task runs a CHUNK of trajectories SEQUENTIALLY on its single core -- task t runs
global indices t*chunk .. t*chunk+chunk-1, one after another (see workflow.worker run_chunk: a plain
in-process loop, no subprocess, no polling). So a <=500-task array can cover >500 trajectories.

Sizing: chunk = max(cfg['slurm']['chunk'] or 1, ceil(n_traj / max_tasks)); n_tasks = ceil(n_traj /
chunk). With max_tasks<=500 the array indices stay 0..499 (<1000). Seeds derive from
(base_seed, GLOBAL idx), so trajectory k is identical to a local or 1-per-task run.

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

from .runner import load_config, traj_dirs


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
# one thread per task -> a task runs its chunk of trajectories sequentially on ONE core
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTHONPATH={rp_mash_root}:$PYTHONPATH
cd {run_dir}
{extra_cmds}
# each task runs trajectories  (SLURM_ARRAY_TASK_ID * {chunk}) .. +{chunk}-1  one at a time
{python} -m workflow.worker --config {config} --task-id $SLURM_ARRAY_TASK_ID --chunk {chunk}
"""


def write_array_sbatch(config_path, path=None):
    """Write an sbatch array script for this config's trajectory grid. Returns the path written."""
    cfg = load_config(config_path)
    run_dir, grid_dir, dirs = traj_dirs(config_path)
    n = len(dirs)

    sl        = cfg.get('slurm', {})
    logdir    = os.path.join(grid_dir, 'logs')
    os.makedirs(logdir, exist_ok=True)

    # ---- chunk sizing: keep the array <= max_tasks (cluster limit), auto-raising chunk if needed ----
    max_tasks = int(sl.get('max_tasks', 500))
    chunk     = max(int(sl.get('chunk', 1) or 1), math.ceil(n / max_tasks))
    n_tasks   = math.ceil(n / chunk)
    if n_tasks > max_tasks:
        raise ValueError(f'{n} trajectories with chunk={chunk} need {n_tasks} array tasks > '
                         f'max_tasks={max_tasks}; raise slurm["chunk"] to >= {math.ceil(n/max_tasks)}')

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
        cpus     = sl.get('cpus_per_task', 1),
        time     = sl.get('time', '01:00:00'),
        logdir   = logdir,
        extra    = '\n'.join(extra_lines),
        rp_mash_root = rp_mash_root,
        run_dir  = run_dir,
        python   = sl.get('python', 'python'),
        config   = os.path.abspath(config_path),
        chunk    = chunk,
        extra_cmds = extra_cmds,
    )

    if path is None:
        path = os.path.join(run_dir, 'submit_grid.sh')
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)
    print(f'[slurm] wrote {path}')
    print(f'[slurm] {n} trajectories -> {n_tasks} array tasks x chunk {chunk}  '
          f'(array 0-{n_tasks-1}{throttle}, cpus/task={sl.get("cpus_per_task", 1)})')
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
