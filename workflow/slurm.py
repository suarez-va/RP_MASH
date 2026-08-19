"""
slurm: generate an sbatch job-array script for the same trajectory grid.

The parallelization backend swaps from local subprocesses to a SLURM job array, but the unit of work
is byte-for-byte identical: `python -m workflow.worker --config <cfg> --idx $SLURM_ARRAY_TASK_ID`.
Because seeds derive from (base_seed, idx), trajectory `k` is reproducible whether it ran locally or
on the cluster.

CLI:
    python -m workflow.slurm --config config.py            # writes submit_grid.sh next to config
    sbatch submit_grid.sh                                  # then submit it
    python -m workflow.average --config config.py --file mapSz.dat   # average when done
"""

import os
import argparse

from .runner import load_config, traj_dirs


_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{last}{throttle}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --output={logdir}/traj_%a.out
#SBATCH --error={logdir}/traj_%a.err
{extra}
# one thread per task -> the array provides the parallelism, not BLAS
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export PYTHONPATH={rp_mash_root}:$PYTHONPATH
cd {run_dir}

{python} -m workflow.worker --config {config} --idx $SLURM_ARRAY_TASK_ID
"""


def write_array_sbatch(config_path, path=None):
    """Write an sbatch array script for this config's trajectory grid. Returns the path written."""
    cfg = load_config(config_path)
    run_dir, grid_dir, dirs = traj_dirs(config_path)
    n = len(dirs)

    sl        = cfg.get('slurm', {})
    logdir    = os.path.join(grid_dir, 'logs')
    os.makedirs(logdir, exist_ok=True)

    throttle  = f"%{sl['max_concurrent']}" if sl.get('max_concurrent') else ''
    extra_lines = []
    for key in ('partition', 'account', 'qos'):
        if sl.get(key):
            extra_lines.append(f'#SBATCH --{key}={sl[key]}')
    if sl.get('mem'):
        extra_lines.append(f"#SBATCH --mem={sl['mem']}")
    for raw in sl.get('extra_directives', []):        # escape hatch for anything not covered above
        extra_lines.append(raw if raw.startswith('#SBATCH') else f'#SBATCH {raw}')

    rp_mash_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = _TEMPLATE.format(
        job_name = sl.get('job_name', 'rpmash_grid'),
        last     = n - 1,
        throttle = throttle,
        cpus     = sl.get('cpus_per_task', 1),
        time     = sl.get('time', '01:00:00'),
        logdir   = logdir,
        extra    = '\n'.join(extra_lines),
        rp_mash_root = rp_mash_root,
        run_dir  = run_dir,
        python   = sl.get('python', 'python'),
        config   = os.path.abspath(config_path),
    )

    if path is None:
        path = os.path.join(run_dir, 'submit_grid.sh')
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)
    print(f'[slurm] wrote {path}  (array 0-{n-1}{throttle})')
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
