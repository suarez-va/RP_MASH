"""
average: collect a per-trajectory output file across all trajectory directories and average it,
with a standard error of the mean over trajectories (the natural error bar for independent runs).

RP_MASH writes one file of each name per trajectory (output.dat, nucR.dat, nucP.dat, mapSx/y/z.dat).
Every trajectory shares the same time grid, so files stack cleanly into an (n_traj, nrows, ncols)
array that we reduce over the trajectory axis.

CLI:
    python -m workflow.average --config config.py --file mapSz.dat
    python -m workflow.average --config config.py --file output.dat --cols 1 2 3
"""

import os
import argparse

import numpy as np

from .runner import load_config, traj_dirs


def collect(config_path, filename, indices=None):
    """
    Load `filename` from every (existing) trajectory directory.

    Returns (data, used) where data has shape (n_used, nrows, ncols) and `used` is the list of
    trajectory indices actually found. Trajectories whose file is missing (e.g. a crash) are skipped
    with a warning, so a partial grid still averages.
    """
    _, _, dirs = traj_dirs(config_path)
    if indices is None:
        indices = range(len(dirs))

    stack, used = [], []
    for i in indices:
        path = os.path.join(dirs[i], filename)
        if not os.path.isfile(path):
            print(f'[average] WARNING: missing {path} (skipping traj {i})')
            continue
        stack.append(np.loadtxt(path))
        used.append(i)

    if not stack:
        raise FileNotFoundError(f'no "{filename}" found in any trajectory directory')
    data = np.array(stack)
    if data.ndim == 2:                    # single-column files -> (n, nrows) -> (n, nrows, 1)
        data = data[:, :, None]
    return data, used


def average_column(config_path, filename, cols=None, indices=None):
    """
    Mean and standard error (over trajectories) of `filename`.

    Returns (mean, sem) each of shape (nrows, ncols) — or the subset selected by `cols`.
    sem = std(ddof=1) / sqrt(n_traj), the error bar on the trajectory-averaged observable.
    """
    data, used = collect(config_path, filename, indices=indices)
    n = data.shape[0]
    mean = data.mean(axis=0)
    sem  = data.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    if cols is not None:
        cols = list(cols)
        mean, sem = mean[:, cols], sem[:, cols]
    print(f'[average] {filename}: averaged {n} trajectories {used}')
    return mean, sem


def mash_population(config_path, indices=None):
    """
    Convenience: trajectory-averaged mapping spin components (Sx, Sy, Sz) with SE, sharing the time
    column from mapSz.dat's column 0. Returns a dict with keys 'time', 'Sx','Sy','Sz' and their
    '<name>_sem' partners. Assumes mapS?.dat is [time, S] per row (adjust `col` if your layout differs).
    """
    out = {}
    for comp, fname in (('Sx', 'mapSx.dat'), ('Sy', 'mapSy.dat'), ('Sz', 'mapSz.dat')):
        data, used = collect(config_path, fname, indices=indices)
        col = data.shape[2] - 1                     # last column = the spin value
        out.setdefault('time', data[0, :, 0])
        n = data.shape[0]
        out[comp]           = data[:, :, col].mean(axis=0)
        out[comp + '_sem']  = (data[:, :, col].std(axis=0, ddof=1) / np.sqrt(n)
                               if n > 1 else np.zeros(data.shape[1]))
    print(f'[average] mapS[xyz]: averaged {n} trajectories')
    return out


def _main():
    ap = argparse.ArgumentParser(description='Average an RP_MASH output file over trajectories.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--file', required=True, help='output filename to average (e.g. mapSz.dat)')
    ap.add_argument('--cols', type=int, nargs='+', default=None, help='column indices to keep')
    ap.add_argument('--out', default=None, help='write [mean | sem] columns to this .dat (default: <file>.avg)')
    a = ap.parse_args()

    mean, sem = average_column(a.config, a.file, cols=a.cols)
    run_dir, _, _ = traj_dirs(a.config)
    out = a.out or os.path.join(run_dir, a.file.replace('.dat', '') + '.avg')
    np.savetxt(out, np.hstack([mean, sem]),
               header='columns: [means...] then [standard errors...]')
    print(f'[average] wrote {out}  shape={mean.shape}')


if __name__ == '__main__':
    _main()
