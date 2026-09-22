"""
consolidate: gather every trajectory's raw .dat files into ONE HDF5 file for analysis.

HDF5 stores each observable as a big on-disk array; slicing it (e.g. f['nucR'][7] or
f['nucR'][:, :, 0, 0]) reads only that slice, so an analysis script never has to hold the whole
run in RAM. Building the file is likewise streaming -- one trajectory is loaded, reshaped, and
written at a time -- so peak memory is ~one trajectory no matter how many trajectories or how long.

Layout written (n = number of trajectories actually found, T = number of time rows):

    /time                     (T,)                shared time grid
    /traj_index               (n,)                original trajectory indices present in the file
    /nucR                     (n, T, nbds, nnuc)  ring-polymer bead positions
    /nucP                     (n, T, nbds, nnuc)  ring-polymer bead momenta
    /mapSx  /mapSy  /mapSz    (n, T, nbds)        mapping spin components per bead
    /output                   (n, T, ncol)        the scalar-observables file (see column_labels attr)
    /weights                  (n, R, 6)           quantum-jump weights, one row per jump (present
                                                  only when the run used Tjump). Row 0 is t=0;
                                                  columns are t, Sz_n, W_PP, W_CP, W_PC, W_CC.
                                                  RAGGED -- zero-padded to the longest trajectory.
    /n_weights                (n,)                valid row count per trajectory for /weights
    root attrs: config_json + convenience scalars (nbds, nnuc, nstates, beta, delt, gamma, n_traj)

Datasets are chunked one-trajectory-per-chunk and gzip-compressed, so `f['nucR'][k]` is a single
cheap read. Missing trajectories (e.g. a crash) are skipped and omitted from /traj_index.
The weights are piecewise constant in time and so are stored only where they change; use
expand_weights(f) to rebuild the dense (n, T) arrays an analysis script wants.

CLI:
    python -m workflow.consolidate --config config.py                 # -> data.hdf next to config
    python -m workflow.consolidate --config config.py --out big.h5 --no-compress
"""

import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py

from .runner import load_config, traj_dirs

# how many trajectories are submitted to the worker pool at once. Executor.map materializes its
# input, so slabs keep the task list small at large n and bound in-flight memory to about
# workers * chunksize * (one trajectory).
_SLAB = 50000


# filename (without .dat) -> how to reshape the post-time columns of one trajectory.
# 'RxN' -> (T, nbds, nnuc);  'N' -> (T, nbds);  'raw' -> (T, ncol) untouched.
_FILE_SHAPES = {
    'nucR':  'RxN',
    'nucP':  'RxN',
    'mapSx': 'N',
    'mapSy': 'N',
    'mapSz': 'N',
    'output': 'raw',
    # 'table' -> a RAGGED per-trajectory table: no shared time grid, a variable number of rows per
    # trajectory, column 0 kept as-is. Stored padded to the longest trajectory plus a row-count
    # dataset (see _TABLE_COUNT). Must stay LAST so files[0] remains the existence probe.
    'weights': 'table',
}

# ragged 'table' files -> the name of the companion per-trajectory row-count dataset
_TABLE_COUNT = {'weights': 'n_weights'}

# columns of weights.dat (see mash_rpmd.write_jump_weights); row n is sample event n, row 0 is t=0
_WEIGHTS_LABELS = ['t', 'Sz_n', 'W_PP', 'W_CP', 'W_PC', 'W_CC']

# human-readable columns of output.dat (see mash_rpmd.print_data)
_OUTPUT_LABELS = ['time', 'etot', 'engke', 'engpe', 'mean_sign_Sz']  # + nucR_com per nuclei


def _load_and_shape(path, kind, nbds, nnuc):
    """Load one .dat, split off the time column, reshape the rest per `kind`. Returns (time, arr)."""
    raw = np.loadtxt(path)
    if kind == 'table':
        # Ragged table: column 0 is a per-row event time, NOT a shared grid, so nothing is stripped.
        # A file with no data rows loads as shape (0,) and must not go through the 1-D promotion
        # below, which would make it (1, 0) and then raise on raw[:, 0].
        if raw.size == 0:
            return None, raw.reshape(0, 0)
        return None, np.atleast_2d(raw)
    if raw.ndim == 1:                      # a single-row file loads as 1D
        raw = raw[None, :]
    time = raw[:, 0]
    body = raw[:, 1:]
    T = body.shape[0]
    if kind == 'RxN':
        arr = body.reshape(T, nbds, nnuc)
    elif kind == 'N':
        arr = body.reshape(T, nbds)
    else:                                  # 'raw'
        arr = body
    return time, arr


def _count_table_rows(path):
    """
    Rows in a ragged 'table' file WITHOUT parsing it: non-blank, non-comment lines.

    Used only to size the padded dataset. Parsing every weights.dat with np.loadtxt just to read
    .shape[0] costs a full parse per trajectory at startup, and the same files are parsed again in
    the write loop -- at 1e6 trajectories that is minutes of pure startup for a number a line count
    gives in milliseconds.
    """
    if not os.path.isfile(path):
        return 0
    with open(path, 'rb') as fh:
        return sum(1 for line in fh if line.strip() and not line.lstrip().startswith(b'#'))


def _load_traj(task):
    """
    Parse every requested file for ONE trajectory; returns {name: array or None}.

    Module level so ProcessPoolExecutor can pickle it. Workers only parse -- h5py here has no MPI
    support (h5py.get_config().mpi is False), so all writing stays in the parent process.
    """
    d, specs, nbds, nnuc = task
    out = {}
    for name, kind in specs:
        p = os.path.join(d, name + '.dat')
        out[name] = _load_and_shape(p, kind, nbds, nnuc)[1] if os.path.isfile(p) else None
    return out


def consolidate(config_path, out=None, files=None, compression='lzf', indices=None,
                workers=1, chunksize=8):
    """
    Combine per-trajectory .dat files into a single HDF5 file. Returns the output path.

    out         : output .hdf path (default: data.hdf next to the config)
    files       : which observables to include (default: all keys of _FILE_SHAPES that exist)
    compression : 'lzf' (default), 'gzip', or None. float64 trajectory data is close to
                  incompressible -- measured ratios are lzf 1.00x, gzip 1.04x, gzip+shuffle 1.20x --
                  so gzip roughly doubles the runtime for a few percent of disk. It also runs in
                  this single writer process, which caps any benefit from `workers`.
    indices     : trajectory indices to include (default: all that exist on disk)
    workers     : processes used to READ AND PARSE trajectories (default 1 = the serial path).
                  Writing is always serial. Gains flatten around 4-8; past that the writer and the
                  inter-process transfer dominate.
    chunksize   : trajectories handed to a pool worker at a time (amortizes IPC overhead)
    """
    cfg = load_config(config_path)
    run_dir, _, dirs = traj_dirs(config_path)
    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])

    if files is None:
        files = list(_FILE_SHAPES)
    if indices is None:
        indices = range(len(dirs))

    # which trajectories are actually present (use the first requested file as the existence probe)
    indices = list(indices)          # materialize: the comprehension below would consume a generator
    probe = files[0]
    present = [i for i in indices if os.path.isfile(os.path.join(dirs[i], probe + '.dat'))]
    present_set = set(present)       # set membership: a list here makes the next line O(n^2)
    missing = [i for i in indices if i not in present_set]
    if not present:
        raise FileNotFoundError(f'no "{probe}.dat" found in any trajectory directory under {run_dir}')
    if missing:
        print(f'[consolidate] WARNING: {len(missing)} trajectories missing {probe}.dat, skipped: '
              f'{missing[:20]}{" ..." if len(missing) > 20 else ""}')
    n = len(present)

    # establish the time grid / row count from the first present trajectory
    time0, _ = _load_and_shape(os.path.join(dirs[present[0]], probe + '.dat'),
                               _FILE_SHAPES[probe], nbds, nnuc)
    T = time0.shape[0]

    if out is None:
        out = os.path.join(run_dir, 'data.hdf')

    comp = dict(compression=compression) if compression else {}

    with h5py.File(out, 'w') as f:
        # ---- metadata -------------------------------------------------------
        f.attrs['config_json'] = json.dumps(cfg, default=str)
        for k in ('nbds', 'nnuc', 'nstates', 'beta', 'delt'):
            if k in cfg:
                f.attrs[k] = cfg[k]
        f.attrs['n_traj'] = n
        gamma = cfg.get('langevin_params', {}).get('gamma')
        if gamma is not None:
            f.attrs['gamma'] = gamma

        f.create_dataset('time', data=time0)
        f.create_dataset('traj_index', data=np.array(present, dtype=int))

        # ---- pre-create one dataset per observable, then stream trajectories in ----
        dsets = {}
        counts = {}          # ragged 'table' observables -> (count dataset, {traj: n_rows})
        for name in files:
            if name not in _FILE_SHAPES:
                print(f'[consolidate] WARNING: unknown file "{name}", skipping')
                continue
            # probe this file's per-trajectory shape from the first present trajectory
            p0 = os.path.join(dirs[present[0]], name + '.dat')
            if not os.path.isfile(p0):
                print(f'[consolidate] WARNING: {name}.dat absent, skipping this observable')
                continue
            _, a0 = _load_and_shape(p0, _FILE_SHAPES[name], nbds, nnuc)
            if _FILE_SHAPES[name] == 'table':
                # Ragged: row counts differ per trajectory, so size the middle axis from the LONGEST
                # trajectory and pad the rest with zeros. A companion count dataset records how many
                # rows are valid. The pre-pass only counts lines -- it must not parse, or startup
                # pays a full np.loadtxt per trajectory for a number it throws away.
                r_max = max(max(_count_table_rows(os.path.join(dirs[t], name + '.dat'))
                                for t in present), 1)
                shape = (n, r_max, a0.shape[1])
                chunks = (1, r_max, a0.shape[1])
                dsets[name] = f.create_dataset(name, shape=shape, dtype='f8', chunks=chunks, **comp)
                # counts are accumulated in memory and written once; one element at a time would be
                # a separate HDF5 write per trajectory (~25 s at 1e6)
                counts[name] = (f.create_dataset(_TABLE_COUNT[name], shape=(n,), dtype='i8'),
                                np.zeros(n, dtype='i8'))
                if name == 'weights':
                    dsets[name].attrs['column_labels'] = list(_WEIGHTS_LABELS)
                continue
            shape = (n,) + a0.shape
            chunks = (1,) + a0.shape          # one trajectory per chunk -> cheap per-traj reads
            dsets[name] = f.create_dataset(name, shape=shape, dtype='f8', chunks=chunks, **comp)
            if name == 'output':
                labels = list(_OUTPUT_LABELS) + [f'nucR_com{j}' for j in range(a0.shape[1] - len(_OUTPUT_LABELS))]
                dsets[name].attrs['column_labels'] = labels

        # ---- write trajectory by trajectory (peak RAM = one trajectory, or one slab in flight) ----
        # Parsing is per-trajectory CPU and parallelizes cleanly; writing must stay in this process
        # because this h5py has no MPI support. Workers therefore only read and parse.
        specs = [(name, _FILE_SHAPES[name]) for name in dsets]
        every = max(1, n // 100)                       # throttle progress: one line per ~1% of the run
        absent = {}                                    # name -> how many trajectories lacked that file
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        try:
            row = 0
            for lo in range(0, n, _SLAB):
                slab = present[lo:lo + _SLAB]
                tasks = [(dirs[t], specs, nbds, nnuc) for t in slab]
                stream = (pool.map(_load_traj, tasks, chunksize=chunksize) if pool
                          else (_load_traj(t) for t in tasks))
                for traj, loaded in zip(slab, stream):
                    for name, dset in dsets.items():
                        arr = loaded[name]
                        if arr is None:
                            # aggregate: one line per trajectory per file would be unusable at 1e6
                            absent[name] = absent.get(name, 0) + 1
                            continue
                        if _FILE_SHAPES[name] == 'table':
                            # ragged: a row count of its own, no shared-time-grid check
                            m = arr.shape[0]
                            if m:
                                dset[row, :m] = arr
                            counts[name][1][row] = m
                            continue
                        if arr.shape[0] != T:
                            raise ValueError(f'traj {traj} {name}.dat has {arr.shape[0]} rows, expected {T} '
                                             f'(time grids must match across trajectories)')
                        dset[row] = arr
                    row += 1
                    if row % every == 0 or row == n:
                        print(f'[consolidate] wrote traj {traj}  ({row}/{n})', flush=True)
        finally:
            if pool is not None:
                pool.shutdown()

        for name, k in sorted(absent.items()):
            print(f'[consolidate] WARNING: {k}/{n} trajectories missing {name}.dat, left as zeros')

        # one batched write per ragged observable instead of one element per trajectory
        for name, (cdset, carr) in counts.items():
            cdset[:] = carr

    print(f'[consolidate] {out}  ({n} trajectories, {T} time rows, observables: {list(dsets)})')
    return out


def open_run(path):
    """Convenience: open a consolidated run read-only. `with open_run("data.hdf") as f: ...`"""
    return h5py.File(path, 'r')


def expand_weights(f, time=None):
    """
    Expand the per-jump quantum-jump weight table into dense (n_traj, T) arrays.

    The weights are stored only where they change (one row per jump, plus row 0 at t=0), because
    they are piecewise constant in time. This rebuilds the dense form the analysis wants:

        W(t) = W^(m)   where   m = #{ jumps with t_jump <= t }

    so W is W^(0) for every t before the first jump, and a jump landing exactly on a grid point
    counts at that point (matching mash_rpmd, which writes the post-jump state at t_jump).

    f    : open h5py.File from open_run()
    time : time grid (default: the run's own /time)

    Returns a dict with keys 'W_PP', 'W_CP', 'W_PC', 'W_CC', each of shape (n_traj, T).

    A run made without Tjump has no weight table; there W is just W^(0) for all t, which the caller
    can build directly from Sz0 = f['mapSz'][:, 0].mean(axis=-1) as (2|Sz0|, 2, 2, 3).
    """
    if 'weights' not in f:
        raise KeyError("no 'weights' dataset in this run -- it was made without Tjump")
    time = f['time'][:] if time is None else np.asarray(time)
    tab  = f['weights'][:]                 # (n, r_max, 6): t, Sz_n, W_PP, W_CP, W_PC, W_CC
    cnt  = f[_TABLE_COUNT['weights']][:]   # (n,)
    names = ('W_PP', 'W_CP', 'W_PC', 'W_CC')
    out = {k: np.empty((tab.shape[0], time.size)) for k in names}
    for k in range(tab.shape[0]):
        m = int(cnt[k])
        ts = tab[k, :m, 0]                                  # sample times, ascending, ts[0] = t0
        idx = np.searchsorted(ts, time, side='right') - 1   # >= 0 since ts[0] <= time[0]
        np.clip(idx, 0, m - 1, out=idx)
        for j, name in enumerate(names):
            out[name][k] = tab[k, :m, 2 + j][idx]
    return out


def _main():
    ap = argparse.ArgumentParser(description='Consolidate trajectory .dat files into one HDF5 file.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='output .hdf (default: data.hdf next to config)')
    ap.add_argument('--files', nargs='+', default=None,
                    help='observables to include (default: all). e.g. --files nucR nucP output')
    ap.add_argument('--compression', default='lzf', choices=('lzf', 'gzip', 'none'),
                    help='HDF5 filter (default lzf; gzip buys ~4%% on this data for ~2x the runtime)')
    ap.add_argument('--no-compress', action='store_true', help='alias for --compression none')
    ap.add_argument('--workers', type=int, default=1,
                    help='processes used to read/parse trajectories (default 1 = serial). '
                         'Writing is always serial; gains flatten around 4-8. Try --workers 8.')
    a = ap.parse_args()
    comp = None if (a.no_compress or a.compression == 'none') else a.compression
    consolidate(a.config, out=a.out, files=a.files, compression=comp, workers=a.workers)


if __name__ == '__main__':
    _main()
