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
    root attrs: config_json + convenience scalars (nbds, nnuc, nstates, beta, delt, gamma, n_traj)

Datasets are chunked one-trajectory-per-chunk and gzip-compressed, so `f['nucR'][k]` is a single
cheap read. Missing trajectories (e.g. a crash) are skipped and omitted from /traj_index.

CLI:
    python -m workflow.consolidate --config config.py                 # -> data.hdf next to config
    python -m workflow.consolidate --config config.py --out big.h5 --no-compress
"""

import os
import json
import argparse

import numpy as np
import h5py

from .runner import load_config, traj_dirs


# filename (without .dat) -> how to reshape the post-time columns of one trajectory.
# 'RxN' -> (T, nbds, nnuc);  'N' -> (T, nbds);  'raw' -> (T, ncol) untouched.
_FILE_SHAPES = {
    'nucR':  'RxN',
    'nucP':  'RxN',
    'mapSx': 'N',
    'mapSy': 'N',
    'mapSz': 'N',
    'output': 'raw',
}

# human-readable columns of output.dat (see mash_rpmd.print_data)
_OUTPUT_LABELS = ['time', 'etot', 'engke', 'engpe', 'mean_sign_Sz']  # + nucR_com per nuclei


def _load_and_shape(path, kind, nbds, nnuc):
    """Load one .dat, split off the time column, reshape the rest per `kind`. Returns (time, arr)."""
    raw = np.loadtxt(path)
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


def consolidate(config_path, out=None, files=None, compression='gzip', indices=None):
    """
    Combine per-trajectory .dat files into a single HDF5 file. Returns the output path.

    out         : output .hdf path (default: data.hdf next to the config)
    files       : which observables to include (default: all keys of _FILE_SHAPES that exist)
    compression : 'gzip' (default), None to disable
    indices     : trajectory indices to include (default: all that exist on disk)
    """
    cfg = load_config(config_path)
    run_dir, _, dirs = traj_dirs(config_path)
    nbds, nnuc = int(cfg['nbds']), int(cfg['nnuc'])

    if files is None:
        files = list(_FILE_SHAPES)
    if indices is None:
        indices = range(len(dirs))

    # which trajectories are actually present (use the first requested file as the existence probe)
    probe = files[0]
    present = [i for i in indices if os.path.isfile(os.path.join(dirs[i], probe + '.dat'))]
    missing = [i for i in indices if i not in present]
    if not present:
        raise FileNotFoundError(f'no "{probe}.dat" found in any trajectory directory under {run_dir}')
    if missing:
        print(f'[consolidate] WARNING: {len(missing)} trajectories missing {probe}.dat, skipped: {missing}')
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
            shape = (n,) + a0.shape
            chunks = (1,) + a0.shape          # one trajectory per chunk -> cheap per-traj reads
            dsets[name] = f.create_dataset(name, shape=shape, dtype='f8', chunks=chunks, **comp)
            if name == 'output':
                labels = list(_OUTPUT_LABELS) + [f'nucR_com{j}' for j in range(a0.shape[1] - len(_OUTPUT_LABELS))]
                dsets[name].attrs['column_labels'] = labels

        # ---- write trajectory by trajectory (peak RAM = one trajectory) -----
        for row, traj in enumerate(present):
            for name, dset in dsets.items():
                path = os.path.join(dirs[traj], name + '.dat')
                if not os.path.isfile(path):
                    print(f'[consolidate] WARNING: traj {traj} missing {name}.dat, left as zeros')
                    continue
                t, arr = _load_and_shape(path, _FILE_SHAPES[name], nbds, nnuc)
                if arr.shape[0] != T:
                    raise ValueError(f'traj {traj} {name}.dat has {arr.shape[0]} rows, expected {T} '
                                     f'(time grids must match across trajectories)')
                dset[row] = arr
            print(f'[consolidate] wrote traj {traj}  ({row + 1}/{n})', flush=True)

    print(f'[consolidate] {out}  ({n} trajectories, {T} time rows, observables: {list(dsets)})')
    return out


def open_run(path):
    """Convenience: open a consolidated run read-only. `with open_run("data.hdf") as f: ...`"""
    return h5py.File(path, 'r')


def _main():
    ap = argparse.ArgumentParser(description='Consolidate trajectory .dat files into one HDF5 file.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='output .hdf (default: data.hdf next to config)')
    ap.add_argument('--files', nargs='+', default=None,
                    help='observables to include (default: all). e.g. --files nucR nucP output')
    ap.add_argument('--no-compress', action='store_true', help='disable gzip compression')
    a = ap.parse_args()
    consolidate(a.config, out=a.out, files=a.files,
                compression=None if a.no_compress else 'gzip')


if __name__ == '__main__':
    _main()
