"""
shards: store trajectory output in SHARD files instead of one directory per trajectory.

Enabled by a `shard_size` key in the run config (e.g. 1000). Trajectory idx belongs to shard
idx // shard_size, and each shard is ONE HDF5 file in <run_dir>/<shard_dir>/ (default 'shards'):

    shard_<s>.hdf     s zero-padded to exactly the digits n_shards-1 needs
                      (n_traj=10M, shard_size=1000 -> 10000 shards -> shard_0000 .. shard_9999)

A shard holds the same data consolidate.py would produce for its trajectories, in the same layout:

    /time                  (T,)                shared time grid
    /traj_index            (n,)                trajectory indices committed to this shard
    /nucR /nucP            (n, T, nbds, nnuc)
    /mapSx /mapSy /mapSz   (n, T, nbds)
    /output                (n, T, ncol)        column_labels attr as in consolidate
    /weights               (n, R, 15)          only with Tjump; ragged, zero-padded (column_labels attr)
    /n_weights             (n,)                valid rows per trajectory in /weights
    root attrs: config_json, nbds, nnuc, nstates, beta, delt, gamma, n_traj (= rows committed),
                plus shard_id, n_shards, shard_size, idx_start, idx_stop

No-concurrent-writer guarantee: a SLURM array task (or a local process) OWNS a contiguous block of
whole shards -- task t writes shards t*K .. t*K+K-1 and nothing else, with K = ceil(n_shards /
max_tasks). Every shard therefore has exactly one writer for the whole run, and n_tasks <= n_shards
always (a task never gets zero shards). With slurm cpus_per_task = c > 1, a task runs c trajectories
at once in child processes that only compute; the task's main process is still the only one that
opens its shards for writing.

Each trajectory still runs in its own directory, because RP_MASH writes hardcoded .dat files into
the cwd -- but that directory lives under a per-task temp dir (tempfile honours $TMPDIR, which SLURM
clusters point at node-local disk), is parsed straight into the shard, and is deleted. The shared
filesystem sees only the shard files.

Custom records: if the driver's run_trajectory RETURNS a dict {name: array}, that dict is stored as
the trajectory's row instead of parsing .dat files (no /time) -- e.g. a thermalization stage that
saves a restart state. Each name becomes a dataset of shape (n,) + array.shape. If the driver module
also defines shard_attrs(cfg) -> dict, those entries are added to every shard's root attrs.

Commit/resume: a trajectory's rows are written first and its traj_index entry LAST, so
len(traj_index) counts only fully written trajectories; a row past it (a write cut off by a kill)
is overwritten by the next append. On (re)start a task skips every trajectory already in
traj_index, so resubmitting the same submit_grid.sh after a timeout picks up where it stopped. A
trajectory that raises is reported and left out; the task exits non-zero at the end.

Analysis: the shards ARE the dataset. Analysis scripts read them directly and in parallel with
workflow.reduce (a map-reduce: each worker process reads whole shards and returns partial sums), as
one multi-core SLURM job -- no merge needed, and no second copy of the data written.

Merge (OPTIONAL: archiving, or handing one file to a tool that cannot read shards) into one
consolidate-style data.hdf (same layout, every trajectory, sorted by index). It is a serial
read-everything-write-everything pass, so it is the slow way to get to an analysis:
    python -m workflow.shards --config config.py               # -> <run_dir>/data.hdf
    python -m workflow.shards --config config.py --out big.hdf --slab 2000
"""

import os
import json
import math
import glob
import shutil
import inspect
import argparse
import tempfile

import numpy as np
import h5py

from .consolidate import (_FILE_SHAPES, _TABLE_COUNT, _TABLE_LABELS, _OUTPUT_LABELS,
                          _load_and_shape)


# ------------------------------------------------------------------ layout
def shard_layout(n_traj, shard_size):
    """(n_shards, digits) for this run."""
    n_shards = math.ceil(int(n_traj) / int(shard_size))
    return n_shards, len(str(max(n_shards - 1, 0)))


def shard_dir(cfg, run_dir):
    return os.path.join(run_dir, cfg.get('shard_dir', 'shards'))


def shard_path(cfg, run_dir, s):
    n_shards, width = shard_layout(cfg['n_traj'], cfg['shard_size'])
    return os.path.join(shard_dir(cfg, run_dir), f'shard_{s:0{width}d}.hdf')


def shard_range(cfg, s):
    """Trajectory indices [start, stop) held by shard s."""
    size = int(cfg['shard_size'])
    return s * size, min((s + 1) * size, int(cfg['n_traj']))


def task_layout(n_shards, max_tasks):
    """(n_tasks, shards_per_task): contiguous blocks of whole shards, at most max_tasks tasks."""
    per = math.ceil(n_shards / max(1, int(max_tasks)))
    return math.ceil(n_shards / per), per


def task_shards(task_id, shards_per_task, n_shards):
    return range(task_id * shards_per_task, min((task_id + 1) * shards_per_task, n_shards))


# ------------------------------------------------------------------ writing one trajectory
def _shard_attrs(cfg, s, n_shards):
    start, stop = shard_range(cfg, s)
    a = {'config_json': json.dumps(cfg, default=str),
         'shard_id': s, 'n_shards': n_shards, 'shard_size': int(cfg['shard_size']),
         'idx_start': start, 'idx_stop': stop}
    for k in ('nbds', 'nnuc', 'nstates', 'beta', 'delt'):
        if k in cfg:
            a[k] = cfg[k]
    gamma = cfg.get('langevin_params', {}).get('gamma')
    if gamma is not None:
        a['gamma'] = gamma
    return a


def parse_traj(workdir, nbds, nnuc):
    """
    Parse one finished trajectory's .dat files -> (time, {name: array}). Files absent from the run
    entirely (e.g. weights.dat without Tjump) are simply not returned.
    """
    time, data = None, {}
    for name, kind in _FILE_SHAPES.items():
        p = os.path.join(workdir, name + '.dat')
        if not os.path.isfile(p):
            continue
        t, arr = _load_and_shape(p, kind, nbds, nnuc)
        if kind == 'table' and arr.shape[1] == 0:          # no jumps: empty table, known width
            arr = np.zeros((0, len(_TABLE_LABELS[name])))
        if t is not None and time is None:
            time = t
        data[name] = arr
    if time is None:
        raise FileNotFoundError(f'no trajectory output (.dat) found in {workdir}')
    return time, data


def committed(path):
    """Set of trajectory indices fully written to the shard at `path`."""
    if not os.path.isfile(path):
        return set()
    with h5py.File(path, 'r') as f:
        return set(f['traj_index'][:].tolist()) if 'traj_index' in f else set()


def _is_table(name):
    return _FILE_SHAPES.get(name) == 'table'


# shard datasets that are bookkeeping rather than one row per trajectory
_BOOKKEEPING = {'time', 'traj_index'} | set(_TABLE_COUNT.values())


def append_traj(path, idx, time, data, attrs, compression='lzf'):
    """
    Append one trajectory as the next row of the shard at `path` (one writer per shard). `data` is
    {name: array}; `time` is the shared time grid, or None for a record without one.
    """
    comp = dict(compression=compression) if compression else {}
    data = {k: np.asarray(v) for k, v in data.items()}
    with h5py.File(path, 'a') as f:
        if 'traj_index' not in f:
            f.attrs.update(attrs)
            if time is not None and 'time' not in f:
                f.create_dataset('time', data=time)
            for name, arr in data.items():
                # `not in f` guards: a kill between creations can leave some datasets behind
                if _is_table(name):
                    if name not in f:
                        ncol = arr.shape[1]
                        rows = max(16, 2 * arr.shape[0])
                        d = f.create_dataset(name, shape=(0, rows, ncol), maxshape=(None, None, ncol),
                                             chunks=(1, rows, ncol), dtype='f8', **comp)
                        d.attrs['column_labels'] = list(_TABLE_LABELS[name])
                    if _TABLE_COUNT[name] not in f:
                        f.create_dataset(_TABLE_COUNT[name], shape=(0,), maxshape=(None,),
                                         chunks=(1024,), dtype='i8')
                elif name not in f:
                    sh = arr.shape
                    d = f.create_dataset(name, shape=(0,) + sh, maxshape=(None,) + sh,
                                         chunks=(1,) + sh if sh else (1024,), dtype=arr.dtype,
                                         **(comp if sh else {}))
                    if name == 'output':
                        d.attrs['column_labels'] = list(_OUTPUT_LABELS) + [
                            f'nucR_com{j}' for j in range(arr.shape[1] - len(_OUTPUT_LABELS))]
            f.create_dataset('traj_index', shape=(0,), maxshape=(None,), chunks=(1024,), dtype='i8')

        if (time is None) != ('time' not in f) or (time is not None
                                                   and not np.array_equal(f['time'][:], time)):
            raise ValueError(f'traj {idx}: time grid differs from the rest of {path}')
        names = sorted(k for k in f if k not in _BOOKKEEPING)
        if names != sorted(data):
            raise ValueError(f'traj {idx}: produced {sorted(data)} but {path} holds {names}')

        n = f['traj_index'].shape[0]
        for name, arr in data.items():
            d = f[name]
            if d.shape[0] < n + 1:
                d.resize(n + 1, axis=0)
            if _is_table(name):
                m = arr.shape[0]
                if m > d.shape[1]:
                    d.resize(m, axis=1)
                row = np.zeros(d.shape[1:])     # full padded row: clears leftovers of a cut-off write
                row[:m] = arr
                d[n] = row
                c = f[_TABLE_COUNT[name]]
                if c.shape[0] < n + 1:
                    c.resize(n + 1, axis=0)
                c[n] = m
            else:
                d[n] = arr
        f.flush()
        f['traj_index'].resize(n + 1, axis=0)
        f['traj_index'][n] = int(idx)
        f.attrs['n_traj'] = n + 1


# ------------------------------------------------------------------ running a task's shards
def _call_system(system, cfg, seed, idx, run_dir):
    """run_trajectory(cfg, seed), plus idx/run_dir when the driver accepts them (the cwd no longer
    sits under the run dir, so a driver resolving paths from the cwd needs run_dir handed to it)."""
    params = inspect.signature(system.run_trajectory).parameters
    kw = {k: v for k, v in (('idx', idx), ('run_dir', run_dir)) if k in params}
    return system.run_trajectory(cfg, seed, **kw)


def _run_traj(cfg, system, run_dir, work_root, idx, seed_for_idx):
    """Run trajectory idx in its own temp dir under work_root -> (time, data). Never writes a shard."""
    workdir = os.path.join(work_root, f'traj{idx}')
    os.makedirs(workdir)
    seed = seed_for_idx(cfg.get('base_seed'), idx)
    os.chdir(workdir)
    try:
        record = _call_system(system, cfg, seed, idx, run_dir)
        if isinstance(record, dict):        # the driver supplied its own row
            return None, record
        return parse_traj(workdir, int(cfg['nbds']), int(cfg['nnuc']))
    finally:
        os.chdir(run_dir)
        shutil.rmtree(workdir, ignore_errors=True)


# per-process state of a pool child, set once by _pool_init
_CHILD = {}


def _pool_init(config_path, work_root):
    """Pool child initializer: load the config and driver once per child. Children are spawned, not
    forked, so they inherit nothing of the parent -- in particular no HDF5 state."""
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    from .runner import load_config, load_system, seed_for_idx
    cfg = load_config(config_path)
    run_dir = os.path.dirname(os.path.abspath(config_path))
    _CHILD.update(cfg=cfg, run_dir=run_dir, work_root=work_root, seed_for_idx=seed_for_idx,
                  system=load_system(os.path.join(run_dir, cfg.get('system_file', 'system.py'))))


def _pool_run(idx):
    """Pool child: run one trajectory and RETURN its data to the parent, the only shard writer."""
    c = _CHILD
    try:
        time, data = _run_traj(c['cfg'], c['system'], c['run_dir'], c['work_root'], idx,
                               c['seed_for_idx'])
        return idx, time, data, None
    except Exception as e:
        return idx, None, None, f'{type(e).__name__}: {e}'


def run_shards(cfg, system, run_dir, shards, seed_for_idx, ncores=1, config_path=None):
    """
    Run every trajectory of `shards` (shard ids this process owns) and append each to its shard.
    Returns the list of trajectory indices that failed.

    ncores > 1 (needs config_path): the trajectories run in a pool of `ncores` child processes that
    only COMPUTE -- each returns its trajectory's data -- and this process alone appends it to the
    shard. So a shard still has exactly one writer however many cores run its trajectories. The pool
    is fed one queue across all of `shards` in index order, so every core stays busy until the block
    runs out; rows are committed in completion order, which traj_index records (readers sort by it).
    """
    n_shards, _ = shard_layout(cfg['n_traj'], cfg['shard_size'])
    os.makedirs(shard_dir(cfg, run_dir), exist_ok=True)
    work_root = tempfile.mkdtemp(prefix='rpmash_', dir=cfg.get('work_dir'))
    extra_attrs = system.shard_attrs(cfg) if hasattr(system, 'shard_attrs') else {}
    comp = cfg.get('shard_compression', 'lzf')
    size = int(cfg['shard_size'])

    # what is left to run, and which shard each trajectory goes to
    plan = {}                                       # shard -> (path, attrs)
    left = {}                                       # shard -> trajectories not yet finished
    queue = []
    for s in shards:
        path  = shard_path(cfg, run_dir, s)
        done  = committed(path)
        start, stop = shard_range(cfg, s)
        todo = [i for i in range(start, stop) if i not in done]
        plan[s] = (path, {**_shard_attrs(cfg, s, n_shards), **extra_attrs})
        left[s] = len(todo)
        queue  += todo
        print(f'[shards] shard {s}: traj {start}..{stop - 1}, {len(done)} already done, '
              f'{len(todo)} to run -> {path}', flush=True)
        if not todo:
            print(f'[shards] shard {s} finished', flush=True)

    failed = []

    def commit(idx, time, data, err):
        """Append one finished trajectory to its shard. Only ever called in THIS process."""
        s = idx // size
        if err is None:
            try:
                append_traj(plan[s][0], idx, time, data, plan[s][1], comp)
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
        if err is not None:
            failed.append(idx)
            print(f'[shards] traj {idx} FAILED: {err}', flush=True)
        left[s] -= 1
        if left[s] == 0:
            print(f'[shards] shard {s} finished', flush=True)

    try:
        if ncores <= 1 or len(queue) <= 1:
            for idx in queue:
                try:
                    time, data = _run_traj(cfg, system, run_dir, work_root, idx, seed_for_idx)
                except Exception as e:
                    commit(idx, None, None, f'{type(e).__name__}: {e}')
                else:
                    commit(idx, time, data, None)
        else:
            _run_pool(queue, ncores, config_path, work_root, commit)
    finally:
        os.chdir(run_dir)
        shutil.rmtree(work_root, ignore_errors=True)
    return sorted(failed)


def _run_pool(queue, ncores, config_path, work_root, commit):
    """
    Run `queue` on `ncores` spawned children, at most 2*ncores in flight (bounded memory), and
    commit() each result in this process as it arrives.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from concurrent.futures.process import BrokenProcessPool
    if config_path is None:
        raise ValueError('run_shards(ncores > 1) needs config_path to start its pool')
    ncores = min(int(ncores), len(queue))
    print(f'[shards] running {len(queue)} trajectories on {ncores} cores '
          f'(this process is the only shard writer)', flush=True)

    # A child that dies hard (segfault, OOM kill) breaks the whole pool: every pending trajectory
    # then fails with BrokenProcessPool, and so does any later submit. What already finished is kept,
    # everything else is reported failed, and a resubmission resumes from the committed rows.
    _DIED = 'BrokenProcessPool: a pool child died (segfault / OOM kill?)'
    state = {'broken': False}

    def result(fut, idx):
        try:
            return fut.result()
        except BrokenProcessPool:
            state['broken'] = True
            return idx, None, None, _DIED

    todo, running = iter(queue), {}
    with ProcessPoolExecutor(ncores, mp_context=mp.get_context('spawn'), initializer=_pool_init,
                             initargs=(os.path.abspath(config_path), work_root)) as ex:
        def fill():
            while not state['broken'] and len(running) < 2 * ncores:
                idx = next(todo, None)
                if idx is None:
                    return
                try:
                    running[ex.submit(_pool_run, idx)] = idx
                except BrokenProcessPool:
                    state['broken'] = True
                    commit(idx, None, None, _DIED)

        fill()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                commit(*result(fut, running.pop(fut)))
            fill()
        for idx in todo:                     # left over only if the pool broke
            commit(idx, None, None, 'not run: the process pool broke')


# ------------------------------------------------------------------ merging into one data.hdf
def merge(config_path, out=None, slab=1000, compression='lzf'):
    """
    Merge every shard of this run into ONE file with consolidate.py's layout (trajectories sorted by
    index; /weights padded to the longest table across all shards). Streams slab rows at a time.
    """
    from .runner import load_config
    cfg = load_config(config_path)
    run_dir = os.path.dirname(os.path.abspath(config_path))
    paths = sorted(glob.glob(os.path.join(shard_dir(cfg, run_dir), 'shard_*.hdf')))
    if not paths:
        raise FileNotFoundError(f'no shard_*.hdf in {shard_dir(cfg, run_dir)}')
    out = out or os.path.join(run_dir, 'data.hdf')

    # pass 1: committed rows, shapes, and the widest ragged table
    shards, shapes, time, rmax = [], None, None, {}
    for p in paths:
        with h5py.File(p, 'r') as f:
            if 'traj_index' not in f or f['traj_index'].shape[0] == 0:
                print(f'[shards] WARNING: {p} has no committed trajectories, skipped')
                continue
            idx = f['traj_index'][:]
            sh = {k: f[k].shape[1:] for k in f if k in _FILE_SHAPES}
            if shapes is None:
                shapes, time, attrs = sh, f['time'][:], dict(f.attrs)
            else:
                same = {k: v for k, v in sh.items() if _FILE_SHAPES[k] != 'table'} == \
                       {k: v for k, v in shapes.items() if _FILE_SHAPES[k] != 'table'}
                if sorted(sh) != sorted(shapes) or not same or not np.array_equal(f['time'][:], time):
                    raise ValueError(f'{p} does not match the layout of {shards[0][0]}')
            for k in sh:
                if _FILE_SHAPES[k] == 'table':
                    cnt = f[_TABLE_COUNT[k]][:idx.size]
                    rmax[k] = max(rmax.get(k, 1), int(cnt.max()) if cnt.size else 1)
        shards.append((p, idx))
    if not shards:
        raise FileNotFoundError('no shard has any committed trajectories')

    order_all = np.concatenate([idx for _, idx in shards])
    if np.unique(order_all).size != order_all.size:
        raise ValueError('a trajectory index appears in more than one shard')
    n = order_all.size
    present = np.sort(order_all)
    comp = dict(compression=compression) if compression else {}

    tmp = out + '.tmp'
    with h5py.File(tmp, 'w') as fo:
        for k in ('config_json', 'nbds', 'nnuc', 'nstates', 'beta', 'delt', 'gamma'):
            if k in attrs:
                fo.attrs[k] = attrs[k]
        fo.attrs['n_traj'] = n
        fo.create_dataset('time', data=time)
        fo.create_dataset('traj_index', data=present)
        dsets, counts = {}, {}
        with h5py.File(shards[0][0], 'r') as f0:
            for k, sh in shapes.items():
                if _FILE_SHAPES[k] == 'table':
                    sh = (rmax[k], sh[1])
                    counts[k] = fo.create_dataset(_TABLE_COUNT[k], shape=(n,), dtype='i8')
                dsets[k] = fo.create_dataset(k, shape=(n,) + sh, dtype='f8',
                                             chunks=(1,) + sh, **comp)
                if 'column_labels' in f0[k].attrs:
                    dsets[k].attrs['column_labels'] = f0[k].attrs['column_labels']

        # shards hold disjoint ascending index ranges, so writing them in order of their smallest
        # index, each sorted internally (a resumed shard can commit out of order), matches present
        row = 0
        for j, (p, idx) in enumerate(sorted(shards, key=lambda t: t[1].min())):
            order = np.argsort(idx, kind='stable')
            with h5py.File(p, 'r') as fi:
                for lo in range(0, idx.size, slab):
                    sel = order[lo:lo + slab]
                    span = slice(int(sel.min()), int(sel.max()) + 1)   # one contiguous read
                    pick = sel - span.start
                    out_rows = slice(row + lo, row + lo + sel.size)
                    for k, d in dsets.items():
                        block = fi[k][span][pick]
                        if _FILE_SHAPES[k] == 'table':
                            r = min(block.shape[1], d.shape[1])
                            padded = np.zeros((sel.size,) + d.shape[1:])
                            padded[:, :r] = block[:, :r]
                            d[out_rows] = padded
                            counts[k][out_rows] = fi[_TABLE_COUNT[k]][span][pick]
                        else:
                            d[out_rows] = block
            row += idx.size
            print(f'[shards] merged {j + 1}/{len(shards)} {os.path.basename(p)}  ({row}/{n})',
                  flush=True)
    os.replace(tmp, out)

    n_cfg = int(cfg['n_traj'])
    if n < n_cfg:
        have = np.zeros(n_cfg, dtype=bool)
        have[present[present < n_cfg]] = True
        miss = np.nonzero(~have)[0]
        print(f'[shards] WARNING: {miss.size}/{n_cfg} trajectories missing: '
              f'{miss[:20].tolist()}{" ..." if miss.size > 20 else ""}')
    print(f'[shards] {out}  ({n} trajectories, {time.size} time rows, observables: {list(dsets)})')
    return out


def _main():
    ap = argparse.ArgumentParser(description='Merge trajectory shards into one consolidate-style data.hdf.')
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='output .hdf (default: data.hdf next to config)')
    ap.add_argument('--slab', type=int, default=1000, help='rows read per batch (default 1000)')
    ap.add_argument('--compression', default='lzf', choices=('lzf', 'gzip', 'none'))
    a = ap.parse_args()
    merge(a.config, out=a.out, slab=a.slab,
          compression=None if a.compression == 'none' else a.compression)


if __name__ == '__main__':
    _main()
