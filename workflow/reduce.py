"""
reduce: parallel map-reduce over a sharded run, for analysis scripts.

The shards (workflow.shards) ARE the dataset: an analysis reads them directly, with no merge into a
monolithic data.hdf first. The standard pattern on a cluster, run as ONE multi-core SLURM job
rather than on a login node:

    map    : each worker process opens whole shard files read-only and returns small partial sums
    reduce : the parent adds the partials up and forms means / block error bars / plots

so the run time is set by how fast the shards can be read in parallel, and no copy of the data is
ever written. Pieces an analysis script needs:

    plan  = shard_plan(path)                 # [(file, traj_index, rank_offset)] in trajectory order
    n     = sum(ti.size for _, ti, _ in plan)
    block = block_ids(n, nblocks)            # contiguous error-bar blocks over the trajectory rank
    for result in map_files(fn, tasks, ncores):   # fn(*task) in parallel, results in task order
        ...

fn must be a module-level function (the workers are spawned and import it by name), and a script
that calls map_files with ncores > 1 must keep its entry point under `if __name__ == '__main__':`.

`path` may also be a single consolidated data.hdf (one task; nothing to parallelize over).

    #SBATCH --cpus-per-task=32
    python my_analysis.py --file shards --ncores $SLURM_CPUS_PER_TASK
"""

import os
import glob
import time

import numpy as np
import h5py


def ncores_source():
    """
    (worker processes to use, where that number came from), in this order:
      1. $SLURM_CPUS_PER_TASK              a job submitted with --cpus-per-task / -c
      2. the CPUs this process may run on  any other SLURM allocation ($SLURM_JOB_ID set): -n N,
                                           --exclusive, an interactive desktop, ... SLURM confines the
                                           job to exactly its allocated CPUs, so the affinity mask
                                           counts them
      3. 1                                 outside SLURM (e.g. a login node, which must not be flooded)
    An explicit ncores / --ncores always overrides this.
    """
    try:
        n = int(os.environ['SLURM_CPUS_PER_TASK'])
        if n >= 1:
            return n, f'$SLURM_CPUS_PER_TASK={n}'
    except (KeyError, ValueError):
        pass
    if os.environ.get('SLURM_JOB_ID'):
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:                       # no sched_getaffinity (not Linux)
            n = os.cpu_count() or 1
        return max(1, n), f'SLURM job {os.environ["SLURM_JOB_ID"]}: {n} allocated CPUs'
    return 1, 'not inside a SLURM job'


def default_ncores():
    """Worker processes to use when none are given; see ncores_source()."""
    return ncores_source()[0]


def open_readonly(path):
    """
    h5py.File(path, 'r') WITHOUT HDF5 file locking. A locked read open makes a concurrent writer's
    open fail ("unable to lock file"), so an analysis run while production is still appending to a
    shard would make production trajectories fail -- and vice versa. Readers here never take the
    lock; a shard caught mid-append may then fail to read, which the callers treat as "still being
    written" and skip (it is never cached).
    """
    try:
        return h5py.File(path, 'r', locking=False)
    except TypeError:                                     # h5py < 3.5: no locking keyword
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        return h5py.File(path, 'r')


def shard_plan(path):
    """
    [(file, traj_index, rank_offset)] for a shard directory (every shard_*.hdf with committed rows)
    or a single data.hdf, sorted by smallest trajectory index. Shards hold disjoint ascending index
    ranges, so rank_offset (the number of trajectories in all earlier files) places each file's rows
    in the global trajectory ranking. Reads only /traj_index.
    """
    if os.path.isdir(path):
        paths = glob.glob(os.path.join(path, 'shard_*.hdf'))
        if not paths:
            raise FileNotFoundError(f'no shard_*.hdf in {path}')
    else:
        paths = [path]
    out = []
    for p in paths:
        try:
            with open_readonly(p) as f:
                ti = f['traj_index'][:] if 'traj_index' in f else np.arange(int(f.attrs['n_traj']))
        except (OSError, KeyError) as e:                  # caught mid-append by a running writer
            print(f'[reduce] WARNING: {p} unreadable right now ({type(e).__name__}: {e}); '
                  f'skipped -- rerun once it is complete', flush=True)
            continue
        if ti.size:
            out.append((p, ti))
    if not out:
        raise FileNotFoundError(f'no committed trajectories in {path}')
    out.sort(key=lambda s: s[1].min())
    offsets = np.cumsum([0] + [ti.size for _, ti in out[:-1]])
    return [(p, ti, int(o)) for (p, ti), o in zip(out, offsets)]


def block_ids(n_traj, nblocks):
    """Block of each global trajectory rank: np.array_split into nblocks contiguous blocks
    (nblocks=None -> one trajectory per block). Returns (block_of (n_traj,), n_blocks)."""
    nb = n_traj if nblocks is None else max(1, min(int(nblocks), n_traj))
    block_of = np.concatenate([np.full(len(b), i, dtype=int)
                               for i, b in enumerate(np.array_split(np.arange(n_traj), nb))])
    return block_of, nb


def _init_worker():
    # read-only workers: HDF5 file locking only gets in the way (and is often broken on
    # Lustre/GPFS); respect an explicit setting from the user
    os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')


def map_files(fn, tasks, ncores=None):
    """
    Yield fn(*task) for every task, IN TASK ORDER, computed by `ncores` spawned worker processes
    (default: default_ncores()). ncores=1, or a single task, runs in this process. A worker that
    raises re-raises here.
    """
    tasks = list(tasks)
    ncores = default_ncores() if ncores is None else max(1, int(ncores))
    ncores = min(ncores, len(tasks))
    if ncores <= 1:
        for t in tasks:
            yield fn(*t)
        return

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(ncores, mp_context=mp.get_context('spawn'),
                             initializer=_init_worker) as ex:
        # at most 2*ncores results waiting at once, so a fast map never piles partials up in memory
        futs, nxt = [], 0
        for i in range(len(tasks)):
            while nxt < len(tasks) and nxt < i + 2 * ncores:
                futs.append(ex.submit(fn, *tasks[nxt]))
                nxt += 1
            yield futs[i].result()
            futs[i] = None


# ------------------------------------------------------------------ cached per-unit partials
#
# For a run that keeps growing (or is analysed more than once), each UNIT of work -- a shard, or a
# slice of at most unit_rows rows of a bigger file -- is reduced to a small partial file once and
# reused: a later analysis computes only the units that are new or changed and reads the rest back.
#
#     units = plan_units(path, unit_rows)                       # [Unit] in trajectory order
#     parts, stats = map_cached(fn, units, cache_dir, version)  # fn(file, lo, hi) -> {name: array}
#
# A partial is cached only for a COMPLETE unit: a data.hdf, or a shard whose committed rows cover its
# whole index range. A shard still being written is analysed afresh every time (and skipped with a
# warning if it cannot be read mid-append), so its partial can never go stale. The cache key holds
# the source file's size, mtime, committed rows, the row range and the analysis `version` string:
# bump `version` whenever the analysis definition changes, and everything is recomputed.

from collections import namedtuple

Unit = namedtuple('Unit', 'file lo hi complete committed')


def plan_units(path, unit_rows=1000):
    """
    Split `path` (a shard directory or a data.hdf) into units of at most unit_rows rows of one
    file, in trajectory order (files sorted by smallest index, rows in file order).
    """
    out = []
    for p, ti, _ in shard_plan(path):
        try:
            with open_readonly(p) as f:
                if 'idx_start' in f.attrs:
                    complete = ti.size >= int(f.attrs['idx_stop']) - int(f.attrs['idx_start'])
                else:
                    complete = True                      # a consolidated data.hdf is final
        except (OSError, KeyError):
            complete = False                             # caught mid-append: being written
        step = max(1, int(unit_rows))
        for lo in range(0, ti.size, step):
            out.append(Unit(p, lo, min(lo + step, ti.size), complete, int(ti.size)))
    return out


def _unit_key(u, version):
    st = os.stat(u.file)
    return f'{version}|{st.st_size}|{st.st_mtime_ns}|{u.committed}|{u.lo}|{u.hi}'


def _cache_path(cache_dir, u):
    stem = os.path.splitext(os.path.basename(u.file))[0]
    return os.path.join(cache_dir, f'{stem}_{u.lo}_{u.hi}.npz')


def _load_partial(path, key):
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z['_key']) != key:
                return None
            return {k: z[k] for k in z.files if k != '_key'}
    except (OSError, KeyError, ValueError):
        return None


def _run_unit(fn, u, cache_file, key):
    """Worker: compute one unit, and cache it if it is complete. Returns (partial or None, note)."""
    try:
        part = fn(u.file, u.lo, u.hi)
    except Exception as e:
        if u.complete:
            raise
        return None, f'{os.path.basename(u.file)} rows {u.lo}-{u.hi} (still being written) ' \
                     f'skipped: {type(e).__name__}: {e}'
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        tmp = f'{cache_file}.{os.getpid()}.tmp.npz'
        np.savez(tmp, _key=np.array(key), **part)
        os.replace(tmp, cache_file)                      # atomic: readers never see half a file
    return part, None


def map_cached(fn, units, cache_dir, version, ncores=None, refresh=False, use_cache=True,
               tag=None, every=60.0):
    """
    Partials of every unit, in unit order: read from `cache_dir` where a valid one exists, else
    fn(file, lo, hi) (-> {name: array}) computed by `ncores` worker processes (default_ncores()).
    Returns (list of partials -- None for a skipped in-progress unit, stats dict). refresh=True
    recomputes (and rewrites) every unit; use_cache=False neither reads nor writes the cache.

    With a `tag` (e.g. '[p2]') it reports as it goes, flushed so it reaches a SLURM .out file at
    once: one line before any unit is computed (units cached / to compute / workers), then a
    progress line (units, trajectories/s, elapsed, ETA) at most every `every` seconds and at the end.
    """
    parts = [None] * len(units)
    stats = {'cached': 0, 'computed': 0, 'in_progress': 0, 'skipped': 0, 'notes': []}
    todo = []
    for i, u in enumerate(units):
        key = _unit_key(u, version)
        if use_cache and u.complete and not refresh:
            part = _load_partial(_cache_path(cache_dir, u), key)
            if part is not None:
                parts[i] = part
                stats['cached'] += 1
                continue
        cache_file = _cache_path(cache_dir, u) if (use_cache and u.complete) else None
        todo.append((i, (fn, u, cache_file, key)))

    ncores = default_ncores() if ncores is None else max(1, int(ncores))
    if tag:
        print(f'{tag} {len(units)} unit(s): {stats["cached"]} from cache, {len(todo)} to compute on '
              f'{min(ncores, len(todo)) if todo else 0} worker(s)', flush=True)
    n_todo = sum(units[i].hi - units[i].lo for i, _ in todo)
    t0 = last = time.monotonic()
    done_units = done_traj = 0
    for (i, task), (part, note) in zip(todo, map_files(_run_unit, [t for _, t in todo], ncores)):
        parts[i] = part
        stats['computed' if units[i].complete else 'in_progress'] += 1
        if note:
            stats['skipped'] += 1
            stats['notes'].append(note)
        done_units += 1
        done_traj += units[i].hi - units[i].lo
        now = time.monotonic()
        if tag and (now - last >= every or done_units == len(todo)):
            last = now
            el = now - t0
            rate = done_traj / el if el > 0 else float('inf')
            eta = (n_todo - done_traj) / rate if rate > 0 else float('nan')
            print(f'{tag} {done_units}/{len(todo)} units ({100.0 * done_units / len(todo):.0f}%), '
                  f'{rate:.3g} traj/s, elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min', flush=True)
    return parts, stats
