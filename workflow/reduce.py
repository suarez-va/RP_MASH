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
        with h5py.File(p, 'r') as f:
            ti = f['traj_index'][:] if 'traj_index' in f else np.arange(int(f.attrs['n_traj']))
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
