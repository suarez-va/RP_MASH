"""
workflow: reusable machinery for running MANY independent RP_MASH trajectories and averaging them.

The library owns the *mechanics* (parallel scheduling, per-trajectory seeding, per-trajectory output
directories, averaging + error bars, SLURM submission). The *physics* stays with the user, who
provides just two files in their run directory:

    config.py   -  a dict CONFIG with orchestration + model parameters
    system.py   -  def run_trajectory(cfg, seed):  build a mash_rpmd object (seed=seed) and run it

Typical local use (from the run directory, with PYTHONPATH=<RP_MASH root>):
    python -m workflow.runner   --config config.py          # fan out n_traj across cores
    python -m workflow.average  --config config.py --file mapSz.dat --sign   # average + SE

See runner.run_trajectories(), average.average_column(), slurm.write_array_sbatch().
"""

from .scheduler   import run_grid_local
from .runner      import load_config, load_system, seed_for_idx, run_trajectories, traj_dirs
from .average     import collect, average_column, mash_population
from .slurm       import write_array_sbatch
from .consolidate import consolidate, open_run

__all__ = [
    'run_grid_local', 'load_config', 'load_system', 'seed_for_idx', 'run_trajectories',
    'traj_dirs', 'collect', 'average_column', 'mash_population', 'write_array_sbatch',
    'consolidate', 'open_run',
]
