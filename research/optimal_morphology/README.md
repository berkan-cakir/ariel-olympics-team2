# Team 2 experiments for ARIEL

Experiment code from Team 2's evolutionary robotics research (VU Amsterdam, Evolutionary Computing, 2026):

- `examples/e_symmetric_coevolution/` — symmetric body evolution with learned sine-wave gaits on a flat 3 m track: the seven runs (baseline, pruning, servo limits, growth, improved servo, double servo) and the long final learning. See its README.
- `examples/d_design_priors/` — the earlier design-priors experiment. See its README.
- `src/ariel/…` — four new library modules both experiments import: tree constraints, prebuilt bodies, symmetric genomes (`ec/genotypes/tree/`) and symmetry helpers (`utils/symmetry.py`).
- `tests/unit/…` — unit tests for those modules.

## Setup

The folders mirror the layout of the ARIEL repository. The code was written against ARIEL commit `fdf454c` (main, 5 August 2026).

1. Clone ARIEL and check out that commit:

   ```bash
   git clone https://github.com/ci-group/ariel.git
   cd ariel
   git checkout fdf454c
   ```

2. Copy the contents of this folder over the repository root, so `examples/`, `src/` and `tests/` merge with ARIEL's. No existing ARIEL file is replaced; all files here are new.

3. Install and check:

   ```bash
   uv sync
   uv run pytest tests/unit/test_ec/test_genotypes/test_tree tests/unit/test_utils/test_symmetry.py
   ```

   The design-priors analysis also uses SciPy: `uv add --group analysis "scipy>=1.16"`.

## Running

Commands for every run are in each experiment's README. Results go to `~/ariel_runs/` by default (outside any synced folder: cloud syncing can corrupt the SQLite run databases). A 100-generation symmetric co-evolution run takes several hours on 7 workers.

Run results, videos and databases are not included; they are about 2.3 GB.
