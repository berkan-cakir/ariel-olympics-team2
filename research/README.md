# Experiment template: evolving locomotion controllers

We evolve a controller that makes a robot with a **fixed body** walk 3 m straight in up to four MuJoCo environments. The body is either an ARIEL prebuilt (`gecko`, `spider`, `snake`) or a custom body loaded from a file. Optuna tunes the evolutionary setup, and runs are repeated over several seeds.

Environments: `0` is a simple flat world. `1`, `2` and `3` are realistic terrains.

## Files

| file | contents |
|---|---|
| `fitness.py` | `FitnessConfig`, body loading, environments, simulation, fitness |
| `controller.py` | controller representations, independent of the body |
| `evolution.py` | `EvolutionConfig`, evaluation, simple and island evolution, migration, monitoring, checkpoints |
| `run.py` | CLI, Optuna tuning, seed repetitions, plots |
| `evaluation.py` | opens a checkpoint, re-scores it, records video |

## Work blocks

- **Simulation and fitness:** `fitness.py`
- **Controllers and EA:** `controller.py`, plus `evaluate` and `evolve` in `evolution.py`
- **Islands, monitoring and plots:** `evolve_islands`, `migrate`, `fitness_stats` and `diversity` in `evolution.py`, plus `run.py` and `evaluation.py`

## Running

```sh
cd template
uv run python run.py --bodies gecko spider path/to/body --environments 0 1 2 3 --n-seeds 5
uv run python run.py --islands --environments 1 2 3 --n-trials 30
uv run python run.py --environments 1 2 3 --aggregate min
uv run python evaluation.py path/to/checkpoint --environments 0 1 --video videos
```

`--tuning` has two modes:
- `fast`: each Optuna trial runs one seed. The best params are then run on all `--n-seeds` seeds.
- `robust`: every trial runs all `--n-seeds` seeds.

The search-space arguments (`--controllers`, `--algorithms`, `--pop-size`, `--mutation-rate`, `--crossover-rate`, `--migration-interval`, `--migration-size`) take the values Optuna chooses from.

`--aggregate` sets how per-environment speeds combine into one fitness:
- `mean` (default): the average speed, weighted by `--weights` if given.
- `min`: the speed on the slowest environment, `min(speed on env 1, speed on env 2, ...)`. Rewards robots that walk on every terrain.

`evaluation.py --video` records each selected environment from 3 or more camera angles, all in one video file.
