"""
Run one condition of the design-priors experiment with one seed.

Example
-------
    uv run python examples/d_design_priors/run_design_priors.py \
        --condition symmetric --seed 1

Outputs go to ``<out>/<condition>/seed_<NN>.db`` plus ``_meta.json`` and
``_reeval.json``. An existing database is never overwritten. See
``dp_core.py`` for what each condition does and ``run_batch.py`` for the full
experiment.
"""

# Standard library
import argparse
from pathlib import Path

# Local libraries
from dp_core import CONDITIONS, RunConfig, run_experiment

DEFAULT_OUT = Path.home() / "ariel_runs" / "design_priors"
DESCRIPTION = "Run one condition of the design-priors experiment."


def parse_args() -> RunConfig:
    """
    Parse the command line into a run configuration.

    Returns
    -------
    RunConfig
        The settings of this run.
    """
    defaults = RunConfig(condition="baseline", seed=0, out=DEFAULT_OUT)
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    flags = {
        "pop": int,
        "gens": int,
        "learn_budget": int,
        "learn_pop": int,
        "duration": float,
        "eval_delay": float,
        "z_penalty_weight": float,
        "seed_fraction": float,
        "max_hinges": int,
        "max_modules": int,
        "max_depth": int,
        "init_size_min": int,
        "spawn_yaw": float,
        "retries": int,
        "reeval_top": int,
        "reeval_seeds": int,
        "eval_workers": int,
    }
    for name, kind in flags.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=kind,
            default=getattr(defaults, name),
        )
    args = parser.parse_args()
    return RunConfig(**vars(args))


def main() -> None:
    """Run the experiment and print where the results are."""
    cfg = parse_args()
    meta = run_experiment(cfg)
    print(  # noqa: T201
        f"Done: {cfg.db_path} ({meta['total_sims']} simulations, "
        f"{meta['wall_seconds']} s, re-evaluated best {meta['reeval_best']})",
    )


if __name__ == "__main__":
    main()
