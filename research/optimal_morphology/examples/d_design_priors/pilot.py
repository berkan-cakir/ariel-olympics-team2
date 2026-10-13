"""
Pilot for the design-priors experiment: measure speed and fix the budget.

Run once, on the machine and with the parallelism the batch will use:

    uv run python examples/d_design_priors/pilot.py --hours 8.5 --jobs 7

Steps:

1. measure simulation throughput with ``--jobs`` parallel processes;
2. report how often the priors need repairs or fallbacks, and how noisy one
   brain-learning run is;
3. learn a brain for every hand-designed seed body;
4. recommend the number of generations that fits ``--hours`` for ``--seeds``
   seeds of all four conditions;
5. run ``--tau-runs`` baseline runs at half that budget and store the median
   of their best scores as ``tau_fit``, the efficiency threshold.

Everything is written to ``<out>/analysis_config.json``, which the analysis
reads. Budget flags must match the ones given to ``run_batch.py``.
"""

# Standard library
import argparse
import json
import math
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

# Local libraries
from dp_core import (
    CONDITIONS,
    EvalResult,
    EvalTask,
    RunConfig,
    RunContext,
    derive_seed,
    initial_body,
    initial_population,
    learn_brain,
    make_offspring,
    random_body,
    run_experiment,
)
from run_batch import SINGLE_THREAD_ENV

from ariel.ec.genotypes.tree.constraints import count_hinges, count_modules
from ariel.ec.genotypes.tree.prebuilt import (
    PREBUILT_TREE_NAMES,
    prebuilt_tree_genome,
)

DEFAULT_OUT = Path.home() / "ariel_runs" / "design_priors"
PILOT_SEED = 900
NOISE_BODIES = 4
NOISE_SEEDS = 3
OFFSPRING_PER_CONDITION = 40


def parse_args() -> tuple[argparse.Namespace, RunConfig]:
    """
    Parse pilot options and the run budget they describe.

    Returns
    -------
    tuple[argparse.Namespace, RunConfig]
        Pilot options and a baseline configuration with the run budget.
    """
    parser = argparse.ArgumentParser(
        description="Measure speed and fix the design-priors budget.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--hours", type=float, default=8.5)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bodies-per-job", type=int, default=3)
    parser.add_argument("--tau-runs", type=int, default=3)
    defaults = RunConfig(condition="baseline", seed=PILOT_SEED, out=Path())
    for name in (
        "pop",
        "learn_budget",
        "learn_pop",
        "duration",
        "eval_delay",
        "reeval_top",
        "reeval_seeds",
    ):
        value = getattr(defaults, name)
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(value),
            default=value,
        )
    args = parser.parse_args()
    cfg = replace(
        defaults,
        out=args.out / "pilot" / time.strftime("%Y%m%d-%H%M%S"),
        pop=args.pop,
        learn_budget=args.learn_budget,
        learn_pop=args.learn_pop,
        duration=args.duration,
        eval_delay=args.eval_delay,
        reeval_top=args.reeval_top,
        reeval_seeds=args.reeval_seeds,
    )
    return args, cfg


def learn_many(
    executor: ProcessPoolExecutor,
    genomes: list[dict[str, Any]],
    cfg: RunConfig,
    salt: int,
) -> tuple[list[EvalResult], float]:
    """
    Learn a brain for each genome in parallel.

    Parameters
    ----------
    executor
        The worker pool.
    genomes
        Bodies to learn brains for.
    cfg
        Settings (budget and simulation).
    salt
        Distinguishes the learner seeds of different pilot steps.

    Returns
    -------
    tuple[list[EvalResult], float]
        The results and the wall-clock seconds they took.
    """
    tasks = [
        EvalTask(
            genome=genome,
            learn_seed=derive_seed(PILOT_SEED, salt, index),
            settings=cfg.eval_settings(),
        )
        for index, genome in enumerate(genomes)
    ]
    started = time.perf_counter()
    results = list(executor.map(learn_brain, tasks))
    return results, time.perf_counter() - started


def prior_statistics(cfg: RunConfig) -> dict[str, Any]:
    """
    Measure how the priors behave when making bodies and offspring.

    Parameters
    ----------
    cfg
        Settings (limits and priors).

    Returns
    -------
    dict[str, Any]
        Per condition: initial-body size and hinges, time per initial body,
        and the attempts and fallback rate of offspring.
    """
    stats: dict[str, Any] = {}
    for condition in CONDITIONS:
        run_cfg = replace(cfg, condition=condition)
        ctx = RunContext(cfg=run_cfg, rng=random.Random(PILOT_SEED))
        random.seed(PILOT_SEED)
        started = time.perf_counter()
        bodies = [initial_body(run_cfg, ctx.rng) for _ in range(20)]
        per_body = (time.perf_counter() - started) / len(bodies)
        parents = list(initial_population(ctx))
        offspring = [
            make_offspring(parents, ctx) for _ in range(OFFSPRING_PER_CONDITION)
        ]
        stats[condition] = {
            "initial_modules_mean": statistics.mean(map(count_modules, bodies)),
            "initial_hinges_mean": statistics.mean(map(count_hinges, bodies)),
            "seconds_per_initial_body": round(per_body, 4),
            "offspring_attempts_mean": statistics.mean(
                float(str(child.tags["attempts"])) for child in offspring
            ),
            "offspring_fallback_rate": statistics.mean(
                float(bool(child.tags["fallback"])) for child in offspring
            ),
        }
    rng = random.Random(PILOT_SEED)
    random_bodies = [random_body(cfg, rng) for _ in range(100)]
    stats["joint_budget"]["share_of_random_bodies_capped"] = statistics.mean(
        float(count_hinges(body) > cfg.max_hinges) for body in random_bodies
    )
    return stats


def recommend_generations(
    args: argparse.Namespace,
    cfg: RunConfig,
    throughput: float,
) -> int:
    """
    Largest number of generations that fits the time budget.

    Parameters
    ----------
    args
        Pilot options (hours, seeds).
    cfg
        Run budget.
    throughput
        Simulations per second across all parallel runs.

    Returns
    -------
    int
        Recommended ``--gens`` (at least 1).
    """
    sims_per_body = cfg.learn_budget * cfg.learn_pop
    total_sims = args.hours * 3600 * throughput
    bodies_per_run = total_sims / (len(CONDITIONS) * args.seeds * sims_per_body)
    reeval_bodies = cfg.reeval_top * cfg.reeval_seeds
    return max(
        1,
        math.floor((bodies_per_run - cfg.pop - reeval_bodies) / cfg.pop),
    )


def measure_learning(
    args: argparse.Namespace,
    cfg: RunConfig,
) -> dict[str, Any]:
    """
    Measure throughput, learning noise and the seed bodies' scores.

    Parameters
    ----------
    args
        Pilot options (parallelism, bodies per job).
    cfg
        Run budget.

    Returns
    -------
    dict[str, Any]
        Throughput, per-body time, scores of random bodies, learning noise
        and seed scores.
    """
    rng = random.Random(PILOT_SEED)
    random.seed(PILOT_SEED)
    n_speed = args.jobs * args.bodies_per_job
    speed = [random_body(cfg, rng).to_dict() for _ in range(n_speed)]
    noise = [random_body(cfg, rng).to_dict() for _ in range(NOISE_BODIES)]
    repeated = [body for body in noise for _ in range(NOISE_SEEDS)]
    seeds = [prebuilt_tree_genome(n).to_dict() for n in PREBUILT_TREE_NAMES]

    with ProcessPoolExecutor(
        max_workers=args.jobs,
        mp_context=get_context("spawn"),
    ) as executor:
        speed_results, seconds = learn_many(executor, speed, cfg, salt=1)
        noise_results, _ = learn_many(executor, repeated, cfg, salt=2)
        seed_results, _ = learn_many(executor, seeds, cfg, salt=3)

    scores = [result.score or 0.0 for result in noise_results]
    per_body = [
        scores[i : i + NOISE_SEEDS] for i in range(0, len(scores), NOISE_SEEDS)
    ]
    return {
        "throughput_sims_per_s": sum(r.n_sims for r in speed_results) / seconds,
        "seconds_per_body_median": statistics.median(
            result.seconds for result in speed_results
        ),
        "random_body_scores": [result.score for result in speed_results],
        "learning_noise": {
            "scores_per_body": per_body,
            "within_body_sd": statistics.mean(map(statistics.stdev, per_body)),
            "between_body_sd": statistics.stdev(map(statistics.mean, per_body)),
        },
        "seed_scores": dict(
            zip(
                PREBUILT_TREE_NAMES,
                [result.score for result in seed_results],
                strict=True,
            ),
        ),
    }


def measure_tau(
    args: argparse.Namespace,
    cfg: RunConfig,
    gens: int,
) -> dict[str, Any]:
    """
    Set tau_fit to the median best score of half-budget baseline runs.

    Parameters
    ----------
    args
        Pilot options (number of runs, parallelism).
    cfg
        Run budget.
    gens
        The recommended number of generations; the runs use half.

    Returns
    -------
    dict[str, Any]
        ``tau_fit`` and the best score of each run.
    """
    configs = [
        replace(cfg, seed=PILOT_SEED + k, gens=max(1, gens // 2))
        for k in range(1, args.tau_runs + 1)
    ]
    with ProcessPoolExecutor(
        max_workers=min(args.jobs, args.tau_runs),
        mp_context=get_context("spawn"),
    ) as executor:
        metas = list(executor.map(run_experiment, configs))
    bests = [float(meta["best_score"]) for meta in metas]
    return {"tau_fit": statistics.median(bests), "tau_fit_runs": bests}


def main() -> None:
    """Run the pilot and write ``analysis_config.json``."""
    args, cfg = parse_args()
    os.environ.update(SINGLE_THREAD_ENV)
    report: dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jobs": args.jobs,
        "hours": args.hours,
        "seeds": args.seeds,
        "budget": {
            key: value
            for key, value in asdict(cfg).items()
            if key not in {"condition", "seed", "out"}
        },
    }
    report |= measure_learning(args, cfg)
    print(  # noqa: T201
        f"1-3. {report['throughput_sims_per_s']:.1f} simulations/s; "
        f"noise {report['learning_noise']}; seeds {report['seed_scores']}",
        flush=True,
    )
    report["priors"] = prior_statistics(cfg)
    print(f"2b. priors {report['priors']}", flush=True)  # noqa: T201

    gens = recommend_generations(args, cfg, report["throughput_sims_per_s"])
    report["recommended_gens"] = gens
    print(f"4. recommended --gens {gens}", flush=True)  # noqa: T201

    if args.tau_runs > 0:
        report |= measure_tau(args, cfg, gens)
        print(f"5. tau_fit {report['tau_fit']:.4f}", flush=True)  # noqa: T201

    config_path = args.out / "analysis_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {config_path}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
