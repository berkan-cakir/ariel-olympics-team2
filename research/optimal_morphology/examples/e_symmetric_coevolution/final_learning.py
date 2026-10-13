"""
Learn the best gait for every robot of a finished run.

Example
-------
    uv run python examples/e_symmetric_coevolution/final_learning.py \
        ~/ariel_runs/symmetric_coevolution/symmetric_pruned_realservo_grow_double/seed_01.db

During evolution every body learns its gait with 192 races. That is enough to
rank bodies, but not to find out what each body can do: the best bodies of a
run, re-learned with fresh seeds, end up about 20% slower on average. Once
the bodies are final, this script gives every survivor of the last
generation a long learning phase:

* every gait family (alternating, synchronous and free, see the README) gets
  ``--restarts`` independent CMA-ES runs, each from the middle gait with its
  own seed;
* a CMA-ES run goes on until its best speed has not improved by
  ``--min-gain`` (relative) in ``--patience`` generations, CMA-ES itself
  stops, or ``--max-gens`` generations are done;
* the body keeps the fastest gait found, or its gait from the run if that is
  faster.

No method guarantees the best gait of a body. But when independent runs end
at the same speed, a much faster gait of this controller is unlikely. A body
counts as converged when at least two runs came within ``--agree`` of its
final speed.

Outputs go next to the database:

* ``seed_NN_final_learning.json``: per body, its speed and gait from the run,
  its final speed and gait, and every CMA-ES run. It is rewritten whenever a
  CMA-ES run finishes, so it also shows the progress;
* ``seed_NN_videos/final_learning_all.mp4``: all bodies racing with their
  final gaits, fastest first.

``view_best.py seed_NN_final_learning.json`` shows or films the fastest body
with its final gait (``--uid`` picks another).
"""

# Standard library
import argparse
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

# Third-party libraries
import cma
import numpy as np

# Local libraries
from run_symmetric_coevolution import SINGLE_THREAD_ENV
from sc_core import (
    FAMILIES,
    EvalSettings,
    Gait,
    RaceResult,
    build_world,
    decode_gait,
    derive_seed,
    gait_layout,
    load_individuals,
    mirror_partners,
    replay_gait,
    run_race,
    settle,
    write_json,
)
from sc_gallery import race_video, runner
from view_best import run_settings

from ariel.parameters.ariel_types import FloatArray

FINAL_SEED_KEY = 3  # keeps these seeds apart from the run's learning seeds
# Free gaits have the most parameters, so their runs are started first.
FAMILY_ORDER = ("free", "alternating", "synchronous")


@dataclass(frozen=True)
class Plan:
    """
    How long and how often every body learns.

    Parameters
    ----------
    families
        Gait families to learn.
    restarts
        Independent CMA-ES runs per family.
    popsize
        Races per CMA-ES generation.
    sigma
        Initial CMA-ES step size.
    max_gens
        Most CMA-ES generations of one run.
    patience
        A run stops when its best speed has not improved by ``min_gain``
        in this many generations.
    min_gain
        Smallest relative improvement that counts.
    agree
        Relative distance to the final speed within which a run counts as
        having found it.
    """

    families: tuple[str, ...]
    restarts: int
    popsize: int
    sigma: float
    max_gens: int
    patience: int
    min_gain: float
    agree: float


@dataclass(frozen=True)
class Task:
    """
    One CMA-ES run: a body, a gait family and a seed.

    Parameters
    ----------
    uid
        The body's uid.
    genome
        Its tree genome as a dict.
    family
        The gait family.
    restart
        Number of the run within the family, from 0.
    seed
        Seed of the CMA-ES sampler.
    settings
        The run's settings.
    plan
        Budget and stopping rule.
    """

    uid: int
    genome: dict[str, Any]
    family: str
    restart: int
    seed: int
    settings: EvalSettings
    plan: Plan


def learn_run(task: Task) -> dict[str, Any]:
    """
    Run CMA-ES on one body and gait family until it stops improving.

    Parameters
    ----------
    task
        The body, family, seed and plan.

    Returns
    -------
    dict[str, Any]
        The best gait found, its dash, the races used, the best speed after
        every generation and why the run stopped. A run that fails gets
        speed -1 and the error, so that the other runs go on.
    """
    started = time.perf_counter()
    try:
        return _learn_run(task, started)
    except Exception as error:  # noqa: BLE001 -- one failure must not stop hours of runs
        return {
            "uid": task.uid,
            "family": task.family,
            "restart": task.restart,
            "seed": task.seed,
            "speed": -1.0,
            "finish_time": None,
            "furthest": 0.0,
            "drift": 0.0,
            "turn": 0.0,
            "gait": None,
            "n_params": 0,
            "races": 0,
            "generations": 0,
            "stopped": f"failed: {error!r}",
            "curve": [],
            "seconds": round(time.perf_counter() - started, 1),
        }


def _learn_run(task: Task, started: float) -> dict[str, Any]:
    plan = task.plan
    dash, learner, n_params = _course(task)
    best: tuple[float, Any, Any] = (-math.inf, None, None)
    curve: list[float] = []
    stopped = ""
    while not stopped:
        vectors = learner.ask()
        outcomes = [dash(np.asarray(vector)) for vector in vectors]
        learner.tell(vectors, [-race.speed for race, _ in outcomes])
        for race, gait in outcomes:
            if race.speed > best[0]:
                best = (race.speed, race, gait)
        curve.append(best[0])
        stopped = _stop_reason(curve, learner, plan)
    _, race, gait = best
    return {
        "uid": task.uid,
        "family": task.family,
        "restart": task.restart,
        "seed": task.seed,
        "speed": race.speed,
        "finish_time": race.finish_time,
        "furthest": race.furthest,
        "drift": race.drift,
        "turn": race.turn,
        "gait": gait.to_dict(),
        "n_params": n_params,
        "races": len(curve) * plan.popsize,
        "generations": len(curve),
        "stopped": stopped,
        "curve": [round(speed, 5) for speed in curve],
        "seconds": round(time.perf_counter() - started, 1),
    }


def _course(
    task: Task,
) -> tuple[
    Callable[[FloatArray], tuple[RaceResult, Gait]],
    cma.CMAEvolutionStrategy,
    int,
]:
    # The body after settling, a dash from there, and a fresh learner.
    settings = task.settings
    _, model, data = build_world(task.genome, settings.spawn_yaw, settings)
    layout = gait_layout(mirror_partners(model), task.family)
    warm_state = settle(model, data, settings.settle)

    def dash(vector: FloatArray) -> tuple[RaceResult, Gait]:
        gait = decode_gait(vector, layout, settings.max_hinge_frequency)
        return run_race(model, data, warm_state, gait, settings), gait

    rng = np.random.default_rng(task.seed)
    learner = cma.CMAEvolutionStrategy(
        np.zeros(layout.n_params),
        task.plan.sigma,
        {
            "popsize": task.plan.popsize,
            # Sample from our own generator; nan leaves numpy's global alone.
            "randn": lambda lam, dim: rng.standard_normal((lam, dim)),
            "seed": math.nan,
            "verbose": -9,
        },
    )
    return dash, learner, layout.n_params


def _stop_reason(
    curve: list[float],
    learner: cma.CMAEvolutionStrategy,
    plan: Plan,
) -> str:
    # Why a CMA-ES run stops after this generation; "" if it goes on.
    if len(curve) > plan.patience:
        gain = curve[-1] - curve[-1 - plan.patience]
        if gain <= plan.min_gain * abs(curve[-1]):
            return f"no gain in {plan.patience} generations"
    if learner.stop():
        return "CMA-ES stopped (" + ", ".join(map(str, learner.stop())) + ")"
    if len(curve) >= plan.max_gens:
        return f"{plan.max_gens} generations"
    return ""


def final_population(db: Path, top: int) -> list[dict[str, Any]]:
    """
    Return the survivors of a run's last generation, fastest first.

    Parameters
    ----------
    db
        The run database.
    top
        Keep only this many; 0 keeps all.

    Returns
    -------
    list[dict[str, Any]]
        Rows of :func:`sc_core.load_individuals`.
    """
    survivors = sorted(
        (row for row in load_individuals(db) if row["alive"]),
        key=lambda row: (row["tags"].get("speed") or 0.0, row["tags"]["uid"]),
        reverse=True,
    )
    return survivors[:top] if top > 0 else survivors


def body_summary(
    rank: int,
    row: dict[str, Any],
    runs: list[dict[str, Any]],
    plan: Plan,
) -> dict[str, Any]:
    """
    Combine a body's CMA-ES runs into its final gait.

    Parameters
    ----------
    rank
        Its place in the run's last generation, from 1.
    row
        The body as stored by the run.
    runs
        Its finished CMA-ES runs.
    plan
        The learning plan (``agree``).

    Returns
    -------
    dict[str, Any]
        The body, its gait from the run, its CMA-ES runs (fastest first) and,
        once every run has finished, its final gait.
    """
    tags = row["tags"]
    from_run = {
        "speed": tags["speed"],
        "finish_time": tags["finish_time"],
        "family": tags["family"],
        "gait": tags["gait"],
    }
    summary: dict[str, Any] = {
        "uid": tags["uid"],
        "run_rank": rank,
        "n_modules": tags["n_modules"],
        "n_hinges": tags["n_hinges"],
        "n_double": tags.get("n_double", 0),
        "genome": row["genotype"]["morph"],
        "run": from_run,
        "runs": sorted(runs, key=lambda run: -run["speed"]),
    }
    if len(runs) < len(plan.families) * plan.restarts:
        return summary
    best = summary["runs"][0]
    final = (
        {key: best[key] for key in ("speed", "finish_time", "family", "gait")}
        | {"found_by": f"{best['family']} run {best['restart'] + 1}"}
        if best["speed"] >= from_run["speed"]
        else from_run | {"found_by": "the run (no CMA-ES run beat it)"}
    )
    agreeing = sum(
        run["speed"] >= (1.0 - plan.agree) * final["speed"] for run in runs
    )
    summary.update({
        "final": final,
        "agreeing_runs": agreeing,
        "converged": agreeing >= 2,
        "family_best": {
            family: max(
                (run["speed"] for run in runs if run["family"] == family),
                default=None,
            )
            for family in plan.families
        },
    })
    return summary


def parse_args() -> tuple[Path, Plan, int, int, bool]:
    """
    Parse the command line.

    Returns
    -------
    tuple[Path, Plan, int, int, bool]
        The run database, the learning plan, how many bodies, the number of
        workers, and whether to film the race.
    """
    parser = argparse.ArgumentParser(
        description="Learn the best gait for every survivor of a finished run.",
    )
    parser.add_argument("db", type=Path, help="the run's seed_NN.db")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILY_ORDER),
        help="gait families to learn (default: all three)",
    )
    for flag, kind, default, text in (
        ("--restarts", int, 4, "CMA-ES runs per gait family"),
        ("--popsize", int, 16, "races per CMA-ES generation"),
        ("--sigma", float, 0.5, "initial CMA-ES step size"),
    ):
        parser.add_argument(flag, type=kind, default=default, help=text)
    parser.add_argument(
        "--max-gens",
        type=int,
        default=200,
        help="most generations of one run",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="stop a run after this many generations without a gain",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.005,
        help="smallest relative gain that counts (default 0.5%%)",
    )
    parser.add_argument(
        "--agree",
        type=float,
        default=0.02,
        help="runs within this relative distance of the best found it too",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="learn only the fastest N survivors (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="worker processes (default: all cores but one)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="do not film the final race",
    )
    args = parser.parse_args()
    if min(args.restarts, args.max_gens, args.patience, args.workers) < 1:
        parser.error(
            "--restarts, --max-gens, --patience and --workers must be >= 1",
        )
    if args.popsize < 2:
        parser.error("--popsize must be at least 2")
    if not (0.0 <= args.min_gain < 1.0 and 0.0 <= args.agree < 1.0):
        parser.error("--min-gain and --agree must be in [0, 1)")
    order = [family for family in FAMILY_ORDER if family in args.families]
    plan = Plan(
        families=tuple(order),
        restarts=args.restarts,
        popsize=args.popsize,
        sigma=args.sigma,
        max_gens=args.max_gens,
        patience=args.patience,
        min_gain=args.min_gain,
        agree=args.agree,
    )
    return args.db, plan, args.top, args.workers, not args.no_video


def film_final(
    db: Path,
    bodies: list[dict[str, Any]],
    settings: EvalSettings,
    seed: int,
) -> Path:
    """
    Film all bodies racing with their final gaits, fastest first.

    Parameters
    ----------
    db
        The run database; the video goes to its videos folder.
    bodies
        Body summaries with a final gait.
    settings
        The run's settings.
    seed
        The run's seed, for the title.

    Returns
    -------
    Path
        The video written.
    """
    ranked = sorted(bodies, key=lambda body: -body["final"]["speed"])
    robots = [
        runner(
            rank,
            body["genome"],
            {
                "uid": body["uid"],
                "gait": body["final"]["gait"],
                "finish_time": body["final"]["finish_time"],
            },
        )
        for rank, body in enumerate(ranked, start=1)
    ]
    folder = db.with_name(f"{db.stem}_videos")
    folder.mkdir(exist_ok=True)
    return race_video(
        robots,
        settings,
        folder / "final_learning_all.mp4",
        f"Seed {seed}: the last generation after long learning",
    )


def make_tasks(
    rows: list[dict[str, Any]],
    plan: Plan,
    settings: EvalSettings,
    seed: int,
) -> list[Task]:
    """
    List every CMA-ES run: each body, gait family and restart.

    Parameters
    ----------
    rows
        The bodies.
    plan
        Families and restarts.
    settings
        The run's settings.
    seed
        The run's seed; every run's seed derives from it.

    Returns
    -------
    list[Task]
        Family by family, in :data:`FAMILY_ORDER`.
    """
    return [
        Task(
            uid=int(row["tags"]["uid"]),
            genome=row["genotype"]["morph"],
            family=family,
            restart=restart,
            seed=derive_seed(
                seed,
                int(row["tags"]["uid"]),
                FAMILIES.index(family),
                restart,
                FINAL_SEED_KEY,
            ),
            settings=settings,
            plan=plan,
        )
        for family in plan.families
        for restart in range(plan.restarts)
        for row in rows
    ]


def learn_all(
    tasks: list[Task],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: Path,
    workers: int,
) -> None:
    """
    Run every CMA-ES run on the workers, saving the summary as they finish.

    Parameters
    ----------
    tasks
        The runs.
    rows
        The bodies, in the run's order.
    summary
        The summary to fill in (``bodies``) and save.
    path
        Where to save it.
    workers
        Worker processes.
    """
    plan = tasks[0].plan
    runs: dict[int, list[dict[str, Any]]] = {
        int(row["tags"]["uid"]): [] for row in rows
    }
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = [executor.submit(learn_run, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            run = future.result()
            runs[run["uid"]].append(run)
            print(  # noqa: T201 -- progress for the log
                f"[{done}/{len(tasks)}] {_describe_run(run)}; "
                f"{(time.perf_counter() - started) / 60:.0f} min",
                flush=True,
            )
            summary["bodies"] = [
                body_summary(rank, row, runs[int(row["tags"]["uid"])], plan)
                for rank, row in enumerate(rows, start=1)
            ]
            write_json(path, summary)


def _describe_run(run: dict[str, Any]) -> str:
    finish = (
        f"3 m in {run['finish_time']:.1f} s"
        if run["finish_time"] is not None
        else f"{run['furthest']:.2f} m"
    )
    return (
        f"uid {run['uid']} {run['family']} run {run['restart'] + 1}: "
        f"{run['speed']:.3f} m/s ({finish}) after {run['races']} races, "
        f"{run['stopped']}"
    )


def report(summary: dict[str, Any]) -> None:
    """
    Print every body's speed before and after, fastest first.

    Parameters
    ----------
    summary
        The finished summary.
    """
    bodies = sorted(summary["bodies"], key=lambda b: -b["final"]["speed"])
    for body in bodies:
        before, after = body["run"], body["final"]
        print(  # noqa: T201 -- the results table
            f"uid {body['uid']:5d} (run rank {body['run_rank']:2d}): "
            f"{before['speed']:.3f} -> {after['speed']:.3f} m/s "
            f"({after['family']}, {after['found_by']}); "
            f"{body['agreeing_runs']} runs agree, "
            f"{'converged' if body['converged'] else 'NOT converged'}",
            flush=True,
        )


def main() -> None:
    """Learn every survivor's gait at length and save the results."""
    db, plan, top, workers, video = parse_args()
    os.environ.update(SINGLE_THREAD_ENV)
    settings = run_settings(db)
    seed = int(db.stem.split("_")[-1])
    rows = final_population(db, top)
    tasks = make_tasks(rows, plan, settings, seed)
    path = db.with_name(f"{db.stem}_final_learning.json")
    summary: dict[str, Any] = {
        "run": str(db),
        "status": "running",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan": asdict(plan),
        "settings": asdict(settings),
        "bodies": [],
    }
    print(  # noqa: T201 -- progress for the log
        f"{len(rows)} bodies x {len(plan.families)} families x "
        f"{plan.restarts} runs = {len(tasks)} CMA-ES runs of up to "
        f"{plan.max_gens} x {plan.popsize} races, on {workers} workers",
        flush=True,
    )
    started = time.perf_counter()
    learn_all(tasks, rows, summary, path, workers)
    for body in summary["bodies"]:
        replayed = replay_gait(body["genome"], body["final"]["gait"], settings)
        body["final"]["replayed_speed"] = replayed.speed
    summary.update({
        "status": "completed",
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": round(time.perf_counter() - started, 1),
        "races": sum(
            run["races"] for body in summary["bodies"] for run in body["runs"]
        ),
    })
    write_json(path, summary)
    print(f"\nWritten {path}", flush=True)  # noqa: T201
    report(summary)
    if video:
        filmed = film_final(db, summary["bodies"], settings, seed)
        print("Filmed", filmed, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
