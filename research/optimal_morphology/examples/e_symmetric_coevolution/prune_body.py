"""
Remove the legs of a robot that never touch the ground during its dash.

Example
-------
    uv run python examples/e_symmetric_coevolution/prune_body.py \
        ~/ariel_runs/symmetric_coevolution/symmetric/seed_01.db --uid 12

The robot runs its learned gait exactly as it was scored, and every module
that touches the floor is recorded, from the start of the dash until it
crosses the finish line or the time runs out. Then every subtree (a leg, or
the end of one) in which no module touched the floor is cut off. A module
that never touches the floor itself stays if something below it does: a
hinge that swings a foot is needed. Mirror twins are judged together, so the
pruned body stays symmetric.

The pruned body is then tested twice:

* with its old gait, i.e. the waves of the hinges that are left;
* with gaits learned from scratch as in the experiment, for ``--relearn``
  fresh seeds, next to the original body re-learned with the same seeds.

Outputs go to ``<run folder>/pruned/<run>_uid<N>*``: a JSON summary with the
pruned genome and its best gait, a picture of the body before and after, and,
with ``--video``, a video of the pruned robot from the top, the side and 45
degrees (``view_best.py``).
"""

# Standard library
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

# Third-party libraries
import numpy as np
from sc_core import (
    EvalResult,
    EvalSettings,
    EvalTask,
    RaceResult,
    derive_seed,
    grounded_modules,
    learn_brain,
    mirror_twins,
    prune_genome,
    replay_gait,
    transfer_gait,
)
from sc_render import BodyCard, save_grid
from view_best import (
    DEFAULT_VIEWS,
    FPS,
    TILE,
    Robot,
    film,
    pick_individual,
    run_settings,
)

# Local libraries
from ariel.ec.genotypes.tree.constraints import count_hinges, count_modules
from ariel.ec.genotypes.tree.symmetry import genome_symmetry
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.utils.video_recorder import VideoRecorder

# Global constants
RELEARN_SEED_OFFSET = 20_000


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        The run database, body uid, number of re-learning seeds, workers and
        video folder.
    """
    parser = argparse.ArgumentParser(
        description="Cut off the legs of a robot that never touch the ground.",
    )
    parser.add_argument("db", type=Path, help="a run database")
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="the body to prune (default: the fastest)",
    )
    parser.add_argument(
        "--relearn",
        type=int,
        default=3,
        help="fresh learning seeds for each of the two bodies (0: skip)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="processes for re-learning",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="also save a video of the pruned robot to this folder",
    )
    return parser.parse_args()


def relearn(
    bodies: dict[str, dict[str, Any]],
    seeds: list[int],
    settings: EvalSettings,
    workers: int,
) -> dict[str, list[EvalResult]]:
    """
    Learn gaits from scratch for each body with the same fresh seeds.

    Parameters
    ----------
    bodies
        Name to tree genome (as a dict).
    seeds
        Learning seeds, shared by all bodies.
    settings
        The run's settings (brain and budget).
    workers
        Processes to learn in.

    Returns
    -------
    dict[str, list[EvalResult]]
        Per body, one result per seed.
    """
    settings = replace(settings, prune=False)
    tasks = [
        EvalTask(genome=genome, learn_seed=seed, settings=settings)
        for genome in bodies.values()
        for seed in seeds
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(tasks))),
        mp_context=get_context("spawn"),
    ) as executor:
        results = list(executor.map(learn_brain, tasks))
    return {
        name: results[i * len(seeds) : (i + 1) * len(seeds)]
        for i, name in enumerate(bodies)
    }


def describe_race(race: RaceResult | EvalResult, goal: float) -> str:
    """
    Summarise one dash.

    Parameters
    ----------
    race
        The dash, or the best dash of a learning run.
    goal
        Distance to cover (m).

    Returns
    -------
    str
        E.g. ``0.163 m/s (0.59 km/h): 3 m in 18.4 s``.
    """
    speed = race.speed or 0.0
    if race.finish_time is None:
        result = f"did not finish, furthest {race.furthest:.2f} m"
    else:
        result = f"{goal:g} m in {race.finish_time:.1f} s"
    return f"{speed:.3f} m/s ({speed * 3.6:.2f} km/h): {result}"


def save_video(
    folder: Path,
    name: str,
    genome: dict[str, Any],
    gait: dict[str, Any],
    settings: EvalSettings,
) -> None:
    """
    Film a body and gait from the top, the side and 45 degrees.

    Parameters
    ----------
    folder
        Output folder.
    name
        File-name stem.
    genome
        The body's tree genome as a dict.
    gait
        Its gait.
    settings
        The run's settings.
    """
    robot = Robot(
        name=name,
        header=name,
        genome=genome,
        gait=gait,
        stored_speed=0.0,
        settings=settings,
    )
    recorder = VideoRecorder(
        file_name=name,
        output_folder=folder,
        width=TILE[1] * len(DEFAULT_VIEWS),
        height=TILE[0],
        fps=FPS,
    )
    film(robot, recorder)


def prune_and_test(
    genome: dict[str, Any],
    gait: dict[str, Any],
    settings: EvalSettings,
) -> dict[str, Any]:
    """
    Prune a body and race it with the waves its remaining hinges had.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    gait
        Its learned gait.
    settings
        The run's settings.

    Returns
    -------
    dict[str, Any]
        Summary: the modules that touched the floor, the subtrees cut off,
        and the original and pruned bodies with their speeds.
    """
    grounded = grounded_modules(genome, gait, settings)
    pruned, cut = prune_genome(genome, grounded, mirror_twins(genome))
    original = TreeGenome.from_dict(genome)
    old_gait = transfer_gait(gait, genome, pruned)
    before = replay_gait(genome, gait, settings)
    after = replay_gait(pruned.to_dict(), old_gait, settings)
    symmetry = genome_symmetry(pruned).combined
    print(  # noqa: T201
        f"{count_modules(original)} modules, {count_hinges(original)} "
        f"hinges; touching the floor: {sorted(grounded)}\n"
        f"Cut off subtrees at nodes {cut}: now {count_modules(pruned)} "
        f"modules, {count_hinges(pruned)} hinges, symmetry {symmetry:.2f}\n"
        f"Original:          {describe_race(before, settings.goal)}\n"
        f"Pruned, old gait:  {describe_race(after, settings.goal)}",
        flush=True,
    )
    return {
        "grounded": sorted(grounded),
        "cut": cut,
        "original": {
            "genome": genome,
            "modules": count_modules(original),
            "hinges": count_hinges(original),
            "speed": before.speed,
        },
        "pruned": {
            "genome": pruned.to_dict(),
            "modules": count_modules(pruned),
            "hinges": count_hinges(pruned),
            "symmetry": symmetry,
            "old_gait": old_gait,
            "old_gait_speed": after.speed,
            "best_gait": old_gait,
            "best_speed": after.speed,
        },
    }


def add_relearning(
    summary: dict[str, Any],
    settings: EvalSettings,
    seeds: list[int],
    workers: int,
) -> None:
    """
    Re-learn both bodies with the same seeds and record the results.

    The pruned body's best gait becomes the best re-learned gait if that is
    faster than its old gait.

    Parameters
    ----------
    summary
        From :func:`prune_and_test`; updated in place.
    settings
        The run's settings.
    seeds
        Fresh learning seeds.
    workers
        Processes to learn in.
    """
    learned = relearn(
        {name: summary[name]["genome"] for name in ("original", "pruned")},
        seeds,
        settings,
        workers,
    )
    for name, results in learned.items():
        speeds = [r.speed or 0.0 for r in results]
        summary[name]["relearned_speeds"] = speeds
        print(  # noqa: T201
            f"Re-learned {name:9s} mean {np.mean(speeds):.3f} m/s "
            f"({np.mean(speeds) * 3.6:.2f} km/h); per seed: "
            + "; ".join(describe_race(r, settings.goal) for r in results),
            flush=True,
        )
    top = max(learned["pruned"], key=lambda r: r.speed or -math.inf)
    if (top.speed or 0.0) > summary["pruned"]["best_speed"]:
        summary["pruned"]["best_gait"] = top.gait
        summary["pruned"]["best_speed"] = top.speed


def save_picture(
    summary: dict[str, Any],
    path: Path,
    title: str,
    spawn_yaw: float,
) -> None:
    """
    Draw the body before and after pruning.

    Parameters
    ----------
    summary
        From :func:`prune_and_test`.
    path
        Output PNG.
    title
        Line above the pictures.
    spawn_yaw
        Yaw (degrees) at spawn.
    """
    cards = [
        BodyCard(
            summary[name]["genome"],
            f"{label}: {summary[name]['modules']} modules, "
            f"{summary[name]['hinges']} hinges",
            highlight=name == "pruned",
        )
        for name, label in (("original", "before"), ("pruned", "after"))
    ]
    save_grid(cards, path, title=title, spawn_yaw=spawn_yaw)


def main() -> None:
    """Prune the chosen body, test it, and save the results."""
    args = parse_args()
    settings = run_settings(args.db)
    ind = pick_individual(args.db, args.uid)
    uid = ind["tags"]["uid"]
    print(f"Body uid {uid}:", end=" ")  # noqa: T201
    summary = prune_and_test(
        ind["genotype"]["morph"],
        ind["tags"]["gait"],
        settings,
    )
    summary["uid"] = uid
    summary["db"] = str(args.db)
    summary["settings"] = asdict(settings)
    if args.relearn > 0:
        seeds = [
            derive_seed(uid, RELEARN_SEED_OFFSET + k)
            for k in range(args.relearn)
        ]
        add_relearning(summary, settings, seeds, args.workers)

    out = args.db.parent / "pruned"
    out.mkdir(exist_ok=True)
    stem = f"{args.db.parent.name}_{args.db.stem}_uid{uid}"
    (out / f"{stem}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    save_picture(
        summary,
        out / f"{stem}.png",
        f"uid {uid}: legs that never touch the ground removed",
        settings.spawn_yaw,
    )
    if args.video is not None:
        pruned = summary["pruned"]
        save_video(
            args.video,
            f"{stem}_pruned",
            pruned["genome"],
            pruned["best_gait"],
            settings,
        )
    print(f"Saved {stem}.json and {stem}.png in {out}")  # noqa: T201


if __name__ == "__main__":
    main()
