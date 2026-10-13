"""
Run one seed of the symmetric body-brain co-evolution experiment.

Example
-------
    uv run python \
        examples/e_symmetric_coevolution/run_symmetric_coevolution.py --seed 1

Mirror-symmetric bodies are evolved to cover 3 m within 60 s as fast as
possible, and every new body learns its own brain from scratch; then the legs
that never touch the ground are cut off and the pruned body learns again
(``--no-prune`` skips that). The hinges behave like hobby servos: stiff
position control (``--servo-gain``), a top speed, a torque limit and a cap on
how fast they oscillate (``--max-hinge-speed``, ``--max-hinge-torque``,
``--max-hinge-frequency``; see ``sc_core.py``). Besides the usual tree
mutations, the ``grow`` mutation adds a module to a limb (``--grow``, its
share of the mutations), and a child can also switch a hinge between one
servo and two in parallel (``--double-servo``, the chance). Outputs go to
``<out>/<brain>_pruned_realservo_grow_double/`` (the name drops ``_pruned``
without pruning, ``_grow`` without growth and ``_double`` without double
servos; with ``--servo-gain 0``, ARIEL's own servo, ``_realservo`` becomes
``_servo``, or nothing without servo limits):

* ``seed_<NN>.db`` -- the EA database, every body with its tags;
* ``seed_<NN>_meta.json`` -- settings, status and one line per generation;
* ``seed_<NN>_reeval.json`` -- the best bodies re-learned with fresh seeds;
* ``seed_<NN>_snapshots/gen_<NNN>.png`` -- pictures of the surviving bodies,
  every ``--snapshot-every`` generations (default: every generation);
* ``seed_<NN>_videos/gen_<NNN>_uid<N>.mp4`` -- the fastest robot from the
  top, the side and 45 degrees, every ``--video-every`` generations (default:
  every generation);
* ``seed_<NN>_videos/gen_<NNN>_all.mp4`` -- all survivors racing side by
  side, fastest first, as often;
* ``seed_<NN>_gallery.html`` -- every generation's videos and picture, on
  one page, rewritten after every generation.

An existing database is never overwritten. ``--resume`` continues its run
instead, from its last finished generation up to ``--gens``, exactly as if it
had never stopped; the settings must be the run's, apart from ``--gens``,
``--workers``, the pictures' and videos' cadence and the final
re-evaluation. The database, metadata and re-evaluation as they were are
first copied to ``seed_<NN>_backup_gen<NNN>/``.
"""

# Standard library
import argparse
import os
from pathlib import Path

# Local libraries
from sc_core import BRAINS, RunConfig, Snapshot, run_experiment
from sc_gallery import gallery_page, survivor_race_videos
from sc_render import population_snapshot
from view_best import best_robot_videos

from ariel.ec import Individual

DEFAULT_OUT = Path.home() / "ariel_runs" / "symmetric_coevolution"
DESCRIPTION = (
    "Evolve mirror-symmetric robots that cover 3 m in 60 s as fast as "
    "possible; every new body learns its own brain."
)
# Learning runs in worker processes; one BLAS thread each avoids
# oversubscribing the cores.
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
FLAGS: dict[str, tuple[type, str]] = {
    "pop": (int, "bodies kept each generation (mu)"),
    "offspring": (int, "new bodies per generation (lambda)"),
    "gens": (int, "generations after the initial population"),
    "tournament": (int, "tournament size of parent selection"),
    "learn_gens": (int, "CMA-ES generations per body, split by gait family"),
    "learn_pop": (int, "races per CMA-ES generation"),
    "learn_sigma": (float, "initial CMA-ES step size"),
    "goal": (float, "distance to cover, in m"),
    "time_limit": (float, "seconds the robot has to cover it"),
    "settle": (float, "seconds without control before the dash"),
    "max_modules": (int, "largest body, in modules"),
    "max_depth": (int, "deepest tree allowed"),
    "init_size_min": (int, "smallest target size of an initial body"),
    "spawn_yaw": (float, "yaw at spawn, in degrees (-90 faces +x)"),
    "retries": (int, "variation attempts per offspring"),
    "reeval_top": (int, "best bodies re-learned at the end"),
    "reeval_seeds": (int, "fresh learning seeds per re-learned body"),
    "snapshot_every": (int, "save body pictures every N generations (0: off)"),
    "video_every": (int, "film the fastest robot every N generations (0: off)"),
    "max_hinge_speed": (float, "servo top speed, in rad/s (0: no limit)"),
    "max_hinge_torque": (float, "servo torque limit, in N m (0: no limit)"),
    "max_hinge_frequency": (float, "fastest hinge oscillation, Hz (0: none)"),
    "grow": (float, "share of mutations that add a module to a limb (0: off)"),
    "servo_gain": (float, "hinge servo gain, N m/rad (0: ARIEL's slow servo)"),
    "double_servo": (
        float,
        "chance that a mutated child also switches a hinge between one "
        "servo and two in parallel (0: off)",
    ),
}


def parse_args() -> tuple[RunConfig, bool]:
    """
    Parse the command line into a run configuration.

    Returns
    -------
    tuple[RunConfig, bool]
        The settings of this run, and whether to resume it.
    """
    defaults = RunConfig(seed=0, out=DEFAULT_OUT)
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, required=True, help="run seed")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output root (keep it out of cloud-synced folders)",
    )
    parser.add_argument(
        "--brain",
        choices=BRAINS,
        default=defaults.brain,
        help="gait families each body learns; 'symmetric' learns the "
        "alternating and the synchronous family and keeps the faster",
    )
    for name, (kind, text) in FLAGS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=kind,
            default=getattr(defaults, name),
            help=text,
        )
    parser.add_argument(
        "--prune",
        action=argparse.BooleanOptionalAction,
        default=defaults.prune,
        help="after learning, cut off the legs that never touch the ground "
        "and learn again",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="processes that learn brains in parallel",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the run in the output folder from its last finished "
        "generation up to --gens, with the run's settings",
    )
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    return RunConfig(**args), resume


def safe(hook: Snapshot, what: str) -> Snapshot:
    """
    Wrap an output hook so that its failure only stops that output.

    Rendering needs an OpenGL context, which a headless machine may lack; a
    long run should not die for a picture or a video.

    Parameters
    ----------
    hook
        The picture or video hook.
    what
        Name of the output, for the message.

    Returns
    -------
    Snapshot
        The hook, which disables itself after its first error.
    """
    enabled = True

    def wrapped(survivors: list[Individual], generation: int) -> None:
        nonlocal enabled
        if not enabled:
            return
        try:
            hook(survivors, generation)
        except Exception as error:  # noqa: BLE001 -- outputs are optional
            enabled = False
            print(f"{what} disabled: {error!r}", flush=True)  # noqa: T201

    return wrapped


def output_hooks(cfg: RunConfig) -> Snapshot:
    """
    Combine the picture, video and gallery hooks that the settings ask for.

    Parameters
    ----------
    cfg
        Run settings (``snapshot_every``, ``video_every``).

    Returns
    -------
    Snapshot
        Called after every generation; each output keeps its own cadence.
    """
    hooks = []
    if cfg.snapshot_every > 0:
        hooks.append(safe(population_snapshot(cfg), "Pictures"))
    if cfg.video_every > 0:
        hooks.extend((
            safe(best_robot_videos(cfg), "Videos"),
            safe(survivor_race_videos(cfg), "Race videos"),
        ))
    hooks.append(safe(gallery_page(cfg), "Gallery"))  # last: links the rest

    def combined(survivors: list[Individual], generation: int) -> None:
        for hook in hooks:
            hook(survivors, generation)

    return combined


def main() -> None:
    """Run the experiment and print where the results are."""
    cfg, resume = parse_args()
    os.environ.update(SINGLE_THREAD_ENV)
    meta = run_experiment(cfg, snapshot=output_hooks(cfg), resume=resume)
    print(  # noqa: T201
        f"Done: {cfg.db_path}\n"
        f"  best speed {meta['best_speed']:.3f} m/s (uid {meta['best_uid']}),"
        f" re-learned best {meta['reeval_best']}\n"
        f"  {meta['total_sims']} races, {meta['sim_hours']} h simulated, "
        f"{meta['wall_seconds']} s wall time\n"
        f"  pictures in {cfg.snapshot_dir}\n"
        f"  videos in {cfg.video_dir}\n"
        f"  gallery: {cfg.run_dir / f'{cfg.stem}_gallery.html'}",
    )


if __name__ == "__main__":
    main()
