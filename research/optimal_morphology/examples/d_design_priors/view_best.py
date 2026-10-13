"""
Watch the best robot of a design-priors run, or save a video of it.

Example
-------
    uv run python examples/d_design_priors/view_best.py \
        ~/ariel_runs/design_priors/symmetric/seed_01.db

The graphics are ARIEL's usual ones: MuJoCo's viewer (as in
``examples/a_mujoco``) or, with ``--video``,
:func:`ariel.utils.renderers.tracking_video_renderer`. This script only loads a
body and its learned brain from the run database and drives the robot with
them, exactly as it was scored: ``eval_delay`` seconds without control, then
the brain walks it toward the target (the red sphere).

Before anything is shown, the stored score is recomputed from the stored
brain, which confirms the replay is the same robot and brain. Brains were only
optimized for the scored ``duration`` (3 s); what happens after that was never
selected for. The video stops there unless ``--seconds`` says otherwise; in the
viewer the simulation keeps running until the window is closed, and Backspace
restarts it.
"""

# Standard library
import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from dp_core import (
    TARGET_POSITION,
    EvalSettings,
    MlpBrain,
    RunConfig,
    build_world,
    load_individuals,
    replay_brain,
)
from mujoco import viewer

from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import tracking_video_renderer
from ariel.utils.symmetry import find_core_body
from ariel.utils.video_recorder import VideoRecorder

# The viewer opens on a side view of the spawn point (left) and the target
# (right): its free camera looks at SCENE_CENTER from 1.5 * SCENE_EXTENT away.
SCENE_CENTER = (0.6, 0.0, 0.1)
SCENE_EXTENT = 2.0
VIEW_AZIMUTH = 90.0
VIEW_ELEVATION = -25.0
TARGET_RADIUS = 0.08
# Video cameras that follow the robot (azimuth, elevation, distance). The
# robot faces +x at spawn: "chase" looks from behind it toward the target,
# "side" looks along +y, so it walks from left to right.
VIDEO_VIEWS = {
    "chase": (0.0, -25.0, 1.5),
    "side": (90.0, -15.0, 1.8),
}
TARGET_RGBA = (0.85, 0.2, 0.2, 0.9)
REPLAY_TOLERANCE = 1e-9

type ControlCallback = Callable[[mujoco.MjModel, mujoco.MjData], None]


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        The run database, optional body uid, video folder and duration.
    """
    parser = argparse.ArgumentParser(
        description="Watch the best robot of a design-priors run.",
    )
    parser.add_argument(
        "db",
        type=Path,
        help="a run database, e.g. ~/ariel_runs/design_priors/symmetric/"
        "seed_01.db",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="show this body (its 'uid' tag) instead of the best one",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="save an MP4 to this folder instead of opening the viewer",
    )
    parser.add_argument(
        "--view",
        choices=VIDEO_VIEWS,
        default="chase",
        help="video camera: behind the robot (chase) or beside it (side)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="seconds of control in the video (default: the scored duration,"
        " 3 s; brains were only optimized for that long)",
    )
    return parser.parse_args()


def run_settings(db: Path) -> EvalSettings:
    """
    Read the simulation settings the run used.

    Parameters
    ----------
    db
        The run database; its ``*_meta.json`` sits next to it.

    Returns
    -------
    EvalSettings
        The run's settings, or the defaults if the metadata is missing.
    """
    meta_path = db.with_name(f"{db.stem}_meta.json")
    if not meta_path.exists():
        return RunConfig("baseline", seed=0, out=db.parent).eval_settings()
    config = json.loads(meta_path.read_text("utf-8"))["config"]
    fields = EvalSettings.__dataclass_fields__
    return EvalSettings(**{name: config[name] for name in fields})


def pick_individual(db: Path, uid: int | None) -> dict[str, Any]:
    """
    Choose the body to show.

    Parameters
    ----------
    db
        The run database.
    uid
        A body's ``uid`` tag, or ``None`` for the best-scoring body.

    Returns
    -------
    dict[str, Any]
        The individual, as returned by :func:`dp_core.load_individuals`.

    Raises
    ------
    SystemExit
        If the database has no evaluated body with that uid.
    """
    scored = [
        ind
        for ind in load_individuals(db)
        if ind["tags"].get("score") is not None
    ]
    if uid is None:
        return max(scored, key=lambda ind: ind["tags"]["score"])
    for ind in scored:
        if ind["tags"]["uid"] == uid:
            return ind
    msg = f"No evaluated body with uid {uid} in {db}"
    raise SystemExit(msg)


def build_scene(
    genome: dict[str, Any],
    settings: EvalSettings,
) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    """
    Spawn the body as during scoring and mark the target.

    The target is a site, so the physics is the same as when it was scored.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    settings
        The run's settings (spawn yaw).

    Returns
    -------
    tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]
        The world, its compiled model and fresh data.
    """
    world, _, _ = build_world(genome, settings.spawn_yaw)
    world.spec.worldbody.add_site(
        name="target",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[TARGET_RADIUS, 0.0, 0.0],
        pos=TARGET_POSITION.tolist(),
        rgba=TARGET_RGBA,
    )
    model = cast("mujoco.MjModel", world.spec.compile())
    model.stat.center[:] = SCENE_CENTER
    model.stat.extent = SCENE_EXTENT
    model.vis.global_.azimuth = VIEW_AZIMUTH
    model.vis.global_.elevation = VIEW_ELEVATION
    return world, model, mujoco.MjData(model)


def make_control(
    world: SimpleFlatWorld,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    brain_vector: list[float],
    eval_delay: float,
) -> ControlCallback:
    """
    Build the control callback: no control while settling, then the brain.

    Parameters
    ----------
    world
        The world the body was spawned in.
    model
        The compiled model.
    data
        The simulation data.
    brain_vector
        The brain parameters stored with the body.
    eval_delay
        Seconds without control before the brain takes over.

    Returns
    -------
    ControlCallback
        A function for :func:`mujoco.set_mjcb_control`.
    """
    brain = MlpBrain(len(get_state_from_data(data)) + 2, model.nu)
    brain.set_params(np.asarray(brain_vector, dtype=np.float64))
    tracker = Tracker(
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)
    controller = Controller(controller_callback_function=brain, tracker=tracker)

    def control(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if data.time < eval_delay:
            data.ctrl[:] = 0.0
        else:
            controller.set_control(model, data)

    return control


def describe(ind: dict[str, Any], replayed: float) -> str:
    """
    Summarise the chosen body and check its replayed score.

    Parameters
    ----------
    ind
        The individual being shown.
    replayed
        Its score recomputed from the stored brain.

    Returns
    -------
    str
        A few lines for the terminal.
    """
    tags = ind["tags"]
    stored = float(tags["score"])
    same = abs(replayed - stored) <= REPLAY_TOLERANCE
    return (
        f"Body uid {tags['uid']} ({tags['origin']}, born in generation "
        f"{ind['birth']}): {tags['n_modules']} modules, {tags['n_hinges']} "
        f"hinges, symmetry {min(tags['sym_position'], tags['sym_axis']):.2f}\n"
        f"Score {stored:.4f} m; replayed from the stored brain: "
        f"{replayed:.4f} m ({'identical' if same else 'DIFFERENT'})"
    )


def save_video(
    args: argparse.Namespace,
    ind: dict[str, Any],
    model: mujoco.MjModel,
    data: mujoco.MjData,
    settings: EvalSettings,
) -> None:
    """
    Record the robot with ARIEL's tracking camera.

    Parameters
    ----------
    args
        Command-line options (video folder, view, seconds).
    ind
        The individual being shown.
    model
        The compiled scene.
    data
        The simulation data; the control callback must already be set.
    settings
        The run's settings (settling time and scored duration).
    """
    name = (
        f"{args.db.parent.name}_{args.db.stem}_uid{ind['tags']['uid']}"
        f"_{args.view}"
    )
    azimuth, elevation, distance = VIDEO_VIEWS[args.view]
    seconds = settings.duration if args.seconds is None else args.seconds
    tracking_video_renderer(
        model,
        data,
        duration=settings.eval_delay + seconds,
        video_recorder=VideoRecorder(file_name=name, output_folder=args.video),
        geom_to_track=mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            find_core_body(model),
        ),
        tracking_distance=distance,
        tracking_azimuth=azimuth,
        tracking_elevation=elevation,
    )
    print(f"Saved {name}_<time>.mp4 in {args.video}")  # noqa: T201


def main() -> None:
    """Show the chosen body in the viewer, or save a video of it."""
    args = parse_args()
    settings = run_settings(args.db)
    ind = pick_individual(args.db, args.uid)
    genome, brain_vector = ind["genotype"]["morph"], ind["tags"]["brain"]
    replayed = replay_brain(genome, brain_vector, settings)
    print(describe(ind, replayed))  # noqa: T201

    world, model, data = build_scene(genome, settings)
    mujoco.set_mjcb_control(
        make_control(world, model, data, brain_vector, settings.eval_delay),
    )
    try:
        if args.video is None:
            viewer.launch(model, data)
        else:
            save_video(args, ind, model, data, settings)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
