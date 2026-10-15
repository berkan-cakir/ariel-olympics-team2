"""
Watch the fastest robot of a symmetric co-evolution run, or save a video.

Example
-------
    uv run python examples/e_symmetric_coevolution/view_best.py \
        ~/ariel_runs/symmetric_coevolution/alternating/seed_01.db

The script loads a body and its learned gait from the run database, or a
pruned body from a summary written by ``prune_body.py``, and drives the robot
exactly as it was scored: ``settle`` seconds without control, then the gait.
A green line marks the start and a red line the finish, ``goal`` metres
ahead.

Before anything is shown, the stored speed is recomputed from the stored
gait, which confirms the replay is the same robot and brain. In MuJoCo's
viewer the simulation keeps running until the window is closed (Backspace
restarts it).

With ``--video``, the robot is filmed by cameras that follow it, side by
side in one video: by default from the top, from the side and at 45 degrees
(``--view`` picks others, e.g. just ``side``). A clock shows the time and the
distance covered. The video covers the dash until the robot finishes (plus
one second) or the time limit, unless ``--seconds`` says otherwise.

    uv run python examples/e_symmetric_coevolution/view_best.py \
        ~/ariel_runs/symmetric_coevolution/symmetric/seed_01.db \
        --uid 12 --video ~/ariel_runs/videos --view top side angle
"""

# Standard library
import argparse
import json
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import cv2
import mujoco
import numpy as np
from mujoco import viewer
from sc_core import (
    CONTROL_STEPS,
    EvalSettings,
    Gait,
    RaceResult,
    RunConfig,
    Snapshot,
    build_world,
    is_due,
    load_individuals,
    replay_gait,
    servo_commands,
    settle,
)

# Local libraries
from ariel.ec import Individual
from ariel.parameters.ariel_types import FloatArray
from ariel.utils.symmetry import find_core_body
from ariel.utils.video_recorder import VideoRecorder

# Video cameras that follow the robot: (azimuth, elevation, distance, label).
# The robot faces +x. "top" looks straight down and "side" along +y from its
# right, so in both it runs from left to right; "angle" looks from its
# front-right, 45 degrees up (as in the body pictures); "chase" from behind.
VIDEO_VIEWS = {
    "top": (90.0, -90.0, 2.4, "top"),
    "side": (90.0, -15.0, 2.0, "side"),
    "angle": (135.0, -45.0, 2.0, "45 deg"),
    "chase": (0.0, -25.0, 1.6, "behind"),
}
DEFAULT_VIEWS = ("top", "side", "angle")
TILE = (480, 640)  # height, width of each camera's picture
FPS = 30
TEXT_SCALE = 0.6
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LINE_HALF_SIZE = (0.01, 1.0, 0.001)  # m: thin, 2 m wide, flat
START_RGBA = (0.2, 0.75, 0.3, 0.9)
FINISH_RGBA = (0.85, 0.2, 0.2, 0.9)
VIEW_AZIMUTH = 90.0
VIEW_ELEVATION = -25.0
HEADLIGHT_SPECULAR = 0.2
VIEWER_SECONDS = 600.0  # commands are prepared for this long, then held
EXTRA_SECONDS = 1.0
REPLAY_TOLERANCE = 1e-9

type ControlCallback = Callable[[mujoco.MjModel, mujoco.MjData], None]


@dataclass(frozen=True)
class Scene:
    """
    A body spawned as during scoring, with start and finish lines.

    Parameters
    ----------
    model
        The compiled scene.
    data
        Its data.
    start
        The core's x when the dash starts (m).
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    start: float


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns
    -------
    argparse.Namespace
        The run database, optional body uid, video folder, view and length.
    """
    parser = argparse.ArgumentParser(
        description="Watch the fastest robot of a symmetric co-evolution run.",
    )
    parser.add_argument(
        "db",
        type=Path,
        help="a run database, e.g. ~/ariel_runs/symmetric_coevolution/"
        "symmetric/seed_01.db, or a summary written by prune_body.py or "
        "final_learning.py",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="show this body (its 'uid' tag) instead of the fastest one; "
        "for a run database or a final_learning.py summary",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="save an MP4 to this folder instead of opening the viewer",
    )
    parser.add_argument(
        "--view",
        nargs="+",
        choices=VIDEO_VIEWS,
        default=list(DEFAULT_VIEWS),
        help="video cameras, shown side by side in this order (default: "
        "top side angle)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="seconds of the dash in the video (default: until the robot "
        "finishes, plus 1 s, or the time limit)",
    )
    return parser.parse_args()


def run_settings(db: Path) -> EvalSettings:
    """
    Read the settings the run used.

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
        return RunConfig(seed=0, out=db.parent).eval_settings()
    config = json.loads(meta_path.read_text("utf-8"))["config"]
    # Settings added later (e.g. prune) keep their defaults for older runs.
    fields = EvalSettings.__dataclass_fields__
    return EvalSettings(**{n: config[n] for n in fields if n in config})


def pick_individual(db: Path, uid: int | None) -> dict[str, Any]:
    """
    Choose the body to show.

    Parameters
    ----------
    db
        The run database.
    uid
        A body's ``uid`` tag, or ``None`` for the fastest body.

    Returns
    -------
    dict[str, Any]
        The individual, as returned by :func:`sc_core.load_individuals`.

    Raises
    ------
    SystemExit
        If the database has no evaluated body with that uid.
    """
    scored = [
        ind
        for ind in load_individuals(db)
        if ind["tags"].get("speed") is not None
    ]
    if uid is None:
        return max(scored, key=lambda ind: ind["tags"]["speed"])
    for ind in scored:
        if ind["tags"]["uid"] == uid:
            return ind
    msg = f"No evaluated body with uid {uid} in {db}"
    raise SystemExit(msg)


def start_x(genome: dict[str, Any], settings: EvalSettings) -> float:
    """
    Return where the core is when the dash starts.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    settings
        The run's settings (spawn yaw, settling time).

    Returns
    -------
    float
        The core's x after settling (m).
    """
    _, model, data = build_world(
        genome,
        settings.spawn_yaw,
        settings,
    )
    settle(model, data, settings.settle)
    return float(data.qpos[0])


def build_scene(genome: dict[str, Any], settings: EvalSettings) -> Scene:
    """
    Spawn the body as during scoring and mark the start and finish lines.

    The lines are sites, so the physics is the same as when it was scored.
    Call this before setting the control callback: compiling runs it.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    settings
        The run's settings (spawn yaw, settling time, goal).

    Returns
    -------
    Scene
        The compiled scene, fresh data and the start line's x.
    """
    start = start_x(genome, settings)
    world, _, _ = build_world(
        genome,
        settings.spawn_yaw,
        settings,
    )
    for name, x, rgba in (
        ("start", start, START_RGBA),
        ("finish", start + settings.goal, FINISH_RGBA),
    ):
        world.spec.worldbody.add_site(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(LINE_HALF_SIZE),
            pos=[x, 0.0, LINE_HALF_SIZE[2]],
            rgba=rgba,
        )
    model = cast("mujoco.MjModel", world.spec.compile())
    model.stat.center[:] = (start + settings.goal / 2, 0.0, 0.1)
    model.stat.extent = settings.goal / 2 + 0.5
    model.vis.global_.azimuth = VIEW_AZIMUTH
    model.vis.global_.elevation = VIEW_ELEVATION
    # A strong headlight highlight washes out flat tops seen from straight
    # above; this only changes the rendering.
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR
    return Scene(model=model, data=mujoco.MjData(model), start=start)


def make_control(
    gait: Gait,
    settings: EvalSettings,
    timestep: float,
) -> ControlCallback:
    """
    Build the control callback: no control while settling, then the gait.

    The hinges get the same commands as in :func:`sc_core.run_race`
    (:func:`sc_core.servo_commands`, with the run's servo speed limit),
    changed every :data:`sc_core.CONTROL_STEPS` physics steps; after
    :data:`VIEWER_SECONDS` the last command is held.

    Parameters
    ----------
    gait
        The learned gait.
    settings
        The run's settings (settling time, servo limits).
    timestep
        The model's physics timestep.

    Returns
    -------
    ControlCallback
        A function for :func:`mujoco.set_mjcb_control`.
    """
    period = CONTROL_STEPS * timestep
    horizon = max(settings.time_limit, VIEWER_SECONDS)
    commands = servo_commands(gait, settings, period, round(horizon / period))
    settle_steps = round(settings.settle / timestep)

    def control(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        step = round(data.time / timestep) - settle_steps
        if step < 0:
            data.ctrl[:] = 0.0
            return
        update = min(step // CONTROL_STEPS, len(commands) - 1)
        data.ctrl[:] = commands[update]

    return control


@dataclass(frozen=True)
class Robot:
    """
    A body and gait to show, with the speed they were stored with.

    Parameters
    ----------
    name
        File-name stem for videos.
    header
        First line of the terminal summary.
    genome
        The body's tree genome as a dict.
    gait
        Its gait, as :meth:`sc_core.Gait.to_dict`.
    stored_speed
        The speed stored with the gait (m/s).
    settings
        The run's settings.
    """

    name: str
    header: str
    genome: dict[str, Any]
    gait: dict[str, Any]
    stored_speed: float
    settings: EvalSettings


def robot_from_run(db: Path, uid: int | None) -> Robot:
    """
    Load a body and its learned gait from a run database.

    Parameters
    ----------
    db
        The run database.
    uid
        A body's ``uid`` tag, or ``None`` for the fastest body.

    Returns
    -------
    Robot
        The body to show.
    """
    ind = pick_individual(db, uid)
    tags = ind["tags"]
    return Robot(
        name=f"{db.parent.name}_{db.stem}_uid{tags['uid']}",
        header=(
            f"Body uid {tags['uid']} ({tags['origin']}, born in generation "
            f"{ind['birth']}): {tags['n_modules']} modules, "
            f"{tags['n_hinges']} hinges ({tags['n_pairs']} mirror pairs, "
            f"{tags['n_plane']} on the sagittal plane), "
            f"{tags.get('family', '')} gait"
        ),
        genome=ind["genotype"]["morph"],
        gait=tags["gait"],
        stored_speed=float(tags["speed"]),
        settings=run_settings(db),
    )


def robot_from_pruning(summary_path: Path) -> Robot:
    """
    Load a pruned body and its best gait from a ``prune_body.py`` summary.

    Parameters
    ----------
    summary_path
        The summary, ``<run folder>/pruned/<run>_uid<N>.json``.

    Returns
    -------
    Robot
        The pruned body to show.
    """
    summary = json.loads(summary_path.read_text("utf-8"))
    if "settings" in summary:
        settings = EvalSettings(**summary["settings"])
    else:  # summaries written before the settings were stored
        match = re.search(r"(seed_\d+)_uid\d+$", summary_path.stem)
        stem = match.group(1) if match else "seed_00"
        settings = run_settings(summary_path.parents[1] / f"{stem}.db")
    pruned = summary["pruned"]
    return Robot(
        name=f"{summary_path.stem}_pruned",
        header=(
            f"Pruned body of uid {summary['uid']}: {pruned['modules']} "
            f"modules, {pruned['hinges']} hinges (was "
            f"{summary['original']['modules']} and "
            f"{summary['original']['hinges']})"
        ),
        genome=pruned["genome"],
        gait=pruned["best_gait"],
        stored_speed=float(pruned["best_speed"]),
        settings=settings,
    )


def robot_from_final_learning(summary_path: Path, uid: int | None) -> Robot:
    """
    Load a body and its final gait from a ``final_learning.py`` summary.

    Parameters
    ----------
    summary_path
        The summary, ``<run folder>/seed_<NN>_final_learning.json``.
    uid
        A body's uid, or ``None`` for the fastest body.

    Returns
    -------
    Robot
        The body to show, with its final gait.

    Raises
    ------
    ValueError
        If the summary has no finished body with that uid.
    """
    summary = json.loads(summary_path.read_text("utf-8"))
    bodies = [body for body in summary["bodies"] if "final" in body]
    if uid is not None:
        bodies = [body for body in bodies if body["uid"] == uid]
    if not bodies:
        msg = f"No finished body{'' if uid is None else f' with uid {uid}'}."
        raise ValueError(msg)
    body = max(bodies, key=lambda candidate: candidate["final"]["speed"])
    final, before = body["final"], body["run"]
    return Robot(
        name=f"{summary_path.parent.name}_{summary_path.stem}_uid{body['uid']}",
        header=(
            f"uid {body['uid']} after long learning: {final['family']} gait "
            f"from {final['found_by']} ({before['speed']:.4f} m/s in the run)"
        ),
        genome=body["genome"],
        gait=final["gait"],
        stored_speed=float(final["speed"]),
        settings=EvalSettings(**summary["settings"]),
    )


def describe(robot: Robot, replayed: RaceResult) -> str:
    """
    Summarise the robot and check its replayed speed.

    Parameters
    ----------
    robot
        The robot being shown.
    replayed
        Its dash, recomputed from the stored gait.

    Returns
    -------
    str
        A few lines for the terminal.
    """
    same = abs(replayed.speed - robot.stored_speed) <= REPLAY_TOLERANCE
    if replayed.finish_time is None:
        result = f"did not finish; furthest {replayed.furthest:.2f} m"
    else:
        result = f"{robot.settings.goal:g} m in {replayed.finish_time:.2f} s"
    return (
        f"{robot.header}\n"
        f"Stored speed {robot.stored_speed:.4f} m/s; replayed from the stored"
        f" gait: {replayed.speed:.4f} m/s "
        f"({'identical' if same else 'DIFFERENT'})\n{result}; drift "
        f"{replayed.drift:+.3f} m, turn {replayed.turn:+.1f} deg; gait at "
        f"{robot.gait['frequency']:.2f} Hz"
    )


def tracking_camera(core: int, view: str) -> mujoco.MjvCamera:
    """
    Make a camera that follows the robot's core from one of the views.

    Parameters
    ----------
    core
        Body id of the robot's core.
    view
        A key of :data:`VIDEO_VIEWS`.

    Returns
    -------
    mujoco.MjvCamera
        A tracking camera.
    """
    azimuth, elevation, distance, _ = VIDEO_VIEWS[view]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = core
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    return camera


def put_text(
    image: FloatArray,
    text: str,
    origin: tuple[int, int],
    scale: float = TEXT_SCALE,
) -> None:
    """
    Write white text with a dark outline onto an RGB image, in place.

    Parameters
    ----------
    image
        The picture.
    text
        ASCII text.
    origin
        Bottom-left corner of the text, in pixels.
    scale
        Font scale.
    """
    for color, thickness in ((BLACK, 3), (WHITE, 1)):
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def race_clock(
    data: mujoco.MjData,
    settings: EvalSettings,
    start: float,
    finish_time: float | None,
) -> str:
    """
    Describe the moment of the dash shown in a frame.

    Parameters
    ----------
    data
        The simulation data.
    settings
        The run's settings (settling time, goal).
    start
        The core's x when the dash starts.
    finish_time
        When the robot crosses the finish line, or ``None``.

    Returns
    -------
    str
        E.g. ``dash 12.34 s, 1.84 m`` or ``3 m in 18.45 s``.
    """
    dash = data.time - settings.settle
    if dash < 0:
        return f"settling {data.time:.1f} s"
    if finish_time is not None and dash >= finish_time:
        return f"{settings.goal:g} m in {finish_time:.2f} s"
    return f"dash {dash:5.2f} s, {data.qpos[0] - start:+.2f} m"


def record_video(
    scene: Scene,
    views: Sequence[str],
    duration: float,
    caption: Callable[[mujoco.MjData], str],
    recorder: VideoRecorder,
) -> None:
    """
    Film the scene with several tracking cameras, side by side.

    The simulation is reset and stepped in real time; the control callback
    must be set.

    Parameters
    ----------
    scene
        The compiled scene and its data.
    views
        Keys of :data:`VIDEO_VIEWS`, left to right.
    duration
        Simulated seconds to film, from the reset.
    caption
        Text for the bottom of each frame.
    recorder
        Writes the frames; its width must fit ``len(views)`` tiles.
    """
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    core = find_core_body(model)
    cameras = [tracking_camera(core, view) for view in views]
    steps = max(1, round(1 / (recorder.fps * model.opt.timestep)))
    with mujoco.Renderer(model, height=TILE[0], width=TILE[1]) as renderer:
        while data.time < duration:
            mujoco.mj_step(model, data, nstep=steps)
            tiles = []
            for view, camera in zip(views, cameras, strict=True):
                renderer.update_scene(data, camera=camera)
                tile = renderer.render()
                put_text(tile, VIDEO_VIEWS[view][3], (12, 28))
                tiles.append(tile)
            frame = np.hstack(tiles)
            put_text(frame, caption(data), (12, TILE[0] - 16))
            recorder.write(frame)
    recorder.release()


class NamedRecorder(VideoRecorder):
    """
    A :class:`VideoRecorder` that keeps its file name (no timestamp).

    It writes H.264, which browsers play (the gallery page), and falls back to
    MPEG-4 Part 2 where OpenCV has no H.264 encoder.
    """

    _add_timestamp_to_file_name = False
    _video_encoding = "avc1"
    _fallback_encoders = [("mp4v", ".mp4"), ("MJPG", ".avi")]  # noqa: RUF012


def film(
    robot: Robot,
    recorder: VideoRecorder,
    views: Sequence[str] = DEFAULT_VIEWS,
    seconds: float | None = None,
) -> RaceResult:
    """
    Film a robot's dash, as it was scored, with tracking cameras.

    The video shows the settling and then the dash until the robot finishes
    (plus one second) or the time limit, unless ``seconds`` says otherwise.

    Parameters
    ----------
    robot
        The body, gait and run settings.
    recorder
        Writes the frames; its width must fit ``len(views)`` tiles.
    views
        Keys of :data:`VIDEO_VIEWS`, left to right.
    seconds
        Seconds of the dash to film, or ``None`` for the default.

    Returns
    -------
    RaceResult
        The dash, recomputed from the stored gait.
    """
    settings = robot.settings
    replayed = replay_gait(robot.genome, robot.gait, settings)
    if seconds is None:
        finish = replayed.finish_time
        seconds = (
            settings.time_limit if finish is None else finish + EXTRA_SECONDS
        )
    scene = build_scene(robot.genome, settings)
    mujoco.set_mjcb_control(
        make_control(
            Gait.from_dict(robot.gait),
            settings,
            scene.model.opt.timestep,
        ),
    )
    try:
        record_video(
            scene,
            views,
            settings.settle + seconds,
            lambda data: race_clock(
                data,
                settings,
                scene.start,
                replayed.finish_time,
            ),
            recorder,
        )
    finally:
        mujoco.set_mjcb_control(None)
    return replayed


def best_robot_videos(cfg: RunConfig) -> Snapshot:
    """
    Return the run's video hook: the fastest robot of a generation.

    Each due generation (``cfg.video_every``, see :func:`sc_core.is_due`)
    gets ``<cfg.video_dir>/gen_<NNN>_uid<N>.mp4``, filmed from the top, the
    side and 45 degrees. When the fastest robot is the one filmed last time,
    that video is linked instead of filmed again.

    Parameters
    ----------
    cfg
        Run settings (output folder, cadence, task and servo settings).

    Returns
    -------
    Snapshot
        A function of the survivors (best first) and the generation.
    """
    settings = cfg.eval_settings()
    filmed: dict[int, Path] = {}

    def hook(survivors: list[Individual], generation: int) -> None:
        if not is_due(generation, cfg.video_every, cfg.gens):
            return
        best = survivors[0]
        uid = int(cast("int", best.tags["uid"]))
        stem = f"gen_{generation:03d}_uid{uid}"
        cfg.video_dir.mkdir(parents=True, exist_ok=True)
        if uid in filmed:
            target = cfg.video_dir / f"{stem}{filmed[uid].suffix}"
            try:
                os.link(filmed[uid], target)
            except OSError:
                shutil.copy2(filmed[uid], target)
            return
        robot = Robot(
            name=stem,
            header=f"generation {generation}, uid {uid}",
            genome=cast("dict[str, Any]", best.genotype)["morph"],
            gait=cast("dict[str, Any]", best.tags["gait"]),
            stored_speed=float(cast("float", best.tags["speed"])),
            settings=settings,
        )
        film(
            robot,
            NamedRecorder(
                file_name=stem,
                output_folder=cfg.video_dir,
                width=TILE[1] * len(DEFAULT_VIEWS),
                height=TILE[0],
                fps=FPS,
            ),
        )
        filmed[uid] = next(cfg.video_dir.glob(f"{stem}.*"))

    return hook


def main() -> None:
    """Show the chosen body in the viewer, or save a video of it."""
    args = parse_args()
    if args.db.name.endswith("_final_learning.json"):
        robot = robot_from_final_learning(args.db, args.uid)
    elif args.db.suffix == ".json":
        robot = robot_from_pruning(args.db)
    else:
        robot = robot_from_run(args.db, args.uid)
    settings = robot.settings
    replayed = replay_gait(robot.genome, robot.gait, settings)
    print(describe(robot, replayed))  # noqa: T201

    if args.video is not None:
        name = f"{robot.name}_{'-'.join(args.view)}"
        recorder = VideoRecorder(
            file_name=name,
            output_folder=args.video,
            width=TILE[1] * len(args.view),
            height=TILE[0],
            fps=FPS,
        )
        film(robot, recorder, args.view, args.seconds)
        print(f"Saved {name}_<time>.mp4 in {args.video}")  # noqa: T201
        return

    scene = build_scene(robot.genome, settings)
    mujoco.set_mjcb_control(
        make_control(
            Gait.from_dict(robot.gait),
            settings,
            scene.model.opt.timestep,
        ),
    )
    try:
        viewer.launch(scene.model, scene.data)
    finally:
        mujoco.set_mjcb_control(None)


if __name__ == "__main__":
    main()
