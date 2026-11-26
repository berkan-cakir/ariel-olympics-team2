import argparse
import itertools
import json
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np
from ariel.ec import Archive, Individual
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from mujoco import viewer
from rich.console import Console
from sqlmodel import create_engine

from controller import CONTROLLERS
from evolution import EvolutionConfig, best_combined
from fitness import CONTROL_STEPS, GOAL, TIME_LIMIT, TIMESTEP, WORLDS, FitnessConfig, build_scene, fitness, load_body, make_world

console = Console()


CAMERAS = ((90.0, -15.0), (135.0, -30.0), (90.0, -89.0))
CAMERA_DISTANCE = 1.5
OVERVIEW = (90.0, -55.0, 8.0)
FRAME_SIZE = (640, 480)
LINE_WIDTH = 0.012
START_RADIUS = 0.15
RING_SEGMENTS = 240
START_RGBA = (1.0, 1.0, 1.0, 1.0)
FINISH_RGBA = (0.1, 0.85, 0.35, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path, help="database.db written by the EA, or the run folder containing it")
    p.add_argument("--environments", type=int, nargs="+", choices=sorted(WORLDS), default=None,
                   help="default: the environments the checkpoint was trained on")
    p.add_argument("--video", type=Path, default=None, help="output folder; enables recording")
    p.add_argument("--duration", type=float, default=TIME_LIMIT, help="video length in seconds (default: the whole race)")
    p.add_argument("--no-viewer", action="store_true", help="do not open the interactive MuJoCo window")
    return p.parse_args()


def load_checkpoint(path: Path) -> tuple[EvolutionConfig, Individual]:
    db = path / "database.db" if path.is_dir() else path
    settings = json.loads((db.parent / "config.json").read_text())
    settings.pop("db_file_path")  # computed from output_folder, not a setting
    cfg = EvolutionConfig.model_validate(settings)
    if not cfg.islands:
        return cfg, Archive(db).best_individual("max" if cfg.is_maximisation else "min")
    return cfg, best_combined(create_engine(f"sqlite:///{db}"))


def build_simulation(cfg: EvolutionConfig, genotype: np.ndarray, env_id: int) -> tuple[mj.MjModel, mj.MjData]:
    world, model, data = build_scene(load_body(cfg.body), env_id)
    brain = CONTROLLERS[cfg.controller.name](genotype, model.nu, CONTROL_STEPS * TIMESTEP, cfg.controller)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
    controller = Controller(brain, time_steps_per_ctrl_step=CONTROL_STEPS, tracker=tracker)
    controller.tracker.setup(world.spec, data)
    mj.set_mjcb_control(controller.set_control)
    return model, data


def tracking_cameras(model: mj.MjModel) -> list[mj.MjvCamera]:
    core = next(i for i in range(model.nbody) if "core" in model.body(i).name)
    cameras = []
    for azimuth, elevation in CAMERAS:
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = core
        camera.distance, camera.azimuth, camera.elevation = CAMERA_DISTANCE, azimuth, elevation
        cameras.append(camera)
    return cameras


def overview_camera(lookat: np.ndarray) -> mj.MjvCamera:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.azimuth, camera.elevation, camera.distance = OVERVIEW
    return camera


def circle(center: np.ndarray, radius: float) -> np.ndarray:
    angles = np.linspace(0.0, 2 * np.pi, RING_SEGMENTS + 1)
    return center + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def on_ground(env_id: int, xy: np.ndarray) -> np.ndarray:
    callback = mj.get_mjcb_control()
    mj.set_mjcb_control(None)
    model = make_world(env_id).spec.compile()
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    mj.set_mjcb_control(callback)
    top, down, geom = 10.0, np.array([0.0, 0.0, -1.0]), np.zeros(1, dtype=np.int32)
    z = [top - mj.mj_ray(model, data, np.array([x, y, top]), down, None, 1, -1, geom) for x, y in xy]
    return np.column_stack([xy, np.array(z) + LINE_WIDTH])


def draw_line(scene: mj.MjvScene, points: np.ndarray, rgba: tuple[float, ...]) -> None:
    for start, end in itertools.pairwise(points):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, dtype=np.float32))
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, LINE_WIDTH, start, end)
        scene.ngeom += 1


def caption(frame: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        position = (16, 34 + 32 * i)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def record_video(cfg: EvolutionConfig, genotype: np.ndarray, env_id: int, out_dir: Path, duration: float) -> None:
    model, data = build_simulation(cfg, genotype, env_id)
    core = next(i for i in range(model.ngeom) if "core" in model.geom(i).name)
    start = data.geom_xpos[core, :2].copy()
    start_ring, finish_ring = on_ground(env_id, circle(start, START_RADIUS)), on_ground(env_id, circle(start, GOAL))
    cameras = [*tracking_cameras(model), overview_camera(np.array([*start, finish_ring[:, 2].mean() / 2]))]
    width, height = FRAME_SIZE
    recorder = VideoRecorder(f"{cfg.body}_{cfg.controller.name}_env{env_id}", out_dir, 2 * width, 2 * height)
    steps_per_frame = max(1, round(1 / (recorder.fps * model.opt.timestep)))
    begin, finished = data.time, None
    with mj.Renderer(model, width=width, height=height) as renderer:
        while data.time < begin + duration:
            mj.mj_step(model, data, nstep=steps_per_frame)
            elapsed = data.time - begin
            distance = float(np.linalg.norm(data.geom_xpos[core, :2] - start))
            if finished is None and distance >= GOAL:
                finished = elapsed
            frames = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                draw_line(renderer.scene, start_ring, START_RGBA)
                draw_line(renderer.scene, finish_ring, FINISH_RGBA)
                frames.append(renderer.render())
            status = f"finished: {GOAL:g} m in {finished:.1f} s" if finished is not None else ""
            caption(frames[-1], [f"time {elapsed:4.1f} s   distance {distance:.2f} / {GOAL:g} m", status])
            recorder.write(np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])]))
    recorder.release()
    result = f"reached {GOAL:g} m after {finished:.1f} s" if finished is not None else f"ended {distance:.2f} m from the start"
    console.log(f"environment {env_id}: {result}; video with {len(cameras)} camera angles saved to {out_dir}")


def view(cfg: EvolutionConfig, genotype: np.ndarray, env_id: int) -> None:
    model, data = build_simulation(cfg, genotype, env_id)
    console.log(f"environment {env_id}: viewer open, close the window to continue")
    viewer.launch(model, data)


def main() -> None:
    args = parse_args()
    cfg, best = load_checkpoint(args.checkpoint)
    genotype = np.asarray(best.genotype)
    value = best.tags["combined"] if cfg.islands else best.fitness
    console.log(f"{cfg.body}, {cfg.controller}, {cfg.algorithm}: best individual born in generation "
                f"{best.time_of_birth}, fitness {value:.4f}")

    environments = tuple(args.environments or cfg.fitness.environments)
    for env_id in environments:
        score = fitness(genotype, cfg.body, cfg.controller, FitnessConfig((env_id,)))
        reached = "yes" if score >= GOAL / TIME_LIMIT else "no"
        console.log(f"environment {env_id}: race speed {score:.4f} m/s, reached {GOAL:g} m: {reached}")

    for env_id in environments:
        if args.video:
            record_video(cfg, genotype, env_id, args.video, args.duration)
        if not args.no_viewer:
            view(cfg, genotype, env_id)


if __name__ == "__main__":
    main()
