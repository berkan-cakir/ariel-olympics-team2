import argparse
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from ariel.ec import Archive, Individual
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from mujoco import viewer
from rich.console import Console

from controller import CONTROLLERS
from evolution import EvolutionConfig
from fitness import CONTROL_STEPS, GOAL, TIME_LIMIT, TIMESTEP, WORLDS, FitnessConfig, build_scene, fitness, load_body

console = Console()


CAMERAS = ((90.0, -15.0), (135.0, -30.0), (90.0, -89.0))
CAMERA_DISTANCE = 1.5
FRAME_SIZE = (640, 480)


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
    best = Archive(db).best_individual("max" if cfg.is_maximisation else "min")
    return cfg, best


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


def record_video(cfg: EvolutionConfig, genotype: np.ndarray, env_id: int, out_dir: Path, duration: float) -> None:
    model, data = build_simulation(cfg, genotype, env_id)
    cameras = tracking_cameras(model)
    width, height = FRAME_SIZE
    recorder = VideoRecorder(f"{cfg.body}_{cfg.controller.name}_env{env_id}", out_dir, width * len(cameras), height)
    steps_per_frame = max(1, round(1 / (recorder.fps * model.opt.timestep)))
    end = data.time + duration
    with mj.Renderer(model, width=width, height=height) as renderer:
        while data.time < end:
            mj.mj_step(model, data, nstep=steps_per_frame)
            frames = []
            for camera in cameras:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())
            recorder.write(np.hstack(frames))
    recorder.release()
    console.log(f"environment {env_id}: video with {len(cameras)} camera angles saved to {out_dir}")


def view(cfg: EvolutionConfig, genotype: np.ndarray, env_id: int) -> None:
    model, data = build_simulation(cfg, genotype, env_id)
    console.log(f"environment {env_id}: viewer open, close the window to continue")
    viewer.launch(model, data)


def main() -> None:
    args = parse_args()
    cfg, best = load_checkpoint(args.checkpoint)
    genotype = np.asarray(best.genotype)
    console.log(f"{cfg.body}, {cfg.controller}, {cfg.algorithm}: best individual born in generation "
                f"{best.time_of_birth}, fitness {best.fitness:.4f}")

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
