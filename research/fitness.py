import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mujoco as mj
import numpy as np
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders import load_graph_from_json
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.ec.genotypes.tree import load_genome
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import BaseWorld, CompoundWorld, SimpleFlatWorld
from ariel.simulation.environments.heightmap_functions import (
    amphitheater_heightmap,
    crater_heightmap,
    smooth_edges_heightmap,
)
from ariel.simulation.tasks.gait_learning import xy_displacement
from ariel.utils.noise_gen import PerlinNoise
from ariel.utils.runners import thread_safe_runner
from ariel.utils.tracker import Tracker

from controller import CONTROLLERS, Brain, BrainConfig

GOAL = 3.0  # m, straight-line distance from the start in any direction
TIME_LIMIT = 30.0  # s of control to cover GOAL
SETTLE = 1.0  # s without control after spawning, so the robot has landed when the clock starts
TIMESTEP = 0.002  # s, MuJoCo's default, which ARIEL's worlds keep
CONTROL_STEPS = 50  # physics steps per brain call, as in ARIEL's Controller
TERRAIN_SEED = 42
TERRAIN_DIMS = (100, 100)
RUGGED_HEIGHT = 0.15  # m between the lowest and highest point of the rugged terrain (ARIEL's default is 0.5)
RAY_SPACING = 0.05  # m between the rays that find the ground under a robot before it spawns
UNSTABLE = (mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL, mj.mjtWarning.mjWARN_BADQACC)
BUILDER_FUNCTION = "custom_robot"  # the function ARIEL's 3D robot builder exports

PREBUILT_BODIES: dict[str, Callable[[], CoreModule]] = {
    "gecko": gecko,
    "spider": spider,
    "snake": john_set.snake,
    "baby_a": john_set.baby_a,
    "baby_b": john_set.baby_b,
    "centipede_3": john_set.centipede_3,
    "centipede_4": john_set.centipede_4,
    "centipede_5": john_set.centipede_5,
    "iguana": john_set.iguana,
    "linkin_modified": john_set.linkin_modified,
    "spider_8": john_set.spider_8,
    "spider_12": john_set.spider_12,
    "spider_16": john_set.spider_16,
    "turtle": john_set.turtle,
}


@dataclass
class FitnessConfig:
    environments: tuple[int, ...] = (0,)
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if unknown := set(self.environments) - WORLDS.keys():
            msg = f"Unknown environments {sorted(unknown)}, choose from {sorted(WORLDS)}."
            raise ValueError(msg)
        if self.weights is not None and len(self.weights) != len(self.environments):
            msg = f"Got {len(self.weights)} weights for {len(self.environments)} environments."
            raise ValueError(msg)


def load_body(source: str) -> CoreModule:
    if source in PREBUILT_BODIES:
        return PREBUILT_BODIES[source]()
    path = Path(source)
    if not path.is_file():
        msg = f"Body {source!r} is neither an ARIEL prebuilt ({', '.join(PREBUILT_BODIES)}) nor a file."
        raise FileNotFoundError(msg)
    match path.suffix:
        case ".py":
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, BUILDER_FUNCTION)()
        case ".json":
            # networkx node-link data (decoders.save_graph_as_json) or a tree genome (TreeGenome.save_json)
            if "directed" in json.loads(path.read_text("utf-8")):
                return construct_mjspec_from_graph(load_graph_from_json(path))
            return construct_mjspec_from_graph(load_genome(str(path)).to_networkx())
    msg = f"Body file {path} must be .py (ARIEL robot builder export) or .json (ARIEL body graph or tree genome)."
    raise ValueError(msg)


@cache
def noise(scale: int) -> np.ndarray:
    # ARIEL's rugged_heightmap, but seeded: it draws a new terrain every time it is called
    return PerlinNoise(seed=TERRAIN_SEED).as_grid(*TERRAIN_DIMS, scale=scale, normalize="none")


def terrain(name: str, depth: float, heightmap: np.ndarray) -> CompoundWorld:
    return CompoundWorld(name=name, floor_size=(10, 10, depth), floor_heightmap=heightmap, checker_floor=False)


# ARIEL's RuggedTerrainWorld, CraterTerrainWorld and AmphitheatreTerrainWorld with their default parameters,
# rebuilt from the same heightmaps with a fixed seed so that every evaluation sees the same ground.
def rugged() -> CompoundWorld:
    # MuJoCo rescales a heightfield to [0, depth / 2], so the depth alone sets how rough the terrain is
    return terrain("rugged-world", 2 * RUGGED_HEIGHT, noise(4) * smooth_edges_heightmap(TERRAIN_DIMS, edge_width=0.2))


def crater() -> CompoundWorld:
    hole = crater_heightmap(TERRAIN_DIMS, crater_depth=1.0, crater_radius=0.3)
    return terrain("crater-world", 2, hole + 0.3 * noise(5))


def amphitheatre() -> CompoundWorld:
    rings = amphitheater_heightmap(TERRAIN_DIMS, ring_inner_radius=0.2, ring_outer_radius=0.45, cone_height=2)
    return terrain("amphitheatre-world", 2, rings + 0.3 * noise(5))


WORLDS: dict[int, Callable[[], BaseWorld]] = {0: SimpleFlatWorld, 1: rugged, 2: crater, 3: amphitheatre}


def make_world(env_id: int) -> BaseWorld:
    world = WORLDS[env_id]()
    world.spec.option.timestep = TIMESTEP
    return world


def reach(body: CoreModule) -> float:
    model = body.spec.compile()
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    return float(np.max(np.linalg.norm(data.geom_xpos[:, :2], axis=1) + np.linalg.norm(model.geom_size, axis=1)))


@cache
def ground_height(env_id: int, radius: float) -> float:
    model = make_world(env_id).spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    top, down, geom = 10.0, np.array([0.0, 0.0, -1.0]), np.zeros(1, dtype=np.int32)
    grid = np.arange(-radius, radius + RAY_SPACING, RAY_SPACING)
    distances = [
        mj.mj_ray(model, data, np.array([x, y, top]), down, None, 1, -1, geom)
        for x in grid
        for y in grid
        if x**2 + y**2 <= radius**2
    ]
    return top - min(d for d in distances if d >= 0)


def build_scene(body: CoreModule, env_id: int) -> tuple[BaseWorld, mj.MjModel, mj.MjData]:
    mj.set_mjcb_control(None)
    world = make_world(env_id)
    # ARIEL puts the robot's lowest point 1 cm above the spawn height, taking that to be the ground.
    # On a terrain the ground is higher, so spawn at the highest ground under the robot.
    world.spawn(body.spec, position=(0.0, 0.0, ground_height(env_id, reach(body))))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_step(model, data, nstep=round(SETTLE / TIMESTEP))
    return world, model, data


def simulate(brain: Brain, body: CoreModule, env_id: int) -> np.ndarray | None:
    world, model, data = build_scene(body, env_id)
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core", ["xpos"])
    tracker.setup(world.spec, data)
    controller = Controller(brain, time_steps_per_ctrl_step=CONTROL_STEPS, time_steps_per_save=1, tracker=tracker)
    thread_safe_runner(model, data, controller, duration=TIME_LIMIT)
    # MuJoCo resets a simulation that blows up, which teleports the robot back to its spawn
    if any(data.warning[warning].number for warning in UNSTABLE):
        return None
    return np.array(tracker.history["xpos"][0])


def race_speed(path: np.ndarray) -> float:
    # Average speed over the race: GOAL / finish time for a finisher, else furthest distance / race time.
    # Both are GOAL / race time at the finish line, so every finisher beats every non-finisher.
    distance = xy_displacement(path[0], path.T)
    crossed = np.flatnonzero(distance >= GOAL)
    if crossed.size == 0:
        return float(distance.max() / ((len(path) - 1) * TIMESTEP))
    step = crossed[0]
    before, after = distance[step - 1], distance[step]
    return float(GOAL / ((step - 1 + (GOAL - before) / (after - before)) * TIMESTEP))


def fitness(genotype: np.ndarray, body: str, controller: BrainConfig, cfg: FitnessConfig) -> float:
    brain_type = CONTROLLERS[controller.name]
    scores = []
    for env_id in cfg.environments:
        robot = load_body(body)
        brain = brain_type(genotype, len(robot.spec.actuators), CONTROL_STEPS * TIMESTEP, controller)
        path = simulate(brain, robot, env_id)
        scores.append(0.0 if path is None else race_speed(path))
    return float(np.average(scores, weights=cfg.weights))
