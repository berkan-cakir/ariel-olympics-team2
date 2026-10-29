import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

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
RUGGED_HEIGHT = 0.1  # m between the lowest and highest point of the rugged terrain (ARIEL's default is 0.5)
LINOLEUM_FRICTION = 0.3  # sliding friction of the slippery floor; every other surface keeps MuJoCo's default of 1
CARPET_FRICTION = 1.5
GRASS_SLIDE_DEGREES = 6.0
SCREEN_GLASS_FRICTION = 0.03 # TODO better measurements!!!!
PRINTED_PLASTIC_FRICTION = 0.35
SILICONE_RUBBER_FRICTION = 1.0
GRASS_FRICTION = float(np.tan(np.radians(GRASS_SLIDE_DEGREES)) * np.sqrt(PRINTED_PLASTIC_FRICTION / SCREEN_GLASS_FRICTION))
SILICONE_FRICTION = float(np.sqrt(PRINTED_PLASTIC_FRICTION * SILICONE_RUBBER_FRICTION))
BUMP_TYPES = ((0.02, 0.005), (0.025, 0.007), (0.03, 0.009))
BUMP_ROW = 0.04
BUMP_GAP = 0.006
BUMP_CELL = 0.004
BUMP_FLOOR = 7.0
TRACK_LENGTH = 1.0
TRACK_WIDTH = 1.5
MAT_THICKNESS = 0.003
MAT_GRASS_GAP = 0.004
GRASS_THICKNESS = 0.001
SLOPE_DEGREES = 10.0  # incline of the carpet funnel around the start
RAY_SPACING = 0.05  # m between the rays that find the ground under a robot before it spawns
UNSTABLE = (mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL, mj.mjtWarning.mjWARN_BADQACC)
LATERAL_PENALTY = 0.0  # m/s of race speed lost per m of mean sideways drift off the straight line, 0 = off
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
    # How per-environment speeds combine: "mean" is the (weighted) average, "min" is the slowest environment
    aggregate: Literal["mean", "min"] = "mean"
    lateral_penalty: float = LATERAL_PENALTY

    def __post_init__(self) -> None:
        if self.lateral_penalty < 0:
            msg = f"Lateral penalty must be non-negative, got {self.lateral_penalty}."
            raise ValueError(msg)
        if unknown := set(self.environments) - WORLDS.keys():
            msg = f"Unknown environments {sorted(unknown)}, choose from {sorted(WORLDS)}."
            raise ValueError(msg)
        if self.weights is not None and len(self.weights) != len(self.environments):
            msg = f"Got {len(self.weights)} weights for {len(self.environments)} environments."
            raise ValueError(msg)
        if self.aggregate == "min" and self.weights is not None:
            msg = "Weights only apply to aggregate='mean'."
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


def surface(world: BaseWorld, friction: float, rgba: tuple[float, float, float, float]) -> BaseWorld:
    # MuJoCo uses the larger friction of two touching geoms unless one has priority,
    # so the floor gets priority to make its own friction count against the robot's default of 1
    for geom in world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM):
        geom.friction = [friction, *geom.friction[1:]]
        geom.priority = 1
        geom.material = ""
        geom.rgba = rgba
    return world


def linoleum() -> BaseWorld:
    return surface(SimpleFlatWorld(), LINOLEUM_FRICTION, (0.78, 0.74, 0.62, 1.0))


def carpet() -> BaseWorld:
    return surface(SimpleFlatWorld(), CARPET_FRICTION, (0.45, 0.12, 0.14, 1.0))


def carpet_slope() -> BaseWorld:
    # A funnel rising SLOPE_DEGREES in every direction up to the edge 5 m out, so no way away from the start
    # is downhill: on a plain tilted floor a robot could score by sliding down or walking along the slope
    xy = np.linspace(-1, 1, TERRAIN_DIMS[0])
    funnel = np.minimum(np.hypot(*np.meshgrid(xy, xy)), 1)
    rise = 5 * np.tan(np.radians(SLOPE_DEGREES))
    return surface(terrain("carpet-slope-world", 2 * rise, funnel), CARPET_FRICTION, (0.45, 0.12, 0.14, 1.0))


def grass() -> BaseWorld:
    return surface(SimpleFlatWorld(), GRASS_FRICTION, (0.30, 0.55, 0.20, 1.0))


def pyramid(side: float, height: float, angle: float) -> np.ndarray:
    reach = int(np.ceil(side / np.sqrt(3) / BUMP_CELL))
    offsets = np.arange(-reach, reach + 1) * BUMP_CELL
    normals = angle + 2 * np.pi * np.arange(3) / 3
    distance = np.max(offsets[None, :, None] * np.cos(normals) + offsets[:, None, None] * np.sin(normals), axis=2)
    return height * np.clip(1 - 2 * np.sqrt(3) * distance / side, 0, None)


@cache
def bumps() -> np.ndarray:
    rng = np.random.default_rng(TERRAIN_SEED)
    n = round(BUMP_FLOOR / BUMP_CELL) + 1
    heightmap = np.zeros((n, n))
    for y in range(round(BUMP_ROW / 2 / BUMP_CELL), n, round(BUMP_ROW / BUMP_CELL)):
        x = -rng.uniform(0, BUMP_ROW)
        while x < BUMP_FLOOR:
            side, height = BUMP_TYPES[rng.integers(len(BUMP_TYPES))]
            bump = pyramid(side, height, rng.uniform(0, 2 * np.pi))
            r, c = len(bump) // 2, round((x + side / 2) / BUMP_CELL)
            if r <= c < n - r and y < n - r:
                rows, cols = slice(y - r, y + r + 1), slice(c - r, c + r + 1)
                heightmap[rows, cols] = np.maximum(heightmap[rows, cols], bump)
            x += side + BUMP_GAP
    return heightmap


def silicone_bumps() -> BaseWorld:
    heightmap = bumps()
    world = CompoundWorld(name="silicone-bumps-world", floor_size=(BUMP_FLOOR, BUMP_FLOOR, 2 * heightmap.max()), floor_heightmap=heightmap, checker_floor=False)
    return surface(world, SILICONE_FRICTION, (0.15, 0.15, 0.16, 1.0))


def grass_slope() -> BaseWorld:
    return surface(carpet_slope(), GRASS_FRICTION, (0.30, 0.55, 0.20, 1.0))


def slab(world: BaseWorld, name: str, heightmap: np.ndarray, size: list[float], pos: list[float], friction: float, rgba: tuple[float, float, float, float]) -> None:
    world.spec.add_hfield(name=name, size=size, nrow=heightmap.shape[0], ncol=heightmap.shape[1], userdata=heightmap.flatten().tolist())
    geom = world.spec.worldbody.add_geom(name=name, type=mj.mjtGeom.mjGEOM_HFIELD, hfieldname=name, pos=pos, priority=1, rgba=rgba)
    geom.friction = [friction, *geom.friction[1:]]


def track() -> BaseWorld:
    world = surface(SimpleFlatWorld(), LINOLEUM_FRICTION, (0.78, 0.74, 0.62, 1.0))
    mat = bumps()[: round(TRACK_LENGTH / BUMP_CELL) + 1, : round(TRACK_WIDTH / BUMP_CELL) + 1].T
    slab(world, "mat", mat, [TRACK_LENGTH / 2, TRACK_WIDTH / 2, mat.max(), MAT_THICKNESS], [TRACK_LENGTH / 2, 0, MAT_THICKNESS], SILICONE_FRICTION, (0.15, 0.15, 0.16, 1.0))
    rise = TRACK_LENGTH * np.tan(np.radians(SLOPE_DEGREES))
    ramp = np.array([[0.0, 0.0, 1.0, 1.0]] * 2)
    slab(world, "grass", ramp, [1.5 * TRACK_LENGTH, TRACK_WIDTH / 2, rise, GRASS_THICKNESS], [2.5 * TRACK_LENGTH + MAT_GRASS_GAP, 0, GRASS_THICKNESS], GRASS_FRICTION, (0.30, 0.55, 0.20, 1.0))
    return world


WORLDS: dict[int, Callable[[], BaseWorld]] = {
    0: SimpleFlatWorld,
    1: rugged,
    2: crater,
    3: amphitheatre,
    4: linoleum,
    5: carpet,
    6: carpet_slope,
    7: grass,
    8: silicone_bumps,
    9: grass_slope,
    10: track,
}


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


def race_end(distance: np.ndarray) -> int:
    # The step where the robot crosses the finish line, or reaches its furthest point if it never does
    crossed = np.flatnonzero(distance >= GOAL)
    return int(crossed[0]) if crossed.size else int(distance.argmax())


def race_speed(path: np.ndarray) -> float:
    # Average speed over the race: GOAL / finish time for a finisher, else furthest distance / race time.
    # Both are GOAL / race time at the finish line, so every finisher beats every non-finisher.
    distance = xy_displacement(path[0], path.T)
    if distance.max() < GOAL:
        return float(distance.max() / ((len(path) - 1) * TIMESTEP))
    step = race_end(distance)
    before, after = distance[step - 1], distance[step]
    return float(GOAL / ((step - 1 + (GOAL - before) / (after - before)) * TIMESTEP))


def lateral_drift(path: np.ndarray) -> float:
    # Mean sideways distance (m) of the path from the straight line between the start and the race end,
    # so a robot that weaves or curves towards its end point scores worse than one that walks straight.
    xy = path[:, :2] - path[0, :2]
    end = race_end(np.linalg.norm(xy, axis=1))
    span = np.linalg.norm(xy[end])
    if span == 0:
        return 0.0
    heading = xy[end] / span
    return float(np.abs(xy[: end + 1] @ np.array([-heading[1], heading[0]])).mean())


def score(path: np.ndarray, lateral_penalty: float) -> float:
    # Floored at 0, the score of an unstable simulation, so drifting is never worse than blowing up
    return max(0.0, race_speed(path) - lateral_penalty * lateral_drift(path))


def fitness(genotype: np.ndarray, body: str, controller: BrainConfig, cfg: FitnessConfig) -> float:
    brain_type = CONTROLLERS[controller.name]
    scores = []
    for env_id in cfg.environments:
        robot = load_body(body)
        brain = brain_type(genotype, len(robot.spec.actuators), CONTROL_STEPS * TIMESTEP, controller)
        path = simulate(brain, robot, env_id)
        scores.append(0.0 if path is None else score(path, cfg.lateral_penalty))
    return AGGREGATES[cfg.aggregate](scores, cfg.weights)


def mean_fitness(scores: list[float], weights: tuple[float, ...] | None) -> float:
    # Original fitness: (weighted) average speed over the environments
    return float(np.average(scores, weights=weights))


def min_fitness(scores: list[float], weights: tuple[float, ...] | None) -> float:  # noqa: ARG001
    # fitness(i) = min over environments of speed(i): the robot is only as good as its worst terrain
    return float(np.min(scores))


AGGREGATES: dict[str, Callable[[list[float], tuple[float, ...] | None], float]] = {
    "mean": mean_fitness,
    "min": min_fitness,
}
