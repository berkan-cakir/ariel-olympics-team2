"""
Fitness: how well does a brain make a robot walk?

The task: a robot gets TIME_LIMIT (30) seconds to get GOAL (3) metres away from where it started, in a
straight line in any direction. It does this in up to four worlds, called environments:

    0  flat floor
    1  rugged: hilly ground
    2  crater: the robot starts at the bottom of a bowl and has to climb out
    3  amphitheatre: a fairly flat middle with steep slopes all around

Scoring one genotype (``fitness``) goes like this, for every environment:

    1. build the robot's body                          load_body
    2. build the world                                 make_world
    3. drop the robot in and let it land               build_scene
    4. let the brain drive it and record its path      simulate
    5. turn the path into a score                      race_speed

and then the scores of the environments are averaged.

The score is the robot's average speed (m/s) over the race: GOAL / the time it took if it got GOAL metres
away, otherwise the furthest it got / the time it had. Higher is better.
"""

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mujoco as mj
import numpy as np

# ARIEL: robot bodies
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders import load_graph_from_json
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.ec.genotypes.tree import load_genome

# ARIEL: worlds
from ariel.simulation.environments import BaseWorld, CompoundWorld, SimpleFlatWorld
from ariel.simulation.environments.heightmap_functions import (
    amphitheater_heightmap,
    crater_heightmap,
    smooth_edges_heightmap,
)
from ariel.utils.noise_gen import PerlinNoise

# ARIEL: running a simulation and measuring the result
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.tasks.gait_learning import xy_displacement
from ariel.utils.runners import thread_safe_runner
from ariel.utils.tracker import Tracker

from controller import CONTROLLERS, Brain, BrainConfig

# --- The task ---
GOAL = 3.0  # metres the robot has to get away from its start (straight line, any direction)
TIME_LIMIT = 30.0  # seconds the brain gets to do that

# --- The simulation ---
TIMESTEP = 0.002  # seconds per physics step (MuJoCo's default, which ARIEL's worlds use)
CONTROL_STEPS = 50  # physics steps between two calls to the brain (so 0.1 s), as in ARIEL's Controller
SETTLE = 1.0  # seconds the robot gets to land after it is dropped in, before the race starts
# MuJoCo warnings that mean the physics blew up, so the run cannot be trusted
UNSTABLE = (mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL, mj.mjtWarning.mjWARN_BADQACC)

# --- The terrains ---
TERRAIN_SEED = 42  # random seed for the bumps, so that every evaluation gets the same terrain
TERRAIN_DIMS = (100, 100)  # heightmap resolution: 100 x 100 heights over the 10 m x 10 m world
RAY_SPACING = 0.05  # metres between the rays that measure the ground under a robot before it is dropped in

# Robots that come with ARIEL, by the name you pass to --bodies
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
    """
    Which environments to test a brain in, and how much each one counts.

    environments: ids of the worlds, see WORLDS below (0 flat, 1 rugged, 2 crater, 3 amphitheatre)
    weights: one weight per environment for the average score; None means they all count the same
    """

    environments: tuple[int, ...] = (0,)
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        # Check the settings right away, so that a typo fails before a long run starts
        unknown = set(self.environments) - set(WORLDS)
        if unknown:
            raise ValueError(f"Unknown environments {sorted(unknown)}, choose from {sorted(WORLDS)}.")
        if self.weights is not None and len(self.weights) != len(self.environments):
            raise ValueError(f"Got {len(self.weights)} weights for {len(self.environments)} environments.")


# --- Bodies ---


def load_body(source: str) -> CoreModule:
    """
    Build a robot body.

    ``source`` is either the name of a robot that comes with ARIEL (a key of PREBUILT_BODIES),
    or the path of a body file:
        .py    a Python file with a ``custom_robot()`` function, as ARIEL's 3D robot builder exports it
        .json  a body that ARIEL saved as JSON, see ``load_json_body``
    """
    if source in PREBUILT_BODIES:
        make_body = PREBUILT_BODIES[source]
        return make_body()

    path = Path(source)
    if not path.is_file():
        names = ", ".join(PREBUILT_BODIES)
        raise FileNotFoundError(f"Body {source!r} is neither an ARIEL robot ({names}) nor a file.")
    if path.suffix == ".py":
        return load_python_body(path)
    if path.suffix == ".json":
        return load_json_body(path)
    raise ValueError(f"Body file {path} must be a .py or a .json file.")


def load_python_body(path: Path) -> CoreModule:
    """Run a Python file and return the body that its ``custom_robot()`` function builds."""
    # This is what `import` does, but for a file anywhere on disk
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.custom_robot()


def load_json_body(path: Path) -> CoreModule:
    """
    Build a body from a JSON file saved by ARIEL.

    ARIEL saves a body as a graph of modules (which module is attached to which face of which other one),
    in one of two formats: a networkx graph (``save_graph_as_json``, which has a "directed" key) or a tree
    genome (``TreeGenome.save_json``). Both are turned into a networkx graph, and ARIEL builds the body
    from that.
    """
    data = json.loads(path.read_text("utf-8"))
    if "directed" in data:
        graph = load_graph_from_json(path)
    else:
        graph = load_genome(str(path)).to_networkx()
    return construct_mjspec_from_graph(graph)


# --- Worlds ---


@cache  # remember the result: the same scale always gives the same noise
def seeded_noise(scale: int) -> np.ndarray:
    """
    Random bumps (Perlin noise) for a terrain, the same every time.

    This is what ARIEL's ``rugged_heightmap`` makes, except that ARIEL picks a new random seed every time,
    so every evaluation would get a different terrain. A fixed seed keeps the scores comparable.
    """
    return PerlinNoise(seed=TERRAIN_SEED).as_grid(*TERRAIN_DIMS, scale=scale, normalize="none")


def make_terrain(name: str, depth: float, heightmap: np.ndarray) -> CompoundWorld:
    """A 10 m x 10 m world whose ground follows ``heightmap``; ``depth`` sets how high the ground can get."""
    return CompoundWorld(name=name, floor_size=(10, 10, depth), floor_heightmap=heightmap, checker_floor=False)


# The three terrains below are ARIEL's RuggedTerrainWorld, CraterTerrainWorld and AmphitheatreTerrainWorld
# with the same settings, but built with seeded_noise so that they are the same every time.


def rugged() -> CompoundWorld:
    """Hilly ground that flattens out towards the edges of the world."""
    hills = seeded_noise(4)
    flat_edges = smooth_edges_heightmap(TERRAIN_DIMS, edge_width=0.2)
    return make_terrain("rugged-world", 1, hills * flat_edges)


def crater() -> CompoundWorld:
    """A bumpy bowl: the robot starts at the bottom and has to climb out."""
    bowl = crater_heightmap(TERRAIN_DIMS, crater_depth=1.0, crater_radius=0.3)
    return make_terrain("crater-world", 2, bowl + 0.3 * seeded_noise(5))


def amphitheatre() -> CompoundWorld:
    """A bumpy, fairly flat middle with steep slopes all around, starting about 2 m out."""
    rings = amphitheater_heightmap(TERRAIN_DIMS, ring_inner_radius=0.2, ring_outer_radius=0.45, cone_height=2)
    return make_terrain("amphitheatre-world", 2, rings + 0.3 * seeded_noise(5))


# Environment id -> the function that builds that world
WORLDS: dict[int, Callable[[], BaseWorld]] = {
    0: SimpleFlatWorld,
    1: rugged,
    2: crater,
    3: amphitheatre,
}


def make_world(env_id: int) -> BaseWorld:
    """Build world ``env_id`` (see WORLDS)."""
    world = WORLDS[env_id]()
    # The brain's timing assumes this step size. ARIEL's worlds already use it; this makes sure.
    world.spec.option.timestep = TIMESTEP
    return world


# --- Putting the robot in the world ---


def body_radius(body: CoreModule) -> float:
    """How far the body reaches from its core, seen from above (m), so a circle of this radius covers it."""
    model = body.spec.compile()  # turn ARIEL's description of the body into a MuJoCo model
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)  # work out where every part of the body is
    part_centres = np.linalg.norm(data.geom_xpos[:, :2], axis=1)  # horizontal distance of each part's centre
    part_sizes = np.linalg.norm(model.geom_size, axis=1)  # how far each part sticks out past its centre
    return float(np.max(part_centres + part_sizes))


@cache  # remember the result: the ground never changes
def highest_ground(env_id: int, radius: float) -> float:
    """
    Height (m) of the highest ground within ``radius`` metres of the spawn point (0, 0) in world ``env_id``.

    It drops rays straight down onto the empty world, every RAY_SPACING metres, and keeps the highest hit.
    """
    model = make_world(env_id).spec.compile()
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    start_height = 10.0  # above every terrain
    down = np.array([0.0, 0.0, -1.0])
    hit_geom = np.zeros(1, dtype=np.int32)  # mj_ray writes the id of the part it hit in here
    steps = np.arange(-radius, radius + RAY_SPACING, RAY_SPACING)
    highest = -np.inf
    for x in steps:
        for y in steps:
            if x**2 + y**2 > radius**2:
                continue  # outside the circle
            start = np.array([x, y, start_height])
            # flg_static=1 includes the ground; geomgroup=None and bodyexclude=-1 mean "skip nothing"
            distance = mj.mj_ray(
                model, data, pnt=start, vec=down, geomgroup=None, flg_static=1, bodyexclude=-1, geomid=hit_geom
            )
            if distance >= 0:  # -1 means that the ray hit nothing
                highest = max(highest, start_height - distance)
    return highest


def build_scene(body: CoreModule, env_id: int) -> tuple[BaseWorld, mj.MjModel, mj.MjData]:
    """
    Put ``body`` in world ``env_id`` at (0, 0), let it land, and return the world, model and data.

    ARIEL's ``spawn`` puts the robot's lowest point 1 cm above the height we give it, which it assumes is the
    ground. On a terrain the ground can be higher than 0, so we give it the height of the highest ground
    under the robot; otherwise the robot would start partly inside a hill.
    """
    mj.set_mjcb_control(None)  # MuJoCo can hold one global controller; remove any that other code left behind

    world = make_world(env_id)
    ground = highest_ground(env_id, body_radius(body))
    world.spawn(body.spec, position=(0.0, 0.0, ground))

    model = world.spec.compile()  # the finished world, ready to simulate
    data = mj.MjData(model)  # everything that changes while it runs: positions, speeds, time, ...

    # Let the robot fall and settle for SETTLE seconds, with its motors holding still, before the race starts
    mj.mj_step(model, data, nstep=round(SETTLE / TIMESTEP))
    return world, model, data


# --- Running and scoring the race ---


def simulate(brain: Brain, body: CoreModule, env_id: int) -> np.ndarray | None:
    """
    Let ``brain`` drive ``body`` in world ``env_id`` for TIME_LIMIT seconds and record the robot's path.

    Returns an array with one row (x, y, z) per physics step: where the robot's core was, starting where it
    landed, TIMESTEP seconds apart. Returns None if the physics became unstable.
    """
    world, model, data = build_scene(body, env_id)

    # The tracker records the position ("xpos") of the robot's core
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core", ["xpos"])
    tracker.setup(world.spec, data)

    # ARIEL's Controller asks the brain for new motor targets every CONTROL_STEPS steps,
    # and has the tracker record the position at every step (time_steps_per_save=1)
    controller = Controller(brain, time_steps_per_ctrl_step=CONTROL_STEPS, time_steps_per_save=1, tracker=tracker)
    thread_safe_runner(model, data, controller, duration=TIME_LIMIT)

    # When the physics blows up, MuJoCo warns and puts the robot back at its start, so the path is useless
    for warning in UNSTABLE:
        if data.warning[warning].number > 0:
            return None

    core_path = tracker.history["xpos"][0]  # the history of the first (and only) part that the tracker follows
    return np.array(core_path)


def race_speed(path: np.ndarray) -> float:
    """
    Score a path: the robot's average speed (m/s) over a race of GOAL metres.

    A robot that got GOAL metres away from its start (in a straight line, any direction) scores
    GOAL / the time it took. A robot that did not scores the furthest it got / the time it had.
    Both give the same number right at the finish line, so every robot that finishes scores higher
    than every robot that does not.
    """
    xs, ys = path[:, 0], path[:, 1]
    # Straight-line distance from the start at every step (ARIEL's formula, for all steps at once)
    distance = xy_displacement((xs[0], ys[0]), (xs, ys))
    race_time = (len(path) - 1) * TIMESTEP

    finished = np.flatnonzero(distance >= GOAL)  # the steps at which the robot was past the finish
    if len(finished) == 0:
        return float(distance.max() / race_time)

    # Between step `first - 1` (short of the finish) and step `first` (past it) the robot crossed the line.
    # Work out when exactly, assuming that it moved at a steady speed during that one step.
    first = finished[0]
    before, after = distance[first - 1], distance[first]
    fraction = (GOAL - before) / (after - before)
    finish_time = (first - 1 + fraction) * TIMESTEP
    return float(GOAL / finish_time)


def fitness(genotype: np.ndarray, body: str, controller: BrainConfig, cfg: FitnessConfig) -> float:
    """
    Score one genotype: race in every environment of ``cfg`` and average the scores. Higher is better.

    genotype: the brain's parameters, as evolved
    body: which robot to use, see ``load_body``
    controller: which kind of brain the genotype is for, see controller.py
    """
    brain_type = CONTROLLERS[controller.name]
    control_period = CONTROL_STEPS * TIMESTEP  # seconds between two calls to the brain

    scores = []
    for env_id in cfg.environments:
        # A new body and a new brain for every race, so that nothing carries over from the previous one
        # (a CPG brain, for example, remembers where it is in its rhythm)
        robot = load_body(body)
        motors = len(robot.spec.actuators)
        brain = brain_type(genotype, motors, control_period, controller)

        path = simulate(brain, robot, env_id)
        if path is None:  # the physics blew up: no score
            scores.append(0.0)
        else:
            scores.append(race_speed(path))

    # Weighted average; with weights=None every environment counts the same
    return float(np.average(scores, weights=cfg.weights))
