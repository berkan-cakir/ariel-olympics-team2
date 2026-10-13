"""
Shared machinery for the design-priors experiment.

One run co-optimizes robot bodies (tree genomes) and brains under one of four
conditions (Liu, Guo & Cangelosi 2025, §3.5.4: inject design knowledge into
body-brain co-optimization):

* ``baseline`` -- random initial bodies, no prior;
* ``seeded`` -- part of the initial population is ARIEL's hand-designed robots
  (:func:`ariel.ec.genotypes.tree.prebuilt.prebuilt_tree_genome`);
* ``symmetric`` -- every body is mirror-symmetric
  (:func:`ariel.ec.genotypes.tree.symmetry.symmetrize`);
* ``joint_budget`` -- every body has at most ``max_hinges`` hinges
  (:func:`ariel.ec.genotypes.tree.constraints.cap_hinges`).

Every body gets a brain (a small MLP, same layout as the template's torch
network) learned from scratch by (diagonal) CMA-ES with population
``learn_pop`` for ``learn_budget`` generations, a fixed cost of ``learn_budget
* learn_pop`` simulations, so all conditions spend the same compute per body.

Template: ``examples/c_genotypes/4_body_brain_joint_evolution_tree_nn_learning
_multiprocessing.py``. Differences: spawn yaw (the robot's FRONT faces the
target), a CMA population equal to the batch size (the template's
``registry["CMA"]`` silently uses a larger one), per-body learner seeds, a
numpy brain, and a warm-up that is simulated once per body and restored for
every candidate.

Notes
-----
    * Do not add ``from __future__ import annotations``: the EA inspects the
      annotations of its operations.
"""

# Standard library
import copy
import json
import math
import os
import platform
import random
import subprocess  # noqa: S404 -- only for `git rev-parse HEAD`
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import mujoco
import nevergrad as ng
import numpy as np
from nevergrad.optimization.base import Optimizer

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.constraints import (
    cap_hinges,
    count_hinges,
    count_modules,
)
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    get_tree_depth,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.prebuilt import (
    PREBUILT_TREE_NAMES,
    prebuilt_tree_genome,
)
from ariel.ec.genotypes.tree.symmetry import (
    genome_symmetry,
    subtree_signature,
    symmetrize,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.parameters.ariel_types import FloatArray
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld

# The TPA step-size rule warns on noisy objectives; it is harmless here
# (see the template and wiki/CMA-ES_Mirrored_Sampling.md).
warnings.filterwarnings(
    "ignore",
    message="TPA: apparent inconsistency",
    category=UserWarning,
    module="cma",
)

# Global constants
CONDITIONS = ("baseline", "seeded", "symmetric", "joint_budget")
SPAWN_POSITION = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.1])
HIDDEN_SIZE = 16
CMA_SIGMA = 0.5
CROSSOVER_RATE = 0.6
ROTATION_MUTATION_RATE = 0.25
MUTATIONS = ("point", "subtree", "shrink", "hoist")
MUTATION_WEIGHTS = (0.45, 0.35, 0.10, 0.10)
MAX_INIT_TRIES = 500
REEVAL_SEED_OFFSET = 10_000
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
SIDES = ("LEFT", "RIGHT")


# -- Configuration ----------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """
    Settings of one run; the CLI flags of ``run_design_priors.py``.

    Parameters
    ----------
    condition
        One of :data:`CONDITIONS`.
    seed
        Run seed; every random choice in the run derives from it.
    out
        Output root; files go to ``out/<condition>/seed_<NN>*``.
    pop
        Population size (mu) and offspring per generation (lambda).
    gens
        Number of generations after the initial population.
    learn_budget
        CMA-ES iterations per body.
    learn_pop
        CMA-ES candidates per iteration (fixed for every body).
    duration
        Seconds of active control per simulation.
    eval_delay
        Seconds of no-control warm-up before scoring.
    z_penalty_weight
        Weight of the mean vertical deviation in the score.
    seed_fraction
        Share of the initial population taken from the prebuilt designs
        (``seeded`` only).
    max_hinges
        Hinge budget (``joint_budget`` only).
    max_modules
        Largest body, in modules, for every condition.
    max_depth
        Deepest tree allowed, for every condition.
    init_size_min
        Smallest random initial body; sizes are uniform in
        ``[init_size_min, max_modules]``.
    spawn_yaw
        Yaw (degrees) at spawn; -90 turns the robot's FRONT toward +x.
    retries
        Variation attempts per child before falling back to a parent clone.
    reeval_top
        Number of best bodies re-learned at the end of the run.
    reeval_seeds
        Fresh learning seeds per re-learned body.
    eval_workers
        Worker processes for body evaluation; 1 evaluates in-process.
    """

    condition: str
    seed: int
    out: Path
    pop: int = 16
    gens: int = 16
    learn_budget: int = 5
    learn_pop: int = 24
    duration: float = 3.0
    eval_delay: float = 2.0
    z_penalty_weight: float = 2.0
    seed_fraction: float = 0.5
    max_hinges: int = 8
    max_modules: int = 20
    max_depth: int = 12
    init_size_min: int = 8
    spawn_yaw: float = -90.0
    retries: int = 12
    reeval_top: int = 3
    reeval_seeds: int = 3
    eval_workers: int = 1

    def __post_init__(self) -> None:
        """
        Reject settings the experiment cannot run with.

        Raises
        ------
        ValueError
            If the condition is unknown or a setting is out of range.
        """
        if self.condition not in CONDITIONS:
            msg = f"Unknown condition {self.condition!r}; use {CONDITIONS}."
            raise ValueError(msg)
        if self.pop < 2 or self.gens < 0 or self.learn_budget < 1:
            msg = "Need pop >= 2, gens >= 0 and learn_budget >= 1."
            raise ValueError(msg)
        if not 1 <= self.init_size_min <= self.max_modules:
            msg = "Need 1 <= init_size_min <= max_modules."
            raise ValueError(msg)

    @property
    def run_dir(self) -> Path:
        """Directory holding this condition's runs."""
        return self.out / self.condition

    @property
    def stem(self) -> str:
        """File-name stem of this run, e.g. ``seed_03``."""
        return f"seed_{self.seed:02d}"

    @property
    def db_path(self) -> Path:
        """SQLite database written by the EA."""
        return self.run_dir / f"{self.stem}.db"

    @property
    def meta_path(self) -> Path:
        """Run metadata, including status and timing."""
        return self.run_dir / f"{self.stem}_meta.json"

    @property
    def reeval_path(self) -> Path:
        """Re-learning results of the best bodies."""
        return self.run_dir / f"{self.stem}_reeval.json"

    def eval_settings(self) -> "EvalSettings":
        """
        Return the settings a worker needs to evaluate a body.

        Returns
        -------
        EvalSettings
            The simulation and learning subset of this configuration.
        """
        return EvalSettings(
            duration=self.duration,
            eval_delay=self.eval_delay,
            z_penalty_weight=self.z_penalty_weight,
            learn_budget=self.learn_budget,
            learn_pop=self.learn_pop,
            spawn_yaw=self.spawn_yaw,
        )


@dataclass(frozen=True)
class EvalSettings:
    """
    Simulation and learning settings sent to evaluation workers.

    Parameters
    ----------
    duration
        Seconds of active control per simulation.
    eval_delay
        Seconds of no-control warm-up before scoring.
    z_penalty_weight
        Weight of the mean vertical deviation in the score.
    learn_budget
        CMA-ES iterations.
    learn_pop
        CMA-ES candidates per iteration.
    spawn_yaw
        Yaw (degrees) at spawn.
    """

    duration: float
    eval_delay: float
    z_penalty_weight: float
    learn_budget: int
    learn_pop: int
    spawn_yaw: float


@dataclass(frozen=True)
class EvalTask:
    """
    One body to learn a brain for.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    learn_seed
        Seed of the CMA-ES sampler.
    settings
        Simulation and learning settings.
    """

    genome: dict[str, Any]
    learn_seed: int
    settings: EvalSettings


@dataclass(frozen=True)
class EvalResult:
    """
    Outcome of learning a brain for one body.

    Parameters
    ----------
    score
        Best score found (higher is better), or ``None`` for a body that
        cannot be simulated.
    brain
        Parameters of the best brain.
    learn_curve
        Best score of each CMA-ES iteration.
    n_params
        Number of brain parameters.
    n_sims
        Simulations used.
    seconds
        Wall-clock time spent.
    """

    score: float | None
    brain: list[float]
    learn_curve: list[float]
    n_params: int
    n_sims: int
    seconds: float


# -- Brain ------------------------------------------------------------------


class MlpBrain:
    """
    The template's MLP brain, in numpy.

    Inputs are the core's orientation, the hinge angles and a 1 Hz sine/cosine
    clock; two ELU hidden layers of ``hidden`` units; ``tanh`` outputs scaled
    to ±90°. Parameters are ordered like the torch network's
    ``parameters()``: layer-by-layer weight (out x in, row-major), then bias.

    Parameters
    ----------
    n_inputs
        Length of the input vector.
    n_outputs
        Number of hinges.
    hidden
        Units per hidden layer.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        hidden: int = HIDDEN_SIZE,
    ) -> None:
        self.shapes = [
            (hidden, n_inputs),
            (hidden,),
            (hidden, hidden),
            (hidden,),
            (n_outputs, hidden),
            (n_outputs,),
        ]
        self.n_params = sum(math.prod(shape) for shape in self.shapes)
        self.layers: list[FloatArray] = []
        self.set_params(np.zeros(self.n_params))

    def set_params(self, vector: FloatArray) -> None:
        """
        Load a flat parameter vector.

        Parameters
        ----------
        vector
            ``n_params`` values in the layout described above.
        """
        self.layers = []
        start = 0
        for shape in self.shapes:
            size = math.prod(shape)
            self.layers.append(vector[start : start + size].reshape(shape))
            start += size

    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> FloatArray:
        """
        Compute hinge targets for the current state.

        Parameters
        ----------
        model
            Unused; part of the controller callback signature.
        data
            Current simulation data.

        Returns
        -------
        FloatArray
            One target angle (radians) per hinge.
        """
        del model
        phase = 2.0 * np.pi * data.time
        x = np.concatenate([
            get_state_from_data(data),
            [2.0 * np.sin(phase), 2.0 * np.cos(phase)],
        ])
        w1, b1, w2, b2, w3, b3 = self.layers
        x = _elu(w1 @ x + b1)
        x = _elu(w2 @ x + b2)
        out: FloatArray = np.tanh(w3 @ x + b3) * (np.pi / 2)
        return out


def _elu(x: FloatArray) -> FloatArray:
    out: FloatArray = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))
    return out


# -- Simulation -------------------------------------------------------------


def build_world(
    genome: dict[str, Any],
    spawn_yaw: float,
) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    """
    Spawn the body of ``genome`` on flat ground and compile it.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    spawn_yaw
        Yaw (degrees) at spawn.

    Returns
    -------
    tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]
        The world, its compiled model and fresh data.
    """
    graph = TreeGenome.from_dict(genome).to_networkx()

    def _build(*, correct_floor: bool) -> tuple[Any, Any, Any]:
        world = SimpleFlatWorld()
        world.spawn(
            construct_mjspec_from_graph(graph).spec,
            position=SPAWN_POSITION,
            rotation=(0.0, 0.0, spawn_yaw),
            correct_collision_with_floor=correct_floor,
        )
        model = world.spec.compile()
        return world, model, mujoco.MjData(model)

    try:
        return _build(correct_floor=True)
    except Exception:  # noqa: BLE001 -- same fallback as the template
        return _build(correct_floor=False)


def learn_brain(task: EvalTask) -> EvalResult:
    """
    Learn a brain for one body with CMA-ES and return the best score.

    The score of a brain is the progress toward the target during
    ``duration`` seconds of control, minus ``z_penalty_weight`` times the
    mean vertical deviation of the core, after ``eval_delay`` seconds without
    control.

    Parameters
    ----------
    task
        The body, learner seed and settings.

    Returns
    -------
    EvalResult
        The best brain and its score; ``score`` is ``None`` if the body
        cannot be built or has no hinges.
    """
    started = time.perf_counter()
    settings = task.settings
    try:
        world, model, data = build_world(task.genome, settings.spawn_yaw)
    except Exception:  # noqa: BLE001 -- an unbuildable body is just invalid
        return EvalResult(None, [], [], 0, 0, time.perf_counter() - started)
    if model.nu == 0:
        return EvalResult(None, [], [], 0, 0, time.perf_counter() - started)

    brain = MlpBrain(len(get_state_from_data(data)) + 2, model.nu)
    learner = _make_learner(brain.n_params, settings, task.learn_seed)
    simulate = _make_simulator(world, model, data, brain, settings)

    best_score = -math.inf
    best_vector: list[float] = []
    curve: list[float] = []
    for _ in range(settings.learn_budget):
        candidates = [learner.ask() for _ in range(settings.learn_pop)]
        scores = [simulate(candidate.value) for candidate in candidates]
        for candidate, score in zip(candidates, scores, strict=True):
            learner.tell(candidate, -score)
            if score > best_score:
                best_score = score
                best_vector = [float(v) for v in candidate.value]
        curve.append(max(scores))

    return EvalResult(
        score=best_score,
        brain=best_vector,
        learn_curve=curve,
        n_params=brain.n_params,
        n_sims=settings.learn_budget * settings.learn_pop,
        seconds=time.perf_counter() - started,
    )


def replay_brain(
    genome: dict[str, Any],
    brain: Sequence[float],
    settings: EvalSettings,
) -> float:
    """
    Score a stored brain on its body exactly as during learning.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    brain
        The brain parameters stored with the body.
    settings
        The run's simulation settings.

    Returns
    -------
    float
        The score; equal to the stored score of the same brain.
    """
    world, model, data = build_world(genome, settings.spawn_yaw)
    mlp = MlpBrain(len(get_state_from_data(data)) + 2, model.nu)
    simulate = _make_simulator(world, model, data, mlp, settings)
    return simulate(np.asarray(brain, dtype=np.float64))


def _make_learner(
    n_params: int,
    settings: EvalSettings,
    learn_seed: int,
) -> Optimizer:
    parametrization = ng.p.Array(shape=(n_params,)).set_mutation(
        sigma=CMA_SIGMA,
    )
    # CMA samples through the parametrization's generator, so this seeds it.
    parametrization.random_state = np.random.RandomState(learn_seed)
    # Not ``registry["CMA"]``: its population is 4 + 8 ln(n_params) (54 for a
    # 5-hinge brain), so asking ``learn_pop`` candidates per batch makes CMA
    # ask for its next generation before the last one was told, which breaks
    # step-size adaptation and can crash it. Setting the population to the
    # batch size makes every batch exactly one generation.
    optimizer = ng.optimizers.ParametrizedCMA(
        popsize=settings.learn_pop,
        diagonal=True,
    )
    return optimizer(
        parametrization=parametrization,
        budget=settings.learn_budget * settings.learn_pop,
        num_workers=settings.learn_pop,
    )


def _make_simulator(
    world: SimpleFlatWorld,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    brain: MlpBrain,
    settings: EvalSettings,
) -> Callable[[FloatArray], float]:
    tracker = Tracker(
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)
    controller = Controller(controller_callback_function=brain, tracker=tracker)
    warm_state = _warm_up(model, data, settings.eval_delay)
    active_steps = max(1, int(settings.duration / model.opt.timestep))

    def simulate(vector: FloatArray) -> float:
        brain.set_params(np.asarray(vector, dtype=np.float64))
        return _rollout(
            model,
            data,
            controller,
            warm_state,
            active_steps=active_steps,
            z_penalty_weight=settings.z_penalty_weight,
        )

    return simulate


def _warm_up(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eval_delay: float,
) -> FloatArray:
    mujoco.mj_resetData(model, data)
    for _ in range(int(eval_delay / model.opt.timestep)):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def _rollout(  # noqa: PLR0913 -- one simulation needs all of these
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Controller,
    warm_state: FloatArray,
    *,
    active_steps: int,
    z_penalty_weight: float,
) -> float:
    mujoco.mj_setState(model, data, warm_state, STATE_SPEC)
    start_distance = float(np.linalg.norm(TARGET_POSITION - data.qpos[0:3]))
    z_ref = float(data.qpos[2])
    z_deviation = 0.0
    for _ in range(active_steps):
        controller.set_control(model, data)
        mujoco.mj_step(model, data)
        z_deviation += abs(float(data.qpos[2]) - z_ref)
    end_distance = float(np.linalg.norm(TARGET_POSITION - data.qpos[0:3]))
    progress = start_distance - end_distance
    return progress - z_penalty_weight * z_deviation / active_steps


def _init_worker() -> None:
    warnings.filterwarnings(
        "ignore",
        message="TPA: apparent inconsistency",
        category=UserWarning,
        module="cma",
    )


# -- Bodies and priors ------------------------------------------------------


def derive_seed(*keys: int) -> int:
    """
    Derive an independent 32-bit seed from integer keys.

    Parameters
    ----------
    *keys
        Integers identifying the random stream, e.g. (run seed, uid).

    Returns
    -------
    int
        A seed in ``[0, 2**32)``.
    """
    return int(np.random.SeedSequence(list(keys)).generate_state(1)[0])


def is_valid_body(genome: TreeGenome, cfg: RunConfig) -> bool:
    """
    Check the limits every condition shares.

    Parameters
    ----------
    genome
        The body to check.
    cfg
        Run settings (module and depth limits).

    Returns
    -------
    bool
        True if the genome is well formed, has 1 to ``max_modules`` modules,
        at least one hinge and depth at most ``max_depth``.
    """
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        return False
    return (
        count_hinges(genome) >= 1
        and count_modules(genome) <= cfg.max_modules
        and get_tree_depth(genome) <= cfg.max_depth
    )


def random_body(cfg: RunConfig, rng: random.Random) -> TreeGenome:
    """
    Draw a valid random body of ``init_size_min`` to ``max_modules`` modules.

    Parameters
    ----------
    cfg
        Run settings.
    rng
        Random source for the size.

    Returns
    -------
    TreeGenome
        A valid random body.

    Raises
    ------
    RuntimeError
        If no valid body is found within ``MAX_INIT_TRIES`` draws.
    """
    for _ in range(MAX_INIT_TRIES):
        genome = random_tree(rng.randint(cfg.init_size_min, cfg.max_modules))
        if is_valid_body(genome, cfg):
            return genome
    msg = "Could not draw a valid random body."
    raise RuntimeError(msg)


def random_symmetric_body(cfg: RunConfig, rng: random.Random) -> TreeGenome:
    """
    Draw a valid symmetric body close to a random target size.

    Random trees are symmetrized until one lands within one module of a
    target drawn uniformly from ``[init_size_min, max_modules]``.

    Parameters
    ----------
    cfg
        Run settings.
    rng
        Random source.

    Returns
    -------
    TreeGenome
        A valid, mirror-symmetric body; the closest size found if none hit
        the target.

    Raises
    ------
    RuntimeError
        If no valid symmetric body is found at all.
    """
    target = rng.randint(cfg.init_size_min, cfg.max_modules)
    best: TreeGenome | None = None
    best_gap = math.inf
    for _ in range(MAX_INIT_TRIES):
        genome = symmetrize(random_tree(rng.randint(1, target)), rng=rng)
        if not is_valid_body(genome, cfg):
            continue
        gap = abs(count_modules(genome) - target)
        if gap <= 1:
            return genome
        if gap < best_gap:
            best, best_gap = genome, gap
    if best is None:
        msg = "Could not draw a valid symmetric body."
        raise RuntimeError(msg)
    return best


def initial_body(cfg: RunConfig, rng: random.Random) -> TreeGenome:
    """
    Draw a random initial body that satisfies the run's prior.

    Parameters
    ----------
    cfg
        Run settings.
    rng
        Random source.

    Returns
    -------
    TreeGenome
        A valid body for ``cfg.condition``.
    """
    match cfg.condition:
        case "symmetric":
            return random_symmetric_body(cfg, rng)
        case "joint_budget":
            return cap_hinges(random_body(cfg, rng), cfg.max_hinges, rng)
        case _:
            return random_body(cfg, rng)


def changed_core_side(child: TreeGenome, parent: TreeGenome) -> str | None:
    """
    Return the core side whose subtree differs from the parent's.

    Parameters
    ----------
    child
        The varied genome.
    parent
        The genome it was varied from.

    Returns
    -------
    str | None
        "LEFT" or "RIGHT" if exactly that side changed, else ``None``.
    """
    changed = [
        side
        for side in SIDES
        if _core_side_signature(child, side)
        != _core_side_signature(parent, side)
    ]
    return changed[0] if len(changed) == 1 else None


def apply_prior(
    genome: TreeGenome,
    cfg: RunConfig,
    rng: random.Random,
    *,
    parent: TreeGenome | None = None,
) -> TreeGenome:
    """
    Enforce the run's prior on a varied body.

    Parameters
    ----------
    genome
        The varied body.
    cfg
        Run settings.
    rng
        Random source.
    parent
        The body it was varied from; the symmetric prior copies the core side
        that changed relative to it.

    Returns
    -------
    TreeGenome
        The body with the prior applied (unchanged for ``baseline`` and
        ``seeded``).
    """
    match cfg.condition:
        case "symmetric":
            side = None if parent is None else changed_core_side(genome, parent)
            return symmetrize(genome, rng=rng, core_source=side)
        case "joint_budget":
            return cap_hinges(genome, cfg.max_hinges, rng)
        case _:
            return genome


def mutate_body(
    genome: TreeGenome,
    cfg: RunConfig,
    rng: random.Random,
) -> TreeGenome:
    """
    Apply the template's mutation mix to a copy of ``genome``.

    Parameters
    ----------
    genome
        The body to mutate; it is not modified.
    cfg
        Run settings (module limit for subtree replacement).
    rng
        Random source for the mutation choice.

    Returns
    -------
    TreeGenome
        The mutant, or an unchanged copy if the mutant is malformed.
    """
    mutant = copy.deepcopy(genome)
    match rng.choices(MUTATIONS, weights=MUTATION_WEIGHTS)[0]:
        case "point":
            mutate_replace_node(mutant)
        case "subtree":
            mutate_subtree_replacement(mutant, max_modules=cfg.max_modules)
        case "shrink":
            mutate_shrink(mutant)
        case _:
            mutate_hoist(mutant)

    non_core = [node for node in sorted(mutant.nodes) if node != IDX_OF_CORE]
    if non_core and rng.random() < ROTATION_MUTATION_RATE:
        node = mutant.nodes[rng.choice(non_core)]
        rotations = ALLOWED_ROTATIONS[ModuleType[node["type"]]]
        node["rotation"] = rng.choice(rotations).name

    try:
        validate_genome_dict(mutant.to_dict())
    except ValueError:
        return copy.deepcopy(genome)
    return mutant


def _core_side_signature(genome: TreeGenome, side: str) -> object:
    for edge in genome.edges:
        if edge["parent"] == IDX_OF_CORE and edge["face"] == side:
            return subtree_signature(genome, edge["child"])
    return None


# -- EA operations ----------------------------------------------------------


@dataclass
class RunContext:
    """
    Mutable state shared by the EA operations of one run.

    Parameters
    ----------
    cfg
        Run settings.
    rng
        The run's random source.
    executor
        Worker pool, or ``None`` to evaluate in-process.
    """

    cfg: RunConfig
    rng: random.Random
    executor: ProcessPoolExecutor | None = None
    next_uid: int = 0
    total_sims: int = 0
    generation: int = 0
    best_score: float = -math.inf
    started: float = field(default_factory=time.perf_counter)


def new_individual(
    genome: TreeGenome,
    ctx: RunContext,
    *,
    origin: str,
    parents: Sequence[Individual] = (),
    attempts: int = 1,
) -> Individual:
    """
    Wrap a body in an unevaluated individual with descriptive tags.

    Parameters
    ----------
    genome
        The body.
    ctx
        Run state (for the uid counter).
    origin
        "random", "seed:<name>", "offspring" or "fallback".
    parents
        Parents the body was made from.
    attempts
        Variation attempts it took.

    Returns
    -------
    Individual
        An individual that still needs evaluation.
    """
    uid = ctx.next_uid
    ctx.next_uid += 1
    if origin.startswith("seed:"):
        ancestry = 1.0
    elif parents:
        ancestry = float(
            np.mean([
                float(cast("float", p.tags["seed_ancestry"])) for p in parents
            ]),
        )
    else:
        ancestry = 0.0
    symmetry = genome_symmetry(genome)

    individual = Individual()
    individual.genotype = {"morph": genome.to_dict()}
    individual.tags = {
        "uid": uid,
        "origin": origin,
        "parents": [int(cast("int", p.tags["uid"])) for p in parents],
        "seed_ancestry": ancestry,
        "attempts": attempts,
        "fallback": origin == "fallback",
        "n_modules": count_modules(genome),
        "n_hinges": count_hinges(genome),
        "depth": get_tree_depth(genome),
        "sym_position": symmetry.position,
        "sym_axis": symmetry.joint_axis,
        "learn_seed": derive_seed(ctx.cfg.seed, uid),
    }
    individual.requires_eval = True
    return individual


def initial_population(ctx: RunContext) -> Population:
    """
    Create the initial population for the run's condition.

    Parameters
    ----------
    ctx
        Run state.

    Returns
    -------
    Population
        ``pop`` unevaluated individuals.
    """
    cfg, rng = ctx.cfg, ctx.rng
    individuals: list[Individual] = []
    if cfg.condition == "seeded":
        n_seeds = min(cfg.pop, round(cfg.seed_fraction * cfg.pop))
        names = [
            name
            for name in PREBUILT_TREE_NAMES
            if is_valid_body(prebuilt_tree_genome(name), cfg)
        ]
        chosen = rng.sample(names, min(n_seeds, len(names)))
        chosen += [rng.choice(names) for _ in range(n_seeds - len(chosen))]
        for index, name in enumerate(chosen):
            genome = prebuilt_tree_genome(name)
            if index >= len(names):
                genome = mutate_body(genome, cfg, rng)
            individuals.append(
                new_individual(genome, ctx, origin=f"seed:{name}"),
            )
    while len(individuals) < cfg.pop:
        genome = initial_body(cfg, rng)
        individuals.append(new_individual(genome, ctx, origin="random"))
    return Population(individuals)


def make_offspring(parents: list[Individual], ctx: RunContext) -> Individual:
    """
    Vary parents into one valid child that satisfies the run's prior.

    Parameters
    ----------
    parents
        The selected parents (at least one).
    ctx
        Run state.

    Returns
    -------
    Individual
        The child; a tagged clone of a parent if every attempt failed.
    """
    cfg, rng = ctx.cfg, ctx.rng
    for attempt in range(1, cfg.retries + 1):
        if len(parents) >= 2 and rng.random() < CROSSOVER_RATE:
            used = rng.sample(parents, 2)
            children = crossover_subtree(_genome(used[0]), _genome(used[1]))
            pick = 0 if rng.random() < 0.5 else 1
            child, base = children[pick], used[pick]
        else:
            base = rng.choice(parents)
            used = [base]
            child = _genome(base)
        child = mutate_body(child, cfg, rng)
        child = apply_prior(child, cfg, rng, parent=_genome(base))
        if is_valid_body(child, cfg):
            return new_individual(
                child,
                ctx,
                origin="offspring",
                parents=used,
                attempts=attempt,
            )
    parent = rng.choice(parents)
    return new_individual(
        _genome(parent),
        ctx,
        origin="fallback",
        parents=[parent],
        attempts=cfg.retries,
    )


def reproduction(population: Population, ctx: RunContext) -> Population:
    """
    Add ``pop`` offspring of the better half of the population.

    Parameters
    ----------
    population
        The alive, evaluated population.
    ctx
        Run state.

    Returns
    -------
    Population
        The population with the offspring appended.
    """
    ctx.generation += 1
    ranked = sorted(population, key=_fitness_key)
    parents = ranked[: max(1, len(ranked) // 2)]
    offspring = [make_offspring(parents, ctx) for _ in range(ctx.cfg.pop)]
    population.extend(offspring)
    return population


def evaluate(population: Population, ctx: RunContext) -> Population:
    """
    Learn brains for every individual that still needs evaluation.

    Parameters
    ----------
    population
        The population, possibly with new offspring.
    ctx
        Run state (worker pool and simulation counter).

    Returns
    -------
    Population
        The population with fitness (-score, minimised) and learning tags.
    """
    pending = [ind for ind in population if ind.requires_eval]
    settings = ctx.cfg.eval_settings()
    tasks = [
        EvalTask(
            genome=cast("dict[str, Any]", ind.genotype)["morph"],
            learn_seed=int(cast("int", ind.tags["learn_seed"])),
            settings=settings,
        )
        for ind in pending
    ]
    if ctx.executor is None:
        results = [learn_brain(task) for task in tasks]
    else:
        results = list(ctx.executor.map(learn_brain, tasks))

    for ind, result in zip(pending, results, strict=True):
        ctx.total_sims += result.n_sims
        if result.score is not None:
            ctx.best_score = max(ctx.best_score, result.score)
        ind.fitness = math.inf if result.score is None else -result.score
        ind.tags = {
            "score": result.score,
            "brain": result.brain,
            "learn_curve": result.learn_curve,
            "n_params": result.n_params,
            "n_sims": result.n_sims,
            "eval_seconds": round(result.seconds, 3),
        }
    return population


def survivor_selection(population: Population, ctx: RunContext) -> Population:
    """
    Keep the ``pop`` fittest individuals (mu + lambda truncation).

    Parameters
    ----------
    population
        Parents and offspring, all evaluated.
    ctx
        Run state.

    Returns
    -------
    Population
        The same individuals, with the losers marked dead.
    """
    ranked = sorted(population, key=_fitness_key)
    for index, ind in enumerate(ranked):
        ind.alive = index < ctx.cfg.pop
    _log_generation(ctx)
    return population


def _fitness_key(individual: Individual) -> float:
    return math.inf if individual.fitness_ is None else individual.fitness_


def _genome(individual: Individual) -> TreeGenome:
    genotype = cast("dict[str, Any]", individual.genotype)
    return TreeGenome.from_dict(copy.deepcopy(genotype["morph"]))


def _log_generation(ctx: RunContext) -> None:
    elapsed = time.perf_counter() - ctx.started
    print(  # noqa: T201 -- progress line for the batch log
        f"[{ctx.cfg.condition} seed {ctx.cfg.seed}] "
        f"gen {ctx.generation}/{ctx.cfg.gens} "
        f"best={ctx.best_score:.4f} sims={ctx.total_sims} "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )


# -- Run --------------------------------------------------------------------


def run_experiment(cfg: RunConfig) -> dict[str, Any]:
    """
    Run one condition with one seed and write its outputs.

    Writes the EA database, ``*_meta.json`` (settings, versions, timing,
    status) and ``*_reeval.json`` (fresh-seed re-learning of the best
    bodies) next to each other in ``cfg.run_dir``.

    Parameters
    ----------
    cfg
        Run settings.

    Returns
    -------
    dict[str, Any]
        The final metadata that was written to ``cfg.meta_path``.
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 -- ARIEL modules use the global
    ctx = RunContext(cfg=cfg, rng=random.Random(cfg.seed))
    meta: dict[str, Any] = {
        "config": {**asdict(cfg), "out": str(cfg.out)},
        "status": "running",
        "pid": os.getpid(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": _versions(),
    }
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.meta_path, meta)

    executor = None
    if cfg.eval_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=cfg.eval_workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
        )
    ctx.executor = executor
    try:
        population = evaluate(initial_population(ctx), ctx)
        _log_generation(ctx)
        ea = EA(
            population,
            operations=[
                EAOperation(reproduction, ctx=ctx),
                EAOperation(evaluate, ctx=ctx),
                EAOperation(survivor_selection, ctx=ctx),
            ],
            num_steps=cfg.gens,
            is_maximisation=False,
            db_file_path=cfg.db_path,
            db_handling="halt",
            quiet=True,
        )
        ea.run()
        reeval = reevaluate_best(cfg, ctx)
    finally:
        if executor is not None:
            executor.shutdown()

    _write_json(cfg.reeval_path, reeval)
    meta.update({
        "status": "completed",
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": round(time.perf_counter() - ctx.started, 1),
        "total_sims": ctx.total_sims,
        "best_score": ctx.best_score,
        "reeval_best": reeval["reeval_best"],
    })
    _write_json(cfg.meta_path, meta)
    return meta


def load_individuals(db_path: Path) -> list[dict[str, Any]]:
    """
    Read every individual of a run database.

    Parameters
    ----------
    db_path
        The EA's SQLite file.

    Returns
    -------
    list[dict[str, Any]]
        One dict per individual with ``id``, ``alive``, ``birth``,
        ``death``, ``fitness``, ``genotype`` and ``tags``.
    """
    import sqlite3  # noqa: PLC0415 -- only needed after a run

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, alive, time_of_birth, time_of_death, fitness_, "
            "genotype_, tags_ FROM individual",
        ).fetchall()
    return [
        {
            "id": row[0],
            "alive": bool(row[1]),
            "birth": row[2],
            "death": row[3],
            "fitness": row[4],
            "genotype": json.loads(row[5]),
            "tags": json.loads(row[6]),
        }
        for row in rows
    ]


def reevaluate_best(cfg: RunConfig, ctx: RunContext) -> dict[str, Any]:
    """
    Re-learn the best bodies with fresh seeds to de-noise the final result.

    A body's score during the run is the best of one noisy learning run, so
    the (mu + lambda) survivors are biased toward lucky evaluations. Each of
    the ``reeval_top`` best bodies is re-learned ``reeval_seeds`` times.

    Parameters
    ----------
    cfg
        Run settings.
    ctx
        Run state (worker pool and simulation counter).

    Returns
    -------
    dict[str, Any]
        Per-body scores and ``reeval_best``, the highest mean re-learned
        score.
    """
    individuals = [
        ind
        for ind in load_individuals(cfg.db_path)
        if ind["tags"].get("score") is not None
    ]
    individuals.sort(key=lambda ind: ind["tags"]["score"], reverse=True)
    top = individuals[: cfg.reeval_top]
    settings = cfg.eval_settings()
    tasks = [
        EvalTask(
            genome=ind["genotype"]["morph"],
            learn_seed=derive_seed(
                cfg.seed,
                ind["tags"]["uid"],
                REEVAL_SEED_OFFSET + k,
            ),
            settings=settings,
        )
        for ind in top
        for k in range(cfg.reeval_seeds)
    ]
    if ctx.executor is None:
        results = [learn_brain(task) for task in tasks]
    else:
        results = list(ctx.executor.map(learn_brain, tasks))
    ctx.total_sims += sum(result.n_sims for result in results)

    bodies = []
    for index, ind in enumerate(top):
        chunk = results[
            index * cfg.reeval_seeds : (index + 1) * cfg.reeval_seeds
        ]
        scores = [r.score for r in chunk if r.score is not None]
        bodies.append({
            "uid": ind["tags"]["uid"],
            "run_score": ind["tags"]["score"],
            "reeval_scores": scores,
            "reeval_mean": float(np.mean(scores)) if scores else None,
        })
    means = [b["reeval_mean"] for b in bodies if b["reeval_mean"] is not None]
    return {"bodies": bodies, "reeval_best": max(means) if means else None}


def _versions() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git on PATH
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "nevergrad": ng.__version__,
        "numpy": np.__version__,
        "git_commit": commit,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
