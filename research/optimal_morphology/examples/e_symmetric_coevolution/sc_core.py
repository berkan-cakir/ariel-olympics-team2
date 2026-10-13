"""
Shared machinery for the symmetric body-brain co-evolution experiment.

Task
----
A 3 m dash on flat ground. After ``settle`` seconds without control, a robot
has ``time_limit`` (60) seconds to move its core ``goal`` (3) metres forward,
along +x, the direction it faces at spawn. Its score is its average forward
speed over the dash (:func:`race_speed`):

* ``goal / finish_time`` if it crosses the finish line (3 m in 15 s scores
  0.2 m/s);
* ``furthest / time_limit`` if it does not, with ``furthest`` the furthest
  forward it got.

Every finisher therefore beats every non-finisher, and the score has no jump
at the finish line.

Optimisation
------------
Body and brain are optimised in two nested loops:

* **evolution** (outer): a (mu + lambda) EA over tree genomes that are
  mirror-symmetric about the robot's sagittal plane, in module positions and
  hinge axes (:func:`ariel.ec.genotypes.tree.symmetry.symmetrize`). Every
  offspring is a body that is not already in the population. Besides the
  usual tree mutations, which shorten limbs far more often than they
  lengthen them, the ``grow`` mutation adds one module to a limb
  (:func:`grow_limb`).
* **learning** (inner): every new body gets its own brain, learned from
  scratch by CMA-ES with a fixed budget of ``learn_gens * learn_pop`` races.
  A body's fitness is the speed of the best brain found for it. A new body is
  thus judged with a brain made for it, not with its parent's brain, which
  would make most new bodies look worse than they are and freeze evolution on
  the first body that works.

Servos
------
The simulated hinges behave like real servos (all three limits can be turned
off with 0):

* **speed** -- the commanded angle moves toward the gait's target at most
  ``max_hinge_speed`` rad/s, like a servo that turns at a constant speed, so a
  hinge is never sent a command it could not follow (:func:`servo_commands`);
* **torque** -- the actuator torque is clamped to ``max_hinge_torque`` N m;
* **reversals** -- no hinge oscillates faster than ``max_hinge_frequency``
  Hz, i.e. at most twice that many direction changes per second, because
  rapid back-and-forth switching overheats a servo.

Every scored body also gets its dash measured (:func:`actuator_effort`): peak
torque, peak hinge speed and direction reversals per second.

Pruning
-------
With ``prune`` on (the default), a body is simplified after learning. Its
best dash is replayed and every module that touches the floor is recorded;
every subtree in which no module touched the floor is cut off, mirror twins
together, so the body stays symmetric (:func:`grounded_modules`,
:func:`prune_genome`). The pruned body then learns its own gait with the same
budget, and it replaces the original in the population, with the faster of
that gait and the old gait of its remaining hinges. The original's size and
speed are kept in the tags.

Brain
-----
An open-loop central pattern generator: every hinge follows a sine wave, all
at one frequency (:class:`Gait`). The waves are tied together by a gait family
(:data:`FAMILIES`, :func:`gait_layout`):

* ``alternating`` -- the body's mirror symmetry, half a period apart. The two
  hinges of a mirror pair share amplitude and offset and run half a period
  apart; a hinge on the sagittal plane gets a wave that is its own mirror
  image half a period later. At ``t + T/2`` the robot is the mirror image of
  itself at ``t``, as in a walk or a trot.
* ``synchronous`` -- the body's mirror symmetry, in phase. Mirror pairs are
  mirror images at every instant, as in a bound; a plane hinge whose mirror
  image is its own negation is held still.
* ``free`` -- every hinge has its own wave.

A symmetric family gives the gait the symmetry of the body, so the robot moves
straight ahead or straight back without having to learn to steer. Which
symmetric family suits a body depends on the body, so the default brain,
``symmetric`` (:data:`BRAINS`), learns both with half the budget each and
keeps the faster gait.

Notes
-----
    * Do not add ``from __future__ import annotations``: the EA inspects the
      annotations of its operations.
"""

# Standard library
import copy
import hashlib
import json
import math
import operator
import os
import platform
import random
import re
import shutil
import subprocess  # noqa: S404 -- only for `git rev-parse HEAD` and `ps`
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import cma
import mujoco
import numpy as np
from sqlalchemy import Engine, create_engine

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
from ariel.ec.ea import DBHandlingMode
from ariel.ec.genotypes.tree.constraints import count_hinges, count_modules
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    get_tree_depth,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    remove_subtree,
)
from ariel.ec.genotypes.tree.symmetry import (
    genome_symmetry,
    subtree_signature,
    symmetrize,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.parameters.ariel_types import FloatArray
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.symmetry import SymmetryScore, find_core_body

# Type Aliases
type Snapshot = Callable[[list[Individual], int], None]

# Global constants
FAMILIES = ("alternating", "synchronous", "free")
SYMMETRIC_FAMILIES = ("alternating", "synchronous")
# A brain learns every family in its tuple, splitting the budget evenly.
BRAINS: dict[str, tuple[str, ...]] = {
    "symmetric": SYMMETRIC_FAMILIES,
    "alternating": ("alternating",),
    "synchronous": ("synchronous",),
    "free": ("free",),
}
SPAWN_POSITION = (-1.5, 0.0, 0.1)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
BAD_QACC = int(mujoco.mjtWarning.mjWARN_BADQACC)
CONTROL_STEPS = 10  # physics steps per control update: 0.02 s at 2 ms
SERVO_LIMIT = math.pi / 2  # ctrlrange of the hinge actuators
FREQUENCY_RANGE = (0.25, 4.0)  # Hz
MIRROR = np.array([-1.0, 1.0, 1.0])  # reflection in the core frame's x = 0
MATCH_TOLERANCE = 1e-3  # m, as in ariel.utils.symmetry
MIN_ABS_COS = 0.99
CROSSOVER_RATE = 0.5
ROTATION_MUTATION_RATE = 0.25
MUTATIONS = ("point", "subtree", "shrink", "hoist", "grow")
# Shares of the first four; "grow" gets ``RunConfig.grow`` and the others
# shrink in proportion.
MUTATION_WEIGHTS = (0.45, 0.35, 0.10, 0.10)
GROWN_TYPES = (ModuleType.BRICK, ModuleType.HINGE)
SERVOS = "servos"  # genome key of a hinge's servo count; absent means one
SERVO_MASS = 0.06  # kg: one more servo, as heavy as a hinge module
# Two servos side by side on the hinge's axis (its parts' local z): 10 cm wide.
DOUBLE_SERVO_SCALE = (1.0, 1.0, 2.0)
DOUBLE_SERVO_RGBA = (0.55, 0.25, 0.8, 1.0)  # purple: a hinge with two servos
SERVO_NAME = re.compile(r"_\d+-(\d+)-\d+-servo$")
HINGE_PART = re.compile(r"(?:^|_)\d+-(\d+)-\d+-(?:stator|rotor)$")
MAX_INIT_TRIES = 500
REEVAL_SEED_OFFSET = 10_000
PRUNED_SEED_KEY = 1  # learning seed of a pruned body: (body seed, this)
INVALID_SPEED = -1.0
SIDES = ("LEFT", "RIGHT")
SWING_BAND = math.radians(5.0)  # smaller back-and-forth is jitter, not a swing
SERVO_ARMATURE = 1e-3  # kg m^2: a hobby servo's motor through its gearbox
# Settings added after runs were made, with the value those runs had.
LEGACY_SETTINGS = {"grow": 0.0, "servo_gain": 0.0, "double_servo": 0.0}
# Settings a resumed run may change; the others must be the stored run's.
RESUMABLE = frozenset({
    "out",
    "gens",
    "workers",
    "snapshot_every",
    "video_every",
    "reeval_top",
    "reeval_seeds",
})
# Metadata of a finished run, dropped while the run goes on.
FINAL_FIELDS = (
    "finished",
    "wall_seconds",
    "total_sims",
    "sim_hours",
    "best_speed",
    "best_uid",
    "reeval_best",
)
# Body names are "<prefix>_<parent>-<child>-<face>-<part>"; <child> is the
# module's node id in the genome. A module's root body is its brick or hinge.
MODULE_BODY = re.compile(r"_\d+-(\d+)-\d+-(?:brick|hinge|stator|rotor)$")
MODULE_ROOT = re.compile(r"_\d+-(\d+)-\d+-(?:brick|hinge)$")


# -- Configuration ----------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """
    Settings of one run; the CLI flags of ``run_symmetric_coevolution.py``.

    Parameters
    ----------
    seed
        Run seed; every random choice in the run derives from it.
    out
        Output root; files go to ``out/<brain>_pruned_servo_grow/seed_<NN>*``
        (see :attr:`run_dir`).
    brain
        Which gait families a body learns; a key of :data:`BRAINS`.
    pop
        Bodies kept each generation (mu).
    offspring
        New bodies per generation (lambda).
    gens
        Generations after the initial population.
    tournament
        Tournament size of parent selection.
    learn_gens
        CMA-ES generations per body, split evenly between the brain's gait
        families.
    learn_pop
        Races per CMA-ES generation.
    learn_sigma
        Initial CMA-ES step size in the learner's parameter space.
    goal
        Distance (m) to cover.
    time_limit
        Seconds of control the robot has to cover it.
    settle
        Seconds without control before the dash starts.
    max_modules
        Largest body, in modules (bricks and hinges).
    max_depth
        Deepest tree allowed.
    init_size_min
        Smallest target size of an initial body; targets are uniform in
        ``[init_size_min, max_modules]``.
    spawn_yaw
        Yaw (degrees) at spawn; -90 turns the robot's front toward +x.
    retries
        Variation attempts per offspring before a random body is used.
    reeval_top
        Best bodies re-learned with fresh seeds at the end of the run.
    reeval_seeds
        Fresh learning seeds per re-learned body.
    workers
        Worker processes for learning; 1 learns in-process.
    snapshot_every
        Save a picture of the population every this many generations, and of
        the first and last; 0 saves none.
    video_every
        Film the fastest robot every this many generations, and in the first
        and last; 0 films none.
    prune
        Cut off the subtrees that never touch the floor after learning, then
        learn again (see the module notes).
    max_hinge_speed
        Top speed (rad/s) of a hinge's commanded angle; 0 means no limit.
    max_hinge_torque
        Largest actuator torque (N m); 0 means no limit.
    max_hinge_frequency
        Fastest oscillation (Hz) of any hinge; 0 means no limit.
    grow
        Share of mutations that lengthen a limb by one module
        (:func:`grow_limb`); the other mutations keep their proportions.
        0 turns it off, as in the runs made before it existed.
    servo_gain
        Position gain (N m/rad) of a hobby-servo-like hinge
        (:func:`set_servos`): at 10 it pushes with full torque from about 6
        degrees off target. 0 keeps ARIEL's servo, which lags about a second
        behind its command, as in the runs made before this setting existed.
    double_servo
        Chance that a mutated child also switches one hinge between one servo
        and two in parallel (:func:`toggle_servos`), which doubles its torque
        and adds a servo's weight. It comes on top of the other mutations,
        whose rates stay as they are. 0 turns it off, as in the runs made
        before it existed.
    """

    seed: int
    out: Path
    brain: str = "symmetric"
    pop: int = 16
    offspring: int = 16
    gens: int = 20
    tournament: int = 3
    learn_gens: int = 16
    learn_pop: int = 12
    learn_sigma: float = 0.5
    goal: float = 3.0
    time_limit: float = 60.0
    settle: float = 2.0
    max_modules: int = 20
    max_depth: int = 12
    init_size_min: int = 8
    spawn_yaw: float = -90.0
    retries: int = 20
    reeval_top: int = 3
    reeval_seeds: int = 3
    workers: int = 1
    snapshot_every: int = 1
    video_every: int = 1
    prune: bool = True
    max_hinge_speed: float = 5.24  # 300 deg/s: 60 deg in 0.2 s
    max_hinge_torque: float = 1.0  # N m, a standard hobby servo
    max_hinge_frequency: float = 1.0  # Hz: 2 reversals per second
    grow: float = 0.4  # lengthens as many children's limbs as are shortened
    servo_gain: float = 10.0  # N m/rad
    double_servo: float = 0.2

    def __post_init__(self) -> None:
        """
        Reject settings the experiment cannot run with.

        Raises
        ------
        ValueError
            If the brain is unknown or a setting is out of range.
        """
        if self.brain not in BRAINS:
            msg = f"Unknown brain {self.brain!r}; use one of {list(BRAINS)}."
            raise ValueError(msg)
        if self.pop < 2 or self.offspring < 1 or self.gens < 0:
            msg = "Need pop >= 2, offspring >= 1 and gens >= 0."
            raise ValueError(msg)
        if self.learn_gens < len(BRAINS[self.brain]) or self.learn_pop < 2:
            msg = (
                "Need learn_pop >= 2 and at least one learn_gens per gait "
                f"family ({len(BRAINS[self.brain])} for brain {self.brain})."
            )
            raise ValueError(msg)
        if self.goal <= 0 or self.time_limit <= 0 or self.settle < 0:
            msg = "Need goal > 0, time_limit > 0 and settle >= 0."
            raise ValueError(msg)
        if not 1 <= self.init_size_min <= self.max_modules:
            msg = "Need 1 <= init_size_min <= max_modules."
            raise ValueError(msg)
        limits = (
            self.max_hinge_speed,
            self.max_hinge_torque,
            self.max_hinge_frequency,
        )
        if min(limits) < 0:
            msg = "Servo limits must be positive, or 0 for no limit."
            raise ValueError(msg)
        if not 0 <= self.grow < 1 or not 0 <= self.double_servo <= 1:
            msg = "Need 0 <= grow < 1 and 0 <= double_servo <= 1."
            raise ValueError(msg)
        if self.servo_gain < 0:
            msg = "Need servo_gain >= 0 (0: ARIEL's servo)."
            raise ValueError(msg)

    @property
    def servo_limited(self) -> bool:
        """Whether any servo limit is on."""
        return (
            max(
                self.max_hinge_speed,
                self.max_hinge_torque,
                self.max_hinge_frequency,
            )
            > 0
        )

    @property
    def run_dir(self) -> Path:
        """Directory of the runs with these settings (see the README)."""
        pruned = "_pruned" if self.prune else ""
        servo = "_servo" if self.servo_limited else ""
        if self.servo_gain > 0:
            servo = "_realservo"
        grow = "_grow" if self.grow > 0 else ""
        double = "_double" if self.double_servo > 0 else ""
        return self.out / f"{self.brain}{pruned}{servo}{grow}{double}"

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
        """Run metadata: settings, status, timing and per-generation log."""
        return self.run_dir / f"{self.stem}_meta.json"

    @property
    def reeval_path(self) -> Path:
        """Re-learning results of the best bodies."""
        return self.run_dir / f"{self.stem}_reeval.json"

    @property
    def snapshot_dir(self) -> Path:
        """Pictures of the population, one per snapshot generation."""
        return self.run_dir / f"{self.stem}_snapshots"

    @property
    def video_dir(self) -> Path:
        """Videos of each filmed generation's fastest robot."""
        return self.run_dir / f"{self.stem}_videos"

    def eval_settings(self) -> "EvalSettings":
        """
        Return the settings a worker needs to learn a brain.

        Returns
        -------
        EvalSettings
            The task, simulation and learning subset of this configuration.
        """
        return EvalSettings(
            brain=self.brain,
            goal=self.goal,
            time_limit=self.time_limit,
            settle=self.settle,
            learn_gens=self.learn_gens,
            learn_pop=self.learn_pop,
            learn_sigma=self.learn_sigma,
            spawn_yaw=self.spawn_yaw,
            prune=self.prune,
            max_hinge_speed=self.max_hinge_speed,
            max_hinge_torque=self.max_hinge_torque,
            max_hinge_frequency=self.max_hinge_frequency,
            servo_gain=self.servo_gain,
        )


@dataclass(frozen=True)
class EvalSettings:
    """
    Task, simulation and learning settings sent to learning workers.

    Parameters
    ----------
    brain
        A key of :data:`BRAINS`.
    goal
        Distance (m) to cover.
    time_limit
        Seconds of control.
    settle
        Seconds without control before the dash.
    learn_gens
        CMA-ES generations, split evenly between the brain's gait families.
    learn_pop
        Races per CMA-ES generation.
    learn_sigma
        Initial CMA-ES step size.
    spawn_yaw
        Yaw (degrees) at spawn.
    prune
        Prune the body after learning and learn again.
    max_hinge_speed
        Top speed (rad/s) of a commanded hinge angle; 0 means no limit.
    max_hinge_torque
        Largest actuator torque (N m); 0 means no limit.
    max_hinge_frequency
        Fastest hinge oscillation (Hz); 0 means no limit.
    servo_gain
        Position gain (N m/rad) of a hobby-servo-like hinge
        (:func:`set_servos`); 0 keeps ARIEL's servo.
    """

    brain: str
    goal: float
    time_limit: float
    settle: float
    learn_gens: int
    learn_pop: int
    learn_sigma: float
    spawn_yaw: float
    prune: bool = False
    # No limits by default, so runs made before the limits replay unchanged.
    max_hinge_speed: float = 0.0
    max_hinge_torque: float = 0.0
    max_hinge_frequency: float = 0.0
    servo_gain: float = 0.0  # ARIEL's servo


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
        Task, simulation and learning settings.
    """

    genome: dict[str, Any]
    learn_seed: int
    settings: EvalSettings


@dataclass(frozen=True)
class RaceResult:
    """
    Outcome of one dash.

    Parameters
    ----------
    speed
        The score (m/s), see :func:`race_speed`; 0 if the simulation went
        unstable.
    finish_time
        Seconds to cover the goal distance, or ``None`` if it did not.
    furthest
        Furthest forward progress (m).
    drift
        Sideways displacement (m, +y is the robot's left) at the end.
    turn
        Change of heading (degrees, counter-clockwise) at the end.
    duration
        Simulated seconds of control; less than the time limit for a
        finisher.
    stable
        False if MuJoCo reported a diverging simulation.
    """

    speed: float
    finish_time: float | None
    furthest: float
    drift: float
    turn: float
    duration: float
    stable: bool


@dataclass(frozen=True)
class EvalResult:
    """
    Outcome of learning a brain for one body.

    Parameters
    ----------
    speed
        Score of the best brain (m/s), or ``None`` for a body that cannot be
        simulated.
    finish_time
        The best brain's time to cover the goal, or ``None``.
    furthest
        The best brain's furthest progress (m).
    drift
        The best brain's sideways displacement (m).
    turn
        The best brain's change of heading (degrees).
    gait
        The best brain, as :meth:`Gait.to_dict`.
    family
        The best brain's gait family (see :data:`FAMILIES`).
    family_speeds
        Best score found in each gait family the brain learned.
    learn_curve
        Best score of each round of CMA-ES generations (one generation per
        gait family).
    n_params
        Number of learned parameters of the best brain's family.
    n_pairs
        Mirror pairs of hinges.
    n_plane
        Hinges on the sagittal plane.
    n_unpaired
        Hinges without a mirror image (0 for a symmetric body).
    n_sims
        Races run.
    sim_seconds
        Simulated seconds of control over all races.
    seconds
        Wall-clock time spent.
    genome
        The body that was scored: the pruned body if pruning cut anything,
        else the body given.
    cut
        Roots of the subtrees cut off by pruning (node ids of the body
        given); empty if nothing was cut.
    unpruned_speed
        Score of the body given, before pruning, or ``None`` without
        pruning.
    effort
        How hard the scored dash works the hinges, from
        :func:`actuator_effort`.
    """

    speed: float | None
    finish_time: float | None
    furthest: float
    drift: float
    turn: float
    gait: dict[str, Any]
    family: str
    family_speeds: dict[str, float]
    learn_curve: list[float]
    n_params: int
    n_pairs: int
    n_plane: int
    n_unpaired: int
    n_sims: int
    sim_seconds: float
    seconds: float
    genome: dict[str, Any] = field(default_factory=dict)
    cut: list[int] = field(default_factory=list)
    unpruned_speed: float | None = None
    effort: dict[str, float] = field(default_factory=dict)


# -- Task -------------------------------------------------------------------


def race_speed(
    *,
    finish_time: float | None,
    furthest: float,
    goal: float,
    time_limit: float,
) -> float:
    """
    Score a dash by its average forward speed.

    Parameters
    ----------
    finish_time
        Seconds it took to cover ``goal``, or ``None`` if it did not.
    furthest
        Furthest forward progress (m) during the dash.
    goal
        Distance (m) to cover.
    time_limit
        Seconds the robot had.

    Returns
    -------
    float
        ``goal / finish_time`` for a finisher, else ``furthest /
        time_limit`` (m/s). Both equal ``goal / time_limit`` at the finish
        line, so every finisher outscores every non-finisher.
    """
    if finish_time is not None:
        return goal / finish_time
    return furthest / time_limit


# -- Brain ------------------------------------------------------------------


@dataclass(frozen=True)
class Gait:
    """
    An open-loop gait: one sine wave per hinge, all at one frequency.

    Hinge ``i`` is driven toward ``offset[i] + amplitude[i] * sin(2 pi *
    frequency * harmonic[i] * t + phase[i])``, clipped to the servo range.

    Parameters
    ----------
    frequency
        Base frequency (Hz).
    amplitude
        Per hinge, in radians.
    offset
        Per hinge, in radians.
    phase
        Per hinge, in radians.
    harmonic
        Per hinge, the multiple of the base frequency (1 or 2).
    """

    frequency: float
    amplitude: FloatArray
    offset: FloatArray
    phase: FloatArray
    harmonic: FloatArray

    def targets(self, t: float) -> FloatArray:
        """
        Return the hinge targets at time ``t``.

        Parameters
        ----------
        t
            Seconds since the start of the dash.

        Returns
        -------
        FloatArray
            One target angle (radians) per hinge, within the servo range.
        """
        angle = 2.0 * math.pi * self.frequency * t
        wave = np.sin(angle * self.harmonic + self.phase)
        out: FloatArray = np.clip(
            self.offset + self.amplitude * wave,
            -SERVO_LIMIT,
            SERVO_LIMIT,
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        """
        Return the gait as JSON-friendly lists.

        Returns
        -------
        dict[str, Any]
            ``frequency``, ``amplitude``, ``offset``, ``phase`` and
            ``harmonic``.
        """
        return {
            "frequency": float(self.frequency),
            "amplitude": self.amplitude.tolist(),
            "offset": self.offset.tolist(),
            "phase": self.phase.tolist(),
            "harmonic": self.harmonic.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Gait":
        """
        Rebuild a gait saved with :meth:`to_dict`.

        Parameters
        ----------
        payload
            The saved gait.

        Returns
        -------
        Gait
            The same gait.
        """
        return cls(
            frequency=float(payload["frequency"]),
            amplitude=np.asarray(payload["amplitude"], dtype=np.float64),
            offset=np.asarray(payload["offset"], dtype=np.float64),
            phase=np.asarray(payload["phase"], dtype=np.float64),
            harmonic=np.asarray(payload["harmonic"], dtype=np.float64),
        )


@dataclass(frozen=True)
class GaitLayout:
    """
    How a learner's parameter vector becomes a :class:`Gait`.

    Every hinge follows one oscillator unit, as ``sign * (offset + amplitude
    * sin(harmonic * w * t + phase + shift))``; the hinges of a mirror pair
    share a unit. The vector holds the frequency and then, per unit, its
    amplitude, offset (if it has one) and phase (see :func:`decode_gait`).

    Parameters
    ----------
    unit
        Per hinge: its unit, or -1 if it is held at 0.
    sign
        Per hinge: +1, or -1 if it follows the negated wave.
    shift
        Per hinge: phase shift (radians) added to its unit's phase.
    harmonic
        Per hinge: multiple of the base frequency (1 or 2).
    has_offset
        Per unit: whether it has an offset parameter.
    """

    unit: tuple[int, ...]
    sign: tuple[float, ...]
    shift: tuple[float, ...]
    harmonic: tuple[int, ...]
    has_offset: tuple[bool, ...]

    @property
    def n_params(self) -> int:
        """Length of the parameter vector."""
        return 1 + sum(3 if offset else 2 for offset in self.has_offset)


def mirror_partners(model: mujoco.MjModel) -> list[tuple[int, float]]:
    """
    Find the mirror image of every actuated hinge.

    Hinges are compared at the rest pose in the core's frame, whose plane
    ``x = 0`` is the robot's sagittal plane (see :mod:`ariel.utils.symmetry`).
    A rotation by ``q`` about axis ``a`` mirrors to a rotation by ``q`` about
    ``-M a``, with ``M`` the reflection (an axis is a pseudovector), so a
    partner with axis ``b`` turns by ``sign * q``, ``sign = sign(b . -M a)``.

    Parameters
    ----------
    model
        A compiled model containing exactly one robot.

    Returns
    -------
    list[tuple[int, float]]
        Per actuator: the actuator of its mirror image and the sign. A hinge
        on the sagittal plane is its own mirror image; a hinge without one
        gets ``(-1, 1.0)``.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    core = find_core_body(model)
    origin = data.xpos[core].copy()
    to_local = data.xmat[core].reshape(3, 3).copy()
    joints = model.actuator_trnid[:, 0]
    anchors = (data.xanchor[joints] - origin) @ to_local
    axes = data.xaxis[joints] @ to_local
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)

    distance = np.linalg.norm(
        (anchors * MIRROR)[:, None, :] - anchors[None, :, :],
        axis=2,
    )
    cosine = -(axes * MIRROR) @ axes.T
    matches = (distance < MATCH_TOLERANCE) & (np.abs(cosine) >= MIN_ABS_COS)

    partners: list[tuple[int, float]] = []
    for actuator in range(model.nu):
        candidates = np.flatnonzero(matches[actuator])
        if candidates.size == 0:
            partners.append((-1, 1.0))
            continue
        partner = int(candidates[np.argmin(distance[actuator, candidates])])
        partners.append((partner, float(np.sign(cosine[actuator, partner]))))
    # A partner must recognise its partner back.
    return [
        (partner, sign)
        if partner >= 0 and partners[partner][0] == a
        else (-1, 1.0)
        for a, (partner, sign) in enumerate(partners)
    ]


def gait_layout(
    partners: Sequence[tuple[int, float]],
    family: str,
) -> GaitLayout:
    """
    Tie the hinges' waves together as the gait ``family`` prescribes.

    In a symmetric family, the lower-numbered hinge of a mirror pair leads
    and the other follows its wave with the mirror sign, half a period later
    (``alternating``) or at once (``synchronous``). A hinge on the sagittal
    plane whose mirror image is itself (sign +1) or its negation (sign -1)
    must satisfy ``q(t + T/2) = sign * q(t)`` (alternating) or ``q(t) = sign
    * q(t)`` (synchronous); that gives a wave at twice the frequency, a wave
    without offset, a free wave, or no motion. Hinges without a mirror image
    always get a free wave.

    Parameters
    ----------
    partners
        Mirror partners from :func:`mirror_partners`.
    family
        One of :data:`FAMILIES`.

    Returns
    -------
    GaitLayout
        The layout of the learner's parameter vector.
    """
    n_hinges = len(partners)
    unit = [-1] * n_hinges
    sign = [1.0] * n_hinges
    shift = [0.0] * n_hinges
    harmonic = [1] * n_hinges
    has_offset: list[bool] = []

    def new_unit(*, offset: bool) -> int:
        has_offset.append(offset)
        return len(has_offset) - 1

    assigned: set[int] = set()
    for hinge, (partner, partner_sign) in enumerate(partners):
        if hinge in assigned:
            continue
        assigned.add(hinge)
        if family == "free" or partner < 0:
            unit[hinge] = new_unit(offset=True)
        elif partner == hinge and family == "alternating":
            # q(t + T/2) = q(t) needs twice the frequency; q(t + T/2) =
            # -q(t) allows no offset.
            unit[hinge] = new_unit(offset=partner_sign > 0)
            harmonic[hinge] = 2 if partner_sign > 0 else 1
        elif partner == hinge:
            if partner_sign > 0:
                unit[hinge] = new_unit(offset=True)
        else:
            assigned.add(partner)
            unit[hinge] = unit[partner] = new_unit(offset=True)
            sign[partner] = partner_sign
            shift[partner] = math.pi if family == "alternating" else 0.0
    return GaitLayout(
        unit=tuple(unit),
        sign=tuple(sign),
        shift=tuple(shift),
        harmonic=tuple(harmonic),
        has_offset=tuple(has_offset),
    )


def decode_gait(
    vector: FloatArray,
    layout: GaitLayout,
    max_frequency: float = 0.0,
) -> Gait:
    """
    Turn a learner's parameter vector into a gait.

    Every entry is unbounded, and 0 is a moderate default: the frequency
    ``f_min * (f_max / f_min) ** logistic(v)`` is 1 Hz at 0; an amplitude
    ``(pi / 2) * logistic(v)`` is 45 degrees; an offset ``(pi / 2) * tanh(v /
    2)`` is 0; a phase is ``2 pi v``.

    Parameters
    ----------
    vector
        ``layout.n_params`` values.
    layout
        How the values drive the hinges.
    max_frequency
        Fastest oscillation (Hz) of any hinge, harmonics included; ``f_max``
        is lowered to respect it. 0 means no limit.

    Returns
    -------
    Gait
        The gait, with the mirror signs and shifts folded into each hinge's
        offset and phase.
    """
    f_min, f_max = FREQUENCY_RANGE
    if max_frequency > 0:
        f_max = min(f_max, max_frequency / max(layout.harmonic, default=1))
        f_min = min(f_min, f_max / 2)
    frequency = f_min * (f_max / f_min) ** _logistic(float(vector[0]))
    amplitudes: list[float] = []
    offsets: list[float] = []
    phases: list[float] = []
    index = 1
    for has_offset in layout.has_offset:
        amplitudes.append(SERVO_LIMIT * _logistic(float(vector[index])))
        index += 1
        offset = 0.0
        if has_offset:
            offset = SERVO_LIMIT * math.tanh(float(vector[index]) / 2.0)
            index += 1
        offsets.append(offset)
        phases.append(2.0 * math.pi * float(vector[index]))
        index += 1

    n_hinges = len(layout.unit)
    amplitude = np.zeros(n_hinges)
    offset_out = np.zeros(n_hinges)
    phase = np.zeros(n_hinges)
    for hinge, unit in enumerate(layout.unit):
        if unit < 0:
            continue
        flip = layout.sign[hinge] < 0
        amplitude[hinge] = amplitudes[unit]
        offset_out[hinge] = layout.sign[hinge] * offsets[unit]
        phase[hinge] = (
            phases[unit] + layout.shift[hinge] + (math.pi if flip else 0.0)
        )
    return Gait(
        frequency=frequency,
        amplitude=amplitude,
        offset=offset_out,
        phase=phase,
        harmonic=np.asarray(layout.harmonic, dtype=np.float64),
    )


def _logistic(value: float) -> float:
    return 0.5 * (1.0 + math.tanh(value / 2.0))


# -- Simulation -------------------------------------------------------------


def build_world(
    genome: dict[str, Any],
    spawn_yaw: float,
    settings: "EvalSettings | None" = None,
) -> tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]:
    """
    Spawn the body of ``genome`` on flat ground and compile it.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    spawn_yaw
        Yaw (degrees) at spawn.
    settings
        The run's settings, for the servos (:func:`set_servos`); ``None``
        keeps ARIEL's servos without limits, which is enough for pictures and
        for the body's shape. The servos are set in the world's spec, so
        later compiles keep them.

    Returns
    -------
    tuple[SimpleFlatWorld, mujoco.MjModel, mujoco.MjData]
        The world, its compiled model and fresh data.
    """
    graph = TreeGenome.from_dict(genome).to_networkx()
    doubled = double_servo_nodes(genome)

    def _build(*, correct_floor: bool) -> tuple[Any, Any, Any]:
        robot = construct_mjspec_from_graph(graph).spec
        add_second_servos(robot, doubled)  # before spawning lifts it off
        world = SimpleFlatWorld()
        world.spawn(
            robot,
            position=SPAWN_POSITION,
            rotation=(0.0, 0.0, spawn_yaw),
            correct_collision_with_floor=correct_floor,
        )
        if settings is not None:
            set_servos(world.spec, settings, doubled)
        model = world.spec.compile()
        return world, model, mujoco.MjData(model)

    try:
        return _build(correct_floor=True)
    except Exception:  # noqa: BLE001 -- the design-priors fallback
        return _build(correct_floor=False)


def set_servos(
    spec: mujoco.MjSpec,
    settings: "EvalSettings",
    doubled: frozenset[int] = frozenset(),
) -> None:
    """
    Give every hinge of a robot the run's servo.

    ARIEL's hinge is a position servo with a gain of 1 N m/rad and a damping
    of 1 N m s/rad, so it lags about a second behind its command: at 1 Hz a
    hinge swings about a sixth of what it is told. With a ``servo_gain``,
    the hinge acts like a hobby servo instead:

    * it pushes with ``servo_gain`` N m per radian off target, up to
      ``max_hinge_torque``, i.e. with full torque from a few degrees off;
    * like a motor, its torque falls with speed, to 0 at
      ``max_hinge_speed`` (joint damping of torque / speed);
    * it has a motor's inertia through the gearbox (:data:`SERVO_ARMATURE`).

    A hinge with two servos in parallel gets twice the gain, torque, damping
    and inertia: two servos that get the same command add their torques.

    Parameters
    ----------
    spec
        The world's spec, before compiling.
    settings
        The run's settings (torque and speed limits, servo gain).
    doubled
        Genome node ids of the hinges with two servos.
    """
    torque, speed, gain = (
        settings.max_hinge_torque,
        settings.max_hinge_speed,
        settings.servo_gain,
    )
    for actuator in spec.actuators:
        servos = _servo_count(actuator.name, doubled)
        if torque > 0:
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = [-servos * torque, servos * torque]
        if gain > 0:
            gainprm, biasprm = np.zeros(10), np.zeros(10)
            gainprm[0] = gain
            biasprm[:3] = [0.0, -gain, 0.0]
        elif servos > 1:  # ARIEL's servo, its damping included
            gainprm = np.array(actuator.gainprm)
            biasprm = np.array(actuator.biasprm)
        else:
            continue
        actuator.gainprm, actuator.biasprm = servos * gainprm, servos * biasprm
    if gain > 0:
        for joint in spec.joints:
            if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                servos = _servo_count(joint.name, doubled)
                joint.armature = servos * SERVO_ARMATURE
                if torque > 0 and speed > 0:
                    # Linear damping, then MuJoCo's two polynomial terms.
                    joint.damping = [servos * torque / speed, 0.0, 0.0]


def add_second_servos(spec: mujoco.MjSpec, doubled: frozenset[int]) -> None:
    """
    Add a second servo to every hinge that has two, and colour the hinge.

    The two servos sit side by side on the hinge's axis, so the hinge's
    housing and rotor get twice as wide along it (:data:`DOUBLE_SERVO_SCALE`):
    10 cm, a block's width, so the hinge still fits wherever one fits. The
    second servo's weight goes to the housing.

    Parameters
    ----------
    spec
        The robot's spec, or a world's spec with the robot, before compiling.
        Give the robot's, before it is spawned, so that spawning lifts the
        wider hinges off the floor too.
    doubled
        Genome node ids of the hinges with two servos.
    """
    if not doubled:
        return
    for body in spec.bodies:
        match = HINGE_PART.search(body.name)
        if match is None or int(match.group(1)) not in doubled:
            continue
        for geom in body.geoms:
            geom.rgba = DOUBLE_SERVO_RGBA
            geom.size = np.array(geom.size) * DOUBLE_SERVO_SCALE
        if body.name.endswith("stator"):
            body.geoms[0].mass += SERVO_MASS


def double_servo_nodes(genome: dict[str, Any]) -> frozenset[int]:
    """
    Return the hinges with two servos.

    Parameters
    ----------
    genome
        A tree genome as a dict.

    Returns
    -------
    frozenset[int]
        Node ids whose :data:`SERVOS` is 2.
    """
    return frozenset(
        int(node)
        for node, attributes in genome["nodes"].items()
        if attributes.get(SERVOS, 1) == 2
    )


def _servo_count(name: str | None, doubled: frozenset[int]) -> int:
    match = SERVO_NAME.search(name or "")
    return 2 if match is not None and int(match.group(1)) in doubled else 1


def settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seconds: float,
) -> FloatArray:
    """
    Let the robot settle without control and return the resulting state.

    Parameters
    ----------
    model
        The compiled model.
    data
        Its data; reset, then stepped.
    seconds
        How long to settle.

    Returns
    -------
    FloatArray
        The full integration state, for :func:`run_race`.
    """
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0
    steps = round(seconds / model.opt.timestep)
    if steps > 0:
        mujoco.mj_step(model, data, nstep=steps)
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def servo_commands(
    gait: Gait,
    settings: EvalSettings,
    period: float,
    updates: int,
) -> FloatArray:
    """
    Return the hinge commands of a dash, one row per control update.

    Each update, the command moves toward the gait's target by at most
    ``max_hinge_speed * period``, like a servo that turns at a constant top
    speed; it starts from 0, where the settled robot holds every hinge.
    Without a speed limit, the commands are the targets themselves.

    Parameters
    ----------
    gait
        The hinge waves.
    settings
        The servo speed limit.
    period
        Seconds between control updates.
    updates
        Number of control updates.

    Returns
    -------
    FloatArray
        ``updates`` x hinges commanded angles (radians).
    """
    times = np.arange(updates) * period
    angle = 2.0 * math.pi * gait.frequency * times[:, None] * gait.harmonic
    targets: FloatArray = np.clip(
        gait.offset + gait.amplitude * np.sin(angle + gait.phase),
        -SERVO_LIMIT,
        SERVO_LIMIT,
    )
    if settings.max_hinge_speed <= 0:
        return targets
    step = settings.max_hinge_speed * period
    commands = np.empty_like(targets)
    command = np.zeros(targets.shape[1])
    for update, target in enumerate(targets):
        command += np.clip(target - command, -step, step)
        commands[update] = command
    return commands


def run_race(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    warm_state: FloatArray,
    gait: Gait,
    settings: EvalSettings,
) -> RaceResult:
    """
    Run one dash from the settled state with an open-loop gait.

    The hinge commands (:func:`servo_commands`) change every
    :data:`CONTROL_STEPS` physics steps. The dash ends when the core has
    moved ``goal`` forward (the crossing time is interpolated between control
    updates) or after ``time_limit`` seconds.

    Parameters
    ----------
    model
        The compiled model.
    data
        Its data; overwritten.
    warm_state
        State after settling, from :func:`settle`.
    gait
        The hinge waves.
    settings
        Goal and time limit.

    Returns
    -------
    RaceResult
        Score and description of the dash.
    """
    mujoco.mj_setState(model, data, warm_state, STATE_SPEC)
    bad_steps = data.warning[BAD_QACC].number
    start = data.qpos[0:2].copy()
    start_yaw = _yaw(data.qpos[3:7])
    period = CONTROL_STEPS * model.opt.timestep
    updates = round(settings.time_limit / period)
    commands = servo_commands(gait, settings, period, updates)

    previous = furthest = 0.0
    finish_time: float | None = None
    stable = True
    done = 0
    for update in range(updates):
        data.ctrl[:] = commands[update]
        mujoco.mj_step(model, data, nstep=CONTROL_STEPS)
        done = update + 1
        if data.warning[BAD_QACC].number != bad_steps:
            stable = False
            break
        progress = float(data.qpos[0] - start[0])
        if progress >= settings.goal:
            fraction = (settings.goal - previous) / (progress - previous)
            finish_time = (update + fraction) * period
            furthest = settings.goal
            break
        furthest = max(furthest, progress)
        previous = progress

    speed = 0.0
    if stable:
        speed = race_speed(
            finish_time=finish_time,
            furthest=furthest,
            goal=settings.goal,
            time_limit=settings.time_limit,
        )
    turn = math.degrees(_yaw(data.qpos[3:7]) - start_yaw)
    return RaceResult(
        speed=speed,
        finish_time=finish_time,
        furthest=furthest,
        drift=float(data.qpos[1] - start[1]),
        turn=(turn + 180.0) % 360.0 - 180.0,
        duration=done * period,
        stable=stable,
    )


def _yaw(quaternion: FloatArray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# -- Learning ---------------------------------------------------------------


def learn_brain(task: EvalTask) -> EvalResult:
    """
    Learn a gait for one body and, if asked, prune the body and learn again.

    Without pruning this is :func:`learn_body`. With ``settings.prune``, the
    body's best dash is replayed to find the modules that touch the floor,
    the subtrees that never do are cut off (:func:`prune_genome`), and the
    pruned body learns its own gait with the same budget. The pruned body is
    scored with the faster of that gait and the old gait of its remaining
    hinges.

    Parameters
    ----------
    task
        The body, learner seed and settings.

    Returns
    -------
    EvalResult
        The scored body (``genome``), its best gait, dash and actuator
        effort; ``speed`` is ``None`` if the body cannot be built or has no
        hinges.
    """
    result = _learn_and_prune(task)
    if result.speed is None:
        return result
    effort = actuator_effort(result.genome, result.gait, task.settings)
    return replace(result, effort=effort)


def _learn_and_prune(task: EvalTask) -> EvalResult:
    started = time.perf_counter()
    settings = task.settings
    first = learn_body(task.genome, settings, task.learn_seed)
    if not settings.prune or first.speed is None:
        return replace(first, genome=task.genome)
    grounded = grounded_modules(task.genome, first.gait, settings)
    pruned, cut = prune_genome(task.genome, grounded, mirror_twins(task.genome))
    if not cut:
        return replace(first, genome=task.genome, unpruned_speed=first.speed)

    body = pruned.to_dict()
    second = learn_body(
        body,
        settings,
        derive_seed(task.learn_seed, PRUNED_SEED_KEY),
    )
    old_gait = transfer_gait(first.gait, task.genome, pruned)
    old = replay_gait(body, old_gait, settings)
    best = second
    if second.speed is None or old.speed > second.speed:
        best = replace(
            second,
            speed=old.speed,
            finish_time=old.finish_time,
            furthest=old.furthest,
            drift=old.drift,
            turn=old.turn,
            gait=old_gait,
            family=first.family,
        )
    return replace(
        best,
        genome=body,
        cut=cut,
        unpruned_speed=first.speed,
        n_sims=first.n_sims + second.n_sims + 1,
        sim_seconds=first.sim_seconds + second.sim_seconds + old.duration,
        seconds=time.perf_counter() - started,
    )


def learn_body(
    genome: dict[str, Any],
    settings: EvalSettings,
    learn_seed: int,
) -> EvalResult:
    """
    Learn a gait for one body with CMA-ES and return the best one.

    Every gait family of the brain (:data:`BRAINS`) gets its own CMA-ES run
    with an equal share of the ``learn_gens`` generations of ``learn_pop``
    races, so every body costs the same number of races. Each run starts from
    the all-zero vector (see :func:`decode_gait`) with step size
    ``learn_sigma``. The fastest gait of any family is kept.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    settings
        Task, simulation and learning settings.
    learn_seed
        Seed of the CMA-ES samplers.

    Returns
    -------
    EvalResult
        The best gait and its dash; ``speed`` is ``None`` if the body cannot
        be built or has no hinges.
    """
    started = time.perf_counter()
    try:
        model, data = build_world(
            genome,
            settings.spawn_yaw,
            settings,
        )[1:]
    except Exception:  # noqa: BLE001 -- an unbuildable body is just invalid
        return _invalid_result(started)
    if model.nu == 0:
        return _invalid_result(started)

    partners = mirror_partners(model)
    course = _Course(model, data, settle(model, data, settings.settle))
    families = BRAINS[settings.brain]
    generations = settings.learn_gens // len(families)
    layouts = {family: gait_layout(partners, family) for family in families}
    learned = {
        family: _learn_family(
            course,
            layouts[family],
            settings,
            (derive_seed(learn_seed, index), generations),
        )
        for index, family in enumerate(families)
    }
    family = max(families, key=lambda name: learned[name][0][0].speed)
    race, gait = learned[family][0]
    return EvalResult(
        speed=race.speed,
        finish_time=race.finish_time,
        furthest=race.furthest,
        drift=race.drift,
        turn=race.turn,
        gait=gait.to_dict(),
        family=family,
        family_speeds={name: learned[name][0][0].speed for name in families},
        learn_curve=np.max(
            [learned[name][1] for name in families],
            axis=0,
        ).tolist(),
        n_params=layouts[family].n_params,
        n_pairs=sum(p not in {-1, h} for h, (p, _) in enumerate(partners)) // 2,
        n_plane=sum(p == h for h, (p, _) in enumerate(partners)),
        n_unpaired=sum(p < 0 for p, _ in partners),
        n_sims=generations * len(families) * settings.learn_pop,
        sim_seconds=sum(learned[name][2] for name in families),
        seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class _Course:
    # A body after settling, ready for dashes from the same start.
    model: mujoco.MjModel
    data: mujoco.MjData
    warm_state: FloatArray


def _learn_family(
    course: _Course,
    layout: GaitLayout,
    settings: EvalSettings,
    budget: tuple[int, int],
) -> tuple[tuple[RaceResult, Gait], list[float], float]:
    seed, generations = budget

    def dash(vector: FloatArray) -> tuple[RaceResult, Gait]:
        gait = decode_gait(
            np.asarray(vector),
            layout,
            settings.max_hinge_frequency,
        )
        race = run_race(
            course.model,
            course.data,
            course.warm_state,
            gait,
            settings,
        )
        return race, gait

    learner = _make_learner(layout.n_params, settings, seed)
    return _optimise(dash, learner, generations)


def _make_learner(
    n_params: int,
    settings: EvalSettings,
    learn_seed: int,
) -> cma.CMAEvolutionStrategy:
    rng = np.random.default_rng(learn_seed)
    return cma.CMAEvolutionStrategy(
        np.zeros(n_params),
        settings.learn_sigma,
        {
            "popsize": settings.learn_pop,
            # Sample from our own generator; nan leaves numpy's global alone.
            "randn": lambda lam, dim: rng.standard_normal((lam, dim)),
            "seed": math.nan,
            "verbose": -9,
        },
    )


def _optimise(
    dash: Callable[[FloatArray], tuple[RaceResult, Gait]],
    learner: cma.CMAEvolutionStrategy,
    generations: int,
) -> tuple[tuple[RaceResult, Gait], list[float], float]:
    best: tuple[RaceResult, Gait] | None = None
    curve: list[float] = []
    sim_seconds = 0.0
    for _ in range(generations):
        vectors = learner.ask()
        outcomes = [dash(vector) for vector in vectors]
        learner.tell(vectors, [-race.speed for race, _ in outcomes])
        for outcome in outcomes:
            sim_seconds += outcome[0].duration
            if best is None or outcome[0].speed > best[0].speed:
                best = outcome
        curve.append(max(race.speed for race, _ in outcomes))
    return cast("tuple[RaceResult, Gait]", best), curve, sim_seconds


def replay_gait(
    genome: dict[str, Any],
    gait: dict[str, Any],
    settings: EvalSettings,
) -> RaceResult:
    """
    Run a stored gait on its body exactly as during learning.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    gait
        The gait stored with the body, as :meth:`Gait.to_dict`.
    settings
        The run's settings.

    Returns
    -------
    RaceResult
        The dash; its speed equals the stored speed of the same gait.
    """
    _, model, data = build_world(
        genome,
        settings.spawn_yaw,
        settings,
    )
    warm_state = settle(model, data, settings.settle)
    return run_race(model, data, warm_state, Gait.from_dict(gait), settings)


def _invalid_result(started: float) -> EvalResult:
    return EvalResult(
        speed=None,
        finish_time=None,
        furthest=0.0,
        drift=0.0,
        turn=0.0,
        gait={},
        family="",
        family_speeds={},
        learn_curve=[],
        n_params=0,
        n_pairs=0,
        n_plane=0,
        n_unpaired=0,
        n_sims=0,
        sim_seconds=0.0,
        seconds=time.perf_counter() - started,
    )


# -- Pruning ----------------------------------------------------------------


def module_of_body(model: mujoco.MjModel) -> np.ndarray:
    """
    Map every body of a model to the genome node it belongs to.

    Parameters
    ----------
    model
        A compiled model with one robot.

    Returns
    -------
    np.ndarray
        Per body id, the node id, or -1 for bodies outside the robot.
    """
    nodes = np.full(model.nbody, -1)
    core = find_core_body(model)
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        match = MODULE_BODY.search(name)
        if match is not None:
            nodes[body] = int(match.group(1))
        elif body == core:
            nodes[body] = IDX_OF_CORE
    return nodes


def grounded_modules(
    genome: dict[str, Any],
    gait: dict[str, Any],
    settings: EvalSettings,
) -> set[int]:
    """
    Replay a dash as it was scored and record the modules on the floor.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    gait
        Its gait, as :meth:`Gait.to_dict`.
    settings
        Goal, time limit and settling time.

    Returns
    -------
    set[int]
        Node ids of the modules that touched the floor at least once between
        the start of the dash and the finish line (or the time limit).
    """
    model, steps = _dash(genome, gait, settings)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    node_of_geom = module_of_body(model)[model.geom_bodyid]
    touched: set[int] = set()
    for data in steps:
        for first, second in np.asarray(data.contact.geom)[: data.ncon]:
            if first == floor:
                touched.add(int(node_of_geom[second]))
            elif second == floor:
                touched.add(int(node_of_geom[first]))
    touched.discard(-1)
    return touched


def actuator_effort(
    genome: dict[str, Any],
    gait: dict[str, Any],
    settings: EvalSettings,
) -> dict[str, float]:
    """
    Measure how hard a scored dash works the hinges.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    gait
        Its gait, as :meth:`Gait.to_dict`.
    settings
        Goal, time limit, settling time and servo limits.

    Returns
    -------
    dict[str, float]
        ``peak_torque`` (N m, of any servo), ``rms_torque`` (N m, the root
        mean square torque of the busiest servo: motor heating grows with the
        square of the current, i.e. of the torque, so compare it with the
        servo's continuous torque rating), ``peak_hinge_speed`` (rad/s),
        ``reversals_per_s`` (direction changes per second of the busiest
        hinge, counting only swings of at least :data:`SWING_BAND`) and
        ``hinge_frequency`` (Hz, the gait's fastest hinge oscillation).
    """
    model, steps = _dash(genome, gait, settings)
    joints = model.actuator_trnid[:, 0]
    doubled = double_servo_nodes(genome)
    # Torques per servo: a hinge's two servos share its torque.
    servos = np.array([
        _servo_count(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
            doubled,
        )
        for actuator in range(model.nu)
    ])
    swings = _SwingCounter(SWING_BAND)
    squared_torque = np.zeros(model.nu)
    peak_torque = peak_speed = 0.0
    count = 0
    for data in steps:
        torque = data.actuator_force / servos
        peak_torque = max(peak_torque, float(np.abs(torque).max(initial=0.0)))
        speed = np.abs(data.qvel[model.jnt_dofadr[joints]])
        peak_speed = max(peak_speed, float(speed.max(initial=0.0)))
        squared_torque += torque**2
        swings.update(data.qpos[model.jnt_qposadr[joints]])
        count += 1
    count = max(count, 1)
    rms = np.sqrt(squared_torque / count).max(initial=0.0)
    fastest = gait["frequency"] * max(gait["harmonic"], default=0.0)
    return {
        "peak_torque": round(peak_torque, 4),
        "rms_torque": round(float(rms), 4),
        "peak_hinge_speed": round(peak_speed, 3),
        "reversals_per_s": round(
            swings.busiest / (count * model.opt.timestep),
            2,
        ),
        "hinge_frequency": round(fastest, 3),
    }


@dataclass
class _SwingCounter:
    # Counts the direction changes of each hinge's signal after a swing of
    # at least `band` (hysteresis), so jitter is not counted as switching.
    band: float
    extreme: FloatArray | None = None
    direction: FloatArray = field(default_factory=lambda: np.zeros(0))
    reversals: FloatArray = field(default_factory=lambda: np.zeros(0))

    def update(self, angles: FloatArray) -> None:
        if self.extreme is None:
            self.extreme = angles.copy()
            self.direction = np.zeros_like(angles)
            self.reversals = np.zeros_like(angles)
            return
        rise = angles - self.extreme
        idle = self.direction == 0
        up, down = self.direction > 0, self.direction < 0
        turn_down = up & (rise < -self.band)
        turn_up = down & (rise > self.band)
        start_up = idle & (rise > self.band)
        start_down = idle & (rise < -self.band)
        self.reversals += turn_down | turn_up
        self.direction = np.where(
            start_up | turn_up,
            1.0,
            np.where(start_down | turn_down, -1.0, self.direction),
        )
        further = (up & (rise > 0)) | (down & (rise < 0))
        moved = further | turn_down | turn_up | start_up | start_down
        self.extreme = np.where(moved, angles, self.extreme)

    @property
    def busiest(self) -> float:
        return float(self.reversals.max(initial=0.0))


def _dash(
    genome: dict[str, Any],
    gait: dict[str, Any],
    settings: EvalSettings,
) -> tuple[mujoco.MjModel, Iterator[mujoco.MjData]]:
    # Replay a dash as it was scored, one physics step at a time (the same
    # physics as run_race), until the finish line or the time limit.
    _, model, data = build_world(
        genome,
        settings.spawn_yaw,
        settings,
    )
    warm_state = settle(model, data, settings.settle)
    mujoco.mj_setState(model, data, warm_state, STATE_SPEC)
    period = CONTROL_STEPS * model.opt.timestep
    updates = round(settings.time_limit / period)
    commands = servo_commands(Gait.from_dict(gait), settings, period, updates)

    def steps() -> Iterator[mujoco.MjData]:
        start = float(data.qpos[0])
        for update in range(updates):
            data.ctrl[:] = commands[update]
            for _ in range(CONTROL_STEPS):
                mujoco.mj_step(model, data)
                yield data
            if data.qpos[0] - start >= settings.goal:
                return

    return model, steps()


def mirror_twins(genome: dict[str, Any]) -> dict[int, int]:
    """
    Pair every module with its mirror image at the rest pose.

    Parameters
    ----------
    genome
        A mirror-symmetric body's tree genome as a dict.

    Returns
    -------
    dict[int, int]
        Node id to the node id of its mirror image (itself on the sagittal
        plane); modules without one are missing.
    """
    _, model, data = build_world(genome, spawn_yaw=0.0)
    mujoco.mj_forward(model, data)
    core = find_core_body(model)
    origin = data.xpos[core].copy()
    to_local = data.xmat[core].reshape(3, 3).copy()
    positions = {IDX_OF_CORE: np.zeros(3)}
    for body in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        match = MODULE_ROOT.search(name)
        if match is not None:
            positions[int(match.group(1))] = (
                data.xpos[body] - origin
            ) @ to_local
    return {
        node: other
        for node, position in positions.items()
        for other, other_position in positions.items()
        if np.linalg.norm(position * MIRROR - other_position) < MATCH_TOLERANCE
    }


def prune_genome(
    genome: dict[str, Any],
    grounded: set[int],
    twins: dict[int, int],
) -> tuple[TreeGenome, list[int]]:
    """
    Cut off every subtree in which no module, or its twin, touched the floor.

    A module that never touches the floor stays if a module below it does.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    grounded
        Node ids of the modules that touched the floor.
    twins
        Mirror images, from :func:`mirror_twins`.

    Returns
    -------
    tuple[TreeGenome, list[int]]
        The pruned genome and the roots of the subtrees that were cut off.
    """
    tree = tree_from_dict(genome)
    touched = grounded | {twins[node] for node in grounded if node in twins}
    children: dict[int, list[int]] = {node: [] for node in tree.nodes}
    for edge in tree.edges:
        children[edge["parent"]].append(edge["child"])

    def needed(node: int) -> bool:
        return node in touched or any(needed(c) for c in children[node])

    cut = sorted(
        edge["child"]
        for edge in tree.edges
        if needed(edge["parent"]) and not needed(edge["child"])
    )
    for root in cut:
        remove_subtree(tree, root)
    return tree, cut


def transfer_gait(
    gait: dict[str, Any],
    genome: dict[str, Any],
    pruned: TreeGenome,
) -> dict[str, Any]:
    """
    Keep the waves of the hinges that are left after pruning.

    Parameters
    ----------
    gait
        The original body's gait, one entry per actuator.
    genome
        The original body's tree genome as a dict.
    pruned
        The pruned genome.

    Returns
    -------
    dict[str, Any]
        A gait for the pruned body, as :meth:`Gait.to_dict`.
    """
    old_hinges = _hinge_nodes(build_world(genome, spawn_yaw=0.0)[1])
    new_hinges = _hinge_nodes(build_world(pruned.to_dict(), spawn_yaw=0.0)[1])
    keep = [old_hinges.index(node) for node in new_hinges]
    result: dict[str, Any] = {"frequency": gait["frequency"]}
    for key in ("amplitude", "offset", "phase", "harmonic"):
        result[key] = [gait[key][actuator] for actuator in keep]
    return result


def _hinge_nodes(model: mujoco.MjModel) -> list[int]:
    nodes = module_of_body(model)
    joint_body = model.jnt_bodyid[model.actuator_trnid[:, 0]]
    return [int(nodes[body]) for body in joint_body]


# -- Bodies -----------------------------------------------------------------


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


def body_key(genome: TreeGenome) -> str:
    """
    Return a short id of the body that ``genome`` builds.

    Parameters
    ----------
    genome
        The genome to identify.

    Returns
    -------
    str
        16 hex digits; genomes with the same subtree signature (see
        :func:`ariel.ec.genotypes.tree.symmetry.subtree_signature`) build
        the same body and get the same id.
    """
    signature = repr(subtree_signature(genome, IDX_OF_CORE)).encode()
    return hashlib.blake2b(signature, digest_size=8).hexdigest()


def is_valid_body(genome: TreeGenome, cfg: RunConfig) -> bool:
    """
    Check a genome against the run's size limits.

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


def mutate_body(
    genome: TreeGenome,
    cfg: RunConfig,
    rng: random.Random,
) -> tuple[TreeGenome, str]:
    """
    Apply the design-priors mutation mix, plus growth, to a copy of a body.

    Parameters
    ----------
    genome
        The body to mutate; it is not modified.
    cfg
        Run settings (share of growth, module limit for subtree replacement).
    rng
        Random source for the mutation choice.

    Returns
    -------
    tuple[TreeGenome, str]
        The mutant, or an unchanged copy if the mutant is malformed, and the
        mutation it got (from :data:`MUTATIONS`, with ``+servo`` if a hinge
        also switched its servos).
    """
    mutant = copy.deepcopy(genome)
    # Without growth this draws exactly as the four-mutation mix did.
    weights = (*(w * (1.0 - cfg.grow) for w in MUTATION_WEIGHTS), cfg.grow)
    mutation = rng.choices(MUTATIONS, weights=weights)[0]
    match mutation:
        case "point":
            mutate_replace_node(mutant)
        case "subtree":
            mutate_subtree_replacement(mutant, max_modules=cfg.max_modules)
        case "shrink":
            mutate_shrink(mutant)
        case "hoist":
            mutate_hoist(mutant)
        case _:
            grow_limb(mutant, rng)
    # Without double servos nothing more is drawn, as before they existed.
    if cfg.double_servo > 0 and rng.random() < cfg.double_servo:
        toggle_servos(mutant, rng)
        mutation += "+servo"

    non_core = [node for node in sorted(mutant.nodes) if node != IDX_OF_CORE]
    if non_core and rng.random() < ROTATION_MUTATION_RATE:
        node = mutant.nodes[rng.choice(non_core)]
        rotations = ALLOWED_ROTATIONS[ModuleType[node["type"]]]
        node["rotation"] = rng.choice(rotations).name

    try:
        validate_genome_dict(mutant.to_dict())
    except ValueError:
        return copy.deepcopy(genome), mutation
    return mutant, mutation


def grow_limb(genome: TreeGenome, rng: random.Random) -> None:
    """
    Lengthen a limb by one module, in place.

    A new brick or hinge with a random rotation goes to a random place along
    a limb: between a module and its parent, which pushes that module and
    everything below it one module outward, or on the front face of a limb's
    last module. Every module is 10 cm long, so the limb grows by 10 cm.

    Parameters
    ----------
    genome
        The body to grow.
    rng
        Random source.
    """
    parents = {edge["parent"] for edge in genome.edges}
    tips = [
        node
        for node in sorted(genome.nodes)
        if node != IDX_OF_CORE and node not in parents
    ]
    places = len(genome.edges) + len(tips)
    if places == 0:
        return
    new = max(genome.nodes) + 1
    kind = rng.choice(GROWN_TYPES)
    rotation = rng.choice(ALLOWED_ROTATIONS[kind]).name
    place = rng.randrange(places)
    if place < len(genome.edges):
        edge = genome.edges[place]
        genome.edges.append({
            "parent": new,
            "child": edge["child"],
            "face": "FRONT",
        })
        edge["child"] = new
    else:
        tip = tips[place - len(genome.edges)]
        genome.edges.append({"parent": tip, "child": new, "face": "FRONT"})
    genome.nodes[new] = {"type": kind.name, "rotation": rotation}


def toggle_servos(genome: TreeGenome, rng: random.Random) -> None:
    """
    Switch a random hinge between one servo and two in parallel, in place.

    The servo count is the genome key :data:`SERVOS` of the hinge's node,
    left out for one servo; mirroring copies it to the hinge's twin.

    Parameters
    ----------
    genome
        The body to change.
    rng
        Random source.
    """
    hinges = [
        node
        for node in sorted(genome.nodes)
        if genome.nodes[node]["type"] == ModuleType.HINGE.name
    ]
    if not hinges:
        return
    attributes = genome.nodes[rng.choice(hinges)]
    if attributes.pop(SERVOS, 1) == 1:
        attributes[SERVOS] = 2


def tree_from_dict(genome: dict[str, Any]) -> TreeGenome:
    """
    Rebuild a tree genome from a dict, keeping its hinges' servo counts.

    :meth:`TreeGenome.from_dict` keeps only each module's type and rotation.

    Parameters
    ----------
    genome
        A tree genome as a dict, e.g. from the database.

    Returns
    -------
    TreeGenome
        The genome, with :data:`SERVOS` where the dict has it.
    """
    tree = TreeGenome.from_dict(genome)
    for node, attributes in genome["nodes"].items():
        if SERVOS in attributes:
            tree.nodes[int(node)][SERVOS] = attributes[SERVOS]
    return tree


def count_double_servos(genome: TreeGenome) -> int:
    """
    Count the hinges with two servos.

    Parameters
    ----------
    genome
        The body.

    Returns
    -------
    int
        Hinges whose :data:`SERVOS` is 2.
    """
    return sum(
        attributes.get(SERVOS, 1) == 2 for attributes in genome.nodes.values()
    )


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
        Worker pool, or ``None`` to learn in-process.
    snapshot
        Called with the survivors and the generation when a picture is due.
    meta
        The run metadata, rewritten after every generation.
    """

    cfg: RunConfig
    rng: random.Random
    executor: ProcessPoolExecutor | None = None
    snapshot: Snapshot | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    next_uid: int = 0
    total_sims: int = 0
    total_sim_seconds: float = 0.0
    generation: int = 0
    best_speed: float = -math.inf
    history: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)


def new_individual(  # noqa: PLR0913 -- one body needs all of these
    genome: TreeGenome,
    ctx: RunContext,
    *,
    origin: str,
    symmetry: SymmetryScore,
    parents: Sequence[Individual] = (),
    attempts: int = 1,
    mutation: str | None = None,
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
        "random", "offspring" or "immigrant" (a random body used when
        variation kept failing).
    symmetry
        The body's measured symmetry.
    parents
        Parents the body was made from.
    attempts
        Variation attempts it took.
    mutation
        The mutation of the attempt that made it (see :func:`mutate_body`).

    Returns
    -------
    Individual
        An individual that still needs evaluation.
    """
    uid = ctx.next_uid
    ctx.next_uid += 1
    individual = Individual()
    individual.genotype = {"morph": genome.to_dict()}
    individual.tags = {
        "uid": uid,
        "origin": origin,
        "parents": [int(cast("int", p.tags["uid"])) for p in parents],
        "attempts": attempts,
        "mutation": mutation,
        "body_key": body_key(genome),
        "n_modules": count_modules(genome),
        "n_hinges": count_hinges(genome),
        "n_double": count_double_servos(genome),
        "depth": get_tree_depth(genome),
        "sym_position": symmetry.position,
        "sym_axis": symmetry.joint_axis,
        "learn_seed": derive_seed(ctx.cfg.seed, uid),
    }
    individual.requires_eval = True
    return individual


def symmetric_body_or_none(
    genome: TreeGenome,
    cfg: RunConfig,
    taken: set[str],
) -> SymmetryScore | None:
    """
    Check that a body is valid, new and exactly symmetric.

    Parameters
    ----------
    genome
        The candidate body.
    cfg
        Run settings (size limits).
    taken
        Ids (:func:`body_key`) of the bodies it must differ from.

    Returns
    -------
    SymmetryScore | None
        Its symmetry, which is perfect, or ``None`` if the body is rejected.
    """
    if not is_valid_body(genome, cfg) or body_key(genome) in taken:
        return None
    symmetry = genome_symmetry(genome)
    return symmetry if symmetry.combined == 1.0 else None


def initial_population(ctx: RunContext) -> Population:
    """
    Create ``pop`` different random symmetric bodies.

    Parameters
    ----------
    ctx
        Run state.

    Returns
    -------
    Population
        Unevaluated individuals.

    Raises
    ------
    RuntimeError
        If not enough different symmetric bodies can be drawn.
    """
    cfg, rng = ctx.cfg, ctx.rng
    individuals: list[Individual] = []
    taken: set[str] = set()
    for _ in range(cfg.pop * MAX_INIT_TRIES):
        if len(individuals) == cfg.pop:
            return Population(individuals)
        genome = random_symmetric_body(cfg, rng)
        symmetry = symmetric_body_or_none(genome, cfg, taken)
        if symmetry is not None:
            taken.add(body_key(genome))
            individuals.append(
                new_individual(genome, ctx, origin="random", symmetry=symmetry),
            )
    msg = f"Drew only {len(individuals)} of {cfg.pop} initial bodies."
    raise RuntimeError(msg)


def tournament(
    pool: Sequence[Individual],
    size: int,
    rng: random.Random,
) -> Individual:
    """
    Pick the fittest of ``size`` random members of ``pool``.

    Parameters
    ----------
    pool
        Evaluated candidates.
    size
        Tournament size.
    rng
        Random source.

    Returns
    -------
    Individual
        The winner; ties go to the younger body.
    """
    contestants = rng.sample(list(pool), min(size, len(pool)))
    return max(contestants, key=_rank_key)


def make_offspring(
    pool: Sequence[Individual],
    ctx: RunContext,
    taken: set[str],
) -> Individual:
    """
    Vary tournament winners into a new valid symmetric body.

    A child comes from subtree crossover of two winners (with probability
    :data:`CROSSOVER_RATE`) or from one winner, is mutated, and is then made
    symmetric again by copying the core side that changed onto the other
    side. It must be valid, exactly symmetric and absent from ``taken``.

    Parameters
    ----------
    pool
        The evaluated population to select parents from.
    ctx
        Run state.
    taken
        Ids of the bodies the child must differ from; its own id is added.

    Returns
    -------
    Individual
        The child, or a random symmetric body ("immigrant") if every attempt
        failed.

    Raises
    ------
    RuntimeError
        If not even a new random body can be found.
    """
    cfg, rng = ctx.cfg, ctx.rng
    for attempt in range(1, cfg.retries + 1):
        if rng.random() < CROSSOVER_RATE:
            used = [
                tournament(pool, cfg.tournament, rng),
                tournament(pool, cfg.tournament, rng),
            ]
            children = crossover_subtree(_genome(used[0]), _genome(used[1]))
            pick = rng.randrange(2)
            child, base = children[pick], used[pick]
        else:
            base = tournament(pool, cfg.tournament, rng)
            used = [base]
            child = _genome(base)
        child, mutation = mutate_body(child, cfg, rng)
        side = changed_core_side(child, _genome(base))
        child = symmetrize(child, rng=rng, core_source=side)
        symmetry = symmetric_body_or_none(child, cfg, taken)
        if symmetry is not None:
            taken.add(body_key(child))
            return new_individual(
                child,
                ctx,
                origin="offspring",
                symmetry=symmetry,
                parents=used,
                attempts=attempt,
                mutation=mutation,
            )
    for _ in range(MAX_INIT_TRIES):
        genome = random_symmetric_body(cfg, rng)
        symmetry = symmetric_body_or_none(genome, cfg, taken)
        if symmetry is not None:
            taken.add(body_key(genome))
            return new_individual(
                genome,
                ctx,
                origin="immigrant",
                symmetry=symmetry,
                attempts=cfg.retries,
            )
    msg = "Could not find a new symmetric body."
    raise RuntimeError(msg)


def reproduction(population: Population, ctx: RunContext) -> Population:
    """
    Add ``offspring`` new bodies, bred by tournament selection.

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
    pool = list(population)
    taken = {str(ind.tags["body_key"]) for ind in pool}
    offspring = [
        make_offspring(pool, ctx, taken) for _ in range(ctx.cfg.offspring)
    ]
    population.extend(offspring)
    return population


def evaluate(population: Population, ctx: RunContext) -> Population:
    """
    Learn a brain for every individual that still needs evaluation.

    Parameters
    ----------
    population
        The population, possibly with new offspring.
    ctx
        Run state (worker pool and counters).

    Returns
    -------
    Population
        The population with fitness (the speed, maximised) and tags; a
        pruned body replaces the body it was cut from.
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
        ctx.total_sim_seconds += result.sim_seconds
        if result.speed is not None:
            ctx.best_speed = max(ctx.best_speed, result.speed)
        ind.fitness = INVALID_SPEED if result.speed is None else result.speed
        ind.tags = {
            "speed": result.speed,
            "finish_time": result.finish_time,
            "reached": result.finish_time is not None,
            "furthest": result.furthest,
            "drift": result.drift,
            "turn": result.turn,
            "gait": result.gait,
            "family": result.family,
            "family_speeds": result.family_speeds,
            "learn_curve": result.learn_curve,
            "n_params": result.n_params,
            "n_pairs": result.n_pairs,
            "n_plane": result.n_plane,
            "n_unpaired": result.n_unpaired,
            "n_sims": result.n_sims,
            "sim_seconds": round(result.sim_seconds, 2),
            "eval_seconds": round(result.seconds, 3),
            "unpruned_speed": result.unpruned_speed,
            "cut": result.cut,
            **result.effort,
        }
        if result.cut:
            _adopt_pruned_body(ind, result.genome)
    return population


def _adopt_pruned_body(individual: Individual, genome: dict[str, Any]) -> None:
    tags = individual.tags
    pruned = tree_from_dict(genome)
    symmetry = genome_symmetry(pruned)
    individual.genotype = {"morph": genome}
    individual.tags = {
        "unpruned_modules": tags["n_modules"],
        "unpruned_hinges": tags["n_hinges"],
        "body_key": body_key(pruned),
        "n_modules": count_modules(pruned),
        "n_hinges": count_hinges(pruned),
        "n_double": count_double_servos(pruned),
        "depth": get_tree_depth(pruned),
        "sym_position": symmetry.position,
        "sym_axis": symmetry.joint_axis,
    }


def survivor_selection(population: Population, ctx: RunContext) -> Population:
    """
    Keep the ``pop`` fittest different bodies (mu + lambda truncation).

    Pruning can turn two different bodies into the same one; only the
    fitter copy (the younger on a tie) can survive.

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
    survivors: list[Individual] = []
    seen: set[str] = set()
    for ind in sorted(population, key=_rank_key, reverse=True):
        key = str(ind.tags["body_key"])
        ind.alive = len(survivors) < ctx.cfg.pop and key not in seen
        if ind.alive:
            survivors.append(ind)
            seen.add(key)
    record_generation(survivors, ctx)
    return population


def record_generation(survivors: list[Individual], ctx: RunContext) -> None:
    """
    Log a generation, rewrite the metadata and save a picture if due.

    Parameters
    ----------
    survivors
        The bodies kept this generation, best first.
    ctx
        Run state.
    """
    cfg = ctx.cfg
    speeds = [float(_speed(ind)) for ind in survivors]
    modules = [int(cast("int", ind.tags["n_modules"])) for ind in survivors]
    hinges = [int(cast("int", ind.tags["n_hinges"])) for ind in survivors]
    best = survivors[0].tags
    finish_time = best.get("finish_time")
    entry = {
        "generation": ctx.generation,
        "best_uid": best["uid"],
        "best_speed": speeds[0],
        "best_finish_time": finish_time,
        "mean_speed": float(np.mean(speeds)),
        "median_speed": float(np.median(speeds)),
        "finishers": sum(bool(ind.tags.get("reached")) for ind in survivors),
        "best_family": best.get("family"),
        "best_modules": best["n_modules"],
        "best_hinges": best["n_hinges"],
        "mean_modules": float(np.mean(modules)),
        "mean_hinges": float(np.mean(hinges)),
        "pruned_survivors": sum(bool(i.tags.get("cut")) for i in survivors),
        **{
            f"{family}_survivors": sum(
                ind.tags.get("family") == family for ind in survivors
            )
            for family in FAMILIES
        },
        "total_sims": ctx.total_sims,
        "sim_hours": round(ctx.total_sim_seconds / 3600, 3),
        "elapsed": round(time.perf_counter() - ctx.started, 1),
    }
    ctx.history.append(entry)
    ctx.meta["history"] = ctx.history
    write_json(cfg.meta_path, ctx.meta)

    reach = (
        f"{cfg.goal:g} m in {float(cast('float', finish_time)):.1f} s"
        if finish_time is not None
        else f"furthest {float(cast('float', best['furthest'])):.2f} m"
    )
    print(  # noqa: T201 -- progress line for the run log
        f"[{cfg.brain} seed {cfg.seed}] gen {ctx.generation}/{cfg.gens} "
        f"best {speeds[0]:.3f} m/s ({reach}, {best.get('family')}), "
        f"finishers {entry['finishers']}/{len(survivors)}, mean "
        f"{entry['mean_speed']:.3f} m/s, sims {ctx.total_sims}, elapsed "
        f"{entry['elapsed']:.0f} s",
        flush=True,
    )

    if ctx.snapshot is not None:
        ctx.snapshot(survivors, ctx.generation)


def is_due(generation: int, every: int, last: int) -> bool:
    """
    Tell whether a periodic output is due in a generation.

    Parameters
    ----------
    generation
        The generation just finished.
    every
        Period in generations; 0 means never.
    last
        The run's last generation, which always gets one.

    Returns
    -------
    bool
        True every ``every`` generations (from 0) and in the last one.
    """
    return every > 0 and (generation % every == 0 or generation == last)


def _rank_key(individual: Individual) -> tuple[float, int]:
    fitness = individual.fitness_
    uid = int(cast("int", individual.tags["uid"]))
    return (-math.inf if fitness is None else fitness, uid)


def _speed(individual: Individual) -> float:
    speed = individual.tags.get("speed")
    return 0.0 if speed is None else float(cast("float", speed))


def _genome(individual: Individual) -> TreeGenome:
    genotype = cast("dict[str, Any]", individual.genotype)
    return tree_from_dict(copy.deepcopy(genotype["morph"]))


# -- Run --------------------------------------------------------------------


def run_experiment(
    cfg: RunConfig,
    snapshot: Snapshot | None = None,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Run one seed of the experiment and write its outputs.

    Writes the EA database, ``*_meta.json`` (settings, versions, status,
    timing and one line per generation) and ``*_reeval.json`` (fresh-seed
    re-learning of the best bodies) to ``cfg.run_dir``.

    Parameters
    ----------
    cfg
        Run settings.
    snapshot
        Called after every generation with the survivors (best first) and
        the generation; it decides what to save (see :func:`is_due`).
    resume
        Continue the run stored in ``cfg.run_dir`` from its last finished
        generation up to generation ``cfg.gens`` (:func:`resume_run`)
        instead of starting a new one.

    Returns
    -------
    dict[str, Any]
        The final metadata, as written to ``cfg.meta_path``.
    """
    ctx = RunContext(cfg=cfg, rng=random.Random(cfg.seed), snapshot=snapshot)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        resume_run(ctx)
    else:
        start_run(ctx)

    if cfg.workers > 1:
        ctx.executor = ProcessPoolExecutor(
            max_workers=cfg.workers,
            mp_context=get_context("spawn"),
        )
    try:
        evolve(ctx, resume=resume)
        reeval = reevaluate_best(cfg, ctx)
    finally:
        if ctx.executor is not None:
            ctx.executor.shutdown()

    write_json(cfg.reeval_path, reeval)
    best = ctx.history[-1]
    ctx.meta.update({
        "status": "completed",
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": round(time.perf_counter() - ctx.started, 1),
        "total_sims": ctx.total_sims,
        "sim_hours": round(ctx.total_sim_seconds / 3600, 3),
        "best_speed": ctx.best_speed,
        "best_uid": best["best_uid"],
        "reeval_best": reeval["reeval_best"],
    })
    write_json(cfg.meta_path, ctx.meta)
    return ctx.meta


def start_run(ctx: RunContext) -> None:
    """
    Seed the random sources and write the metadata of a new run.

    Parameters
    ----------
    ctx
        Run state, fresh.

    Raises
    ------
    FileExistsError
        If the run's database exists; it is never overwritten.
    """
    cfg = ctx.cfg
    if cfg.db_path.exists():
        msg = f"{cfg.db_path} exists; continue that run with --resume."
        raise FileExistsError(msg)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 -- ARIEL modules use the global
    ctx.meta = {
        "config": _config_dict(cfg),
        "status": "running",
        "pid": os.getpid(),
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": _versions(),
    }
    write_json(cfg.meta_path, ctx.meta)


def evolve(ctx: RunContext, *, resume: bool) -> None:
    """
    Run the EA up to generation ``cfg.gens``.

    A new run first draws and scores its initial population (generation 0);
    a resumed run goes on in its database from ``ctx.generation``.

    Parameters
    ----------
    ctx
        Run state, from :func:`start_run` or :func:`resume_run`.
    resume
        Whether the run was restored by :func:`resume_run`.
    """
    cfg = ctx.cfg
    operations = [
        EAOperation(reproduction, ctx=ctx),
        EAOperation(evaluate, ctx=ctx),
        EAOperation(survivor_selection, ctx=ctx),
    ]
    if resume:
        ea: EA = _ContinuedEA(
            Population([]),  # every step fetches the survivors from the db
            operations,
            first_generation_id=ctx.generation,
            num_steps=cfg.gens - ctx.generation,
            is_maximisation=True,
            db_file_path=cfg.db_path,
            quiet=True,
        )
    else:
        population = evaluate(initial_population(ctx), ctx)
        record_generation(sorted(population, key=_rank_key, reverse=True), ctx)
        ea = EA(
            population,
            operations,
            num_steps=cfg.gens,
            is_maximisation=True,
            db_file_path=cfg.db_path,
            db_handling="halt",
            quiet=True,
        )
    ea.run()


class _ContinuedEA(EA):
    """The EA engine, going on in the database of an earlier run."""

    def _setup_database(  # noqa: PLR6301 -- overrides EA's method
        self,
        db_file_path: Path,
        db_handling: DBHandlingMode,  # noqa: ARG002 -- the database is kept
    ) -> Engine:
        return create_engine(f"sqlite:///{db_file_path}")


def resume_run(ctx: RunContext) -> None:
    """
    Restore a stopped or finished run so that it can go on.

    The run's database, metadata and re-evaluation are first copied to
    ``<stem>_backup_gen<NNN>/``. The survivors of the last finished
    generation are in the database, where the EA picks them up; the
    counters and the per-generation log are restored from the metadata.
    The random sources are brought to where they were after that generation
    by redoing the variation of every generation, which is quick (nothing is
    learned), and checking that it makes the stored bodies
    (:func:`replay_variation`). The run then goes on exactly as if it had
    never stopped. If the bodies differ, e.g. because the variation code
    changed, the random sources are seeded from the run seed and the
    generation instead.

    Parameters
    ----------
    ctx
        Run state, fresh. ``ctx.cfg`` must have the stored run's settings,
        apart from those in :data:`RESUMABLE`, and ``gens`` at least its
        last generation (equal: only the final re-evaluation is redone).

    Raises
    ------
    FileNotFoundError
        If there is no run to resume.
    RuntimeError
        If the run is still running.
    """
    cfg = ctx.cfg
    if not cfg.db_path.exists():
        msg = f"No run to resume at {cfg.db_path}."
        raise FileNotFoundError(msg)
    stored = json.loads(cfg.meta_path.read_text(encoding="utf-8"))
    if _still_running(stored):
        msg = f"The run is still running (pid {stored['pid']}); stop it first."
        raise RuntimeError(msg)
    rows = sorted(load_individuals(cfg.db_path), key=operator.itemgetter("id"))
    individuals = [_stored_individual(row) for row in rows]
    last = max(ind.time_of_death for ind in individuals)
    _check_resumable(cfg, stored, last)
    backup = _backup(cfg, last)

    differs = replay_variation(ctx, individuals, last)
    if differs is None:
        random_state = "replayed"
    else:
        seed = derive_seed(cfg.seed, last)
        random.seed(seed)
        np.random.seed(seed)  # noqa: NPY002 -- as in start_run
        ctx.rng = random.Random(seed)
        random_state = f"reseeded: the replay differs in generation {differs}"
    _restore_counters(ctx, individuals, stored["history"], last)
    ctx.meta = {k: v for k, v in stored.items() if k not in FINAL_FIELDS}
    ctx.meta.update({
        "config": _config_dict(cfg),
        "status": "running",
        "pid": os.getpid(),
        "host": platform.node(),
        "history": ctx.history,
        "resumed": [
            *stored.get("resumed", []),
            {
                "generation": last,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "random_state": random_state,
                "versions": _versions(),
            },
        ],
    })
    write_json(cfg.meta_path, ctx.meta)
    print(  # noqa: T201 -- for the run log
        f"[{cfg.brain} seed {cfg.seed}] resuming after generation {last} of "
        f"{cfg.gens}; random state {random_state}; backup in {backup}",
        flush=True,
    )


def replay_variation(
    ctx: RunContext,
    individuals: Sequence[Individual],
    last: int,
) -> int | None:
    """
    Redo the variation of generations 0 to ``last`` of a stored run.

    Variation is the only user of the run's random sources in the main
    process (learning has its own seeds), and what it varies, the survivors
    of the previous generation, is in the database. Redoing it brings the
    random sources to where they were after generation ``last``.

    Parameters
    ----------
    ctx
        Run state, fresh; its random sources and counters are advanced.
    individuals
        Every individual of the run, in database order.
    last
        The run's last finished generation.

    Returns
    -------
    int | None
        ``None`` if every body made again is the stored one, else the first
        generation in which one is not.
    """
    cfg = ctx.cfg
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # noqa: NPY002 -- as in start_run
    ctx.rng = random.Random(cfg.seed)
    stored = {int(cast("int", ind.tags["uid"])): ind for ind in individuals}
    made: list[Individual] = list(initial_population(ctx))
    for generation in range(last + 1):
        if generation > 0:
            pool = [
                ind
                for ind in individuals
                if ind.time_of_birth < generation <= ind.time_of_death
            ]
            made = list(reproduction(Population(pool), ctx))[len(pool) :]
        if not all(_made_as_stored(ind, stored, generation) for ind in made):
            return generation
    return None


def _made_as_stored(
    made: Individual,
    stored: dict[int, Individual],
    generation: int,
) -> bool:
    original = stored.get(int(cast("int", made.tags["uid"])))
    if original is None or original.time_of_birth != generation:
        return False
    if any(
        made.tags[key] != original.tags.get(key)
        for key in ("origin", "parents", "attempts")
    ):
        return False
    genome = _genome(made)
    for root in cast("list[int]", original.tags.get("cut") or []):
        remove_subtree(genome, root)  # pruned after learning, as stored
    return body_key(genome) == original.tags["body_key"]


def _stored_individual(row: dict[str, Any]) -> Individual:
    return Individual(
        alive=row["alive"],
        time_of_birth=row["birth"],
        time_of_death=row["death"],
        requires_eval=False,
        fitness_=row["fitness"],
        genotype_=row["genotype"],
        tags_=row["tags"],
    )


def _restore_counters(
    ctx: RunContext,
    individuals: Sequence[Individual],
    history: list[dict[str, Any]],
    last: int,
) -> None:
    entry = history[last]
    speeds = [ind.tags.get("speed") for ind in individuals]
    ctx.generation = last
    ctx.next_uid = 1 + max(int(cast("int", i.tags["uid"])) for i in individuals)
    ctx.total_sims = int(entry["total_sims"])
    ctx.total_sim_seconds = float(entry["sim_hours"]) * 3600
    ctx.best_speed = max(
        (float(cast("float", s)) for s in speeds if s is not None),
        default=-math.inf,
    )
    ctx.history = history[: last + 1]
    # Elapsed time counts on from where the run stopped.
    ctx.started = time.perf_counter() - float(entry["elapsed"])


def _check_resumable(
    cfg: RunConfig,
    stored: dict[str, Any],
    last: int,
) -> None:
    current = _config_dict(cfg)
    settings = {**LEGACY_SETTINGS, **stored["config"]}
    changed = [
        f"{name} ({settings.get(name)!r} -> {value!r})"
        for name, value in current.items()
        if name not in RESUMABLE and settings.get(name) != value
    ]
    if changed:
        msg = f"Settings differ from the stored run: {', '.join(changed)}."
        raise ValueError(msg)
    if cfg.gens < last:
        msg = f"The run already has {last} generations; ask for more (--gens)."
        raise ValueError(msg)
    logged = [entry["generation"] for entry in stored["history"]]
    if logged[: last + 1] != list(range(last + 1)):
        msg = "The metadata does not log every generation in the database."
        raise ValueError(msg)


def _still_running(meta: dict[str, Any]) -> bool:
    if meta.get("status") != "running" or meta.get("host") != platform.node():
        return False
    command = subprocess.run(  # noqa: S603 -- a pid, no shell
        ["ps", "-p", str(meta.get("pid")), "-o", "command="],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return "run_symmetric_coevolution" in command


def _backup(cfg: RunConfig, generation: int) -> Path:
    folder = cfg.run_dir / f"{cfg.stem}_backup_gen{generation:03d}"
    folder.mkdir(exist_ok=True)
    for path in (cfg.db_path, cfg.meta_path, cfg.reeval_path):
        if path.exists() and not (folder / path.name).exists():
            shutil.copy2(path, folder / path.name)
    return folder


def _config_dict(cfg: RunConfig) -> dict[str, Any]:
    return {**asdict(cfg), "out": str(cfg.out)}


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

    A body's speed is the best of one learning run, so the survivors of a
    (mu + lambda) EA are biased toward lucky learning runs. Each of the
    ``reeval_top`` fastest bodies is re-learned ``reeval_seeds`` times.

    Parameters
    ----------
    cfg
        Run settings.
    ctx
        Run state (worker pool and counters).

    Returns
    -------
    dict[str, Any]
        Per body its run speed and re-learned speeds and finish times, and
        ``reeval_best``, the highest mean re-learned speed.
    """
    individuals = [
        ind
        for ind in load_individuals(cfg.db_path)
        if ind["tags"].get("speed") is not None
    ]
    individuals.sort(key=lambda ind: ind["tags"]["speed"], reverse=True)
    top = individuals[: cfg.reeval_top]
    # The stored bodies are already pruned; re-learn them as they are.
    settings = replace(cfg.eval_settings(), prune=False)
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
    ctx.total_sim_seconds += sum(result.sim_seconds for result in results)

    bodies = []
    for index, ind in enumerate(top):
        chunk = results[
            index * cfg.reeval_seeds : (index + 1) * cfg.reeval_seeds
        ]
        speeds = [r.speed for r in chunk if r.speed is not None]
        bodies.append({
            "uid": ind["tags"]["uid"],
            "run_speed": ind["tags"]["speed"],
            "run_finish_time": ind["tags"]["finish_time"],
            "reeval_speeds": speeds,
            "reeval_finish_times": [r.finish_time for r in chunk],
            "reeval_mean": float(np.mean(speeds)) if speeds else None,
        })
    means = [b["reeval_mean"] for b in bodies if b["reeval_mean"] is not None]
    return {"bodies": bodies, "reeval_best": max(means) if means else None}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Write ``payload`` to ``path`` atomically.

    Parameters
    ----------
    path
        Destination file.
    payload
        JSON-serialisable data.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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
        "cma": cma.__version__,
        "numpy": np.__version__,
        "git_commit": commit,
    }
