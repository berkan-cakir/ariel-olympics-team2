from dataclasses import dataclass

import numpy as np
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import BaseWorld

from controller import Brain, BrainConfig


@dataclass
class FitnessConfig:
    environments: tuple[int, ...] = (0,)
    weights: tuple[float, ...] | None = None


def load_body(source: str) -> CoreModule:
    raise NotImplementedError


def make_world(env_id: int) -> BaseWorld:
    raise NotImplementedError


def simulate(brain: Brain, body: CoreModule, env_id: int):
    raise NotImplementedError


def fitness(genotype: np.ndarray, body: str, controller: BrainConfig, cfg: FitnessConfig) -> float:
    raise NotImplementedError
