import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import mujoco as mj
import numpy as np
import torch
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency
from ariel.simulation.controllers.simple_cpg import SimpleCPG
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from torch import nn
from torch.nn.utils import vector_to_parameters

MAX_CPG_DT = 0.01
CLOCK_HZ = (0.5, 2.5)
ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "elu": nn.ELU, "relu": nn.ReLU}


@dataclass(frozen=True)
class BrainConfig:
    name: Literal["na_cpg", "simple_cpg", "mlp", "mlp_clock"] = "na_cpg"
    hidden: tuple[int, ...] = (16,)
    activation: Literal["tanh", "elu", "relu"] = "tanh"


def parse_hidden(spec: str) -> tuple[int, ...]:
    return tuple(int(h) for h in spec.split(",") if int(h) > 0)


def scale(genes: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (genes + 1) / 2 * (high - low)


class Brain(ABC):
    @classmethod
    @abstractmethod
    def genome_size(cls, n_actuators: int, cfg: BrainConfig) -> int: ...

    @abstractmethod
    def __init__(self, genotype: np.ndarray, n_actuators: int, dt: float, cfg: BrainConfig) -> None: ...

    @abstractmethod
    def __call__(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray: ...


class CPGBrain(Brain):
    groups: tuple[tuple[str, float, float], ...]

    @classmethod
    def genome_size(cls, n_actuators: int, cfg: BrainConfig) -> int:
        return len(cls.groups) * n_actuators

    def __init__(self, genotype: np.ndarray, n_actuators: int, dt: float, cfg: BrainConfig) -> None:
        self.substeps = math.ceil(dt / MAX_CPG_DT)
        genes = np.asarray(genotype).reshape(len(self.groups), n_actuators)
        params = {name: scale(g, low, high) for (name, low, high), g in zip(self.groups, genes, strict=True)}
        self.cpg = self.build(params, n_actuators, dt / self.substeps)

    @abstractmethod
    def build(self, params: dict[str, np.ndarray], n_actuators: int, dt: float) -> NaCPG | SimpleCPG: ...

    def __call__(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray:
        angles = self.cpg.forward(time=data.time)
        for _ in range(self.substeps - 1):
            angles = self.cpg.forward()
        return angles.numpy()


class NaCPGBrain(CPGBrain):
    groups = (
        ("phase", -np.pi, np.pi),
        ("w", 2.0, 15.0),
        ("amplitudes", 0.0, np.pi / 2),
        ("ha", -0.5, 0.5),
    )

    def build(self, params: dict[str, np.ndarray], n_actuators: int, dt: float) -> NaCPG:
        cpg = NaCPG(create_fully_connected_adjacency(n_actuators), dt=dt)
        cpg.set_param_with_dict(params)
        phase = torch.as_tensor(params["phase"], dtype=torch.float32)
        xy = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)
        cpg.initial_state.update(xy=xy, xy_dot_old=xy.clone())
        cpg.reset()
        return cpg


class SimpleCPGBrain(CPGBrain):
    groups = (
        ("phase", -np.pi / 2, np.pi / 2),
        ("w", 2.0, 15.0),
        ("amplitudes", 0.0, 1.5),
        ("initial_phase", -np.pi, np.pi),
    )

    def build(self, params: dict[str, np.ndarray], n_actuators: int, dt: float) -> SimpleCPG:
        cpg = SimpleCPG({i: [] for i in range(n_actuators)}, dt=dt)
        initial_phase = torch.as_tensor(params.pop("initial_phase"), dtype=torch.float32)
        cpg.set_param_with_dict({**params, "b": np.zeros(n_actuators)})
        cpg.initial_state.update(x=torch.cos(initial_phase), y=torch.sin(initial_phase))
        cpg.reset()
        return cpg


class MLPBrain(Brain):
    clock = False

    @classmethod
    def network(cls, n_actuators: int, cfg: BrainConfig) -> nn.Sequential:
        sizes = (3 + n_actuators + (2 if cls.clock else 0), *cfg.hidden, n_actuators)
        layers: list[nn.Module] = []
        for n_in, n_out in pairwise(sizes):
            layers += [nn.Linear(n_in, n_out), ACTIVATIONS[cfg.activation]()]
        layers[-1] = nn.Tanh()
        return nn.Sequential(*layers).requires_grad_(False)

    @classmethod
    def genome_size(cls, n_actuators: int, cfg: BrainConfig) -> int:
        return sum(p.numel() for p in cls.network(n_actuators, cfg).parameters()) + int(cls.clock)

    def __init__(self, genotype: np.ndarray, n_actuators: int, dt: float, cfg: BrainConfig) -> None:
        genes = np.asarray(genotype, dtype=np.float32)
        self.net = self.network(n_actuators, cfg)
        n_weights = len(genes) - int(self.clock)
        vector_to_parameters(torch.from_numpy(genes[:n_weights]), self.net.parameters())
        if self.clock:
            self.omega = 2 * np.pi * scale(genes[-1], *CLOCK_HZ)

    @torch.inference_mode()
    def __call__(self, model: mj.MjModel, data: mj.MjData) -> np.ndarray:
        x = get_state_from_data(data)
        if self.clock:
            x = np.append(x, [np.sin(self.omega * data.time), np.cos(self.omega * data.time)])
        return self.net(torch.as_tensor(x, dtype=torch.float32)).numpy() * np.pi / 2


class MLPClockBrain(MLPBrain):
    clock = True


CONTROLLERS: dict[str, type[Brain]] = {
    "na_cpg": NaCPGBrain,
    "simple_cpg": SimpleCPGBrain,
    "mlp": MLPBrain,
    "mlp_clock": MLPClockBrain,
}
