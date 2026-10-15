import random
from collections.abc import Callable
from functools import partial
from multiprocessing import get_context
from multiprocessing.pool import Pool
from typing import Literal

from sqlmodel import Session, select

import ariel
import numpy as np
import torch
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    EASettings,
    FloatMutator,
    FloatsGenerator,
    Individual,
    Population,
    generators,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.optimizers.revde import RevDE
from pydantic import Field

from controller import CONTROLLERS, BrainConfig
from fitness import FitnessConfig, fitness, load_body
from islands import IslandEA


class EvolutionConfig(EASettings):
    body: str = "gecko"
    controller: BrainConfig = Field(default_factory=BrainConfig)
    algorithm: str = "ga"
    num_steps: int = 50
    target_population_size: int = 30
    mutation_rate: float = 0.1
    mutation_std: float = 0.1
    crossover_rate: float = 0.5
    crossover_kind: Literal["one_point", "uniform"] = "uniform"
    revde_scaling_factor: float = 0.5
    parent_tournament_size: int = 3
    survivor_tournament_size: int = 3
    islands: bool = False
    migration_interval: int = 10
    migration_size: int = 2
    seed: int = 0
    workers: int = 1
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    for rng in (generators._rng, ariel.RNG):
        rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


def limit_threads() -> None:
    torch.set_num_threads(1)


def count_actuators(body: str) -> int:
    world = SimpleFlatWorld()
    world.spawn(load_body(body).spec)
    return world.spec.compile().nu


def initial_population(cfg: EvolutionConfig, n_genes: int) -> Population:
    individuals = []
    for _ in range(cfg.target_population_size):
        ind = Individual()
        ind.genotype = FloatsGenerator.uniform(-1.0, 1.0, size=n_genes)
        individuals.append(ind)
    return Population(individuals)


@EAOperation
def evaluate(population: Population, cfg: EvolutionConfig, pool: Pool) -> Population:
    todo = population.unevaluated
    score = partial(fitness, body=cfg.body, controller=cfg.controller, cfg=cfg.fitness)
    genotypes = [np.asarray(ind.genotype) for ind in todo]
    for ind, value in zip(todo, pool.imap(score, genotypes, chunksize=1), strict=True):
        ind.fitness = value
    return population


@EAOperation
def parent_selection(population: Population, tournament_size: int) -> Population:
    for ind in population:
        ind.tags = {"ps": 0}
    for _ in range(population.size // 2 * 2):
        winner = population.sample(tournament_size).best(sort="max")[0]
        winner.tags = {"ps": winner.tags["ps"] + 1}
    return population


@EAOperation
def crossover(population: Population, rate: float, kind: str) -> Population:
    parents = [ind for ind in population for _ in range(ind.tags.get("ps", 0))]
    random.shuffle(parents)
    operator = getattr(Crossover, kind)
    for parent_i, parent_j in zip(parents[::2], parents[1::2], strict=False):
        genotypes = (parent_i.genotype, parent_j.genotype)
        if random.random() < rate:
            genotypes = operator(*genotypes)
        for genotype in genotypes:
            child = Individual()
            child.genotype = list(genotype)
            child.tags = {"mut": True}
            population.append(child)
    return population


@EAOperation
def mutation(population: Population, rate: float, std: float) -> Population:
    for ind in population.where(lambda ind: ind.tags.get("mut", False)):
        ind.genotype = FloatMutator.gaussian(
            ind.genotype, std=std, mutation_probability=rate, lower_bound=-1.0, upper_bound=1.0
        )
        ind.tags = {"mut": False}
    return population


@EAOperation
def survivor_selection(population: Population, size: int, tournament_size: int) -> Population:
    while population.alive.size > size:
        loser = population.alive.sample(tournament_size).best(sort="min")[0]
        loser.alive = False
    return population


@EAOperation
def revde_variation(population: Population, scaling_factor: float) -> Population:
    revde = RevDE(scaling_factor)
    parents = population.alive.shuffle()
    for i in range(0, parents.size // 3 * 3, 3):
        triplet = [np.asarray(ind.genotype, dtype=float) for ind in parents[i : i + 3]]
        for genotype in revde.mutate(*triplet):
            child = Individual()
            child.genotype = np.clip(genotype, -1.0, 1.0).tolist()
            population.append(child)
    return population


@EAOperation
def truncation_selection(population: Population, size: int) -> Population:
    for ind in population.alive.sort(sort="max")[size:]:
        ind.alive = False
    return population


def ga(cfg: EvolutionConfig, pool: Pool) -> list[EAOperation]:
    return [
        parent_selection(tournament_size=cfg.parent_tournament_size),
        crossover(rate=cfg.crossover_rate, kind=cfg.crossover_kind),
        mutation(rate=cfg.mutation_rate, std=cfg.mutation_std),
        evaluate(cfg=cfg, pool=pool),
        survivor_selection(size=cfg.target_population_size, tournament_size=cfg.survivor_tournament_size),
    ]


def revde(cfg: EvolutionConfig, pool: Pool) -> list[EAOperation]:
    return [
        revde_variation(scaling_factor=cfg.revde_scaling_factor),
        evaluate(cfg=cfg, pool=pool),
        truncation_selection(size=cfg.target_population_size),
    ]


ALGORITHMS: dict[str, Callable[[EvolutionConfig, Pool], list[EAOperation]]] = {"ga": ga, "revde": revde}


def evolve(cfg: EvolutionConfig) -> EA:
    seed_everything(cfg.seed)
    n_genes = CONTROLLERS[cfg.controller.name].genome_size(count_actuators(cfg.body), cfg.controller)
    with get_context("spawn").Pool(cfg.workers, initializer=limit_threads) as pool:
        population = evaluate(cfg=cfg, pool=pool)(initial_population(cfg, n_genes))
        ea = EA(
            population,
            ALGORITHMS[cfg.algorithm](cfg, pool),
            num_steps=cfg.num_steps,
            first_generation_id=cfg.first_generation_id,
            quiet=cfg.quiet,
            db_file_path=cfg.db_file_path,
            db_handling=cfg.db_handling,
        )
        (cfg.output_folder / "config.json").write_text(cfg.model_dump_json(indent=2))
        ea.run()
    return ea


def evolve_islands(cfg: EvolutionConfig):
    seed_everything(cfg.seed)
    n_genes = CONTROLLERS[cfg.controller.name].genome_size(count_actuators(cfg.body), cfg.controller)
    with get_context("spawn").Pool(cfg.workers, initializer=limit_threads) as pool:
        population = evaluate(cfg=cfg, pool=pool)(initial_population(cfg, n_genes))
        ea = IslandEA(
            population,
            ALGORITHMS[cfg.algorithm](cfg, pool),
            num_steps=cfg.num_steps,
            first_generation_id=cfg.first_generation_id,
            quiet=cfg.quiet,
            db_file_path=cfg.db_file_path,
            db_handling=cfg.db_handling,
            migration_interval=cfg.migration_interval,
            migration_count=cfg.migration_size,
        )
        (cfg.output_folder / "config.json").write_text(cfg.model_dump_json(indent=2))
        ea.run()
    return ea


def fitness_stats(ea: EA) -> dict[str, float | np.typing.NDArray[np.float64]]:
    """Calculate fitness stats for an evolution run.

    Parameters
    ----------
    ea : EA
        An ``EA`` object that has finished running. This function will query
        its database to fetch the run's history

    Returns
    -------
    dict[str, float | list[float]]
        A dictionary containing several fitness stats:
        - best_overall : float
        - mean : list[float]
        - std_dev : list[float]
        - best : list[float]
    """
    data = [[] for _ in range(ea.num_steps + 1)]

    with Session(ea.engine) as session:
        # fitness, birth, death = map(np.array, zip(*session.exec(select(Individual.fitness_, Individual.time_of_birth, Individual.time_of_death)).all()))

        # print(fitness)

        for fitness, birth, death in session.exec(select(Individual.fitness_, Individual.time_of_birth, Individual.time_of_death)):
            # time of death is inclusive
            for generation in range(birth, death + 1):
                data[generation].append(fitness)

        # print([len(bucket) for bucket in data])

        # This only works if the total population size stays constant
        data = np.array(data)

    return {
        "best_overall": data.max(),
        "mean": data.mean(axis=1),
        "std_dev": data.std(axis=1),
        "best": data.max(axis=1),
    }


def diversity(ea: EA, vector_conversion: Callable[..., list[float]] = lambda x: x) -> np.typing.NDArray[np.float64]:
    """Calculate diversity for an evolution run.

    Parameters
    ----------
    ea : EA
        An ``EA`` object that has finished running. This function will query
        its database to fetch the run's history

    vector_conversion : callable, default identitiy
        The diversity calculation assumes that each genotype is a vector.
        If this is not the case, the caller can provide a function to convert a genotype into a vector.

    Returns
    -------
    list[float]
        The standard deviation of the population's genotype, per generation
    """
    data = [[] for _ in range(ea.num_steps + 1)]

    with Session(ea.engine) as session:
        for genotype, birth, death in session.exec(select(Individual.genotype_, Individual.time_of_birth, Individual.time_of_death)):
            for generation in range(birth, death + 1):
                data[generation].append(vector_conversion(genotype))

    data = np.array(data)

    return data.std(axis=1).mean(axis=1)
