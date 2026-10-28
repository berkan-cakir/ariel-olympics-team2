import random
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from multiprocessing import get_context
from multiprocessing.pool import Pool
from typing import Any, Literal

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
from sqlalchemy import Engine
from sqlmodel import Session, select

from controller import CONTROLLERS, BrainConfig
from fitness import FitnessConfig, fitness, load_body
from islands import IslandEA

FINAL_ELITES = 10


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
    island_sizes: list[int] = Field(default_factory=list)
    island_kwargs: list[dict[str, Any]] = Field(default_factory=list)
    seed: int = 0
    workers: int = 1
    fitness: FitnessConfig = Field(default_factory=FitnessConfig)
    flat_pretraining: int = 0


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


def score(job: tuple[np.ndarray, FitnessConfig], body: str, controller: BrainConfig) -> float:
    genotype, cfg = job
    return fitness(genotype, body, controller, cfg)


@EAOperation
def evaluate(
    population: Population,
    body: str,
    controller: BrainConfig,
    fitness_config: FitnessConfig,
    pool: Pool,
    island_configs: list[FitnessConfig] | None = None,
    **kwargs,
) -> Population:
    todo = population.unevaluated
    jobs = [(np.asarray(ind.genotype), island_configs[ind.tags["island"]] if island_configs else fitness_config) for ind in todo]
    for ind, value in zip(todo, pool.imap(partial(score, body=body, controller=controller), jobs, chunksize=1), strict=True):
        ind.fitness = value
    return population


@EAOperation
def parent_selection(population: Population, tournament_size: int, **kwargs) -> Population:
    for ind in population:
        ind.tags = {"ps": 0}
        # ind.tags["ps"] = 0
    for _ in range(population.size // 2 * 2):
        winner = population.sample(tournament_size).best(sort="max")[0]
        winner.tags = {"ps": winner.tags["ps"] + 1}
        # winner.tags["ps"] += 1
    return population


@EAOperation
def crossover(population: Population, rate: float, kind: str, **kwargs) -> Population:
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
            # child.tags["mut"] = True
            population.append(child)
    return population


@EAOperation
def mutation(population: Population, rate: float, std: float, **kwargs) -> Population:
    for ind in population.where(lambda ind: ind.tags.get("mut", False)):
        ind.genotype = FloatMutator.gaussian(
            ind.genotype, std=std, mutation_probability=rate, lower_bound=-1.0, upper_bound=1.0
        )
        ind.tags = {"mut": False}
        # ind.tags["mut"] = False
    return population


@EAOperation
def survivor_selection(population: Population, target_population_size: int, tournament_size: int, **kwargs) -> Population:
    while population.alive.size > target_population_size:
        loser = population.alive.sample(tournament_size).best(sort="min")[0]
        loser.alive = False
    return population


@EAOperation
def revde_variation(population: Population, scaling_factor: float, **kwargs) -> Population:
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
def truncation_selection(population: Population, target_population_size: int, **kwargs) -> Population:
    for ind in population.alive.sort(sort="max")[target_population_size:]:
        ind.alive = False
    return population


def ga(cfg: EvolutionConfig, pool: Pool) -> list[EAOperation]:
    return [
        parent_selection(tournament_size=cfg.parent_tournament_size),
        crossover(rate=cfg.crossover_rate, kind=cfg.crossover_kind),
        mutation(rate=cfg.mutation_rate, std=cfg.mutation_std),
        evaluate(body=cfg.body, controller=cfg.controller, fitness_config=cfg.fitness, pool=pool),
        survivor_selection(target_population_size=cfg.target_population_size, tournament_size=cfg.survivor_tournament_size),
    ]


def revde(cfg: EvolutionConfig, pool: Pool) -> list[EAOperation]:
    return [
        revde_variation(scaling_factor=cfg.revde_scaling_factor),
        evaluate(body=cfg.body, controller=cfg.controller, fitness_config=cfg.fitness, pool=pool),
        truncation_selection(target_population_size=cfg.target_population_size),
    ]


ALGORITHMS: dict[str, Callable[[EvolutionConfig, Pool], list[EAOperation]]] = {"ga": ga, "revde": revde}


def evolve(cfg: EvolutionConfig) -> EA:
    seed_everything(cfg.seed)
    n_genes = CONTROLLERS[cfg.controller.name].genome_size(count_actuators(cfg.body), cfg.controller)
    with get_context("spawn").Pool(cfg.workers, initializer=limit_threads) as pool:
        population = evaluate(body=cfg.body, controller=cfg.controller, fitness_config=cfg.fitness, pool=pool)(initial_population(cfg, n_genes))
        ea = EA(
            population=population,
            operations=ALGORITHMS[cfg.algorithm](cfg, pool),
            num_steps=cfg.num_steps,
            first_generation_id=cfg.first_generation_id,
            quiet=cfg.quiet,
            db_file_path=cfg.db_file_path,
            db_handling=cfg.db_handling,
        )
        (cfg.output_folder / "config.json").write_text(cfg.model_dump_json(indent=2))
        ea.run()
    return ea


def island_sizes(population_size: int, rel_sizes: list[int]) -> list[int]:
    num_islands = len(rel_sizes)
    rel_size_sum = sum(rel_sizes)
    # return [population_size // num_islands + (i < population_size % num_islands) for i in range(num_islands)]
    buckets = [int(population_size * s / rel_size_sum) for s in rel_sizes]

    while sum(buckets) < population_size:
        buckets[max(range(num_islands), key=lambda i: buckets[i])] += 1

    return buckets


def offspring(population_size: int, algorithm: str) -> int:
    return population_size // 3 * 3 if algorithm == "revde" else population_size // 2 * 2


def evolve_islands(cfg: EvolutionConfig):
    seed_everything(cfg.seed)
    n_genes = CONTROLLERS[cfg.controller.name].genome_size(count_actuators(cfg.body), cfg.controller)

    # cfg.island_sizes = island_sizes(cfg.target_population_size, cfg.island_sizes)
    print(f"{cfg.island_sizes = }")
    print(f"{cfg.num_steps = }")
    cfg.island_kwargs = [
        {"fitness_config": replace(cfg.fitness, environments=(env,), weights=None), "target_population_size": size}
        for env, size in zip(cfg.fitness.environments, cfg.island_sizes, strict=True)
    ]

    with get_context("spawn").Pool(cfg.workers, initializer=limit_threads) as pool:
        if cfg.flat_pretraining > 0:
            pretrain_cfg = cfg.model_copy(update={'fitness': FitnessConfig()})

            pretrain_ea = EA(
                population=evaluate(body=cfg.body, controller=cfg.controller, fitness_config=pretrain_cfg.fitness, pool=pool)(initial_population(pretrain_cfg, n_genes)),
                operations=ALGORITHMS[pretrain_cfg.algorithm](pretrain_cfg, pool),
                num_steps= pretrain_cfg.flat_pretraining,
                quiet=pretrain_cfg.quiet,
                db_file_path=pretrain_cfg.db_file_path,
                db_handling=pretrain_cfg.db_handling,
            )

            pretrain_ea.run()
            print("finished pretraining")
            # print(*(ind.time_of_death for ind in pretrain_ea._fetch()), sep='\n')

            ea = IslandEA(
                # population=pretrain_ea.population.alive,
                continue_from=cfg.db_file_path,
                operations=ALGORITHMS[cfg.algorithm](cfg, pool),
                num_steps=cfg.num_steps - cfg.flat_pretraining,
                first_generation_id=cfg.flat_pretraining,
                quiet=cfg.quiet,
                db_file_path=cfg.db_file_path,
                migration_interval=cfg.migration_interval,
                migration_count=cfg.migration_size,
                island_kwargs=cfg.island_kwargs,
                island_sizes=cfg.island_sizes,
            )
        else:
            ea = IslandEA(
                population=initial_population(cfg, n_genes),
                operations=ALGORITHMS[cfg.algorithm](cfg, pool),
                num_steps=cfg.num_steps,
                first_generation_id=cfg.first_generation_id,
                quiet=cfg.quiet,
                db_file_path=cfg.db_file_path,
                db_handling=cfg.db_handling,
                migration_interval=cfg.migration_interval,
                migration_count=cfg.migration_size,
                island_kwargs=cfg.island_kwargs,
                island_sizes=cfg.island_sizes,
            )
        (cfg.output_folder / "config.json").write_text(cfg.model_dump_json(indent=2))
        ea.run()
        score_elites(ea, cfg, pool)
    return ea


def score_elites(ea: IslandEA, cfg: EvolutionConfig, pool: Pool) -> None:
    with Session(ea.engine) as session:
        everyone = Population(list(session.exec(select(Individual))))
        chosen = {ind.id: ind for ind in everyone if ind.alive}
        for island in range(ea.num_islands):
            for ind in everyone.where(lambda ind, island=island: ind.tags.get("island") == island).best(n=FINAL_ELITES):
                chosen[ind.id] = ind
        jobs = [(np.asarray(ind.genotype), cfg.fitness) for ind in chosen.values()]
        scores = pool.imap(partial(score, body=cfg.body, controller=cfg.controller), jobs, chunksize=1)
        for ind, value in zip(chosen.values(), scores, strict=True):
            ind.tags = {"combined": value}
        session.commit()


def best_combined(engine: Engine) -> Individual:
    with Session(engine) as session:
        scored = [ind for ind in session.exec(select(Individual)) if "combined" in ind.tags]
    return max(scored, key=lambda ind: ind.tags["combined"])


def best_fitness(ea: EA) -> float:
    if isinstance(ea, IslandEA):
        return best_combined(ea.engine).tags["combined"]
    return ea.get_solution("best", only_alive=False).fitness


def fitness_stats(population: Population) -> dict[str, float]:
    f = np.array([ind.fitness for ind in population])
    return {"best": float(f.max()), "mean": float(f.mean()), "std": float(f.std())}


def diversity(population: Population) -> float:
    return float(np.array([ind.genotype for ind in population]).std(axis=0).mean())


# Quick-and-dirty, for testing the framework
def main():
    evolve_islands(EvolutionConfig(
        islands=True,
        island_sizes=[2, 2, 2],
        num_steps=4,
        flat_pretraining=2,
        target_population_size=6,
        fitness=FitnessConfig(environments=(0, 1, 2))
    ))


if __name__ == '__main__':
    main()
