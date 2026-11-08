from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, func
from sqlmodel import Session, select

from ariel.ec import Individual, Population
from ariel.ec.ea import EA, EAOperation

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


class IslandEA(EA):
    def __init__(
            self,
            population: Population | None = None,
            operations: list[EAOperation] | None = None,
            *,
            continue_from: str | Path | None = None,
            migration_interval: int = 10,
            migration_count: int = 2,
            island_sizes: list[int] = [20, 20, 20],
            island_kwargs: list[dict[str, Any]] = [{}, {}, {}],
            termination_condition: Callable[[EA], bool] | None = None,
            **kwargs
    ) -> None:
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.island_sizes = island_sizes
        self.island_kwargs = island_kwargs

        self.operations = operations or []
        self._terminated = False
        self.termination_condition = termination_condition

        if continue_from is not None:
            continue_from = Path(continue_from)
            if not continue_from.exists():
                raise FileNotFoundError(f"Continue database not found: {continue_from}")

            source_engine = create_engine(f"sqlite:///{continue_from}")
            with Session(source_engine) as session:
                continue_gen = session.exec(select(func.max(Individual.time_of_birth))).one()

                pretrained = Population(session.exec(select(Individual)
                                                     .where(Individual.alive)
                                                     .where(Individual.requires_eval == False)
                                                    ).all())

                population = Population([])
                for individual in pretrained.shuffle():
                    individual.alive = False
                    copy = Individual()
                    copy.genotype = list(individual.genotype)
                    population.append(copy)
                session.commit()

            kwargs |= {
                'db_file_path': continue_from,
                'db_handling': 'continue',  # Somewhat hacky way to avoid the original DB being deleted;
                'first_generation_id': continue_gen,
                'restart': None,
            }

        if len(island_sizes) != len(island_kwargs):
            raise ValueError(f"Number of islands does not match number of island kwargs dicts {len(island_sizes)} != {len(island_kwargs)}")

        if population is not None:
            if len(population) != sum(island_sizes):
                raise ValueError(f"Population size ({len(population)}) must match total island size ({sum(island_sizes)})")

            # print("assigning islands")

            # Assign the appropriate number of individuals to each island
            for island, individual in zip(chain(*(repeat(i, size) for i, size in enumerate(island_sizes))), population, strict=True):
                # individual.tags['island'] = island
                individual.tags = {'island': island}

            self.evaluate(population)

        # print([ind.tags for ind in population])

        super().__init__(
            population=population,
            operations=operations,
            # first_generation_id=first_generation_id,
            **kwargs
        )

    @property
    def num_islands(self) -> int:
        return len(self.island_sizes)

    def split(self, population: Population) -> list[Population]:
        island_pops = [Population([]) for _ in self.island_sizes]
        for individual in population:
            island_pops[individual.tags['island']].append(individual)

        return island_pops

    def island_pops(self) -> list[Population]:
        return self.split(self.population)

    def island_operation(self, op: EAOperation, island: int) -> EAOperation:
        return EAOperation(op.func, *op.args, **op.kwargs | self.island_kwargs[island] | {'island': island, 'current_generation': self.current_generation})

    def evaluate(self, population: Population) -> None:
        configs = [kwargs.get("fitness_config") for kwargs in self.island_kwargs]
        for op in self.operations:
            if op.name == "evaluate":
                EAOperation(op.func, *op.args, **op.kwargs | {"island_configs": configs})(population)
                return

    # def migrate(self) -> None:
    #     arrivals = Population([])
    #     for island, island_pop in enumerate(self.island_pops()):
    #         target = (island + 1) % self.num_islands
    #         for emigrant in island_pop.alive.best(n=self.migration_count):
    #             emigrant.alive = False
    #             arrival = Individual()
    #             arrival.genotype = list(emigrant.genotype)
    #             arrival.tags = {"island": target}
    #             arrivals.append(arrival)
    #     self.evaluate(arrivals)
    #     self.population.extend(arrivals)

    def terminated(self) -> bool:
        if self._terminated:
            return True

        if self.termination_condition is None:
            return False

        if self.termination_condition(self):
            self._terminated = True
            return True

        return False

    def step(self) -> None:
        self.current_generation += 1
        self.fetch_population()

        # at = next((i for i, op in enumerate(self.operations) if op.name == "evaluate"), len(self.operations))
        # island_pops = self.island_pops()

        # for island, island_pop in enumerate(island_pops):
        #     for op in self.operations[:at]:
        #         island_pop = self.island_operation(op, island)(island_pop)

        #     for individual in island_pop:
        #         individual.tags['island'] = island

        #     island_pops[island] = island_pop

        # self.evaluate(Population(list(chain(*island_pops))))

        # new_population = []
        # for island, island_pop in enumerate(island_pops):
        #     for op in self.operations[at + 1:]:
        #         island_pop = self.island_operation(op, island)(island_pop)

        #     new_population.extend(island_pop)

        # self.population = Population(new_population)

        # if self.current_generation % self.migration_interval == 0:
        #     self.migrate()

        island_pops = self.island_pops()
        for op in self.operations:
            # Evaluate the entire population in one function, to make optimal use of the worker pool.
            # TODO maybe rewrite operations as async to avoid needing to do this?
            if op.name == "evaluate":
                self.evaluate(Population(list(chain(*island_pops))))
                continue

            for island, pop in enumerate(island_pops):
                pop: Population = self.island_operation(op, island)(pop)

                for ind in pop:
                    ind.tags = {'island': island}

                island_pops[island] = pop

                # If we terminate mid-step, immediately end the algorithm and
                # do not write to DB, to preserve correctness and fairness between runs
                if self.terminated():
                    return

        self.population = sum(island_pops, start=Population([]))

        self._commit()

    def run(self) -> None:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Running EA",
                total=self.num_steps,
            )
            for step in range(1, self.num_steps + 1):
                self.step()

                if self.terminated():
                    progress.stop()
                    self.console.rule("[green]EA Finished (termination condition reached)")
                    break

                progress.advance(task)

        self.console.rule("[green]EA Finished")


class MigrationPattern(ABC):
    """
    Abstract base class to represent migration logic for island-based evolution.

    Used by wrapping an instance's `emigrate` and `immigrate` functions with
    `EAOperation` and calling those in order. All islands must call `emigrate`
    before any call `immigrate` to guarantee correctness.

    Parameters
    ----------
    migration_interval : int
        The number of generations between migration events. Specifically,
        migration occurs when the `current_generation` argument passed to the
        `*migrate` functions divisible by `migration_interval`.
    migration_size : int
        Normally, this is the number of individuals that migrate along **each**
        migration path, meaning the number of individuals leaving from and arriving
        to an island is an integer multiple of this number. Subclasses are free
        to interpret this number differently.
    keep_original : bool, optional
        Whether a migrating individual is also kept alive on its original island.
        Default is `False`
    selection_mode : {"worst", "random"}, optional
        How to select the migrants leaving an island:
        - `best`: Select the `n` fittest individuals.
        - `random`: Select `n` random individuals.

        Default is `best`.
    replacement_mode : {"worst", "random"}, optional
        How to select which individuals are killed to make room for immigrants:
        - `worst`: Select the `n` individuals with the lowest fitness.
        - `random`: Select `n` random individuals.

        If `keep_original` is set to `False`, this parameter is ignored, as
        immigrants simple replace emigrants.
        Default is `worst`
    """
    def __init__(
            self,
            migration_interval: int,
            migration_size: int,
            island_sizes: list[int],
            keep_original: bool = False,
            selection_mode: Literal['best', 'random'] = 'best',
            replacement_mode: Literal['worst', 'random'] = 'worst',
    ) -> None:
        self.migration_interval = migration_interval
        self._migration_size = migration_size
        self.keep_original = keep_original
        self.selection_mode = selection_mode
        self.replacement_mode = replacement_mode
        self.island_sizes = island_sizes

        self.immigration_queues = [Population([]) for _ in range(self.num_islands)]

    @staticmethod
    def get_by_name(name: str):
        match name:
            case 'single_ring': return SingleRingMigration
            case 'double_ring': return DoubleRingMigration
            case 'all2all': return All2AllMigration
            case 'star': return StarMigration
            case 'no_migration': return NoMigration
            case _: raise ValueError(f'migraton topology {name} cannot be resolved')

    @property
    def num_islands(self) -> int:
        return len(self.island_sizes)

    def num_migrants(self, island: int) -> int:
        return self._migration_size

    @abstractmethod
    def migration_targets(self, source: int) -> Iterator[int]:
        raise NotImplementedError()

    def get_emigrants(self, population: Population, num: int) -> Population:
        match self.selection_mode:
            case "best": result = population.alive.best(n=num).shuffle()
            case "random": result = population.alive.shuffle()[:num]
            case _: raise ValueError("Invalid migration selection mode")

        if len(result) < num:
            raise ValueError("Island population too small for emigration")

        return result

    def _emigrate(self, population: Population, island: int, **kwargs) -> Population:
        total_size = self.num_migrants(island)
        migrants = self.get_emigrants(population, total_size)

        for ind, target in zip(migrants, self.migration_targets(island), strict=True):
            ind.alive = self.keep_original
            self.immigration_queues[target].append(Individual(
                genotype_=ind.genotype,  # This is a copy by reference, so in-place mutation will affect both copies if the original remains alive
                tags_=ind.tags | {'island': target},
                requires_init=ind.requires_init,
            ))

        return population

    def _immigrate(self, population: Population, island: int, target_population_size: int, **kwargs) -> Population:
        num_immigrants = len(self.immigration_queues[island])
        if num_immigrants > target_population_size:
            raise ValueError(f"Immigrants exceed target population size on island {island}")

        # print(f"immigrating {num_immigrants} individuals to island {island}")

        num_to_kill = len(population.alive) + num_immigrants - target_population_size

        if num_to_kill < 0:
            raise ValueError(f"Not enough immigrants to maintain population on island {island}")

        match self.replacement_mode:
            case "worst": to_kill = population.alive.best(sort='min', n=num_to_kill)
            case "random": to_kill = population.alive.sample(n=num_to_kill)
            case _: raise ValueError("Invalid migration replacement mode")

        for ind in to_kill:
            ind.alive = False

        population.extend(self.immigration_queues[island])
        # Clear population for next cycle
        self.immigration_queues[island] = Population.empty()

        return population

    def emigrate(self, population: Population, island: int, current_generation: int, **kwargs) -> Population:
        if current_generation % self.migration_interval != 0:
            return population

        return self._emigrate(population, island, **kwargs)

    def immigrate(self, population: Population, current_generation: int, island: int, **kwargs) -> Population:
        if current_generation % self.migration_interval != 0:
            return population

        return self._immigrate(population, island, **kwargs)


class NoMigration(MigrationPattern):
    """
    Dummy migration pattern so we don't need to change the list of operators for no-island EA.
    """
    def num_migrants(self, island: int) -> int:
        return 0

    def migration_targets(self, source: int) -> Iterator[int]:
        return
        yield  # Incredibly hacky way to tell Python this is a generator so typechecking stops complaining

    def emigrate(self, population: Population, island: int, current_generation: int, **kwargs) -> Population:
        return population

    def immigrate(self, population: Population, current_generation: int, island: int, **kwargs) -> Population:
        return population

class SingleRingMigration(MigrationPattern):
    def migration_targets(self, source: int) -> Iterator[int]:
        yield from repeat((source + 1) % self.num_islands, self._migration_size)

class DoubleRingMigration(MigrationPattern):
    def num_migrants(self, island: int) -> int:
        return 2 * self._migration_size

    def migration_targets(self, source: int) -> Iterator[int]:
        yield from repeat((source + 1) % self.num_islands, self._migration_size)
        yield from repeat((source - 1) % self.num_islands, self._migration_size)

class All2AllMigration(MigrationPattern):
    def num_migrants(self, island: int) -> int:
        return (self.num_islands - 1) * self._migration_size

    def migration_targets(self, source: int) -> Iterator[int]:
        for target in range(self.num_islands):
            if target == source: continue

            yield from repeat(target, self._migration_size)

class StarMigration(MigrationPattern):
    def num_migrants(self, island: int) -> int:
        if island != 0:
            return self._migration_size

        return (self.num_islands - 1) * self._migration_size

    def migration_targets(self, source: int) -> Iterator[int]:
        if source == 0:
            for target in range(1, self.num_islands):
                yield from repeat(target, self._migration_size)
        else:
            yield from repeat(0, self._migration_size)


class SimThreshold(BaseModel):
    """Termination condition tracking the total number of simulation runs."""
    # def __init__(self, max_sims: int | None = None) -> None:
    #     self.max_sims = max_sims
    #     self.total_sims = 0

    max_sims: int | None = Field(default=None)
    total_sims: int = Field(default=0)

    def threshold_exceeded(self, *_) -> bool:
        if self.max_sims is None:
            return False

        return self.total_sims > self.max_sims
