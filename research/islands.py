from itertools import chain, repeat
from pathlib import Path
from typing import Any

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
            **kwargs
    ) -> None:
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.island_sizes = island_sizes
        self.island_kwargs = island_kwargs
        self.operations = operations or []

        if continue_from is not None:
            continue_from = Path(continue_from)
            if not continue_from.exists():
                raise FileNotFoundError(f"Continue database not found: {continue_from}")

            source_engine = create_engine(f"sqlite:///{continue_from}")
            with Session(source_engine) as session:
                continue_gen = session.exec(select(func.max(Individual.time_of_birth))).one()

                pretrained = Population(session.exec(select(Individual).where(Individual.alive).where(Individual.requires_eval == False)).all())

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
        return EAOperation(op.func, *op.args, **op.kwargs | self.island_kwargs[island])

    def evaluate(self, population: Population) -> None:
        configs = [kwargs.get("fitness_config") for kwargs in self.island_kwargs]
        for op in self.operations:
            if op.name == "evaluate":
                EAOperation(op.func, *op.args, **op.kwargs | {"island_configs": configs})(population)

    def migrate(self) -> None:
        arrivals = Population([])
        for island, island_pop in enumerate(self.island_pops()):
            target = (island + 1) % self.num_islands
            for emigrant in island_pop.alive.best(n=self.migration_count):
                emigrant.alive = False
                arrival = Individual()
                arrival.genotype = list(emigrant.genotype)
                arrival.tags = {"island": target}
                arrivals.append(arrival)
        self.evaluate(arrivals)
        self.population.extend(arrivals)

    def step(self) -> None:
        self.current_generation += 1
        self.fetch_population()

        at = next((i for i, op in enumerate(self.operations) if op.name == "evaluate"), len(self.operations))
        island_pops = self.island_pops()

        for island, island_pop in enumerate(island_pops):
            for op in self.operations[:at]:
                island_pop = self.island_operation(op, island)(island_pop)

            for individual in island_pop:
                individual.tags['island'] = island

            island_pops[island] = island_pop

        self.evaluate(Population(list(chain(*island_pops))))

        new_population = []
        for island, island_pop in enumerate(island_pops):
            for op in self.operations[at + 1:]:
                island_pop = self.island_operation(op, island)(island_pop)

            new_population.extend(island_pop)

        self.population = Population(new_population)

        if self.current_generation % self.migration_interval == 0:
            self.migrate()

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
                # if step % self.migration_interval == 0:
                #     self.migrate()

                progress.advance(task)

        self.console.rule("[green]EA Finished")
