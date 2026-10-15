from itertools import chain, repeat

from ariel.ec import Population
from ariel.ec.ea import EA

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
            migration_interval=10,
            migration_count=2,
            island_sizes=[20, 20, 20],
            island_kwargs=[{}, {}, {}],
            **kwargs
    ) -> None:
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.island_sizes = island_sizes
        self.island_kwargs = island_kwargs

        if population is not None:
            if len(population) != sum(island_sizes):
                raise ValueError(f"Population size ({len(population)}) must match total island size({sum(island_sizes)})")

            # Assign the appropriate number of individuals to each island
            for island, individual in zip(chain(*(repeat(i, size) for i, size in enumerate(island_sizes))), population):
                individual.tags['island'] = island

        super().__init__(population=population, **kwargs)

    @property
    def num_islands(self) -> int:
        return len(self.island_sizes)

    # TODO consider making this a property
    def island_pops(self) -> list[Population]:
        self.fetch_population()

        island_pops = [Population([]) for _ in self.island_sizes]
        for individual in self.population:
            island_pops[individual.tags['island']].append(individual)

        return island_pops

    def migrate(self) -> None:
        island_pops = self.island_pops()
        island_emigrants = [island.best(n=self.migration_count) for island in island_pops]

        # Simply cycle the best n individuals one island to the right, looping around
        for emigrant in chain(*island_emigrants):
            emigrant.tags['island'] = (emigrant.tags['island'] + 1) % self.num_islands

    def step(self) -> None:
        self.current_generation += 1

        island_pops = self.island_pops()
        new_population = []

        for island, (island_pop, kwargs) in enumerate(zip(island_pops, self.island_kwargs)):
            for op in self.operations:
                island_pop = op(island_pop, **kwargs)

            for individual in island_pop:
                individual.tags['island'] = island

            new_population.extend(island_pop)

        self.population = Population(new_population)

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
                if step % self.migration_interval == 0:
                    self.migrate()

                progress.advance(task)

        self.console.rule("[green]EA Finished")
