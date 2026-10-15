# Imports
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

# from mujoco import viewer
import matplotlib.pyplot as plt
import numpy as np

# Type Checking
# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install
from sqlmodel import Session, select

# Ariel Imports
# New EA engine (ea.py)
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    EASettings,
    Individual,
    Population,
    config,
)

from islands import IslandEA
from evolution import fitness_stats, diversity

if TYPE_CHECKING:
    from networkx import DiGraph

# Initialize rich console and traceback handler
install()
console = Console()

# Will probably have to fix the paths at some point
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / "Island_Test")
DATA.mkdir(exist_ok=True)

# Show cwd
# print(f"Current working directory: {CWD}")
# print(f"Saving data to {DATA}")

# A seed is optional, but it helps with reproducibility
SEED = None  # e.g., 42

NUM_MODULES = 20
GENE_SIZE = 64  # default is 64, change it in the decoder/NDE
GENE_RANGE = 64
POP_SIZE = 60  # Default is 3 islands * 20 pops
NUM_GENERATIONS = 50

# Initialize RNG
RNG = np.random.default_rng(SEED)


def evaluation(pop: Population) -> Population:
    for individual in pop:
        if not individual.requires_eval:
            continue

        individual.fitness = sum(individual.genotype)

    return pop

def mutation(pop: Population, mutation_rate: float = 0.5) -> Population:
    return Population([Individual(genotype_=[allele ^ (np.random.random() < mutation_rate) for allele in individual.genotype]) for individual in pop])

def parent_selection(pop: Population, num_pairs: int) -> Population:
    for _ in range(num_pairs):
        p1, p2 = np.random.randint(0, len(pop), 2)
        pop[p1].tags.setdefault('mates', []).append(p2)

    return pop

def crossover(pop: Population) -> Population:
    for individual in pop:
        for mate in individual.tags.get('mates', []):
            c1, c2 = Crossover.one_point(individual.genotype, pop[mate].genotype)
            pop.append(Individual(genotype_=list(map(int, c1))))
            pop.append(Individual(genotype_=list(map(int, c2))))

        individual.tags['mates'] = []

    return pop


def survivor_selection(pop: Population, pop_size: int = 60) -> Population:
    return pop.best(n=pop_size)


def evolve() -> EA:
    """Entry point for the evolutionary algorithm.

    Initializes the population, evaluates it, defines the evolutionary
    steps (parent selection, crossover, mutation, learning, survivor selection),
    and runs the EA.

    Returns
    -------
    EA
        The completed EA object containing the run history and solutions.
    """
    console.log("Initializing population...")

    # Initialise Body & Hivemind Population
    population_list = [Individual(genotype_=list(map(int, np.random.randint(2, size=GENE_SIZE))), time_of_birth=0) for _ in range(POP_SIZE)]

    # Define Evolution Loop
    # Operators work for both NDEs and Network Weight Vectors
    ops = [
        # Default EA operators
        EAOperation(parent_selection, num_pairs=10),  # Select parents for bodies
        EAOperation(crossover),  # Crossover Body
        EAOperation(mutation, mutation_rate=0.3),  # Mutation Body
        # Learning acts as a fitness function
        EAOperation(evaluation),  # Do learning for all the bodies
        EAOperation(survivor_selection, pop_size=20),
    ]

    # Initialise EA object
    # ea = EA(
    #     Population(population_list),
    #     operations=ops,
    #     num_steps=NUM_GENERATIONS,
    #     db_file_path=config.db_file_path,
    #     db_handling="delete",
    # )

    ea = IslandEA(
        evaluation(Population(population_list)),
        operations=ops,
        num_steps=NUM_GENERATIONS,
        db_file_path=DATA / 'database.db',
        db_handling="delete",
    )
    ea.run()

    return ea

def main():
    start = time.time()

    ea = evolve()

    best_fitness = ea.get_solution("best", only_alive=False).fitness

    console.log(f"Best fitness found: {best_fitness:.3f}")
    console.log(f"Best fitness possible: ")

    end = time.time()

    time_taken = end - start

    # Literally just to see the results better while testing
    if time_taken < 60:
        console.log(f"Code took {time_taken:.3f} seconds to run")
    elif time_taken < 60 * 60:
        console.log(f"Code took {time_taken / 60:.3f} minutes to run")
    else:
        console.log(f"Code took {time_taken / (60 * 60):.3f} hours to run")

    for name, stat in fitness_stats(ea).items():
        if not isinstance(stat, np.ndarray):
            continue

        fig, ax = plt.subplots()

        ax.plot(np.arange(stat.size), stat)
        ax.set_xlabel('generation')

        fig.savefig(DATA / f'fitness_{name}.pdf')


    div = diversity(ea)

    fig, ax = plt.subplots()

    ax.plot(np.arange(div.size), div)
    ax.set_xlabel('generation')
    ax.set_ylabel('diversity')

    fig.savefig(DATA / 'diversity.pdf')

if __name__ == '__main__':
    main()
