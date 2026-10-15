"""
Size and actuator constraints for tree genomes.

Counting works on the genome alone: every HINGE node becomes exactly one joint
and one actuator, so no MuJoCo compile is needed.
"""

# Standard library
import copy
import random

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleType

from .tree_genome import TreeGenome

# Global constants
BODY_MODULES = frozenset({ModuleType.BRICK.name, ModuleType.HINGE.name})


def count_modules(genome: TreeGenome) -> int:
    """
    Count the bricks and hinges in ``genome`` (the core is not counted).

    Parameters
    ----------
    genome
        The genome to inspect.

    Returns
    -------
    int
        Number of BRICK and HINGE nodes.
    """
    return sum(node["type"] in BODY_MODULES for node in genome.nodes.values())


def count_hinges(genome: TreeGenome) -> int:
    """
    Count the hinges, i.e. the actuated joints, in ``genome``.

    Parameters
    ----------
    genome
        The genome to inspect.

    Returns
    -------
    int
        Number of HINGE nodes, which equals ``model.nu`` of the built body.
    """
    return sum(
        node["type"] == ModuleType.HINGE.name for node in genome.nodes.values()
    )


def cap_hinges(
    genome: TreeGenome,
    max_hinges: int,
    rng: random.Random,
) -> TreeGenome:
    """
    Return a copy of ``genome`` with at most ``max_hinges`` hinges.

    Randomly chosen excess hinges become bricks with the same rotation. A
    hinge's only child sits on its FRONT face, which bricks also have at the
    same distance (0.1 m), so the rest of the body does not move.

    Parameters
    ----------
    genome
        The genome to copy; it is not modified.
    max_hinges
        Largest number of hinges allowed; at least 1.
    rng
        Random source for choosing which hinges to convert.

    Returns
    -------
    TreeGenome
        A copy with at most ``max_hinges`` hinges.

    Raises
    ------
    ValueError
        If ``max_hinges`` is smaller than 1.
    """
    if max_hinges < 1:
        msg = f"max_hinges must be at least 1, got {max_hinges}."
        raise ValueError(msg)

    result = copy.deepcopy(genome)
    hinges = sorted(
        node_id
        for node_id, node in result.nodes.items()
        if node["type"] == ModuleType.HINGE.name
    )
    excess = len(hinges) - max_hinges
    if excess > 0:
        for node_id in rng.sample(hinges, excess):
            result.nodes[node_id]["type"] = ModuleType.BRICK.name
    return result
