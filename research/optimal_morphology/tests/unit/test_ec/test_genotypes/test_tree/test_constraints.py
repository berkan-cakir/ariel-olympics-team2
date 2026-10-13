"""Test: size and actuator constraints for tree genomes."""

import copy
import random
import re

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.ec.genotypes.tree.constraints import (
    cap_hinges,
    count_hinges,
    count_modules,
)
from ariel.ec.genotypes.tree.prebuilt import prebuilt_tree_genome
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

MODULE_BODY = re.compile(r"^\d+-(\d+)-\d+-(?:brick|hinge)$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_frames(genome: TreeGenome) -> dict[int, np.ndarray]:
    # Node id -> world position of that module's frame.
    model = construct_mjspec_from_graph(genome.to_networkx()).spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames = {}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        match = MODULE_BODY.match(name or "")
        if match is not None:
            frames[int(match.group(1))] = data.xpos[body_id].copy()
    return frames


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_counts_of_the_spider() -> None:
    """The spider has 12 modules, 8 of them hinges."""
    spider = prebuilt_tree_genome("spider")
    assert count_modules(spider) == 12
    assert count_hinges(spider) == 8


def test_counts_ignore_core_and_none_nodes() -> None:
    """Neither the core nor NONE placeholders are body modules."""
    genome = TreeGenome(
        nodes={
            0: {"type": "CORE", "rotation": "DEG_0"},
            1: {"type": "NONE", "rotation": "DEG_0"},
        },
        edges=[{"parent": 0, "child": 1, "face": "FRONT"}],
    )
    assert count_modules(genome) == 0
    assert count_hinges(genome) == 0


def test_hinge_count_matches_actuators() -> None:
    """Every hinge adds exactly one actuator to the compiled body."""
    spider = prebuilt_tree_genome("spider")
    model = construct_mjspec_from_graph(spider.to_networkx()).spec.compile()
    assert model.nu == count_hinges(spider)


# ---------------------------------------------------------------------------
# cap_hinges
# ---------------------------------------------------------------------------


def test_cap_hinges_turns_excess_hinges_into_bricks() -> None:
    """Capping keeps the module count and lowers the hinge count."""
    spider = prebuilt_tree_genome("spider")
    original = copy.deepcopy(spider)
    capped = cap_hinges(spider, 3, random.Random(0))
    validate_genome_dict(capped.to_dict())
    assert count_hinges(capped) == 3
    assert count_modules(capped) == 12
    assert spider == original


def test_cap_hinges_does_not_move_the_body() -> None:
    """A brick puts its FRONT child exactly where the hinge did."""
    spider = prebuilt_tree_genome("spider")
    before = _module_frames(spider)
    after = _module_frames(cap_hinges(spider, 1, random.Random(0)))
    assert before.keys() == after.keys()
    for node_id, position in before.items():
        np.testing.assert_allclose(after[node_id], position, atol=1e-9)


def test_cap_hinges_under_the_limit_is_unchanged() -> None:
    """A body within budget is returned as an equal copy."""
    spider = prebuilt_tree_genome("spider")
    assert cap_hinges(spider, 8, random.Random(0)) == spider


def test_cap_hinges_requires_at_least_one_hinge() -> None:
    """A body without hinges cannot move, so the cap must be positive."""
    with pytest.raises(ValueError, match="at least 1"):
        cap_hinges(prebuilt_tree_genome("spider"), 0, random.Random(0))
