"""Test: tree-genome versions of the prebuilt robots."""

import re
from collections.abc import Callable

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.ec.genotypes.tree.constraints import count_hinges, count_modules
from ariel.ec.genotypes.tree.prebuilt import (
    PREBUILT_TREE_NAMES,
    prebuilt_tree_genome,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict

GEOM_KIND = re.compile(r"(core|brick|stator|rotor)$")
TOL = 1e-5

ORIGINALS: dict[str, Callable[[], CoreModule]] = {
    "spider": spider,
    "gecko": john_set.gecko,
    "baby_a": john_set.baby_a,
    "baby_b": john_set.baby_b,
    "snake": john_set.snake,
    "linkin_modified": john_set.linkin_modified,
    "spider_8": john_set.spider_8,
    "turtle": john_set.turtle,
}

# Expected module and hinge counts per robot.
EXPECTED_COUNTS = {
    "spider": (12, 8),
    "gecko": (12, 6),
    "baby_a": (15, 8),
    "baby_b": (20, 10),
    "snake": (15, 8),
    "linkin_modified": (15, 12),
    "spider_8": (16, 8),
    "turtle": (19, 13),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _geometry(
    model: mujoco.MjModel,
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    # (kind, position) per module geom and (anchor, axis) per joint.
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geoms = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        match = GEOM_KIND.search(name or "")
        if match is not None:
            geoms.append((match.group(1), data.geom_xpos[geom_id].copy()))
    joints = [
        (data.xanchor[joint_id].copy(), data.xaxis[joint_id].copy())
        for joint_id in range(model.njnt)
    ]
    return geoms, joints


def _assert_same_body(a: mujoco.MjModel, b: mujoco.MjModel) -> None:
    """Assert both models have the same geoms and hinge axis lines."""
    geoms_a, joints_a = _geometry(a)
    geoms_b, joints_b = _geometry(b)
    assert len(geoms_a) == len(geoms_b)
    assert len(joints_a) == len(joints_b)
    for kind, position in geoms_a:
        assert any(
            other == kind and np.linalg.norm(position - where) < TOL
            for other, where in geoms_b
        ), f"no {kind} geom at {position}"
    for anchor, axis in joints_a:
        assert any(
            np.linalg.norm(anchor - where) < TOL
            and abs(float(np.dot(axis, line))) > 1 - TOL
            for where, line in joints_b
        ), f"no joint at {anchor} along {axis}"


# ---------------------------------------------------------------------------
# prebuilt_tree_genome
# ---------------------------------------------------------------------------


def test_names_cover_all_originals() -> None:
    """Every listed genome has an original robot to compare against."""
    assert set(PREBUILT_TREE_NAMES) == set(ORIGINALS)


@pytest.mark.parametrize("name", PREBUILT_TREE_NAMES)
def test_genome_is_valid(name: str) -> None:
    """Each genome passes structural validation."""
    validate_genome_dict(prebuilt_tree_genome(name).to_dict())


@pytest.mark.parametrize("name", PREBUILT_TREE_NAMES)
def test_genome_counts(name: str) -> None:
    """Each genome has the original's module and hinge counts."""
    genome = prebuilt_tree_genome(name)
    assert (count_modules(genome), count_hinges(genome)) == (
        EXPECTED_COUNTS[name]
    )


@pytest.mark.parametrize("name", PREBUILT_TREE_NAMES)
def test_genome_builds_the_original_body(name: str) -> None:
    """Each genome builds the prebuilt robot's exact geometry."""
    genome = prebuilt_tree_genome(name)
    built = construct_mjspec_from_graph(genome.to_networkx()).spec.compile()
    original = ORIGINALS[name]().spec.compile()
    assert built.nu == original.nu
    _assert_same_body(built, original)
    _assert_same_body(original, built)


def test_genomes_are_fresh_copies() -> None:
    """Changing a returned genome does not affect the next one."""
    first = prebuilt_tree_genome("spider")
    first.nodes[1]["type"] = "BRICK"
    assert prebuilt_tree_genome("spider").nodes[1]["type"] == "HINGE"


def test_unknown_name_raises() -> None:
    """Names outside the table are rejected."""
    with pytest.raises(KeyError, match="iguana"):
        prebuilt_tree_genome("iguana")
