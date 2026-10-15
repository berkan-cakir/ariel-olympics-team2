"""Test: sagittal mirror symmetry of compiled robot models."""

from typing import cast

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.symmetry import (
    SymmetryScore,
    find_core_body,
    sagittal_symmetry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _genome(*rows: tuple[int, int, str, str, str]) -> TreeGenome:
    # Rows are (parent, child, face, type, rotation).
    nodes = {0: {"type": "CORE", "rotation": "DEG_0"}}
    edges = []
    for parent, child, face, module_type, rotation in rows:
        nodes[child] = {"type": module_type, "rotation": rotation}
        edges.append({"parent": parent, "child": child, "face": face})
    return TreeGenome(nodes=nodes, edges=edges)


def _model(genome: TreeGenome) -> mujoco.MjModel:
    core = construct_mjspec_from_graph(genome.to_networkx())
    return cast("mujoco.MjModel", core.spec.compile())


# ---------------------------------------------------------------------------
# SymmetryScore
# ---------------------------------------------------------------------------


def test_combined_is_the_stricter_score() -> None:
    """Combined symmetry is the minimum of the two scores."""
    assert SymmetryScore(position=0.8, joint_axis=0.5).combined == 0.5
    assert SymmetryScore(position=0.2, joint_axis=0.9).combined == 0.2


# ---------------------------------------------------------------------------
# find_core_body
# ---------------------------------------------------------------------------


def test_find_core_body_in_robot_model() -> None:
    """The core of a bare robot model is the body named 'core'."""
    model = _model(_genome())
    core_id = find_core_body(model)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, core_id) == (
        "core"
    )


def test_find_core_body_without_core_raises() -> None:
    """A model without a core body is rejected."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="exactly one core"):
        find_core_body(spec.compile())


def test_find_core_body_with_two_robots_raises() -> None:
    """A world with two robots has an ambiguous core."""
    world = SimpleFlatWorld()
    for x in (0.0, 1.0):
        core = construct_mjspec_from_graph(_genome().to_networkx())
        world.spawn(core.spec, position=(x, 0, 0.2))
    with pytest.raises(ValueError, match="found 2"):
        find_core_body(world.spec.compile())


# ---------------------------------------------------------------------------
# sagittal_symmetry
# ---------------------------------------------------------------------------


def test_core_only_has_nothing_to_compare() -> None:
    """A body with no modules scores 1.0 on both levels."""
    score = sagittal_symmetry(_model(_genome()))
    assert score == SymmetryScore(position=1.0, joint_axis=1.0)


def test_bricks_only_have_no_joints_to_compare() -> None:
    """Without hinges the joint-axis score is 1.0."""
    genome = _genome(
        (0, 1, "LEFT", "BRICK", "DEG_0"),
        (0, 2, "RIGHT", "BRICK", "DEG_0"),
    )
    assert sagittal_symmetry(_model(genome)).joint_axis == 1.0


def test_prebuilt_spider_is_symmetric() -> None:
    """The four-legged spider is perfectly symmetric."""
    score = sagittal_symmetry(spider().spec.compile())
    assert score.combined == 1.0


def test_one_sided_limb_is_not_symmetric() -> None:
    """A limb on the LEFT face only has no mirror partner."""
    genome = _genome(
        (0, 1, "LEFT", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
    )
    score = sagittal_symmetry(_model(genome))
    assert score == SymmetryScore(position=0.0, joint_axis=0.0)


def test_limb_on_the_plane_is_symmetric() -> None:
    """A FRONT chain lies on the sagittal plane."""
    genome = _genome(
        (0, 1, "FRONT", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "HINGE", "DEG_90"),
        (2, 3, "FRONT", "BRICK", "DEG_0"),
    )
    assert sagittal_symmetry(_model(genome)).combined == 1.0


def test_rotated_joints_break_axis_symmetry_only() -> None:
    """Two 45° hinges sit symmetrically, but their axes do not mirror."""
    genome = _genome(
        (0, 1, "LEFT", "HINGE", "DEG_45"),
        (0, 2, "RIGHT", "HINGE", "DEG_45"),
    )
    score = sagittal_symmetry(_model(genome))
    assert score.position == 1.0
    assert score.joint_axis == 0.0


def test_spawned_robot_is_measured_in_its_own_frame() -> None:
    """Spawn position, yaw and the world's free joint do not matter."""
    world = SimpleFlatWorld()
    world.spawn(
        spider().spec,
        position=(0.3, -0.2, 0.2),
        rotation=(0, 0, -90),
        correct_collision_with_floor=False,
    )
    assert sagittal_symmetry(world.spec.compile()).combined == 1.0
