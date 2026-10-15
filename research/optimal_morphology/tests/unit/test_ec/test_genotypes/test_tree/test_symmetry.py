"""Test: mirror-symmetric tree genomes."""

import copy
import random

import pytest

from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.prebuilt import prebuilt_tree_genome
from ariel.ec.genotypes.tree.symmetry import (
    canonicalize_rotations,
    genome_symmetry,
    mirror_subtree,
    subtree_signature,
    symmetrize,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

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


def _core_child(genome: TreeGenome, face: str) -> int | None:
    for edge in genome.edges:
        if edge["parent"] == 0 and edge["face"] == face:
            return edge["child"]
    return None


def _left_limb() -> TreeGenome:
    # A LEFT limb whose spins and faces exercise every mirroring rule.
    return _genome(
        (0, 1, "LEFT", "HINGE", "DEG_90"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
        (2, 3, "TOP", "HINGE", "DEG_0"),
        (2, 4, "LEFT", "BRICK", "DEG_90"),
        (4, 5, "RIGHT", "HINGE", "DEG_0"),
        (4, 6, "BOTTOM", "BRICK", "DEG_0"),
    )


def _assert_symmetric(genome: TreeGenome) -> None:
    validate_genome_dict(genome.to_dict())
    assert genome_symmetry(genome).combined == 1.0


# ---------------------------------------------------------------------------
# canonicalize_rotations
# ---------------------------------------------------------------------------


def test_canonicalize_replaces_only_45_degree_rotations() -> None:
    """45° spins become 0° or 90°; other spins and the input are kept."""
    genome = _genome(
        (0, 1, "FRONT", "HINGE", "DEG_45"),
        (1, 2, "FRONT", "BRICK", "DEG_90"),
    )
    original = copy.deepcopy(genome)
    result = canonicalize_rotations(genome, random.Random(0))
    assert result.nodes[1]["rotation"] in {"DEG_0", "DEG_90"}
    assert result.nodes[2]["rotation"] == "DEG_90"
    assert genome == original


# ---------------------------------------------------------------------------
# mirror_subtree
# ---------------------------------------------------------------------------


def test_mirror_subtree_builds_the_mirror_image() -> None:
    """Mirroring a LEFT limb onto RIGHT gives a symmetric body."""
    genome = _left_limb()
    mirror_subtree(genome, 1, parent=0, face="RIGHT")
    assert len(genome.nodes) == 13
    _assert_symmetric(genome)


def test_mirror_subtree_leaves_the_source_unchanged() -> None:
    """The copied subtree itself is not modified."""
    genome = _left_limb()
    before = subtree_signature(genome, 1)
    mirror_subtree(genome, 1, parent=0, face="RIGHT")
    assert subtree_signature(genome, 1) == before


@pytest.mark.parametrize("flip", [0, 180])
def test_mirror_subtree_rejects_45_degree_spins(flip: int) -> None:
    """A 45° spin has no mirror image in the genome's alphabet."""
    genome = _genome((0, 1, "LEFT", "HINGE", "DEG_45"))
    with pytest.raises(ValueError, match="45° rotation"):
        mirror_subtree(genome, 1, parent=0, face="RIGHT", flip=flip)


# ---------------------------------------------------------------------------
# symmetrize
# ---------------------------------------------------------------------------


def test_symmetrize_rejects_unknown_core_source() -> None:
    """Only LEFT, RIGHT or None may be copied at the core."""
    with pytest.raises(ValueError, match="core_source"):
        symmetrize(_genome(), rng=random.Random(0), core_source="FRONT")


def test_symmetrize_copies_the_requested_side() -> None:
    """The chosen core side survives and the other side mirrors it."""
    genome = _left_limb()
    genome.nodes[7] = {"type": "BRICK", "rotation": "DEG_0"}
    genome.edges.append({"parent": 0, "child": 7, "face": "RIGHT"})
    left_before = subtree_signature(genome, 1)

    result = symmetrize(genome, rng=random.Random(0), core_source="LEFT")

    left = _core_child(result, "LEFT")
    right = _core_child(result, "RIGHT")
    assert left is not None
    assert right is not None
    assert subtree_signature(result, left) == left_before
    assert result.nodes[right]["type"] == "HINGE"
    _assert_symmetric(result)


def test_symmetrize_from_an_empty_side_empties_both() -> None:
    """Copying an empty side removes the subtree on the other side."""
    result = symmetrize(
        _left_limb(),
        rng=random.Random(0),
        core_source="RIGHT",
    )
    assert _core_child(result, "LEFT") is None
    assert _core_child(result, "RIGHT") is None


def test_symmetrize_keeps_a_body_on_the_plane() -> None:
    """A FRONT chain without side limbs is already symmetric."""
    genome = _genome(
        (0, 1, "FRONT", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
    )
    result = symmetrize(genome, rng=random.Random(0))
    assert result == genome


def test_symmetrize_pairs_top_and_bottom_of_a_flipped_brick() -> None:
    """A 90° brick on the plane pairs TOP with BOTTOM, not LEFT with RIGHT."""
    genome = _genome(
        (0, 1, "FRONT", "BRICK", "DEG_90"),
        (1, 2, "TOP", "HINGE", "DEG_0"),
        (2, 3, "FRONT", "BRICK", "DEG_0"),
        (1, 4, "LEFT", "BRICK", "DEG_0"),
        (4, 5, "TOP", "HINGE", "DEG_0"),
        (1, 6, "RIGHT", "BRICK", "DEG_90"),
        (6, 7, "LEFT", "HINGE", "DEG_0"),
    )
    result = symmetrize(genome, rng=random.Random(0))
    assert len(result.nodes) == 12
    _assert_symmetric(result)


def test_symmetrize_random_bodies() -> None:
    """Random bodies of every size come out perfectly symmetric."""
    random.seed(1)
    rng = random.Random(1)
    for _ in range(40):
        genome = random_tree(rng.randint(1, 20))
        _assert_symmetric(symmetrize(genome, rng=rng))


# ---------------------------------------------------------------------------
# subtree_signature / genome_symmetry
# ---------------------------------------------------------------------------


def test_signature_ignores_node_ids() -> None:
    """Equal structures with different ids have equal signatures."""
    a = _genome((0, 1, "FRONT", "HINGE", "DEG_0"))
    b = _genome((0, 9, "FRONT", "HINGE", "DEG_0"))
    assert subtree_signature(a, 0) == subtree_signature(b, 0)


def test_signature_sees_rotations() -> None:
    """A different spin gives a different signature."""
    a = _genome((0, 1, "FRONT", "HINGE", "DEG_0"))
    b = _genome((0, 1, "FRONT", "HINGE", "DEG_90"))
    assert subtree_signature(a, 0) != subtree_signature(b, 0)


def test_genome_symmetry_of_prebuilt_spider() -> None:
    """The spider genome builds a symmetric body."""
    assert genome_symmetry(prebuilt_tree_genome("spider")).combined == 1.0
