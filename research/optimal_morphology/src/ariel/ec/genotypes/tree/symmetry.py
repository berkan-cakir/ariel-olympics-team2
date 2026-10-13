"""
Mirror-symmetric tree genomes.

The robot's sagittal plane contains the core and its FRONT and BACK faces; the
core's LEFT and RIGHT faces are mirror images of each other. ``symmetrize``
rewrites a genome so that the body it builds is exactly mirror-symmetric about
that plane, in module positions *and* hinge axes:

* for every mirror pair of faces, the subtree on one side is replaced by a
  mirrored copy of the subtree on the other side (``mirror_subtree``);
* modules that lie on the plane stay where they are, and their own children
  are paired or made symmetric in turn.

Mirroring rules
---------------
A module's spin θ (its rotation about its attachment axis) is combined with an
inherited flip φ, 0° or 180°. Mirroring negates the spin: ψ = (φ - θ) mod
360°. The half-turn part of ψ, E = 180° if ψ ≥ 180° else 0°, is a flip, and
the rest, ψ - E, is the stored rotation. θ = 45° leaves 135°, which the
genome cannot store, so 45° rotations are changed to 0° or 90° first
(``canonicalize_rotations``).

The flip E decides how a module's children are rewritten:

* E = 0°: children on LEFT and RIGHT trade places; every child starts
  unflipped (φ = 0°).
* E = 180°: children on TOP and BOTTOM trade places; FRONT, LEFT and RIGHT
  children inherit the flip (φ = 180°), TOP and BOTTOM children do not.

For a module on the plane the same state decides its mirror pair: LEFT/RIGHT
if E = 0°, TOP/BOTTOM if E = 180°. Its other faces point along the plane.

These rules were checked against the compiled MuJoCo bodies with
:func:`ariel.utils.symmetry.sagittal_symmetry`.
"""

# Standard library
import copy
import random

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.utils.symmetry import SymmetryScore, sagittal_symmetry

from .operators import remove_subtree
from .tree_genome import TreeGenome

# Type Aliases
type Signature = (
    tuple[str, str, tuple[tuple[str, Signature], ...]]
    | tuple[
        str,
        str,
        tuple[tuple[str, Signature], ...],
        tuple[tuple[str, object], ...],
    ]
)

# Global constants
HALF_TURN = 180
SPIN_DEGREES = {"DEG_0": 0, "DEG_45": 45, "DEG_90": 90}
SPIN_NAMES = {0: "DEG_0", 90: "DEG_90"}
SWAP_LEFT_RIGHT = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
SWAP_TOP_BOTTOM = {"TOP": "BOTTOM", "BOTTOM": "TOP"}
INHERITS_FLIP = frozenset({"FRONT", "LEFT", "RIGHT"})


def canonicalize_rotations(
    genome: TreeGenome,
    rng: random.Random,
) -> TreeGenome:
    """
    Return a copy of ``genome`` with every 45° rotation set to 0° or 90°.

    Parameters
    ----------
    genome
        The genome to copy; it is not modified.
    rng
        Random source for the 0°/90° choice.

    Returns
    -------
    TreeGenome
        A copy without 45° rotations.
    """
    result = copy.deepcopy(genome)
    for node_id in sorted(result.nodes):
        node = result.nodes[node_id]
        if node["rotation"] == "DEG_45":
            node["rotation"] = rng.choice(["DEG_0", "DEG_90"])
    return result


def mirror_subtree(
    genome: TreeGenome,
    source: int,
    *,
    parent: int,
    face: str,
    flip: int = 0,
) -> None:
    """
    Attach a mirrored copy of a subtree to ``parent`` at ``face``, in place.

    Parameters
    ----------
    genome
        The genome to extend.
    source
        Root of the subtree to copy; the subtree itself is left unchanged.
    parent
        Node that receives the copy.
    face
        Face of ``parent`` the copy is attached to. It must be free.
    flip
        Flip φ inherited by the copy's root, 0 or 180 (see module notes).

    Raises
    ------
    ValueError
        If the subtree has a 45° rotation, which has no mirror image.
    """
    next_id = max(genome.nodes) + 1
    stack = [(source, parent, face, flip)]
    while stack:
        src, dst_parent, dst_face, phi = stack.pop()
        mirrored = _mirror_spin(phi, genome.nodes[src]["rotation"])
        if mirrored is None:
            msg = f"Node {src} has a 45° rotation, which cannot be mirrored."
            raise ValueError(msg)
        spin, is_flipped = mirrored

        node_id = next_id
        next_id += 1
        # Any other attributes of a module (e.g. its servos) are mirrored as is.
        genome.nodes[node_id] = {**genome.nodes[src], "rotation": spin}
        genome.edges.append({
            "parent": dst_parent,
            "child": node_id,
            "face": dst_face,
        })

        swap = SWAP_TOP_BOTTOM if is_flipped else SWAP_LEFT_RIGHT
        for child_face, child in _children(genome, src).items():
            stack.append((
                child,
                node_id,
                swap.get(child_face, child_face),
                _child_flip(is_flipped=is_flipped, face=child_face),
            ))


def symmetrize(
    genome: TreeGenome,
    *,
    rng: random.Random,
    core_source: str | None = None,
) -> TreeGenome:
    """
    Return a mirror-symmetric version of ``genome``.

    45° rotations are first set to 0° or 90° at random. Then, starting at the
    core, every mirror pair of faces is made equal by replacing one side with
    a mirrored copy of the other.

    Parameters
    ----------
    genome
        The genome to symmetrize; it is not modified.
    rng
        Random source for rotation changes and for picking which side of a
        pair to copy.
    core_source
        Side of the core ("LEFT" or "RIGHT") to copy onto the other side,
        e.g. the side a mutation just changed. If that side is empty, both
        sides end up empty. ``None`` picks a random non-empty side; deeper
        pairs always do.

    Returns
    -------
    TreeGenome
        A genome whose body is mirror-symmetric about the sagittal plane.
        It may have up to about twice as many modules as ``genome``.

    Raises
    ------
    ValueError
        If ``core_source`` is not "LEFT", "RIGHT" or ``None``.
    """
    if core_source not in {None, "LEFT", "RIGHT"}:
        msg = f"core_source must be 'LEFT', 'RIGHT' or None: {core_source!r}"
        raise ValueError(msg)

    result = canonicalize_rotations(genome, rng)
    stack = [(IDX_OF_CORE, 0, core_source)]
    while stack:
        node_id, phi, source_side = stack.pop()
        is_flipped = _is_flipped(phi, result.nodes[node_id]["rotation"])
        pair = ("TOP", "BOTTOM") if is_flipped else ("LEFT", "RIGHT")
        children = _children(result, node_id)
        source = source_side or _pick_side(children, pair, rng)
        if source is not None:
            target = pair[1] if source == pair[0] else pair[0]
            if target in children:
                remove_subtree(result, children[target])
            if source in children:
                mirror_subtree(
                    result,
                    children[source],
                    parent=node_id,
                    face=target,
                )

        for child_face, child in children.items():
            if child_face not in pair:
                stack.append((
                    child,
                    _child_flip(is_flipped=is_flipped, face=child_face),
                    None,
                ))
    return result


def subtree_signature(genome: TreeGenome, node: int) -> Signature:
    """
    Describe the subtree at ``node`` independently of node ids.

    Two subtrees with the same signature build the same body.

    Parameters
    ----------
    genome
        The genome containing the subtree.
    node
        Root of the subtree.

    Returns
    -------
    Signature
        ``(type, rotation, ((face, child_signature), ...))`` with the
        children sorted by face. A module with other attributes (e.g. its
        servos) gets them appended as sorted ``(key, value)`` pairs, so
        modules without any keep the signature they always had.
    """
    children = _children(genome, node)
    attributes = genome.nodes[node]
    signature = (
        attributes["type"],
        attributes["rotation"],
        tuple(
            (face, subtree_signature(genome, children[face]))
            for face in sorted(children)
        ),
    )
    extra = tuple(
        sorted(
            (key, value)
            for key, value in attributes.items()
            if key not in {"type", "rotation"}
        ),
    )
    return (*signature, extra) if extra else signature


def genome_symmetry(genome: TreeGenome) -> SymmetryScore:
    """
    Build the body of ``genome`` and measure its sagittal symmetry.

    Parameters
    ----------
    genome
        The genome to measure.

    Returns
    -------
    SymmetryScore
        See :func:`ariel.utils.symmetry.sagittal_symmetry`.
    """
    core = construct_mjspec_from_graph(genome.to_networkx())
    return sagittal_symmetry(core.spec.compile())


def _children(genome: TreeGenome, node: int) -> dict[str, int]:
    return {
        edge["face"]: edge["child"]
        for edge in genome.edges
        if edge["parent"] == node
    }


def _is_flipped(phi: int, rotation: str) -> bool:
    return (phi - SPIN_DEGREES[rotation]) % 360 >= HALF_TURN


def _mirror_spin(phi: int, rotation: str) -> tuple[str, bool] | None:
    is_flipped = _is_flipped(phi, rotation)
    psi = (phi - SPIN_DEGREES[rotation]) % 360
    name = SPIN_NAMES.get(psi - HALF_TURN if is_flipped else psi)
    if name is None:
        return None
    return name, is_flipped


def _child_flip(*, is_flipped: bool, face: str) -> int:
    return HALF_TURN if is_flipped and face in INHERITS_FLIP else 0


def _pick_side(
    children: dict[str, int],
    pair: tuple[str, str],
    rng: random.Random,
) -> str | None:
    occupied = [side for side in pair if side in children]
    return rng.choice(occupied) if occupied else None
