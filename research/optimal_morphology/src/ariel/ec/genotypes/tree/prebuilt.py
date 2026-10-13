"""
Tree-genome versions of ARIEL's hand-designed prebuilt robots.

The prebuilt robots in :mod:`ariel.body_phenotypes.robogen_lite.prebuilt_robots`
are assembled directly from module objects, so evolution cannot use them as
genomes. The genomes below build the same bodies: every module geom and every
hinge axis line matches the original to within 1e-5 m (hinge axis *signs* may
differ, which a learned controller absorbs).

Rotations the genome cannot store were rewritten: a spin of -90° or ±180° is
an allowed spin plus a half-turn flip, and a flipped module's LEFT/RIGHT and
TOP/BOTTOM children trade places (FRONT, LEFT and RIGHT children inherit the
flip; TOP and BOTTOM children do not).

Not included:

* ``gecko.gecko`` and ``john_set.iguana`` use -45° spins (DEG_135 is
  disabled in the config);
* ``john_set.spider_12``, ``spider_16`` and the centipedes have more than 20
  modules;
* ``random_20`` and ``spider_with_blocks`` are not hand designs.
"""

# Standard library
from typing import Any

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE

from .tree_genome import TreeGenome

# Type Aliases
# (parent, child, face, module type, rotation)
type ModuleRow = tuple[int, int, str, str, str]

# Global constants
PREBUILT_TREE_ROWS: dict[str, tuple[ModuleRow, ...]] = {
    "spider": (
        (0, 1, "FRONT", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "HINGE", "DEG_90"),
        (2, 3, "FRONT", "BRICK", "DEG_0"),
        (0, 4, "LEFT", "HINGE", "DEG_0"),
        (4, 5, "FRONT", "HINGE", "DEG_90"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (0, 7, "RIGHT", "HINGE", "DEG_0"),
        (7, 8, "FRONT", "HINGE", "DEG_90"),
        (8, 9, "FRONT", "BRICK", "DEG_0"),
        (0, 10, "BACK", "HINGE", "DEG_0"),
        (10, 11, "FRONT", "HINGE", "DEG_90"),
        (11, 12, "FRONT", "BRICK", "DEG_0"),
    ),
    "gecko": (
        (0, 1, "FRONT", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
        (2, 3, "FRONT", "HINGE", "DEG_0"),
        (3, 4, "FRONT", "BRICK", "DEG_0"),
        (4, 5, "LEFT", "HINGE", "DEG_90"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (4, 7, "RIGHT", "HINGE", "DEG_90"),
        (7, 8, "FRONT", "BRICK", "DEG_0"),
        (0, 9, "RIGHT", "HINGE", "DEG_90"),
        (9, 10, "FRONT", "BRICK", "DEG_0"),
        (0, 11, "LEFT", "HINGE", "DEG_90"),
        (11, 12, "FRONT", "BRICK", "DEG_0"),
    ),
    "baby_a": (
        (0, 1, "RIGHT", "HINGE", "DEG_90"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
        (0, 3, "LEFT", "HINGE", "DEG_90"),
        (3, 4, "FRONT", "HINGE", "DEG_90"),
        (4, 5, "FRONT", "BRICK", "DEG_0"),
        (5, 6, "FRONT", "HINGE", "DEG_90"),
        (6, 7, "FRONT", "BRICK", "DEG_0"),
        (0, 8, "FRONT", "HINGE", "DEG_0"),
        (8, 9, "FRONT", "BRICK", "DEG_0"),
        (9, 10, "FRONT", "HINGE", "DEG_0"),
        (10, 11, "FRONT", "BRICK", "DEG_0"),
        (11, 12, "RIGHT", "HINGE", "DEG_90"),
        (12, 13, "FRONT", "BRICK", "DEG_0"),
        (11, 14, "LEFT", "HINGE", "DEG_90"),
        (14, 15, "FRONT", "BRICK", "DEG_0"),
    ),
    "baby_b": (
        (0, 1, "BACK", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
        (2, 3, "FRONT", "HINGE", "DEG_90"),
        (3, 4, "FRONT", "BRICK", "DEG_0"),
        (4, 5, "FRONT", "HINGE", "DEG_90"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (0, 7, "RIGHT", "HINGE", "DEG_0"),
        (7, 8, "FRONT", "BRICK", "DEG_0"),
        (8, 9, "FRONT", "HINGE", "DEG_90"),
        (9, 10, "FRONT", "BRICK", "DEG_0"),
        (10, 11, "FRONT", "HINGE", "DEG_90"),
        (11, 12, "FRONT", "BRICK", "DEG_0"),
        (0, 13, "FRONT", "HINGE", "DEG_0"),
        (13, 14, "FRONT", "BRICK", "DEG_0"),
        (0, 15, "LEFT", "HINGE", "DEG_0"),
        (15, 16, "FRONT", "BRICK", "DEG_0"),
        (16, 17, "FRONT", "HINGE", "DEG_90"),
        (17, 18, "FRONT", "BRICK", "DEG_0"),
        (18, 19, "FRONT", "HINGE", "DEG_90"),
        (19, 20, "FRONT", "BRICK", "DEG_0"),
    ),
    "snake": (
        (0, 1, "RIGHT", "HINGE", "DEG_90"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
        (2, 3, "FRONT", "HINGE", "DEG_90"),
        (3, 4, "FRONT", "BRICK", "DEG_0"),
        (4, 5, "FRONT", "HINGE", "DEG_90"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (6, 7, "FRONT", "HINGE", "DEG_90"),
        (7, 8, "FRONT", "BRICK", "DEG_0"),
        (8, 9, "FRONT", "HINGE", "DEG_90"),
        (9, 10, "FRONT", "BRICK", "DEG_0"),
        (10, 11, "FRONT", "HINGE", "DEG_90"),
        (11, 12, "FRONT", "BRICK", "DEG_0"),
        (12, 13, "FRONT", "HINGE", "DEG_90"),
        (13, 14, "FRONT", "BRICK", "DEG_0"),
        (14, 15, "FRONT", "HINGE", "DEG_90"),
    ),
    "linkin_modified": (
        (0, 1, "LEFT", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "HINGE", "DEG_0"),
        (2, 3, "FRONT", "HINGE", "DEG_0"),
        (3, 4, "FRONT", "HINGE", "DEG_90"),
        (4, 5, "FRONT", "BRICK", "DEG_0"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (5, 7, "BOTTOM", "HINGE", "DEG_0"),
        (7, 8, "FRONT", "HINGE", "DEG_0"),
        (5, 9, "TOP", "HINGE", "DEG_90"),
        (9, 10, "FRONT", "HINGE", "DEG_90"),
        (10, 11, "FRONT", "HINGE", "DEG_0"),
        (11, 12, "FRONT", "HINGE", "DEG_90"),
        (12, 13, "FRONT", "HINGE", "DEG_0"),
        (0, 14, "FRONT", "HINGE", "DEG_90"),
        (14, 15, "FRONT", "BRICK", "DEG_0"),
    ),
    "spider_8": (
        (0, 1, "BACK", "HINGE", "DEG_0"),
        (1, 2, "FRONT", "BRICK", "DEG_0"),
        (2, 3, "FRONT", "HINGE", "DEG_90"),
        (3, 4, "FRONT", "BRICK", "DEG_0"),
        (0, 5, "RIGHT", "HINGE", "DEG_0"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (6, 7, "FRONT", "HINGE", "DEG_90"),
        (7, 8, "FRONT", "BRICK", "DEG_0"),
        (0, 9, "FRONT", "HINGE", "DEG_0"),
        (9, 10, "FRONT", "BRICK", "DEG_0"),
        (10, 11, "FRONT", "HINGE", "DEG_90"),
        (11, 12, "FRONT", "BRICK", "DEG_0"),
        (0, 13, "LEFT", "HINGE", "DEG_0"),
        (13, 14, "FRONT", "BRICK", "DEG_0"),
        (14, 15, "FRONT", "HINGE", "DEG_90"),
        (15, 16, "FRONT", "BRICK", "DEG_0"),
    ),
    "turtle": (
        (0, 1, "RIGHT", "BRICK", "DEG_0"),
        (1, 2, "RIGHT", "HINGE", "DEG_90"),
        (1, 3, "LEFT", "HINGE", "DEG_0"),
        (3, 4, "FRONT", "HINGE", "DEG_90"),
        (4, 5, "FRONT", "BRICK", "DEG_0"),
        (5, 6, "FRONT", "BRICK", "DEG_0"),
        (6, 7, "BOTTOM", "HINGE", "DEG_90"),
        (5, 8, "TOP", "HINGE", "DEG_0"),
        (8, 9, "FRONT", "BRICK", "DEG_0"),
        (9, 10, "FRONT", "BRICK", "DEG_0"),
        (9, 11, "BOTTOM", "HINGE", "DEG_90"),
        (11, 12, "FRONT", "HINGE", "DEG_90"),
        (9, 13, "TOP", "HINGE", "DEG_0"),
        (13, 14, "FRONT", "BRICK", "DEG_0"),
        (14, 15, "TOP", "HINGE", "DEG_0"),
        (14, 16, "BOTTOM", "HINGE", "DEG_90"),
        (16, 17, "FRONT", "HINGE", "DEG_90"),
        (17, 18, "FRONT", "HINGE", "DEG_0"),
        (18, 19, "FRONT", "HINGE", "DEG_0"),
    ),
}
PREBUILT_TREE_NAMES: tuple[str, ...] = tuple(PREBUILT_TREE_ROWS)


def prebuilt_tree_genome(name: str) -> TreeGenome:
    """
    Return a fresh tree genome that builds the prebuilt robot ``name``.

    Parameters
    ----------
    name
        One of :data:`PREBUILT_TREE_NAMES`, e.g. ``"spider"``. ``"gecko"`` is
        ``john_set.gecko``.

    Returns
    -------
    TreeGenome
        A new genome; changing it does not affect later calls.

    Raises
    ------
    KeyError
        If ``name`` is not a known prebuilt robot.
    """
    if name not in PREBUILT_TREE_ROWS:
        msg = f"Unknown prebuilt robot {name!r}; known: {PREBUILT_TREE_NAMES}"
        raise KeyError(msg)
    return TreeGenome.from_dict(_rows_to_dict(PREBUILT_TREE_ROWS[name]))


def _rows_to_dict(rows: tuple[ModuleRow, ...]) -> dict[str, Any]:
    nodes = {str(IDX_OF_CORE): {"type": "CORE", "rotation": "DEG_0"}}
    edges: list[dict[str, Any]] = []
    for parent, child, face, module_type, rotation in rows:
        nodes[str(child)] = {"type": module_type, "rotation": rotation}
        edges.append({"parent": parent, "child": child, "face": face})
    return {"nodes": nodes, "edges": edges}
