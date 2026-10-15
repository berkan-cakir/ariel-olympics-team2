"""
Sagittal (left-right) mirror symmetry of a compiled robot model.

The robot's own sagittal plane is the plane ``x = 0`` of the core body's
frame: the core's LEFT and RIGHT faces sit at ``-x`` and ``+x`` and its FRONT
face at ``+y``. Symmetry is measured at the rest pose (``qpos0``) on two
levels:

* **position** -- every module geom (brick, stator or rotor) has a mirror
  image of the same kind;
* **joint axis** -- every hinge has a mirror image with the same anchor and
  the same axis line (the sign of the axis is ignored).

Module rotations only turn a module about its attachment axis, which moves no
geom (bricks are cubes and hinge boxes have square cross-sections) but does
turn the hinge axis. Checking positions alone would therefore call a body
symmetric even when its joints are not.

Notes
-----
    * The built-in ``MorphologicalMeasures.symmetry`` places modules on a grid
      that ignores parent frames and rotations; use this module instead.
"""

# Standard library
import re
from dataclasses import dataclass

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries
from ariel.parameters.ariel_types import FloatArray

# Global constants
MODULE_GEOM_PATTERN = re.compile(r"(brick|stator|rotor)$")
MIRROR_X = np.array([-1.0, 1.0, 1.0])


@dataclass(frozen=True)
class SymmetryScore:
    """
    Fractions of a robot's modules and hinges that have a mirror partner.

    Parameters
    ----------
    position
        Fraction of module geoms whose mirror image is a geom of the same
        kind, in ``[0, 1]``.
    joint_axis
        Fraction of hinges whose mirror image is a hinge with the same anchor
        and axis line, in ``[0, 1]``.
    """

    position: float
    joint_axis: float

    @property
    def combined(self) -> float:
        """The stricter of the two scores."""
        return min(self.position, self.joint_axis)


def find_core_body(model: mj.MjModel) -> int:
    """
    Return the id of the robot's core body.

    The core body is named ``core``, or ``<prefix>_core`` once the robot has
    been spawned into a world.

    Parameters
    ----------
    model
        A compiled MuJoCo model that contains exactly one robot.

    Returns
    -------
    int
        The body id of the core.

    Raises
    ------
    ValueError
        If the model has no core body or more than one.
    """
    core_ids = [
        body_id
        for body_id in range(model.nbody)
        if _is_core_name(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id))
    ]
    if len(core_ids) != 1:
        msg = f"Expected exactly one core body, found {len(core_ids)}."
        raise ValueError(msg)
    return core_ids[0]


def sagittal_symmetry(
    model: mj.MjModel,
    *,
    tol: float = 1e-3,
    min_abs_cos: float = 0.99,
) -> SymmetryScore:
    """
    Measure how mirror-symmetric a robot is about its own sagittal plane.

    Parameters
    ----------
    model
        A compiled MuJoCo model that contains exactly one robot, either on
        its own or spawned into a world.
    tol
        Largest distance (m) at which a mirrored point still matches.
    min_abs_cos
        Smallest ``|cos|`` between a mirrored hinge axis and its partner's
        axis for the two to count as the same line.

    Returns
    -------
    SymmetryScore
        Position and joint-axis symmetry; a body with nothing to compare
        scores 1.0.
    """
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    core_id = find_core_body(model)
    origin = data.xpos[core_id].copy()
    to_local = data.xmat[core_id].reshape(3, 3).copy().T

    kinds: list[str] = []
    positions: list[FloatArray] = []
    for geom_id in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id) or ""
        match = MODULE_GEOM_PATTERN.search(name)
        if match is not None:
            kinds.append(match.group(1))
            positions.append(to_local @ (data.geom_xpos[geom_id] - origin))

    anchors: list[FloatArray] = []
    axes: list[FloatArray] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mj.mjtJoint.mjJNT_HINGE:
            anchors.append(to_local @ (data.xanchor[joint_id] - origin))
            axes.append(to_local @ data.xaxis[joint_id])

    return SymmetryScore(
        position=_position_score(kinds, positions, tol=tol),
        joint_axis=_joint_axis_score(
            anchors,
            axes,
            tol=tol,
            min_abs_cos=min_abs_cos,
        ),
    )


def _is_core_name(name: str | None) -> bool:
    return name is not None and (name == "core" or name.endswith("_core"))


def _position_score(
    kinds: list[str],
    positions: list[FloatArray],
    *,
    tol: float,
) -> float:
    if not positions:
        return 1.0
    points = np.asarray(positions)
    mirrored = points * MIRROR_X
    labels = np.asarray(kinds)
    close = _pairwise_distance(mirrored, points) < tol
    same_kind = labels[:, None] == labels[None, :]
    return float(np.mean(np.any(close & same_kind, axis=1)))


def _joint_axis_score(
    anchors: list[FloatArray],
    axes: list[FloatArray],
    *,
    tol: float,
    min_abs_cos: float,
) -> float:
    if not anchors:
        return 1.0
    points = np.asarray(anchors)
    lines = np.asarray(axes)
    lines /= np.linalg.norm(lines, axis=1, keepdims=True)
    close = _pairwise_distance(points * MIRROR_X, points) < tol
    aligned = np.abs((lines * MIRROR_X) @ lines.T) >= min_abs_cos
    return float(np.mean(np.any(close & aligned, axis=1)))


def _pairwise_distance(a: FloatArray, b: FloatArray) -> FloatArray:
    difference = a[:, None, :] - b[None, :, :]
    distances: FloatArray = np.linalg.norm(difference, axis=2)
    return distances
