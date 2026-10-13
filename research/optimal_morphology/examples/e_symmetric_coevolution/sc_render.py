"""
Pictures of the robot bodies of the symmetric co-evolution experiment.

* :func:`render_body` -- one body as spawned (facing +x, all hinges at 0),
  from a three-quarter camera on its front-right, framed to its size;
* :func:`save_grid` -- several bodies with captions in one PNG;
* :func:`population_snapshot` -- the hook ``run_symmetric_coevolution.py``
  uses to save the surviving bodies, fastest first, every
  ``--snapshot-every`` generations.
"""

# Standard library
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import mujoco
from matplotlib.figure import Figure
from sc_core import RunConfig, Snapshot, build_world, is_due

# Local libraries
from ariel.ec import Individual
from ariel.parameters.ariel_types import FloatArray
from ariel.utils.symmetry import find_core_body

# Global constants
TILE = (240, 320)  # height, width in pixels
CAMERA_AZIMUTH = 135.0  # looks at the robot's front-right
CAMERA_ELEVATION = -35.0
FRAMING = 1.45  # camera distance per metre of body size
MODULE_MARGIN = 0.2  # metres; the core is 0.2 m wide
MIN_EXTENT = 0.3
COLUMNS = 5
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"


@dataclass(frozen=True)
class BodyCard:
    """
    One body to draw, with its caption.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    caption
        Text under the picture; may span several lines.
    highlight
        Draw the caption in bold and dark ink.
    """

    genome: dict[str, Any]
    caption: str
    highlight: bool = False


def render_body(genome: dict[str, Any], spawn_yaw: float) -> FloatArray:
    """
    Render one body as it is spawned in the experiment.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.
    spawn_yaw
        Yaw (degrees) at spawn.

    Returns
    -------
    FloatArray
        An RGB image of :data:`TILE` pixels.
    """
    _, model, data = build_world(genome, spawn_yaw)
    mujoco.mj_forward(model, data)
    root = model.body_rootid[find_core_body(model)]
    robot_geoms = model.body_rootid[model.geom_bodyid] == root
    points = data.geom_xpos[robot_geoms]
    low, high = points.min(axis=0), points.max(axis=0)
    extent = max(float((high - low).max()) + MODULE_MARGIN, MIN_EXTENT)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (low + high) / 2
    camera.distance = FRAMING * extent
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    with mujoco.Renderer(model, height=TILE[0], width=TILE[1]) as renderer:
        renderer.update_scene(data, camera=camera)
        image: FloatArray = renderer.render()
    return image


def save_grid(
    cards: Sequence[BodyCard],
    path: Path,
    *,
    title: str,
    spawn_yaw: float,
    columns: int = COLUMNS,
) -> None:
    """
    Draw bodies on a grid, in order, and save it as a PNG.

    Parameters
    ----------
    cards
        The bodies and their captions, at least one.
    path
        Output PNG; its folder is created if needed.
    title
        Line above the grid.
    spawn_yaw
        Yaw (degrees) at spawn, as in the run.
    columns
        Most pictures per row.
    """
    n_columns = min(columns, len(cards))
    n_rows = math.ceil(len(cards) / n_columns)
    fig = Figure(
        figsize=(2.6 * n_columns, 2.45 * n_rows + 0.45),
        facecolor=SURFACE,
    )
    axes = fig.subplots(n_rows, n_columns, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, card in zip(axes.flat, cards, strict=False):
        ax.imshow(render_body(card.genome, spawn_yaw))
        ax.set_title(
            card.caption,
            fontsize=7.5,
            color=INK if card.highlight else INK_SECONDARY,
            fontweight="bold" if card.highlight else "normal",
            linespacing=1.3,
        )
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(h_pad=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=SURFACE)


def body_caption(tags: dict[str, Any], goal: float, *, rank: int) -> str:
    """
    Describe a body's result and size in three short lines.

    Parameters
    ----------
    tags
        The individual's tags (speed, finish time, size, uid).
    goal
        The run's goal distance (m).
    rank
        Its place in the population, from 1.

    Returns
    -------
    str
        Three lines, e.g. ``#1 uid 37: 0.142 m/s``, ``3 m in 21.1 s,
        synchronous`` (the gait family) and ``12 modules, 6 hinges (from
        20)`` (the size before pruning, if pruned).
    """
    speed = tags.get("speed")
    finish_time = tags.get("finish_time")
    if speed is None:
        result = "cannot be simulated"
    elif finish_time is not None:
        result = f"{goal:g} m in {float(finish_time):.1f} s"
    else:
        result = f"furthest {float(tags['furthest']):.2f} m"
    if tags.get("family"):
        result += f", {tags['family']}"
    shown_speed = "-" if speed is None else f"{float(speed):.3f} m/s"
    size = f"{tags['n_modules']} modules, {tags['n_hinges']} hinges"
    if tags.get("n_double"):
        size += f" ({tags['n_double']} double)"
    if tags.get("unpruned_modules") is not None:
        size += f" (from {tags['unpruned_modules']})"
    return f"#{rank}  uid {tags['uid']}: {shown_speed}\n{result}\n{size}"


def population_snapshot(cfg: RunConfig) -> Snapshot:
    """
    Return the run's picture hook: the survivors of a generation, best first.

    Each due generation (``cfg.snapshot_every``, see :func:`sc_core.is_due`)
    gets ``<cfg.snapshot_dir>/gen_<NNN>.png``.

    Parameters
    ----------
    cfg
        Run settings (output folder, goal, spawn yaw, cadence).

    Returns
    -------
    Snapshot
        A function of the survivors and the generation.
    """

    def snapshot(survivors: list[Individual], generation: int) -> None:
        if not is_due(generation, cfg.snapshot_every, cfg.gens):
            return
        cards = [
            BodyCard(
                genome=cast("dict[str, Any]", ind.genotype)["morph"],
                caption=body_caption(
                    cast("dict[str, Any]", ind.tags),
                    cfg.goal,
                    rank=rank,
                ),
                highlight=rank == 1,
            )
            for rank, ind in enumerate(survivors, start=1)
        ]
        save_grid(
            cards,
            cfg.snapshot_dir / f"gen_{generation:03d}.png",
            title=(
                f"Generation {generation} of {cfg.gens} (seed {cfg.seed}, "
                f"{cfg.brain} brain): survivors, fastest first; robots face"
                " the lower right"
            ),
            spawn_yaw=cfg.spawn_yaw,
        )

    return snapshot
