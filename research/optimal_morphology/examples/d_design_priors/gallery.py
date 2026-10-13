"""
Render the robot bodies of the design-priors experiment as image grids.

Example
-------
    uv run python examples/d_design_priors/gallery.py

Writes to ``<out>/analysis/``:

* ``designs.png`` -- the 8 hand-designed robots used as seeds;
* ``best_bodies.png`` -- the best-scoring body of every completed run, one row
  per condition.

Bodies are shown as spawned (facing +x, before settling), from the same
three-quarter camera, each framed to its own size.
"""

# Standard library
import argparse
from pathlib import Path
from typing import Any

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from dp_core import CONDITIONS, build_world, load_individuals

from ariel.ec.genotypes.tree.constraints import count_hinges, count_modules
from ariel.ec.genotypes.tree.prebuilt import (
    PREBUILT_TREE_NAMES,
    prebuilt_tree_genome,
)
from ariel.ec.genotypes.tree.symmetry import genome_symmetry
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.parameters.ariel_types import FloatArray

DEFAULT_OUT = Path.home() / "ariel_runs" / "design_priors"
SPAWN_YAW = -90.0
TILE = (240, 320)  # height, width in pixels
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -40.0
FRAMING = 1.6  # camera distance per metre of body size
MODULE_MARGIN = 0.2  # metres; the core is 0.2 m wide
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
LABELS = {
    "baseline": "Baseline",
    "seeded": "Seeded",
    "symmetric": "Symmetric",
    "joint_budget": "Joint budget",
}


def render_body(genome: dict[str, Any]) -> FloatArray:
    """
    Render one body as spawned in the experiment.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.

    Returns
    -------
    FloatArray
        An RGB image of ``TILE`` pixels.
    """
    _, model, data = build_world(genome, SPAWN_YAW)
    mujoco.mj_forward(model, data)
    # Frame the bounding box of the module centres, plus one module of margin.
    body_geoms = data.geom_xpos[model.geom_bodyid > 0]
    low, high = body_geoms.min(axis=0), body_geoms.max(axis=0)
    size = max(float((high - low).max()) + MODULE_MARGIN, 0.3)

    camera = mujoco.MjvCamera()
    camera.lookat[:] = (low + high) / 2
    camera.distance = FRAMING * size
    camera.azimuth = CAMERA_AZIMUTH
    camera.elevation = CAMERA_ELEVATION
    with mujoco.Renderer(model, height=TILE[0], width=TILE[1]) as renderer:
        renderer.update_scene(data, camera=camera)
        image: FloatArray = renderer.render()
    return image


def describe(genome: dict[str, Any]) -> str:
    """
    Summarise a body's size and symmetry in one line.

    Parameters
    ----------
    genome
        The body's tree genome as a dict.

    Returns
    -------
    str
        E.g. ``"12 modules, 8 hinges, symmetric"``.
    """
    tree = TreeGenome.from_dict(genome)
    symmetry = genome_symmetry(tree).combined
    shape = "symmetric" if symmetry == 1.0 else f"symmetry {symmetry:.2f}"
    return (
        f"{count_modules(tree)} modules, {count_hinges(tree)} hinges, {shape}"
    )


def plot_designs(path: Path) -> None:
    """
    Draw the 8 hand-designed seed robots in a 2 x 4 grid.

    Parameters
    ----------
    path
        Output PNG.
    """
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, name in zip(axes.flat, PREBUILT_TREE_NAMES, strict=True):
        genome = prebuilt_tree_genome(name).to_dict()
        ax.imshow(render_body(genome))
        ax.set_title(
            f"{name}\n{describe(genome)}",
            fontsize=9,
            color=INK_SECONDARY,
        )
        ax.axis("off")
    fig.suptitle(
        "Hand-designed seed robots",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(h_pad=3.0)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def best_bodies(out: Path) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    """
    Find the best-scoring body of every completed run.

    Parameters
    ----------
    out
        The experiment's output root.

    Returns
    -------
    dict[str, list[tuple[int, dict[str, Any]]]]
        Per condition, ``(seed, individual)`` pairs ordered by seed.
    """
    bests: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for condition in CONDITIONS:
        bests[condition] = []
        for db in sorted((out / condition).glob("seed_*.db")):
            scored = [
                ind
                for ind in load_individuals(db)
                if ind["tags"].get("score") is not None
            ]
            if scored:
                best = max(scored, key=lambda ind: ind["tags"]["score"])
                bests[condition].append((int(db.stem.split("_")[1]), best))
    return bests


def plot_best_bodies(out: Path, path: Path) -> None:
    """
    Draw the best body of every run: one row per condition.

    Parameters
    ----------
    out
        The experiment's output root.
    path
        Output PNG.
    """
    bests = best_bodies(out)
    columns = max(len(runs) for runs in bests.values())
    fig, axes = plt.subplots(
        len(CONDITIONS),
        columns,
        figsize=(2.4 * columns, 2.3 * len(CONDITIONS)),
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    for row, condition in enumerate(CONDITIONS):
        for col in range(columns):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(bests[condition]):
                continue
            seed, ind = bests[condition][col]
            tags = ind["tags"]
            ax.imshow(render_body(ind["genotype"]["morph"]))
            ax.set_title(
                f"seed {seed}: {tags['score']:.2f} m\n"
                f"{tags['n_modules']} mod, {tags['n_hinges']} hinges",
                fontsize=7,
                color=INK_SECONDARY,
            )
        axes[row][0].text(
            -0.08,
            0.5,
            LABELS[condition],
            transform=axes[row][0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=10,
            color=INK,
        )
    fig.suptitle(
        "Best body of every run (score during the run)",
        x=0.01,
        ha="left",
        color=INK,
        fontsize=12,
    )
    fig.tight_layout(h_pad=1.5)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    """Render both grids into ``<out>/analysis``."""
    parser = argparse.ArgumentParser(
        description="Render the design-priors robot bodies as image grids.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    analysis = args.out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    plot_designs(analysis / "designs.png")
    plot_best_bodies(args.out, analysis / "best_bodies.png")
    print(f"Wrote designs.png and best_bodies.png to {analysis}")  # noqa: T201


if __name__ == "__main__":
    main()
