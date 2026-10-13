"""
Plot the progress of symmetric co-evolution runs and their best bodies.

Example
-------
    uv run python examples/e_symmetric_coevolution/plot_run.py \
        ~/ariel_runs/symmetric_coevolution/alternating/seed_01.db

For every run database given, this writes to ``--out`` (default: a ``plots``
folder next to the first database):

* ``<run>_progress.png`` -- six panels: the speed of the fastest and of the
  average survivor; the speed of every body by the generation it was born
  in; how many survivors finish; the gait family of the survivors; the size
  of the fastest body; and the learning curves of all bodies;
* ``<run>_best_bodies.png`` -- the fastest body at up to eight generations;
* ``<run>_generations.csv`` and ``<run>_bodies.csv`` -- the numbers behind
  the panels: one row per generation and one row per body.

With several databases it also writes ``runs_progress.png``: the fastest
speed per generation of every run, one colour per condition (the run's
folder, e.g. ``symmetric`` or ``symmetric_pruned``), with the median of each
condition that has several runs. A run that is still
going can be plotted too; it shows the generations finished so far.
"""

# Standard library
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party libraries
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from sc_core import load_individuals
from sc_render import BodyCard, body_caption, save_grid

# Global constants
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CONTEXT = "#c3c2b7"  # de-emphasised lines
BLUE = "#2a78d6"  # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
AQUA = "#1baf7a"  # categorical slot 3
# Colour follows the gait family in every panel and every run.
FAMILY_COLORS = {"alternating": BLUE, "synchronous": ORANGE, "free": AQUA}
# Conditions in a comparison, in the order given; the three slots pass the
# palette checks for every pair.
CONDITION_COLORS = (BLUE, ORANGE, AQUA)
LINE_WIDTH = 1.0  # points: 2 px at 144 dpi
HAIRLINE = 0.5
MARKER_SIZE = 4.2  # points: an 8 px dot
DOT_AREA = 18.0  # points squared, for scatter dots
DPI = 144
FINISH_TIMES = (60, 45, 30, 20, 15, 10, 7.5, 5, 4, 3)  # s, reference lines
STRIP_BODIES = 8
BODY_COLUMNS = (
    "uid",
    "origin",
    "birth",
    "death",
    "alive",
    "speed",
    "finish_time",
    "furthest",
    "family",
    "n_modules",
    "n_hinges",
    "n_pairs",
    "n_plane",
    "drift",
    "turn",
    "unpruned_modules",
    "unpruned_hinges",
    "unpruned_speed",
    "cut",
    "peak_torque",
    "rms_torque",
    "peak_hinge_speed",
    "reversals_per_s",
    "hinge_frequency",
)
FONT = {"family": ["Arial", "DejaVu Sans"]}


@dataclass(frozen=True)
class Run:
    """
    One run's settings, per-generation log and bodies.

    Parameters
    ----------
    name
        Short name, e.g. ``symmetric_seed_01``.
    condition
        The run's folder, e.g. ``symmetric_pruned``; runs of one condition
        differ only in their seed.
    config
        The run's settings.
    history
        One entry per generation, from the run's metadata.
    individuals
        Every body, from :func:`sc_core.load_individuals`.
    """

    name: str
    condition: str
    config: dict[str, Any]
    history: list[dict[str, Any]]
    individuals: list[dict[str, Any]]


def load_run(db: Path) -> Run:
    """
    Read a run's database and metadata.

    Parameters
    ----------
    db
        The run database; its ``*_meta.json`` sits next to it.

    Returns
    -------
    Run
        The run.
    """
    meta_path = db.with_name(f"{db.stem}_meta.json")
    meta = json.loads(meta_path.read_text("utf-8"))
    return Run(
        name=f"{db.parent.name}_{db.stem}",
        condition=db.parent.name,
        config=meta["config"],
        history=meta.get("history", []),
        individuals=load_individuals(db),
    )


def style_axes(ax: Axes, title: str, x_label: str, y_label: str) -> None:
    """
    Apply the quiet chart chrome: hairline axes, muted ticks, left title.

    Parameters
    ----------
    ax
        The panel.
    title
        Panel title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    """
    ax.set_facecolor(SURFACE)
    # The pad leaves room for a legend row between title and plot.
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=22)
    ax.set_xlabel(x_label, fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel(y_label, fontsize=8.5, color=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(HAIRLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=8, width=HAIRLINE)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def add_line(
    ax: Axes,
    x: list[float],
    y: list[float],
    *,
    color: str,
    label: str,
) -> None:
    """
    Draw one series as a 2 px line with an 8 px end dot in a surface ring.

    Parameters
    ----------
    ax
        The panel.
    x
        X values.
    y
        Y values.
    color
        Series color.
    label
        Legend label.
    """
    ax.plot(
        x,
        y,
        color=color,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=label,
        zorder=3,
    )
    ax.plot(
        x[-1:],
        y[-1:],
        marker="o",
        markersize=MARKER_SIZE,
        color=color,
        markeredgecolor=SURFACE,
        markeredgewidth=1.0,
        zorder=4,
    )


def end_label(ax: Axes, x: float, y: float, text: str) -> None:
    """
    Write a direct label just right of a line's end, in secondary ink.

    Parameters
    ----------
    ax
        The panel.
    x
        X of the line's end.
    y
        Y of the line's end.
    text
        The label.
    """
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=7.5,
        color=INK_SECONDARY,
        va="center",
    )


def add_legend(ax: Axes) -> None:
    """
    Add a frameless one-row legend between the title and the plot.

    Above the plot it never covers data, the finish-time labels on the left
    or the line-end labels on the right.

    Parameters
    ----------
    ax
        The panel.
    """
    legend = ax.legend(
        frameon=False,
        fontsize=7.5,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=len(ax.get_legend_handles_labels()[0]),
        borderaxespad=0.2,
        handlelength=1.6,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def finish_time_lines(ax: Axes, config: dict[str, Any], top: float) -> None:
    """
    Draw hairlines at the speeds that cover the goal in round times.

    A speed of ``goal / t`` covers the goal in ``t`` seconds; these lines
    replace the usual gridlines on a speed axis. Their labels sit on the left,
    away from the line-end labels on the right.

    Parameters
    ----------
    ax
        A panel with speed (m/s) on the y-axis.
    config
        The run's settings (goal, time limit).
    top
        Upper y limit; lines above it are skipped.
    """
    goal, time_limit = config["goal"], config["time_limit"]
    for seconds in FINISH_TIMES:
        speed = goal / seconds
        # Keep the top of the panel free for the legend row.
        if seconds > time_limit or speed > 0.92 * top:
            continue
        ax.axhline(speed, color=GRID, linewidth=HAIRLINE, zorder=1)
        ax.annotate(
            f"{goal:g} m in {seconds:g} s",
            (0.0, speed),
            xycoords=("axes fraction", "data"),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.5,
            color=INK_MUTED,
            ha="left",
            va="bottom",
        )


def plot_speed(ax: Axes, run: Run) -> None:
    """
    Panel: speed of the fastest and of the average survivor per generation.

    Parameters
    ----------
    ax
        The panel.
    run
        The run.
    """
    gens = [h["generation"] for h in run.history]
    best = [h["best_speed"] for h in run.history]
    mean = [h["mean_speed"] for h in run.history]
    # Show at least the speed that just finishes within the time limit.
    just_finishes = run.config["goal"] / run.config["time_limit"]
    top = 1.2 * max(*best, just_finishes)
    finish_time_lines(ax, run.config, top)
    add_line(ax, gens, best, color=BLUE, label="fastest survivor")
    add_line(ax, gens, mean, color=ORANGE, label="survivor average")
    end_label(ax, gens[-1], best[-1], f"{best[-1]:.3f} m/s")
    ax.set_ylim(0, top)
    style_axes(ax, "Speed over the dash", "generation", "speed (m/s)")
    add_legend(ax)


def plot_bodies(ax: Axes, run: Run) -> None:
    """
    Panel: the speed of every body, by the generation it was born in.

    Parameters
    ----------
    ax
        The panel.
    run
        The run.
    """
    scored = [
        ind for ind in run.individuals if ind["tags"].get("speed") is not None
    ]
    if not scored:
        return
    top = 1.2 * max(ind["tags"]["speed"] for ind in scored)
    finish_time_lines(ax, run.config, top)
    rng = np.random.default_rng(0)  # fixed jitter, so reruns look the same
    for family, color in FAMILY_COLORS.items():
        bodies = [ind for ind in scored if ind["tags"].get("family") == family]
        if not bodies:
            continue
        births = np.array([ind["birth"] for ind in bodies], dtype=float)
        ax.scatter(
            births + rng.uniform(-0.25, 0.25, len(births)),
            [ind["tags"]["speed"] for ind in bodies],
            s=DOT_AREA,
            color=color,
            edgecolors=SURFACE,
            linewidths=0.6,
            alpha=0.85,
            label=family,
            zorder=3,
        )
    ax.set_ylim(0, top)
    style_axes(ax, "Every body, by birth", "generation born", "speed (m/s)")
    add_legend(ax)


def plot_families(ax: Axes, run: Run) -> None:
    """
    Panel: how many survivors use each gait family, per generation.

    Parameters
    ----------
    ax
        The panel.
    run
        The run.
    """
    gens = [h["generation"] for h in run.history]
    ax.grid(axis="y", color=GRID, linewidth=HAIRLINE)
    ax.set_axisbelow(True)
    for family, color in FAMILY_COLORS.items():
        counts = [h.get(f"{family}_survivors", 0) for h in run.history]
        if any(counts):
            add_line(ax, gens, counts, color=color, label=family)
            end_label(ax, gens[-1], counts[-1], f"{counts[-1]}")
    ax.set_ylim(0, run.config["pop"] * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    style_axes(
        ax,
        "Gait family of the survivors",
        "generation",
        f"bodies (of {run.config['pop']})",
    )
    add_legend(ax)


def plot_finishers(ax: Axes, run: Run) -> None:
    """
    Panel: survivors that cover the goal within the time limit.

    Parameters
    ----------
    ax
        The panel.
    run
        The run.
    """
    gens = [h["generation"] for h in run.history]
    finishers = [h["finishers"] for h in run.history]
    ax.grid(axis="y", color=GRID, linewidth=HAIRLINE)
    ax.set_axisbelow(True)
    add_line(ax, gens, finishers, color=BLUE, label="finishers")
    end_label(ax, gens[-1], finishers[-1], f"{finishers[-1]}")
    ax.set_ylim(0, run.config["pop"] * 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    goal, limit = run.config["goal"], run.config["time_limit"]
    style_axes(
        ax,
        f"Survivors that cover {goal:g} m within {limit:g} s",
        "generation",
        f"bodies (of {run.config['pop']})",
    )


def plot_body_size(ax: Axes, run: Run) -> None:
    """
    Panel: modules and hinges of the fastest body per generation.

    Parameters
    ----------
    ax
        The panel.
    run
        The run.
    """
    gens = [h["generation"] for h in run.history]
    modules = [h["best_modules"] for h in run.history]
    hinges = [h["best_hinges"] for h in run.history]
    ax.grid(axis="y", color=GRID, linewidth=HAIRLINE)
    ax.set_axisbelow(True)
    add_line(ax, gens, modules, color=BLUE, label="modules")
    add_line(ax, gens, hinges, color=ORANGE, label="hinges")
    end_label(ax, gens[-1], modules[-1], f"{modules[-1]} modules")
    end_label(ax, gens[-1], hinges[-1], f"{hinges[-1]} hinges")
    ax.set_ylim(0, run.config["max_modules"] + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    style_axes(ax, "Size of the fastest body", "generation", "count")
    add_legend(ax)


def plot_learning(ax: Axes, run: Run) -> None:
    """
    Panel: how fast learning finds a good gait, for every body of the run.

    Each grey line is one body's best speed so far over its CMA-ES
    generations; the blue line is their median.

    Parameters
    ----------
    ax
        The panel.
    run
        The run.
    """
    curves = [
        np.maximum.accumulate(ind["tags"]["learn_curve"])
        for ind in run.individuals
        if ind["tags"].get("learn_curve")
    ]
    if not curves:
        return
    steps = list(range(1, len(curves[0]) + 1))
    for curve in curves:
        ax.plot(steps, curve, color=CONTEXT, linewidth=0.4, alpha=0.5)
    ax.plot([], [], color=CONTEXT, linewidth=LINE_WIDTH, label="each body")
    median = np.median(np.vstack(curves), axis=0)
    add_line(ax, steps, median.tolist(), color=BLUE, label="median body")
    end_label(ax, steps[-1], float(median[-1]), f"{median[-1]:.3f} m/s")
    top = float(np.max(np.vstack(curves))) * 1.1
    finish_time_lines(ax, run.config, top)
    ax.set_ylim(0, top)
    per_family = run.config["brain"] == "symmetric"
    style_axes(
        ax,
        "Learning: best speed so far per body",
        "CMA-ES generation" + (" (in each gait family)" if per_family else ""),
        "speed (m/s)",
    )
    add_legend(ax)


def plot_run(run: Run, out: Path) -> None:
    """
    Write a run's progress figure, best-body strip and generation table.

    Parameters
    ----------
    run
        The run.
    out
        Output folder.
    """
    fig = Figure(figsize=(16, 8.6), facecolor=SURFACE)
    axes = fig.subplots(2, 3)
    plot_speed(axes[0][0], run)
    plot_bodies(axes[0][1], run)
    plot_finishers(axes[0][2], run)
    plot_families(axes[1][0], run)
    plot_body_size(axes[1][1], run)
    plot_learning(axes[1][2], run)
    cfg = run.config
    fig.suptitle(
        f"{run.name}: {cfg['brain']} brain, {cfg['pop']} + "
        f"{cfg['offspring']} bodies, {cfg['learn_gens']} x "
        f"{cfg['learn_pop']} races per body",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(h_pad=2.5, w_pad=3.0)
    fig.savefig(out / f"{run.name}_progress.png", dpi=DPI, facecolor=SURFACE)

    write_table(run, out / f"{run.name}_generations.csv")
    write_bodies(run, out / f"{run.name}_bodies.csv")
    plot_best_bodies(run, out / f"{run.name}_best_bodies.png")


def write_table(run: Run, path: Path) -> None:
    """
    Write the per-generation log as CSV: the table view of the figure.

    Parameters
    ----------
    run
        The run.
    path
        Output CSV.
    """
    if not run.history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run.history[0]))
        writer.writeheader()
        writer.writerows(run.history)


def write_bodies(run: Run, path: Path) -> None:
    """
    Write one row per body: the table view of the per-body panel.

    Parameters
    ----------
    run
        The run.
    path
        Output CSV.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(BODY_COLUMNS)
        for ind in sorted(run.individuals, key=lambda i: i["tags"]["uid"]):
            row = {**ind["tags"], **ind}
            writer.writerow([row.get(column) for column in BODY_COLUMNS])


def plot_best_bodies(run: Run, path: Path) -> None:
    """
    Draw the fastest body at up to eight evenly spaced generations.

    Parameters
    ----------
    run
        The run.
    path
        Output PNG.
    """
    if not run.history:
        return
    by_uid = {ind["tags"]["uid"]: ind for ind in run.individuals}
    last = len(run.history) - 1
    picks = sorted({
        round(i * last / (STRIP_BODIES - 1)) for i in range(STRIP_BODIES)
    })
    cards = []
    for index in picks:
        entry = run.history[index]
        ind = by_uid.get(entry["best_uid"])
        if ind is None:  # still being written by a running experiment
            continue
        caption = body_caption(ind["tags"], run.config["goal"], rank=1)
        caption = caption.replace("#1 ", f"gen {entry['generation']} ", 1)
        cards.append(BodyCard(genome=ind["genotype"]["morph"], caption=caption))
    if cards:
        save_grid(
            cards,
            path,
            title=f"{run.name}: the fastest body as evolution went on",
            spawn_yaw=run.config["spawn_yaw"],
            columns=len(cards),
        )


def plot_runs(runs: list[Run], path: Path) -> None:
    """
    Compare conditions: the fastest speed per generation of every run.

    Each condition (run folder) gets one colour: a single run is drawn as a
    line; several runs as thin lines with their median on top. The legend
    gives each condition's final speed. Only the first three conditions are
    drawn.

    Parameters
    ----------
    runs
        Two or more runs.
    path
        Output PNG.
    """
    fig = Figure(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax = fig.subplots()
    conditions = list(dict.fromkeys(run.condition for run in runs))
    top = 1.2 * max(h["best_speed"] for run in runs for h in run.history)
    finish_time_lines(ax, runs[0].config, top)
    for condition, color in zip(conditions, CONDITION_COLORS, strict=False):
        series = [
            [h["best_speed"] for h in run.history]
            for run in runs
            if run.condition == condition
        ]
        length = max(len(speeds) for speeds in series)
        line = [
            float(np.median([s[g] for s in series if g < len(s)]))
            for g in range(length)
        ]
        label = condition
        if len(series) > 1:
            for speeds in series:
                ax.plot(
                    range(len(speeds)),
                    speeds,
                    color=color,
                    linewidth=0.6,
                    alpha=0.35,
                )
            label += f" (median of {len(series)})"
        add_line(
            ax,
            list(range(length)),
            line,
            color=color,
            label=f"{label}: {line[-1]:.3f} m/s",
        )
    ax.set_ylim(0, top)
    style_axes(
        ax,
        "Fastest survivor per generation",
        "generation",
        "speed (m/s)",
    )
    add_legend(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def main() -> None:
    """Plot every run given on the command line."""
    parser = argparse.ArgumentParser(
        description="Plot symmetric co-evolution runs.",
    )
    parser.add_argument("dbs", type=Path, nargs="+", help="run databases")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    args = parser.parse_args()
    out = args.out or args.dbs[0].parent / "plots"
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl  # noqa: PLC0415 -- fonts for every figure

    mpl.rc("font", **FONT)
    runs = [load_run(db) for db in args.dbs]
    for run in runs:
        plot_run(run, out)
    if len(runs) > 1:
        plot_runs(runs, out / "runs_progress.png")
    print(f"Wrote plots for {len(runs)} run(s) to {out}")  # noqa: T201


if __name__ == "__main__":
    main()
