"""
Analyse the design-priors experiment.

Example
-------
    uv run --group analysis python \
        examples/d_design_priors/analyse_design_priors.py

Reads every completed run under ``<out>/<condition>/`` and writes to
``<out>/analysis/``:

* ``runs.csv`` -- one row per run with every metric;
* ``stats.csv`` -- Kruskal-Wallis across the four conditions, then each prior
  against the baseline (two-sided Mann-Whitney U, Holm-corrected over the
  three comparisons, Vargha-Delaney A12 = P(prior better than baseline));
* ``summary.md`` -- the same as readable tables;
* ``best_so_far.png``, ``primary_metrics.png``, ``morphology.png``.

Metrics
-------
* ``auc`` (primary): mean best-so-far score over the simulation budget, i.e.
  the area under best-so-far vs. cumulative simulations, divided by the
  budget.
* ``reeval_best`` (primary): the best mean score of the top bodies re-learned
  with fresh seeds at the end of the run.
* ``run_best``: the best score seen during the run.
* ``sims_to_tau``: simulations until the best-so-far score first reaches
  ``tau_fit`` (from ``analysis_config.json``, written by ``pilot.py``); runs
  that never reach it are censored and ranked last.
"""

# Standard library
import argparse
import json
import math
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

# Third-party libraries
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Local libraries
from dp_core import CONDITIONS, load_individuals
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from scipy import stats  # pyright: ignore[reportMissingTypeStubs]

DEFAULT_OUT = Path.home() / "ariel_runs" / "design_priors"

# Baseline is the gray reference; the three priors take the first three
# categorical slots, which pass the palette validator on all pairs.
COLORS = {
    "baseline": "#52514e",
    "seeded": "#2a78d6",
    "symmetric": "#eb6834",
    "joint_budget": "#1baf7a",
}
LABELS = {
    "baseline": "Baseline",
    "seeded": "Seeded",
    "symmetric": "Symmetric",
    "joint_budget": "Joint budget",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GRID_POINTS = 200

# Axis label and whether a larger value is better, per metric.
METRICS = {
    "auc": ("Mean best-so-far score over the budget (m)", True),
    "reeval_best": ("Re-evaluated final best score (m)", True),
    "run_best": ("Best score during the run (m)", True),
    "sims_to_tau": ("Simulations to reach tau_fit", False),
}
MORPHOLOGY = {
    "n_modules": "Modules",
    "n_hinges": "Hinges",
    "symmetry": "Symmetry (min of position, axis)",
    "seed_ancestry": "Share of seed ancestry",
}


@dataclass
class Run:
    """
    One completed run and its derived curves.

    Parameters
    ----------
    condition
        Condition name.
    seed
        Run seed.
    individuals
        Every individual of the run database.
    meta
        The run's metadata.
    reeval
        The run's re-learning results.
    """

    condition: str
    seed: int
    individuals: list[dict[str, Any]]
    meta: dict[str, Any]
    reeval: dict[str, Any]
    sims: np.ndarray = field(init=False)
    best: np.ndarray = field(init=False)
    spent: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Build the best-so-far curve in evaluation order."""
        evaluated = sorted(
            (ind for ind in self.individuals if ind["tags"].get("n_sims", 0)),
            key=lambda ind: (ind["birth"], ind["tags"]["uid"]),
        )
        spent = np.array([ind["tags"]["n_sims"] for ind in evaluated], float)
        scores = np.array(
            [
                -math.inf
                if ind["tags"]["score"] is None
                else ind["tags"]["score"]
                for ind in evaluated
            ],
        )
        self.spent = spent
        self.sims = np.cumsum(spent)
        self.best = np.maximum.accumulate(scores)

    def best_at(self, budget: np.ndarray) -> np.ndarray:
        """
        Best-so-far score after ``budget`` simulations.

        Parameters
        ----------
        budget
            Simulation counts.

        Returns
        -------
        np.ndarray
            The best score reached by each count (NaN before the first
            evaluation finished).
        """
        index = np.searchsorted(self.sims, budget, side="right") - 1
        values = np.where(
            index >= 0,
            self.best[np.clip(index, 0, None)],
            np.nan,
        )
        return values.astype(float)

    def metrics(self, tau: float | None) -> dict[str, Any]:
        """
        Compute the per-run metrics.

        Parameters
        ----------
        tau
            Efficiency threshold, or ``None`` if unknown.

        Returns
        -------
        dict[str, Any]
            One value per metric, plus the budget and whether tau was reached.
        """
        total = float(self.sims[-1])
        reached = np.nonzero(self.best >= tau)[0] if tau is not None else []
        return {
            "condition": self.condition,
            "seed": self.seed,
            "budget_sims": total,
            "auc": float(np.sum(self.best * self.spent) / total),
            "run_best": float(self.best[-1]),
            "reeval_best": self.reeval.get("reeval_best"),
            "sims_to_tau": float(self.sims[reached[0]])
            if len(reached)
            else None,
            "reached_tau": bool(len(reached)),
        }

    def survivor_means(self, generation: int) -> dict[str, float]:
        """
        Mean body descriptors of the survivors of one generation.

        Parameters
        ----------
        generation
            Generation index (0 is the initial population).

        Returns
        -------
        dict[str, float]
            Mean of each descriptor in :data:`MORPHOLOGY`.
        """
        survivors = [
            ind["tags"]
            for ind in self.individuals
            if ind["birth"] <= generation
            and (ind["death"] > generation or ind["alive"])
        ]
        return {
            "n_modules": statistics.mean(t["n_modules"] for t in survivors),
            "n_hinges": statistics.mean(t["n_hinges"] for t in survivors),
            "symmetry": statistics.mean(
                min(t["sym_position"], t["sym_axis"]) for t in survivors
            ),
            "seed_ancestry": statistics.mean(
                t["seed_ancestry"] for t in survivors
            ),
        }


def load_runs(out: Path) -> list[Run]:
    """
    Load every completed run under ``out``.

    Parameters
    ----------
    out
        The experiment's output root.

    Returns
    -------
    list[Run]
        Completed runs, ordered by condition and seed.
    """
    runs = []
    for condition in CONDITIONS:
        for meta_path in sorted((out / condition).glob("seed_*_meta.json")):
            meta = json.loads(meta_path.read_text("utf-8"))
            if meta.get("status") != "completed":
                continue
            stem = meta_path.name.removesuffix("_meta.json")
            reeval_path = meta_path.with_name(f"{stem}_reeval.json")
            runs.append(
                Run(
                    condition=condition,
                    seed=int(meta["config"]["seed"]),
                    individuals=load_individuals(
                        meta_path.with_name(f"{stem}.db"),
                    ),
                    meta=meta,
                    reeval=json.loads(reeval_path.read_text("utf-8")),
                ),
            )
    return runs


def resolve_tau(
    out: Path,
    runs: list[Run],
    override: float | None,
) -> tuple[float | None, str]:
    """
    Pick the efficiency threshold.

    Parameters
    ----------
    out
        The experiment's output root (holds ``analysis_config.json``).
    runs
        The loaded runs, used for the fallback rule.
    override
        A threshold given on the command line.

    Returns
    -------
    tuple[float | None, str]
        The threshold and where it came from.
    """
    if override is not None:
        return override, "command line"
    config_path = out / "analysis_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text("utf-8"))
        if config.get("tau_fit") is not None:
            return float(config["tau_fit"]), "pilot (analysis_config.json)"
    baseline = [run for run in runs if run.condition == "baseline"]
    if not baseline:
        return None, "unavailable"
    halves = [
        float(run.best_at(np.array([run.sims[-1] / 2]))[0]) for run in baseline
    ]
    return statistics.median(
        halves,
    ), "fallback: median baseline best at half budget"


def a12(better: list[float], worse: list[float]) -> float:
    """
    Vargha-Delaney A12: probability a value of ``better`` beats ``worse``.

    Parameters
    ----------
    better
        Sample oriented so that larger means better.
    worse
        The comparison sample, same orientation.

    Returns
    -------
    float
        A12 in [0, 1]; 0.5 means no difference.
    """
    wins = sum((x > y) + 0.5 * (x == y) for x in better for y in worse)
    return wins / (len(better) * len(worse))


def holm(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjustment.

    Parameters
    ----------
    p_values
        Raw p-values.

    Returns
    -------
    list[float]
        Adjusted p-values, in the input order.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def compare(table: pl.DataFrame, metric: str) -> list[dict[str, Any]]:
    """
    Test one metric across conditions and each prior against the baseline.

    Parameters
    ----------
    table
        The per-run metrics.
    metric
        Column to test.

    Returns
    -------
    list[dict[str, Any]]
        One row for Kruskal-Wallis and one per prior.
    """
    _, higher_is_better = METRICS[metric]
    budget = float(cast("float", table["budget_sims"].max()))
    censor = budget + 1 if metric == "sims_to_tau" else None
    samples: dict[str, list[float]] = {}
    for condition in CONDITIONS:
        column = table.filter(pl.col("condition") == condition)[metric]
        values = [censor if v is None else float(v) for v in column.to_list()]
        samples[condition] = [v for v in values if v is not None]

    rows: list[dict[str, Any]] = []
    groups = [values for values in samples.values() if len(values) >= 2]
    if len(groups) >= 2:
        kruskal = stats.kruskal(*groups)
        rows.append({
            "metric": metric,
            "comparison": "all conditions",
            "test": "Kruskal-Wallis",
            "statistic": float(kruskal.statistic),
            "p": float(kruskal.pvalue),
            "p_holm": None,
            "a12": None,
        })

    sign = 1.0 if higher_is_better else -1.0
    baseline = samples["baseline"]
    prior_rows = []
    for prior in CONDITIONS[1:]:
        values = samples[prior]
        if len(values) < 2 or len(baseline) < 2:
            continue
        test = stats.mannwhitneyu(values, baseline, alternative="two-sided")
        prior_rows.append({
            "metric": metric,
            "comparison": f"{prior} vs baseline",
            "test": "Mann-Whitney U",
            "statistic": float(test.statistic),
            "p": float(test.pvalue),
            "a12": a12(
                [sign * v for v in values],
                [sign * v for v in baseline],
            ),
        })
    for row, adjusted in zip(
        prior_rows,
        holm([row["p"] for row in prior_rows]),
        strict=True,
    ):
        row["p_holm"] = adjusted
    return rows + prior_rows


def success_tests(table: pl.DataFrame) -> list[dict[str, Any]]:
    """
    Fisher's exact test on how many runs reach tau, prior vs baseline.

    Parameters
    ----------
    table
        The per-run metrics.

    Returns
    -------
    list[dict[str, Any]]
        One row per prior, with Holm-adjusted p-values.
    """

    def counts(condition: str) -> tuple[int, int]:
        reached = table.filter(pl.col("condition") == condition)["reached_tau"]
        hits = int(reached.sum())
        return hits, len(reached) - hits

    rows = []
    base_hits, base_misses = counts("baseline")
    for prior in CONDITIONS[1:]:
        hits, misses = counts(prior)
        if hits + misses == 0 or base_hits + base_misses == 0:
            continue
        test = stats.fisher_exact([[hits, misses], [base_hits, base_misses]])
        rows.append({
            "metric": "reached_tau",
            "comparison": f"{prior} vs baseline",
            "test": "Fisher exact",
            "statistic": float(test.statistic),
            "p": float(test.pvalue),
            "a12": None,
        })
    for row, adjusted in zip(
        rows,
        holm([row["p"] for row in rows]),
        strict=True,
    ):
        row["p_holm"] = adjusted
    return rows


# -- Figures ----------------------------------------------------------------


def _style(ax: Axes, *, ylabel: str, xlabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(visible=True, axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)


def _new_figure(ncols: int, width: float) -> tuple[Any, list[Axes]]:
    fig, axes = plt.subplots(1, ncols, figsize=(width, 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, list(axes[0])


def _label_line_ends(ax: Axes, x_end: float, ends: dict[str, float]) -> None:
    # Direct labels at the right end of each line, pushed apart so that
    # conditions ending close together stay readable.
    low, high = ax.get_ylim()
    gap = 0.05 * (high - low)
    previous = -math.inf
    for condition in sorted(ends, key=ends.__getitem__):
        y = max(ends[condition], previous + gap)
        previous = y
        ax.text(
            x_end * 1.012,
            y,
            LABELS[condition],
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )


def plot_best_so_far(runs: list[Run], path: Path) -> None:
    """
    Median best-so-far score vs. simulations, with interquartile bands.

    Parameters
    ----------
    runs
        The loaded runs.
    path
        Output PNG.
    """
    budget = max(float(run.sims[-1]) for run in runs)
    grid = np.linspace(0, budget, GRID_POINTS)
    fig, (ax,) = _new_figure(1, 8.0)
    ends: dict[str, float] = {}
    for condition in CONDITIONS:
        group = [run for run in runs if run.condition == condition]
        if not group:
            continue
        curves = np.vstack([run.best_at(grid) for run in group])
        with warnings.catch_warnings():
            # Before the first evaluation every curve is NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(curves, axis=0)
            low, high = np.nanpercentile(curves, [25, 75], axis=0)
        color = COLORS[condition]
        ax.fill_between(grid / 1000, low, high, color=color, alpha=0.10, lw=0)
        ax.plot(
            grid / 1000,
            median,
            color=color,
            lw=2,
            solid_capstyle="round",
            label=f"{LABELS[condition]} (n={len(group)})",
        )
        ends[condition] = float(median[-1])
    _label_line_ends(ax, grid[-1] / 1000, ends)
    _style(
        ax,
        ylabel="Best-so-far score (m)",
        xlabel="Simulations (thousands)",
    )
    ax.set_title(
        "Best-so-far score: median and interquartile range across runs",
        color=INK,
        fontsize=11,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_primary_metrics(table: pl.DataFrame, path: Path) -> None:
    """
    One dot per run and the median, for each primary metric.

    Parameters
    ----------
    table
        The per-run metrics.
    path
        Output PNG.
    """
    fig, axes = _new_figure(2, 10.0)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes, ("auc", "reeval_best"), strict=True):
        for position, condition in enumerate(CONDITIONS):
            values = [
                float(v)
                for v in table.filter(pl.col("condition") == condition)[metric]
                if v is not None
            ]
            if not values:
                continue
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(
                position + jitter,
                values,
                s=40,
                color=COLORS[condition],
                edgecolor=SURFACE,
                linewidth=2,
                zorder=3,
            )
            ax.hlines(
                statistics.median(values),
                position - 0.25,
                position + 0.25,
                color=INK,
                lw=2,
                zorder=4,
            )
        ax.set_xticks(range(len(CONDITIONS)), [LABELS[c] for c in CONDITIONS])
        _style(ax, ylabel=METRICS[metric][0])
    axes[0].set_title(
        "Primary metrics per run (bar = median)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_morphology(runs: list[Run], path: Path) -> None:
    """
    Mean survivor body descriptors per generation, one panel per descriptor.

    Parameters
    ----------
    runs
        The loaded runs.
    path
        Output PNG.
    """
    gens = int(min(run.meta["config"]["gens"] for run in runs))
    generations = np.arange(gens + 1)
    fig, axes = _new_figure(len(MORPHOLOGY), 16.0)
    for ax, (key, label) in zip(axes, MORPHOLOGY.items(), strict=True):
        for condition in CONDITIONS:
            group = [run for run in runs if run.condition == condition]
            # Only seeded runs have seed ancestry; the others are 0 throughout.
            if not group or (key == "seed_ancestry" and condition != "seeded"):
                continue
            means = np.array([
                [run.survivor_means(int(g))[key] for g in generations]
                for run in group
            ])
            ax.plot(
                generations,
                means.mean(axis=0),
                color=COLORS[condition],
                lw=2,
                label=LABELS[condition],
            )
        _style(ax, ylabel=label, xlabel="Generation")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_title(
        "Survivor bodies over generations (mean across runs)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# -- Report -----------------------------------------------------------------


def _quartiles(rows: pl.DataFrame, metric: str) -> str:
    # Median [Q1, Q3]; runs that never reached tau count as "not reached",
    # so a quartile past them is reported as such rather than dropped.
    censored = metric == "sims_to_tau"
    values = [
        math.inf if v is None else float(v)
        for v in rows[metric]
        if v is not None or censored
    ]
    if not values:
        return "n/a"
    method = "lower" if censored else "linear"
    quartiles = np.percentile(values, [25, 50, 75], method=method)
    q1, q2, q3 = (
        "not reached" if math.isinf(q) else f"{q:.4g}" for q in quartiles
    )
    return f"{q2} [{q1}, {q3}]"


def summary_markdown(
    table: pl.DataFrame,
    tests: list[dict[str, Any]],
    tau: float | None,
    tau_source: str,
) -> str:
    """
    Render the per-condition medians and the tests as Markdown.

    Parameters
    ----------
    table
        The per-run metrics.
    tests
        Rows from :func:`compare` and :func:`success_tests`.
    tau
        The efficiency threshold.
    tau_source
        Where the threshold came from.

    Returns
    -------
    str
        The report.
    """
    lines = [
        "# Design-priors experiment: results",
        "",
        f"tau_fit = {tau} ({tau_source})",
        "",
        "## Per condition: median [interquartile range]",
        "",
        "| condition | runs | " + " | ".join(METRICS) + " | reached tau |",
        "|---" * (len(METRICS) + 3) + "|",
    ]
    for condition in CONDITIONS:
        rows = table.filter(pl.col("condition") == condition)
        if rows.height == 0:
            continue
        cells = [_quartiles(rows, metric) for metric in METRICS]
        reached = int(rows["reached_tau"].sum())
        lines.append(
            f"| {LABELS[condition]} | {rows.height} | "
            + " | ".join(cells)
            + f" | {reached}/{rows.height} |",
        )
    lines += [
        "",
        "## Tests",
        "",
        "A12 = P(prior better than baseline); 0.5 = no difference. "
        "p_holm is Holm-corrected over the three priors.",
        "",
        "| metric | comparison | test | statistic | p | p_holm | A12 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in tests:
        p_holm = "" if row.get("p_holm") is None else f"{row['p_holm']:.4g}"
        effect = "" if row.get("a12") is None else f"{row['a12']:.2f}"
        lines.append(
            f"| {row['metric']} | {row['comparison']} | {row['test']} | "
            f"{row['statistic']:.4g} | {row['p']:.4g} | {p_holm} | {effect} |",
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """
    Load the runs, compute metrics and tests, and write the report.

    Raises
    ------
    SystemExit
        If there are no completed runs to analyse.
    """
    parser = argparse.ArgumentParser(
        description="Analyse the design-priors experiment.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()

    runs = load_runs(args.out)
    if not runs:
        msg = f"No completed runs under {args.out}"
        raise SystemExit(msg)
    tau, tau_source = resolve_tau(args.out, runs, args.tau)
    table = pl.DataFrame([run.metrics(tau) for run in runs])
    budgets = set(table["budget_sims"].to_list())
    if len(budgets) > 1:
        print(f"Warning: runs used different budgets: {sorted(budgets)}")  # noqa: T201

    tests = [row for metric in METRICS for row in compare(table, metric)]
    tests += success_tests(table)

    analysis = args.out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    table.write_csv(analysis / "runs.csv")
    pl.DataFrame(tests).write_csv(analysis / "stats.csv")
    report = summary_markdown(table, tests, tau, tau_source)
    (analysis / "summary.md").write_text(report, encoding="utf-8")
    plot_best_so_far(runs, analysis / "best_so_far.png")
    plot_primary_metrics(table, analysis / "primary_metrics.png")
    plot_morphology(runs, analysis / "morphology.png")
    print(report)  # noqa: T201
    print(f"Wrote {analysis}")  # noqa: T201


if __name__ == "__main__":
    main()
