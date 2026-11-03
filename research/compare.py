# mkdir -p /scratch-shared/$USER/extract
# sbatch -p genoa -c 24 -t 3:00:00 -J extract -o extract_%j.out --wrap "export TMPDIR=/scratch-shared/$USER/extract OMP_NUM_THREADS=1; ../.venv/bin/python compare.py extract runs/snellius/islands/*_* runs/snellius/features/* runs/snellius/gecko_lab/* $HOME/robot-olympics/ariel-olympics-team2/research/runs/snellius/*_*"

import argparse
import csv
import json
import lzma
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
from ariel.ec import Archive, Individual, Population
from scipy.stats import mannwhitneyu
from sqlmodel import Session, select

from controller import CONTROLLERS, BrainConfig
from evolution import best_combined, diversity, fitness_stats
from fitness import CONTROL_STEPS, GOAL, TIME_LIMIT, TIMESTEP, load_body, race_speed, simulate
from run import BLUE, ORANGE, SURFACE, TEXT, TEXT_MUTED, console

COLORS = {"ga": BLUE, "revde": ORANGE}
LABELS = {"ga": "GA", "revde": "RevDE"}
OFFSETS = {"ga": -0.18, "revde": 0.18}
BODIES = ("spider", "gecko", "snake", "single-sprinter")
TERRAINS = {4: "linoleum", 5: "carpet", 6: "carpet slope", 7: "grass", 8: "silicone bumps", 9: "grass slope", 10: "lab track"}
GOAL_SPEED = GOAL / TIME_LIMIT
HIDDEN = {"0": "none", "16": "16", "32": "32", "16,16": "16+16", "32,32": "32+32"}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["plot", "extract"])
    p.add_argument("runs", type=Path, nargs="+", help="run folders named <body>_<algorithm>_<controller>")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "runs/snellius/plots")
    p.add_argument("--no-simulation", action="store_true", help="skip re-simulating the best robots per terrain")
    p.add_argument("--force", action="store_true", help="extract: redo runs that already have a progress.csv")
    return p.parse_args()


def cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


@contextmanager
def database(seed_dir: Path) -> Iterator[Path]:
    if (seed_dir / "database.db").exists():
        yield seed_dir / "database.db"
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "database.db"
        with lzma.open(seed_dir / "database.db.xz") as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        yield path


def load_run(folder: Path) -> dict | None:
    body, algorithm, controller = folder.name.split("_", 2)
    log = folder / "logs" / "final.log"
    if not log.exists() or not (folder / "final" / body / "best_params.json").exists():
        return None
    seeds = {int(s): float(f) for s, f in re.findall(r"seed (\d+): best fitness ([0-9.]+)", log.read_text())}
    study = optuna.load_study(study_name=body, storage=f"sqlite:///{folder / 'tune' / body / 'optuna.db'}")
    progress = folder / "final" / body / "progress.csv"
    return {
        "folder": folder, "body": body, "algorithm": algorithm, "controller": controller, "seeds": seeds,
        "trials": [t for t in study.trials if t.value is not None], "best_trial": study.best_trial,
        "progress": list(csv.DictReader(progress.open())) if progress.exists() else None,
    }


def legend_handles(algorithms: list[str]) -> list:
    return [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COLORS[a], label=LABELS[a]) for a in algorithms]


def finish(fig: plt.Figure, ax: plt.Axes | None, title: str, path: Path) -> None:
    if ax is None:
        fig.set_layout_engine("constrained")
        fig.suptitle(title, x=0.01, ha="left")
    else:
        ax.set_title(title, loc="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.log(f"saved {path}")


def goal_line(ax: plt.Axes) -> None:
    ax.axhline(GOAL_SPEED, color=TEXT_MUTED, linewidth=1, zorder=1)
    ax.annotate("3 m in 30 s on every terrain", (1, GOAL_SPEED), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=TEXT_MUTED)


def strips(runs: list[dict], values_of, ylabel: str, title: str, path: Path, highlight=None, compare: bool = True) -> None:
    bodies = [b for b in BODIES if any(r["body"] == b for r in runs)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, body in enumerate(bodies):
        group = {r["algorithm"]: r for r in runs if r["body"] == body}
        for algorithm, run in group.items():
            values = np.asarray(values_of(run))
            x = i + OFFSETS[algorithm]
            ax.scatter(x - 0.04 + rng.uniform(-0.05, 0.05, len(values)), values, s=36, color=COLORS[algorithm],
                       edgecolors=SURFACE, linewidths=1, zorder=3)
            if highlight is not None:
                ax.scatter(x - 0.04, highlight(run), s=90, facecolors="none", edgecolors=TEXT, linewidths=1.5, zorder=4)
            ax.errorbar(x + 0.1, values.mean(), yerr=values.std(ddof=1), fmt="_", markersize=12, color=TEXT,
                        elinewidth=1.5, capsize=3, zorder=4)
        if compare and len(group) == 2:
            p = mannwhitneyu(values_of(group["ga"]), values_of(group["revde"])).pvalue
            top = max(max(values_of(r)) for r in group.values())
            ax.annotate(f"p = {p:.4f}" if p >= 0.0001 else "p < 0.0001", (i, top), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT_MUTED)
    top = max(max(values_of(r)) for r in runs)
    if compare:
        goal_line(ax)
        top = max(top, GOAL_SPEED)
    ax.set_xticks(range(len(bodies)), bodies)
    ax.set_xlim(-0.6, len(bodies) - 0.4)
    ax.set_ylim(0, top * 1.15)
    ax.set_ylabel(ylabel)
    handles = legend_handles(sorted({r["algorithm"] for r in runs}))
    handles.append(plt.Line2D([], [], marker="_", markersize=12, color=TEXT, label="mean ± sd"))
    if highlight is not None:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="none",
                                  markeredgecolor=TEXT, label="tuning winner"))
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    finish(fig, ax, title, path)


def dumbbell(runs: list[dict], path: Path) -> None:
    rows = sorted(runs, key=lambda r: np.mean(list(r["seeds"].values())))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4))
    for y, run in enumerate(rows):
        final = np.array(list(run["seeds"].values()))
        tune = run["best_trial"].value
        color = COLORS[run["algorithm"]]
        ax.plot([tune, final.mean()], [y, y], color=color, linewidth=2, zorder=2)
        ax.errorbar(final.mean(), y, xerr=final.std(ddof=1), fmt="none", ecolor=color, elinewidth=1, capsize=3, zorder=2)
        ax.scatter(tune, y, s=60, facecolors=SURFACE, edgecolors=color, linewidths=2, zorder=3)
        ax.scatter(final.mean(), y, s=60, color=color, edgecolors=SURFACE, linewidths=1, zorder=3)
        ax.annotate(f"×{final.mean() / tune:.1f}", (final.mean() + final.std(ddof=1), y), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TEXT_MUTED)
    ax.axvline(GOAL_SPEED, color=TEXT_MUTED, linewidth=1, zorder=1)
    ax.annotate("3 m in 30 s on every terrain", (GOAL_SPEED, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                textcoords="offset points", ha="left", va="top", fontsize=8, color=TEXT_MUTED)
    ax.set_yticks(range(len(rows)), [f"{r['body']} · {LABELS[r['algorithm']]}" for r in rows])
    ax.set_xlim(left=0)
    ax.set_xlabel("fitness: speed on the slowest terrain (m/s)")
    handles = legend_handles(sorted({r["algorithm"] for r in runs})) + [
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE, markeredgecolor=TEXT_MUTED,
                   label="tuning winner, 1,000 evaluations"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=TEXT_MUTED, label="final mean ± sd, 20,000 evaluations"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    finish(fig, ax, "From tuning to final runs: what 20× more evaluations bought", path)


def hyperparameters(runs: list[dict], path: Path) -> None:
    panels = [
        ("MLP hidden layers", "mlp_hidden", list(HIDDEN), lambda v: HIDDEN[v]),
        ("activation", "mlp_activation", ["tanh", "relu", "elu"], str),
        ("population size", "pop_size", [24, 48, 72], str),
        ("mutation std (GA)", "mutation_std", [0.1, 0.2, 0.3, 0.5], str),
    ]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 3.8), sharey=True,
                             gridspec_kw={"width_ratios": [len(c) for _, _, c, _ in panels]})
    for ax, (name, key, categories, label) in zip(axes, panels, strict=True):
        for algorithm in sorted({r["algorithm"] for r in runs}):
            for i, category in enumerate(categories):
                scores = [t.value / r["best_trial"].value for r in runs if r["algorithm"] == algorithm
                          for t in r["trials"] if t.params.get(key) == category]
                if not scores:
                    continue
                x = i + OFFSETS[algorithm] * 0.8
                ax.scatter(x + rng.uniform(-0.05, 0.05, len(scores)), scores, s=24, color=COLORS[algorithm],
                           edgecolors=SURFACE, linewidths=1, zorder=3)
                ax.plot([x - 0.1, x + 0.1], [np.mean(scores)] * 2, color=TEXT, linewidth=2, zorder=4)
        ax.set_xticks(range(len(categories)), [label(c) for c in categories])
        ax.set_xlim(-0.6, len(categories) - 0.4)
        ax.set_xlabel(name)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("tuning score / best score of its study")
    axes[0].set_ylim(0, 1.08)
    axes[-1].legend(handles=legend_handles(sorted({r["algorithm"] for r in runs}))
                    + [plt.Line2D([], [], color=TEXT, linewidth=2, label="mean")], frameon=False, fontsize=8, loc="lower right")
    n = sum(len(r["trials"]) for r in runs)
    finish(fig, None, f"Tuning: how each setting scored ({n} trials, each a mean over 10 seeds)", path)


def best_robot_speeds(run: dict) -> dict[int, float]:
    best_seed = max(run["seeds"], key=run["seeds"].get)
    seed_dir = run["folder"] / "final" / run["body"] / "trials" / "trial_000" / f"seed_{best_seed}"
    cfg = json.loads((seed_dir / "config.json").read_text())
    brain = BrainConfig(name=cfg["controller"]["name"], hidden=tuple(cfg["controller"]["hidden"]),
                        activation=cfg["controller"]["activation"])
    with database(seed_dir) as db:
        archive = Archive(db)
        best = best_combined(archive.engine) if cfg["islands"] else archive.best_individual("max")
    genotype = np.asarray(best.genotype)
    speeds = {}
    for env in cfg["fitness"]["environments"]:
        robot = load_body(cfg["body"])
        path = simulate(CONTROLLERS[brain.name](genotype, len(robot.spec.actuators), CONTROL_STEPS * TIMESTEP, brain), robot, env)
        speeds[env] = 0.0 if path is None else race_speed(path)
    return speeds


def terrain_bars(runs: list[dict], path: Path) -> None:
    bodies = [b for b in BODIES if any(r["body"] == b for r in runs)]
    fig, axes = plt.subplots(1, len(bodies), figsize=(4 * len(bodies), 3.8), sharey=True, squeeze=False)
    for ax, body in zip(axes[0], bodies, strict=True):
        for run in [r for r in runs if r["body"] == body]:
            speeds = best_robot_speeds(run)
            x = np.arange(len(speeds)) + OFFSETS[run["algorithm"]]
            ax.bar(x, list(speeds.values()), width=0.34, color=COLORS[run["algorithm"]], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.set_xticks(range(len(speeds)), [TERRAINS.get(e, str(e)) for e in speeds])
        ax.axhline(GOAL_SPEED, color=TEXT_MUTED, linewidth=1, zorder=4)
        ax.set_title(body)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("race speed of the best robot (m/s)")
    axes[0][-1].legend(handles=legend_handles(sorted({r["algorithm"] for r in runs})), frameon=False, fontsize=8, loc="upper right")
    goal_line(axes[0][-1])
    finish(fig, None, "Best final robot of each configuration on each terrain", path)


def progress_lines(runs: list[dict], column: str, ylabel: str, title: str, path: Path) -> None:
    runs = [r for r in runs if r["progress"]]
    if not runs:
        console.log(f"no progress.csv found, skipped {path.name} (run compare.py extract on Snellius first)")
        return
    bodies = [b for b in BODIES if any(r["body"] == b for r in runs)]
    islands = list(dict.fromkeys(row["island"] for r in runs for row in r["progress"]))
    fig, axes = plt.subplots(len(islands), len(bodies), figsize=(4 * len(bodies), 3.4 * len(islands)),
                             sharex=True, sharey=True, squeeze=False)
    for ax_row, island in zip(axes, islands, strict=True):
        for ax, body in zip(ax_row, bodies, strict=True):
            for run in [r for r in runs if r["body"] == body]:
                rows = [row for row in run["progress"] if row["island"] == island]
                seeds = sorted({int(row["seed"]) for row in rows})
                per_seed = [[row for row in rows if int(row["seed"]) == s] for s in seeds]
                n = min(len(r) for r in per_seed)
                x = np.array([float(row["simulations"]) for row in per_seed[0][:n]]) / 1000
                values = np.array([[float(row[column]) for row in r[:n]] for r in per_seed])
                w = max(1, n // 50)
                values = np.array([np.convolve(v, np.ones(w) / w, mode="valid") for v in values])
                x = x[w - 1:]
                color = COLORS[run["algorithm"]]
                ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=color, alpha=0.15, linewidth=0)
                ax.plot(x, values.mean(axis=0), color=color, linewidth=2)
            ax.set_title(body if island == "all" else f"{body} · {island} island")
    for ax in axes[-1]:
        ax.set_xlabel("terrain simulations (thousands)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0][-1].legend(handles=legend_handles(sorted({r["algorithm"] for r in runs})), frameon=False, fontsize=8)
    finish(fig, None, title, path)


def populations(archive: Archive) -> Iterator[tuple[int, list[Individual], Population]]:
    first, last = archive.generation_range
    alive: list[Individual] = []
    with Session(archive.engine) as session:
        for generation in range(first, last + 1):
            born = [i for i in session.exec(select(Individual).where(Individual.time_of_birth == generation)) if not i.requires_eval]
            alive = [i for i in [*alive, *born] if i.time_of_death > generation or (i.alive and i.time_of_death >= generation)]
            yield generation, born, Population(alive)


def seed_progress(seed_dir: Path) -> tuple[Path, list[dict]]:
    cfg = json.loads((seed_dir / "config.json").read_text())
    environments = cfg["fitness"]["environments"]
    per_evaluation = 1 if cfg["islands"] else len(environments)
    rows, evaluations = [], 0
    with database(seed_dir) as db:
        for generation, born, population in populations(Archive(db)):
            evaluations += len(born)
            groups = ({TERRAINS.get(env, str(env)): population.where(lambda i, k=k: i.tags.get("island") == k)
                       for k, env in enumerate(environments)} if cfg["islands"] else {"all": population})
            if cfg["islands"]:
                groups["flat"] = population.where(lambda i: "island" not in i.tags)
            rows.extend({"seed": cfg["seed"], "generation": generation, "simulations": evaluations * per_evaluation,
                         "island": island, **fitness_stats(group), "diversity": diversity(group)}
                        for island, group in groups.items() if len(group))
    return seed_dir, rows


def extract(folders: list[Path], force: bool) -> None:
    tasks = {}
    for folder in folders:
        body = folder.name.split("_", 1)[0]
        out = folder / "final" / body / "progress.csv"
        if not (folder / "final" / body / "best_params.json").exists():
            continue
        if out.exists() and not force:
            console.log(f"{folder.name}: progress.csv exists, skipped (use --force to redo it)")
            continue
        tasks[folder] = sorted((folder / "final" / body / "trials").glob("trial_*/seed_*"))
    seeds = [seed for folder in tasks for seed in tasks[folder]]
    console.log(f"{len(tasks)} runs, {len(seeds)} seeds, {cpus()} workers")
    pending = {folder: len(tasks[folder]) for folder in tasks}
    rows = {folder: [] for folder in tasks}
    with Pool(min(cpus(), max(len(seeds), 1))) as pool:
        for seed_dir, seed_rows in pool.imap_unordered(seed_progress, seeds):
            folder = seed_dir.parents[4]
            rows[folder].extend(seed_rows)
            pending[folder] -= 1
            if pending[folder]:
                continue
            out = folder / "final" / folder.name.split("_", 1)[0] / "progress.csv"
            with out.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[folder][0]))
                writer.writeheader()
                writer.writerows(sorted(rows[folder], key=lambda r: (r["seed"], r["generation"])))
            console.log(f"{folder.name}: {len(tasks[folder])} seeds, {len(rows[folder])} rows -> {out}")


def main() -> None:
    args = parse_args()
    if args.mode == "extract":
        extract(args.runs, args.force)
        return
    runs = [run for run in map(load_run, args.runs) if run is not None]
    console.log(f"{len(runs)} complete runs: {', '.join(r['folder'].name for r in runs)}")
    args.out.mkdir(parents=True, exist_ok=True)
    strips(runs, lambda r: list(r["seeds"].values()), "fitness: speed on the slowest terrain (m/s)",
           "Final runs: best robot of each seed (10 seeds, 20,000 evaluations each)", args.out / "final_seeds.png")
    strips(runs, lambda r: [t.value for t in r["trials"]], "tuning score: mean over 10 seeds (m/s)",
           "Tuning: the 10 parameter sets of each configuration (1,000 evaluations per run)",
           args.out / "tuning_trials.png", highlight=lambda r: r["best_trial"].value, compare=False)
    dumbbell(runs, args.out / "tuning_to_final.png")
    hyperparameters(runs, args.out / "hyperparameters.png")
    if not args.no_simulation:
        terrain_bars(runs, args.out / "best_robot_terrains.png")
    progress_lines(runs, "mean", "mean fitness of the population (m/s)",
                   "Mean fitness during the final runs (mean of 10 seeds, band = min to max, moving average over 2% of the run)",
                   args.out / "progress_fitness.png")
    progress_lines(runs, "diversity", "genotype diversity",
                   "Diversity during the final runs (mean of 10 seeds, band = min to max, moving average over 2% of the run)",
                   args.out / "progress_diversity.png")


if __name__ == "__main__":
    main()
