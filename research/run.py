import argparse
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from ariel.ec import Archive, Population
from matplotlib.ticker import MaxNLocator
from rich.console import Console

from controller import BrainConfig, parse_hidden
from evolution import EvolutionConfig, diversity, evolve, evolve_islands, fitness_stats
from fitness import FitnessConfig

HERE = Path(__file__).resolve().parent
console = Console()

MLP_CONTROLLERS = ("mlp", "mlp_clock")
GA_PARAMS = (
    "mutation_rate",
    "mutation_std",
    "crossover_rate",
    "crossover_kind",
    "parent_tournament_size",
    "survivor_tournament_size",
)
REVDE_PARAMS = ("revde_scaling_factor",)
ISLAND_PARAMS = ("migration_interval", "migration_size")

SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_argument_group("experiment")
    g.add_argument("--bodies", nargs="+", default=["gecko"],
                   help="ARIEL prebuilt body names (see fitness.PREBUILT_BODIES) or .py / .json body files")
    g.add_argument("--environments", type=int, nargs="+", choices=[0, 1, 2, 3], default=[0])
    g.add_argument("--weights", type=float, nargs="+", default=None)
    g.add_argument("--aggregate", choices=["mean", "min"], default="mean",
                   help="combine per-environment speeds: mean (weighted average) or min (slowest environment)")
    g.add_argument("--islands", action="store_true")
    g.add_argument("--generations", type=int, default=50)
    g.add_argument("--evaluations", type=int, default=None,
                   help="evaluation budget per run, initial population included; overrides --generations")
    g.add_argument("--workers", type=int, default=os.cpu_count())
    g.add_argument("--out-dir", type=Path, default=HERE / "runs")

    g = p.add_argument_group("seeds and tuning")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-seeds", type=int, default=3)
    g.add_argument("--n-trials", type=int, default=1)
    g.add_argument("--tuning", choices=["fast", "robust"], default="fast",
                   help="fast: 1 seed per trial, then best params on all seeds; robust: all seeds per trial")

    g = p.add_argument_group("search space")
    g.add_argument("--controllers", nargs="+", default=["na_cpg"], choices=["na_cpg", "simple_cpg", "mlp", "mlp_clock"])
    g.add_argument("--mlp-hidden", nargs="+", default=["16"],
                   help="MLP hidden layer sizes, comma-separated per architecture, e.g. 0 16 32,32 (0 = linear)")
    g.add_argument("--mlp-activation", nargs="+", default=["tanh"], choices=["tanh", "elu", "relu"])
    g.add_argument("--algorithms", nargs="+", default=["ga"], choices=["ga", "revde"])
    g.add_argument("--pop-size", type=int, nargs="+", default=[30])
    g.add_argument("--mutation-rate", type=float, nargs="+", default=[0.1])
    g.add_argument("--mutation-std", type=float, nargs="+", default=[0.1])
    g.add_argument("--crossover-rate", type=float, nargs="+", default=[0.5])
    g.add_argument("--crossover-kind", nargs="+", default=["uniform"], choices=["one_point", "uniform"])
    g.add_argument("--parent-tournament-size", type=int, nargs="+", default=[3])
    g.add_argument("--survivor-tournament-size", type=int, nargs="+", default=[3])
    g.add_argument("--revde-scaling-factor", type=float, nargs="+", default=[0.5])
    g.add_argument("--migration-interval", type=int, nargs="+", default=[10])
    g.add_argument("--migration-size", type=int, nargs="+", default=[2])
    return p.parse_args()


def body_folder(args: argparse.Namespace, body: str) -> Path:
    return args.out_dir / Path(body).stem


def seeds(args: argparse.Namespace) -> range:
    return range(args.seed, args.seed + args.n_seeds)


def fitness_config(args: argparse.Namespace) -> FitnessConfig:
    return FitnessConfig(tuple(args.environments), tuple(args.weights) if args.weights else None, args.aggregate)


def suggest_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    params = {}

    def pick(*names: str) -> None:
        for name in names:
            params[name] = trial.suggest_categorical(name, getattr(args, name))

    params["controller"] = trial.suggest_categorical("controller", args.controllers)
    if params["controller"] in MLP_CONTROLLERS:
        pick("mlp_hidden", "mlp_activation")
    params["algorithm"] = trial.suggest_categorical("algorithm", args.algorithms)
    pick("pop_size")
    pick(*(GA_PARAMS if params["algorithm"] == "ga" else REVDE_PARAMS))
    if args.islands:
        pick(*ISLAND_PARAMS)
    return params


def generations(args: argparse.Namespace, params: dict) -> int:
    if args.evaluations is None:
        return args.generations
    pop = params["pop_size"]
    offspring = pop // 3 * 3 if params["algorithm"] == "revde" else pop // 2 * 2
    return max(1, (args.evaluations - pop) // offspring)


def make_config(args: argparse.Namespace, body: str, params: dict, seed: int, folder: Path) -> EvolutionConfig:
    mlp = {}
    if params["controller"] in MLP_CONTROLLERS:
        mlp = {"hidden": parse_hidden(params["mlp_hidden"]), "activation": params["mlp_activation"]}
    ea_params = {name: params[name] for name in GA_PARAMS + REVDE_PARAMS + ISLAND_PARAMS if name in params}
    return EvolutionConfig(
        body=body,
        controller=BrainConfig(name=params["controller"], **mlp),
        algorithm=params["algorithm"],
        num_steps=generations(args, params),
        target_population_size=params["pop_size"],
        islands=args.islands,
        seed=seed,
        workers=args.workers,
        fitness=fitness_config(args),
        output_folder=folder,
        **ea_params,
    )


def run_one(cfg: EvolutionConfig) -> float:
    start = time.perf_counter()
    ea = evolve_islands(cfg) if cfg.islands else evolve(cfg)
    best = ea.get_solution("best", only_alive=False).fitness
    console.log(f"seed {cfg.seed}: best fitness {best:.4f} in {time.perf_counter() - start:.0f} s, saved to {cfg.output_folder}")
    return best


def objective(trial: optuna.Trial, args: argparse.Namespace, body: str) -> float:
    params = suggest_params(trial, args)
    console.rule(f"{body}: trial {trial.number}")
    console.log(params)
    folder = body_folder(args, body) / "trials" / f"trial_{trial.number:03d}"
    trial_seeds = seeds(args) if args.tuning == "robust" else [args.seed]
    return float(np.mean([run_one(make_config(args, body, params, seed, folder / f"seed_{seed}")) for seed in trial_seeds]))


def run_seeds(args: argparse.Namespace, body: str, params: dict) -> None:
    console.rule(f"{body}: best params on {args.n_seeds} seeds")
    console.log(params)
    for seed in seeds(args):
        run_one(make_config(args, body, params, seed, body_folder(args, body) / "best" / f"seed_{seed}"))


def survivors(archive: Archive, gen: int) -> Population:
    return archive.by_generation(gen).where(lambda ind: ind.alive or ind.time_of_death > gen)


def generation_stats(db: Path) -> dict[str, list[float]]:
    archive = Archive(db)
    first, last = archive.generation_range
    populations = [survivors(archive, gen) for gen in range(first, last + 1)]
    fitness = [fitness_stats(pop) for pop in populations]
    return {key: [f[key] for f in fitness] for key in fitness[0]} | {"diversity": [diversity(pop) for pop in populations]}


def plot_generations(series: dict[str, np.ndarray], ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, values), color in zip(series.items(), (BLUE, ORANGE), strict=False):
        gens = np.arange(values.shape[1])
        ax.fill_between(gens, values.min(axis=0), values.max(axis=0), color=color, alpha=0.15, linewidth=0)
        ax.plot(gens, values.mean(axis=0), color=color, linewidth=2, label=label)
    ax.set(xlabel="generation", ylabel=ylabel)
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if len(series) > 1:
        ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_study(study: optuna.Study, path: Path) -> None:
    trials = [t for t in study.trials if t.value is not None]
    numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(numbers, values, s=64, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3, label="trial")
    ax.step(numbers, np.maximum.accumulate(values), where="post", color=ORANGE, linewidth=2, label="best so far")
    ax.set(xlabel="trial", ylabel="objective (best fitness)")
    ax.set_title(f"Optuna tuning: {study.study_name}", loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_plots(out_dir: Path) -> None:
    best = json.loads((out_dir / "best_params.json").read_text())
    runs = [generation_stats(db) for db in sorted((out_dir / best["runs"]).glob("seed_*/database.db"))]
    n_gens = min(len(run["best"]) for run in runs)
    series = {key: np.array([run[key][:n_gens] for run in runs]) for key in runs[0]}
    note = f"mean over {len(runs)} seeds, band = min to max"
    plot_generations({key: series[key] for key in ("best", "mean")}, "fitness",
                     f"Fitness per generation: {out_dir.name}\n{note}", out_dir / "fitness.png")
    plot_generations({"diversity": series["diversity"]}, "diversity",
                     f"Diversity per generation: {out_dir.name}\n{note}", out_dir / "diversity.png")
    plot_study(optuna.load_study(study_name=out_dir.name, storage=f"sqlite:///{out_dir / 'optuna.db'}"),
               out_dir / "optuna.png")
    console.log(f"plots saved to {out_dir}")


def tune(args: argparse.Namespace, body: str) -> None:
    folder = body_folder(args, body)
    folder.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=folder.name,
        storage=f"sqlite:///{folder / 'optuna.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(lambda trial: objective(trial, args, body), n_trials=args.n_trials)
    best = study.best_trial
    console.rule(f"{body}: best trial {best.number}, value {best.value:.4f}")
    if args.tuning == "fast":
        run_seeds(args, body, best.params)
        runs = "best"
    else:
        runs = f"trials/trial_{best.number:03d}"
    summary = {"trial": best.number, "value": best.value, "params": best.params, "runs": runs}
    (folder / "best_params.json").write_text(json.dumps(summary, indent=2))
    make_plots(folder)


def main() -> None:
    args = parse_args()
    fitness_config(args)
    for body in args.bodies:
        tune(args, body)


if __name__ == "__main__":
    main()
