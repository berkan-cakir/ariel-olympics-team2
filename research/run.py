import argparse
import os
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_argument_group("experiment")
    g.add_argument("--bodies", nargs="+", default=["gecko"], help="ARIEL body names or paths to body files")
    g.add_argument("--environments", type=int, nargs="+", choices=[0, 1, 2, 3], default=[0])
    g.add_argument("--weights", type=float, nargs="+", default=None)
    g.add_argument("--islands", action="store_true")
    g.add_argument("--generations", type=int, default=50)
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
    g.add_argument("--crossover-rate", type=float, nargs="+", default=[0.5])
    g.add_argument("--revde-scaling-factor", type=float, nargs="+", default=[0.5])
    g.add_argument("--migration-interval", type=int, nargs="+", default=[10])
    g.add_argument("--migration-size", type=int, nargs="+", default=[2])
    return p.parse_args()


def objective(trial: optuna.Trial, args: argparse.Namespace, body: str) -> float:
    raise NotImplementedError


def run_seeds(args: argparse.Namespace, body: str, params: dict) -> None:
    raise NotImplementedError


def make_plots(out_dir: Path) -> None:
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
