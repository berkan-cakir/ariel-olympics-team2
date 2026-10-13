"""
Run the whole design-priors experiment: every condition x every seed.

Example
-------
    caffeinate -dims uv run python examples/d_design_priors/run_batch.py \
        --seeds 10 --jobs 7

Runs start seed by seed (all conditions of seed 1, then seed 2, ...), so the
conditions stay balanced if the batch is stopped early. Each run is a
separate single-threaded process with its own log in ``<out>/logs``.

The batch can be stopped and restarted at any time: completed runs are
skipped, and the files of a run that did not finish are renamed to
``*.partial-<time>`` and the run starts again.

Run budget: if ``pilot.py`` wrote ``<out>/analysis_config.json``, its
recommended ``--gens`` and the budget it measured with are used, so the
batch matches the pilot. Arguments after ``--`` are passed to every
``run_design_priors.py`` call and override those.
"""

# Standard library
import argparse
import json
import os
import platform
import signal
import subprocess  # noqa: S404 -- launches our own run script
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Local libraries
from dp_core import CONDITIONS

DEFAULT_OUT = Path.home() / "ariel_runs" / "design_priors"
RUN_SCRIPT = Path(__file__).with_name("run_design_priors.py")
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
POLL_SECONDS = 2.0
PILOT_BUDGET_FLAGS = frozenset({
    "pop",
    "learn_budget",
    "learn_pop",
    "duration",
    "eval_delay",
    "reeval_top",
    "reeval_seeds",
})


@dataclass(frozen=True)
class Job:
    """
    One run of the batch.

    Parameters
    ----------
    condition
        Condition name.
    seed
        Run seed.
    out
        Output root.
    """

    condition: str
    seed: int
    out: Path

    @property
    def stem(self) -> Path:
        """Path prefix shared by this run's files."""
        return self.out / self.condition / f"seed_{self.seed:02d}"

    @property
    def log(self) -> Path:
        """Log file of this run."""
        return self.out / "logs" / f"{self.condition}_seed_{self.seed:02d}.log"

    def is_complete(self) -> bool:
        """
        Tell whether this run already finished.

        Returns
        -------
        bool
            True if its metadata says ``completed``.
        """
        meta = Path(f"{self.stem}_meta.json")
        if not meta.exists():
            return False
        return json.loads(meta.read_text("utf-8")).get("status") == "completed"

    def live_pid(self) -> int | None:
        """
        Return the PID of a still-running attempt of this run, if any.

        A batch that died can leave its runs going; restarting them would
        write two runs into the same files.

        Returns
        -------
        int | None
            The PID if this run's metadata says ``running`` on this host and
            that process is still a design-priors run, else ``None``.
        """
        meta_path = Path(f"{self.stem}_meta.json")
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text("utf-8"))
        pid = meta.get("pid")
        if (
            meta.get("status") != "running"
            or meta.get("host") != platform.node()
        ):
            return None
        return pid if isinstance(pid, int) and _is_run_process(pid) else None

    def set_aside_partial_files(self) -> None:
        """Rename files left by an unfinished attempt of this run."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in (".db", "_meta.json", "_reeval.json"):
            path = Path(f"{self.stem}{suffix}")
            if path.exists():
                path.rename(path.with_name(f"{path.name}.partial-{stamp}"))


def _is_run_process(pid: int) -> bool:
    try:
        command = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            ["ps", "-p", str(pid), "-o", "command="],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return False
    return RUN_SCRIPT.name in command


def _raise_interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """
    Parse batch options; everything after ``--`` goes to each run.

    Returns
    -------
    tuple[argparse.Namespace, list[str]]
        Batch options and the arguments forwarded to every run.
    """
    parser = argparse.ArgumentParser(
        description="Run every condition x seed of the design-priors study.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, default=10, help="seeds 1..N")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=7)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    return parser.parse_args(argv[:split]), argv[split + 1 :]


def pilot_run_args(out: Path) -> list[str]:
    """
    Return the budget flags fixed by ``pilot.py``, if it has run.

    Parameters
    ----------
    out
        The experiment's output root, holding ``analysis_config.json``.

    Returns
    -------
    list[str]
        ``--gens`` and the budget flags the pilot measured with, or an empty
        list if there is no pilot result.
    """
    config_path = out / "analysis_config.json"
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text("utf-8"))
    flags = ["--gens", str(config["recommended_gens"])]
    for name, value in config["budget"].items():
        if name in PILOT_BUDGET_FLAGS:
            flags += [f"--{name.replace('_', '-')}", str(value)]
    return flags


def launch(job: Job, run_args: list[str]) -> subprocess.Popen[bytes]:
    """
    Start one run in the background, logging to ``job.log``.

    Parameters
    ----------
    job
        The run to start.
    run_args
        Extra arguments for ``run_design_priors.py``.

    Returns
    -------
    subprocess.Popen[bytes]
        The running process.
    """
    job.log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--condition",
        job.condition,
        "--seed",
        str(job.seed),
        "--out",
        str(job.out),
        "--eval-workers",
        "1",
        *run_args,
    ]
    with job.log.open("ab") as log:
        return subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **SINGLE_THREAD_ENV},
        )


def main() -> None:
    """
    Run all pending jobs, at most ``--jobs`` at a time.

    Raises
    ------
    KeyboardInterrupt
        Re-raised after stopping the running jobs, so the batch exits.
    """
    args, user_args = parse_args()
    # Explicit arguments come last, so they override the pilot's budget.
    run_args = pilot_run_args(args.out) + user_args
    print(f"Run arguments: {' '.join(run_args) or '(defaults)'}", flush=True)  # noqa: T201
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    jobs = [
        Job(condition, seed, args.out)
        for seed in seeds
        for condition in args.conditions
    ]
    pending = [job for job in jobs if not job.is_complete()]
    print(  # noqa: T201
        f"{len(jobs) - len(pending)} of {len(jobs)} runs already complete; "
        f"running {len(pending)} with {args.jobs} in parallel.",
        flush=True,
    )

    signal.signal(signal.SIGTERM, _raise_interrupt)
    started = time.perf_counter()
    running: dict[Job, subprocess.Popen[bytes]] = {}
    failed: list[Job] = []
    done = 0
    try:
        while pending or running:
            while pending and len(running) < args.jobs:
                job = pending.pop(0)
                pid = job.live_pid()
                if pid is not None:
                    print(  # noqa: T201
                        f"skipped: {job.condition} seed {job.seed} is still "
                        f"running as process {pid}",
                        flush=True,
                    )
                    continue
                job.set_aside_partial_files()
                running[job] = launch(job, run_args)
            time.sleep(POLL_SECONDS)
            for job, process in list(running.items()):
                if process.poll() is None:
                    continue
                del running[job]
                done += 1
                ok = process.returncode == 0 and job.is_complete()
                if not ok:
                    failed.append(job)
                elapsed = time.perf_counter() - started
                print(  # noqa: T201
                    f"{'done' if ok else 'FAILED'}: {job.condition} seed "
                    f"{job.seed} ({done} finished, {len(running)} running, "
                    f"{len(pending)} queued, {elapsed / 60:.1f} min)",
                    flush=True,
                )
    except KeyboardInterrupt:
        for process in running.values():
            process.terminate()
        print("Interrupted; rerun to resume.", flush=True)  # noqa: T201
        raise

    for job in failed:
        print(f"Failed run, see {job.log}", flush=True)  # noqa: T201
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
