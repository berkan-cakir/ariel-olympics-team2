#!/bin/bash
# One line per run: phase, finished trials / runs / seeds, progress of the current EA run, time left in the phase.
#   bash progress.sh                        (both the non-island and the island runs)
#   bash progress.sh <runs folder> ...      (other folders)
PY=$HOME/robot-olympics/ariel-olympics-team2/.venv/bin/python
ROOTS=("$@")
(( ${#ROOTS[@]} )) || ROOTS=(
  "$HOME/robot-olympics/ariel-olympics-team2/research/runs/snellius"
  "$HOME/robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/islands"
  "$HOME/robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/features"
  "$HOME/robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/gecko_lab"
)
"$PY" - "${ROOTS[@]}" <<'EOF'
import json, re, sqlite3, sys, time
from datetime import datetime, timedelta
from pathlib import Path

TRIALS = {"flat": 5, "sizes": 4}


def current_run(phase: Path, logs: list[Path]) -> tuple[int, int, float]:
    dbs = sorted(phase.glob("**/trials/trial_*/seed_*/database.db"), key=lambda p: p.stat().st_mtime)
    if not dbs or not logs:
        return 0, 1, 0.0
    log = max(logs, key=lambda p: p.stat().st_mtime).read_text()
    log = log[log.rfind("=== "):]
    config = dbs[-1].parent / "config.json"
    printed = re.findall(r"num_steps = (\d+)", log)
    steps = json.loads(config.read_text())["num_steps"] if config.exists() else int(printed[-1]) if printed else 1
    db = sqlite3.connect(f"file:{dbs[-1]}?mode=ro", uri=True)
    gen = db.execute("select max(time_of_death) from individual").fetchone()[0] or 0
    if not log.startswith("=== "):
        return gen, steps, time.time() - config.stat().st_mtime if config.exists() else 0.0
    begun = datetime.strptime(log[4:23], "%Y-%m-%d %H:%M:%S").timestamp()
    finished = sum(int(t) for t in re.findall(r"best fitness [0-9.]+ in (\d+) s", log))
    return gen, steps, time.time() - begun - finished


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds == seconds else "?"


for root in map(Path, sys.argv[1:]):
    if not root.is_dir():
        continue
    print(f"== {root}")
    seeds = 1 if root.name == "gecko_lab" else 10
    for run in sorted(p for p in root.iterdir() if (p / "logs").is_dir()):
        phase = "final" if (run / "logs/final.log").exists() else "tune"
        if list((run / "final").glob("*/best_params.json")):
            print(f"  {run.name:28} done")
            continue
        n = TRIALS.get(run.name.rsplit("_", 1)[-1], 10)
        logs = sorted((run / "logs").glob(f"{phase}*.log"))
        log = "".join(p.read_text() for p in logs)
        times = [int(t) for t in re.findall(r"best fitness [0-9.]+ in (\d+) s", log)]
        trials = len(re.findall(r"finished with value", log))
        gen, steps, elapsed = current_run(run / phase, logs)
        fraction = min(gen / steps, 0.99)
        per_run = sum(times) / len(times) if times else (elapsed / fraction if fraction else float("nan"))
        left = ((10 * n if phase == "tune" else seeds) - len(times) - fraction) * per_run
        eta = f", ETA {datetime.now() + timedelta(seconds=left):%a %H:%M}" if left == left else ""
        done = f"trials {trials}/{n}, runs {len(times)}/{10 * n}" if phase == "tune" else f"seeds {len(times)}/{seeds}"
        pace = f", {elapsed / gen:.1f} s per generation" if gen else ", first generation still running"
        print(f"  {run.name:28} {phase:5} {done}, current run at generation {gen}/{steps} ({fraction:.1%}){pace}, ~{per_run / 60:.0f} min per run, ~{hours(left)} left in {phase}{eta}")
EOF
