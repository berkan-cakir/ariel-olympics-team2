#!/bin/bash
# Island version of pipeline.sbatch: tuning + final run for one body x algorithm x controller, comparable with the
# runs without islands: same terrains (one island per terrain: 4 linoleum, 5 carpet, 6 carpet slope), same search
# space plus migration settings, same seeds, and the same number of terrain simulations (an island evaluation is one
# terrain, so 1000 / 20000 evaluations here buy 3000 / 60000 island evaluations, see run.py generations()).
# The result of a run is its best score on all three terrains (speed on the slowest), like a run without islands.
#   sbatch --job-name=isl_spider_ga_mlp islands.sbatch spider ga mlp
#   sbatch --job-name=isl_snake_revde_simple_cpg --time=4-00:00:00 islands.sbatch snake revde simple_cpg
# 1. tune:  10 Optuna trials, each = one parameter set on seeds 0-9
# 2. best:  tuning winner saved to <run>/best/ (best_params.json, final_args.txt, best tuning robot as checkpoint/)
# 3. final: the winner's parameters on fresh seeds 10-19
# Results go to research/runs/snellius/islands/<body>_<algorithm>_<controller>/, run.py output to its logs/ folder.
# Resubmitting the same command skips finished phases; a phase that was cut off is moved aside to
# <phase>_partial_<date> and restarted, because run.py cannot continue a half-finished study or run.
#SBATCH --job-name=islands
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out

set -u
BODY=${1:?usage: sbatch islands.sbatch <body> [ga|revde] [mlp|simple_cpg]} ALGORITHM=${2:-ga} CONTROLLER=${3:-mlp}
REPO=$HOME/robot-olympics/islands/ariel-olympics-team2
PY=$REPO/.venv/bin/python
O=$REPO/research/runs/snellius/islands/${BODY}_${ALGORITHM}_${CONTROLLER}
WORKERS=${SLURM_CPUS_PER_TASK:-24}
export OMP_NUM_THREADS=1

FITNESS=(--bodies "$BODY" --environments 4 5 6 --aggregate min --lateral-penalty 0 --islands)
TUNE_EVALUATIONS=1000 FINAL_EVALUATIONS=20000 TRIALS=10 SEEDS=10 FINAL_SEED=10
SEARCH_SPACE=(
  --pop-size 24 48 72
  --mutation-std 0.1 0.2 0.3 0.5
  --crossover-rate 0.3 0.6 0.9
  --crossover-kind one_point uniform
  --parent-tournament-size 2 3
  --survivor-tournament-size 2 3
  --revde-scaling-factor 0.3 0.5 0.7 0.9
  --migration-interval 5 10 20
  --migration-size 1 2 4
)
CPG_SPACE=(--mutation-rate 0.1 0.2 0.4 0.6)
MLP_SPACE=(--mutation-rate 0.02 0.05 0.1 0.2 --mlp-hidden 0 16 32 16,16 32,32 --mlp-activation tanh relu elu)

if [[ ! $ALGORITHM =~ ^(ga|revde)$ || ! $CONTROLLER =~ ^(mlp|simple_cpg)$ ]]; then
  echo "usage: sbatch islands.sbatch <body> [ga|revde] [mlp|simple_cpg]"; exit 1
fi
mkdir -p "$O/logs"
cd "$REPO/research" || exit 1

say() { echo "$(date '+%F %T') $*"; }

run() {  # run <phase> <run.py args...>
  local phase=$1; shift
  say "$phase started"
  local t0=$SECONDS
  { echo "=== $(date '+%F %T') run.py $*"; "$PY" run.py "$@" --workers "$WORKERS"; } >> "$O/logs/$phase.log" 2>&1
  local code=$?
  say "$phase finished with exit code $code after $(( (SECONDS - t0) / 60 )) min"
  return "$code"
}

prepare() {  # prepare <phase>: 0 if it still has to run, 1 if it is already done
  if [[ -f $O/$1/$BODY/best_params.json ]]; then say "$1 already done, skipped"; return 1; fi
  if [[ -e $O/$1 ]]; then
    local aside; aside=$O/$1_partial_$(date +%Y%m%d_%H%M%S)
    mv "$O/$1" "$aside"
    say "$1 was cut off earlier, its results moved to $aside; restarting it"
  fi
  return 0
}

tune() {
  prepare tune || return 0
  local space; if [[ $CONTROLLER == mlp ]]; then space=("${MLP_SPACE[@]}"); else space=("${CPG_SPACE[@]}"); fi
  run tune "${FITNESS[@]}" --algorithms "$ALGORITHM" --controllers "$CONTROLLER" \
    --evaluations "$TUNE_EVALUATIONS" --n-trials "$TRIALS" --n-seeds "$SEEDS" --seed 0 --tuning robust \
    "${SEARCH_SPACE[@]}" "${space[@]}" --out-dir "$O/tune"
}

save_best() {  # tuning winner -> best/: its parameters, the final run's arguments and the best tuning robot on all terrains
  "$PY" - "$O" "$BODY" <<'EOF'
import json, shutil, sqlite3, sys
from pathlib import Path
out, body = Path(sys.argv[1]), sys.argv[2]
best = json.loads((out / f"tune/{body}/best_params.json").read_text())
flags = {"controller": "--controllers", "algorithm": "--algorithms"}
args = [f"{flags.get(k, '--' + k.replace('_', '-'))} {v}" for k, v in best["params"].items()]
dest = out / "best"
(dest / "checkpoint").mkdir(parents=True, exist_ok=True)
shutil.copy(out / f"tune/{body}/best_params.json", dest / "best_params.json")
(dest / "final_args.txt").write_text(" ".join(args) + "\n")
combined = "select max(json_extract(tags_, '$.combined')) from individual"
score = lambda db: sqlite3.connect(db).execute(combined).fetchone()[0] or 0
winner = max((out / f"tune/{body}/trials").glob("trial_*/seed_*/database.db"), key=score)
for name in ("database.db", "config.json"):
    shutil.copy(winner.parent / name, dest / "checkpoint" / name)
(dest / "checkpoint" / "source.txt").write_text(f"{winner.parent}\nbest score on all terrains {score(winner):.4f}\n")
print(f"tuning winner: trial {best['trial']}, mean best score on all terrains {best['value']:.4f} over its seeds")
print(f"final run arguments: {' '.join(args)}")
print(f"best tuning robot ({score(winner):.4f} on all terrains) copied from {winner.parent.relative_to(out)} to best/checkpoint")
EOF
}

final() {
  prepare final || return 0
  local best; read -r -a best < "$O/best/final_args.txt"
  run final "${FITNESS[@]}" "${best[@]}" --evaluations "$FINAL_EVALUATIONS" \
    --n-trials 1 --n-seeds "$SEEDS" --seed "$FINAL_SEED" --tuning robust --out-dir "$O/final"
}

summary() {
  "$PY" - "$O" "$BODY" <<'EOF'
import sqlite3, sys
from pathlib import Path
out, body = Path(sys.argv[1]), sys.argv[2]
runs = sorted((out / f"final/{body}/trials").glob("trial_*/seed_*/database.db"), key=lambda p: int(p.parent.name[5:]))
combined = "select max(json_extract(tags_, '$.combined')) from individual"
best = [sqlite3.connect(db).execute(combined).fetchone()[0] or 0 for db in runs]
for db, value in zip(runs, best):
    print(f"  {db.parent.name:8} best score on all terrains {value:.4f}")
if best:
    top = runs[max(range(len(best)), key=best.__getitem__)].parent
    print(f"  mean {sum(best) / len(best):.4f} over {len(best)} seeds, best robot in {top.relative_to(out.parent.parent.parent.parent)}")
EOF
}

say "islands pipeline $BODY + $ALGORITHM + $CONTROLLER on ${SLURM_JOB_PARTITION:-local}, $WORKERS workers, results in $O"
tune || { say "tuning failed, see $O/logs/tune.log"; exit 1; }
[[ -f $O/best/final_args.txt ]] || save_best || { say "saving the tuning winner failed"; exit 1; }
final || { say "final run failed, see $O/logs/final.log"; exit 1; }
say "done"
summary
