#!/bin/bash
# Island features against the plain island runs in runs/snellius/islands (the benchmark), one body x controller x feature per job:
#   flat:  the first part of the run on the flat world, then the islands; fraction of the run tuned over 0.1 0.2 0.3 0.4 0.5 (islands 1:1:1)
#   sizes: unequal islands, linoleum < carpet < carpet slope, tuned over 7:8:9 6:8:10 5:8:11 4:8:12 (no flat phase)
# Everything else is the benchmark's: GA with its tuned parameters (best/final_args.txt), terrains, simulation budgets and seeds.
#   sbatch --job-name=flat_spider_mlp features.sbatch spider mlp flat
#   for b in spider gecko snake; do for c in mlp simple_cpg; do for f in flat sizes; do sbatch --job-name=${f}_${b}_${c} features.sbatch $b $c $f; done; done; done
# 1. tune:  every value once on seeds 0-9 with 1000 evaluations, like one benchmark tuning trial (Optuna's sampler repeats values on a list this short)
# 2. best:  the value with the highest mean, saved to <run>/best/
# 3. final: that value on seeds 10-19 with 20000 evaluations, then compared with the benchmark's final seeds (Mann-Whitney U)
# Results go to research/runs/snellius/features/<body>_ga_<controller>_<feature>/, run.py output to its logs/ folder.
# Resubmitting the same command skips finished phases; a phase that was cut off is moved aside to <phase>_partial_<date> and restarted.
#SBATCH --job-name=features
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-12:00:00
#SBATCH --output=%x_%j.out

set -u
BODY=${1:?usage: sbatch features.sbatch <body> [mlp|simple_cpg] [flat|sizes]} CONTROLLER=${2:-mlp} FEATURE=${3:-flat}
REPO=$HOME/robot-olympics/islands/ariel-olympics-team2
PY=$REPO/.venv/bin/python
BENCH=$REPO/research/runs/snellius/islands/${BODY}_ga_${CONTROLLER}
O=$REPO/research/runs/snellius/features/${BODY}_ga_${CONTROLLER}_${FEATURE}
WORKERS=${SLURM_CPUS_PER_TASK:-24}
export OMP_NUM_THREADS=1

FITNESS=(--bodies "$BODY" --environments 4 5 6 --aggregate min --lateral-penalty 0 --islands)
TUNE_EVALUATIONS=1000 FINAL_EVALUATIONS=20000 SEEDS=10 FINAL_SEED=10
case $FEATURE in
  flat) FLAG=--flat-pretraining; VALUES=(0.1 0.2 0.3 0.4 0.5) ;;
  sizes) FLAG=--island-sizes; VALUES=(7,8,9 6,8,10 5,8,11 4,8,12) ;;
  *) echo "usage: sbatch features.sbatch <body> [mlp|simple_cpg] [flat|sizes]"; exit 1 ;;
esac
if [[ ! $CONTROLLER =~ ^(mlp|simple_cpg)$ ]]; then
  echo "usage: sbatch features.sbatch <body> [mlp|simple_cpg] [flat|sizes]"; exit 1
fi
if [[ ! -f $BENCH/best/final_args.txt ]]; then
  echo "benchmark parameters not found: $BENCH/best/final_args.txt"; exit 1
fi
read -r -a BASE < "$BENCH/best/final_args.txt"
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

prepare() {  # prepare <phase folder>: 0 if it still has to run, 1 if it is already done
  if [[ -f $O/$1/$BODY/best_params.json ]]; then say "$1 already done, skipped"; return 1; fi
  if [[ -e $O/$1 ]]; then
    local aside; aside=$O/$1_partial_$(date +%Y%m%d_%H%M%S)
    mv "$O/$1" "$aside"
    say "$1 was cut off earlier, its results moved to $aside; restarting it"
  fi
  return 0
}

tune() {
  local value
  for value in "${VALUES[@]}"; do
    prepare "tune/$value" || continue
    run "tune_$value" "${FITNESS[@]}" "${BASE[@]}" "$FLAG" "$value" --evaluations "$TUNE_EVALUATIONS" --n-trials 1 --n-seeds "$SEEDS" --seed 0 --tuning robust --out-dir "$O/tune/$value" || return 1
  done
}

save_best() {  # best tuning value -> best/: the final run's arguments and every value's tuning score
  "$PY" - "$O" "$BODY" "$BENCH" "$FLAG" "${VALUES[@]}" <<'EOF'
import json, sys
from pathlib import Path
out, body, bench, flag, values = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5:]
scores = {v: json.loads((out / f"tune/{v}/{body}/best_params.json").read_text())["value"] for v in values}
best = max(values, key=scores.__getitem__)
benchmark = json.loads((bench / "best/best_params.json").read_text())["value"]
(out / "best").mkdir(exist_ok=True)
(out / "best/final_args.txt").write_text(" ".join([*(bench / "best/final_args.txt").read_text().split(), flag, best]) + "\n")
(out / "best/tuning.json").write_text(json.dumps({"best": best, "scores": scores, "benchmark": benchmark}, indent=2))
for v in values:
    print(f"  {flag} {v}: mean best score on all terrains {scores[v]:.4f} over seeds 0-9{'  <- best' if v == best else ''}")
print(f"  benchmark tuning winner (best of its 10 trials, same seeds and budget): {benchmark:.4f}")
print(f"final run arguments: {(out / 'best/final_args.txt').read_text().strip()}")
EOF
}

final() {
  prepare final || return 0
  local best; read -r -a best < "$O/best/final_args.txt"
  run final "${FITNESS[@]}" "${best[@]}" --evaluations "$FINAL_EVALUATIONS" --n-trials 1 --n-seeds "$SEEDS" --seed "$FINAL_SEED" --tuning robust --out-dir "$O/final"
}

summary() {  # final seeds against the benchmark's final seeds, as in the islands vs no islands comparison
  "$PY" - "$O" "$BENCH" <<'EOF'
import re, sys
from pathlib import Path
from statistics import mean
from scipy.stats import mannwhitneyu
out, bench = Path(sys.argv[1]), Path(sys.argv[2])
scores = lambda run: {int(s): float(v) for s, v in re.findall(r"seed (\d+): best fitness ([0-9.]+)", (run / "logs/final.log").read_text())}
new, old = scores(out), scores(bench)
for seed in sorted(new):
    print(f"  seed {seed}: best score on all terrains {new[seed]:.4f}   benchmark {old.get(seed, float('nan')):.4f}")
a, b = list(new.values()), list(old.values())
print(f"  mean {mean(a):.4f} vs benchmark {mean(b):.4f} ({mean(a) / mean(b) - 1:+.0%}), reached 3 m in 30 s: {sum(x >= 0.1 for x in a)}/{len(a)} vs {sum(x >= 0.1 for x in b)}/{len(b)}")
print(f"  Mann-Whitney U two-sided p = {mannwhitneyu(a, b, alternative='two-sided').pvalue:.4f} (significant over all 12 jobs, Bonferroni: p < 0.0042)")
EOF
}

say "features: $FEATURE for $BODY + ga + $CONTROLLER on ${SLURM_JOB_PARTITION:-local}, $WORKERS workers, results in $O"
say "benchmark parameters from $BENCH/best/final_args.txt: ${BASE[*]}"
tune || { say "tuning failed, see $O/logs"; exit 1; }
[[ -f $O/best/final_args.txt ]] || save_best || { say "choosing the best value failed"; exit 1; }
final || { say "final run failed, see $O/logs/final.log"; exit 1; }
say "done"
summary
