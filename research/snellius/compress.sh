#!/bin/bash
# Compresses the run databases: database.db -> database.db.xz.
# Finished phases completely; running final phases only their finished seeds (database untouched for an hour), so run.py's
# fitness.png / diversity.png and the sprinter summary at the end show only the last seed (compare.py's progress.csv reads all seeds).
# Running tune phases are skipped (the sprinter jobs read every tuning database afterwards). Safe to rerun at any time:
#   sbatch -p genoa -c 24 -t 2:00:00 -J compress -o compress_%j.out --wrap "bash $PWD/compress.sh"


set -u
THREADS=${SLURM_CPUS_PER_TASK:-4}
ROOTS=("$@")
(( ${#ROOTS[@]} )) || ROOTS=(
  "$HOME/robot-olympics/ariel-olympics-team2/research/runs/snellius"
  "$HOME/robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/islands"
  "$HOME/robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/features"
  "$HOME/robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/gecko_lab"
)

finished() {  # finished <config folder> <phase folder>
  case $2 in
    */tune) [[ -f $1/best/final_args.txt ]] ;;
    */final) [[ -n $(find "$2" -maxdepth 2 -name optuna.png -mmin +10) ]] ;;
    *_partial_*) true ;;
  esac
}

for root in "${ROOTS[@]}"; do
  for phase in "$root"/*/tune "$root"/*/final "$root"/*/tune_partial_* "$root"/*/final_partial_*; do
    [[ -d $phase ]] || continue
    idle=0
    if ! finished "$(dirname "$phase")" "$phase"; then
      [[ $phase == */final ]] || { echo "skip (still running): $phase"; continue; }
      idle=60
    fi
    todo=$(find "$phase" -name database.db -mmin +$idle | wc -l)
    (( todo )) || continue
    before=$(du -sm "$phase" | cut -f1)
    find "$phase" -name database.db -mmin +$idle -print0 | xargs -0 -n 1 -P "$THREADS" nice xz -f -T1 -6
    echo "compressed $todo databases in $phase: $before MB -> $(du -sm "$phase" | cut -f1) MB"
  done
done
command -v myquota >/dev/null && myquota | head -4
