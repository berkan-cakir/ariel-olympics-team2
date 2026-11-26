#!/bin/bash
#  Downloads what the analysis needs, without the tuning runs' databases:
# logs, best parameters, Optuna studies and plots of every run, plus the best final seed of each run (videos, speeds).
#   bash download.sh                      (all runs: without islands, islands, features, gecko_lab)
#   bash download.sh runs '*_mlp'         (runs without islands whose name ends in _mlp)
#   bash download.sh islands              (island runs only)
#   bash download.sh features             (island features: flat pretraining, island sizes)
#   bash download.sh gecko_lab            (the long gecko run on the lab terrains)
# Rerun it any time: rsync only fetches what is new.
set -u
WHICH=${1:-all} PATTERN=${2:-*}
HOST=igoloborodko@snellius.surf.nl
LOCAL=$HOME/Desktop/ariel-olympics-team2/research/runs/snellius
REMOTE_RUNS=robot-olympics/ariel-olympics-team2/research/runs/snellius
REMOTE_ISLANDS=robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/islands
REMOTE_FEATURES=robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/features
REMOTE_LAB=robot-olympics/islands/ariel-olympics-team2/research/runs/snellius/gecko_lab
SSH="ssh -o ControlMaster=auto -o ControlPath=$HOME/.ssh/snellius-%r@%h:%p -o ControlPersist=10m"
FILTER=(
  --exclude=trials/ --exclude=checkpoint/ --exclude=*_partial_*/ --exclude=calibrate/
  --include="/$PATTERN/" --exclude=/*/
  --include=*/ --include=*.log --include=*.json --include=*.txt --include=*.png --include=*.csv --include=optuna.db
  --exclude=*
)

if [[ ! $WHICH =~ ^(all|runs|islands|features|gecko_lab)$ ]]; then
  echo "usage: bash download.sh [all|runs|islands|features|gecko_lab] [run name pattern, e.g. '*_mlp']"; exit 1
fi

fetch() {  # fetch <remote root> <local root>
  mkdir -p "$2"
  echo "$(date +%T) logs, parameters, Optuna studies and plots of $1/$PATTERN"
  rsync -az --prune-empty-dirs -e "$SSH" "${FILTER[@]}" "$HOST:$1/" "$2/"
  local list; list=$(mktemp)
  for log in "$2"/$PATTERN/logs/final.log; do
    [[ -f $log ]] || continue
    local run; run=$(basename "$(dirname "$(dirname "$log")")")
    local seed; seed=$(grep -o "seed [0-9]*: best fitness [0-9.]*" "$log" | sort -k5 -g | tail -1 | awk '{print $2}' | tr -d :)
    [[ -n $seed ]] && echo "$run/final/${run%%_*}/trials/trial_000/seed_$seed/" >> "$list"
  done
  if [[ -s $list ]]; then
    echo "$(date +%T) best final seed of $(wc -l < "$list" | tr -d ' ') runs:"
    sed 's/^/  /' "$list"
    rsync -az -r --progress -e "$SSH" --files-from="$list" "$HOST:$1/" "$2/"
  fi
  rm -f "$list"
}

[[ $WHICH == all || $WHICH == runs ]] && fetch "$REMOTE_RUNS" "$LOCAL"
[[ $WHICH == all || $WHICH == islands ]] && fetch "$REMOTE_ISLANDS" "$LOCAL/islands"
[[ $WHICH == all || $WHICH == features ]] && fetch "$REMOTE_FEATURES" "$LOCAL/features"
[[ $WHICH == all || $WHICH == gecko_lab ]] && fetch "$REMOTE_LAB" "$LOCAL/gecko_lab"
echo "$(date +%T) done"
du -sh "$LOCAL"
