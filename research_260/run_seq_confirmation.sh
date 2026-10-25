#!/bin/zsh
# Six-pair confirmation of the neighbour sequence screen. Pre-registered 14.09.2026 before the
# January/July decision was read in full: same recipe, seeds 13 and 21, K=24, W=90, no changes per pair.
# Builds missing core tables with research_250/fallback250_cv.py (no variants), then one screen per pair,
# sequentially. Usage: research_260/run_seq_confirmation.sh <log-dir>
set -e
cd "$(dirname "$0")/.."
LOG=${1:-.cache/rmse260/logs}; mkdir -p "$LOG"
for pair in "2,8" "3,9" "4,10" "5,11" "6,12"; do
  tag=${pair/,/_}
  if [ -f ".cache/rmse260/seq-screen/seq_screen_result_${tag}.json" ] && grep -q '"decision"' ".cache/rmse260/seq-screen/seq_screen_result_${tag}.json"; then echo "skip $tag"; continue; fi
  .venv/bin/python -u -B research_250/fallback250_cv.py --project . --output .cache/rmse260 --pairs "$pair" --variants '' > "$LOG/core_${tag}.log" 2>&1
  TR=""; VA=""
  for j in .cache/rmse260/core_*.json; do
    n=$(.venv/bin/python -c "import json;m=json.load(open('$j'))['source_months'];print(len(m), m)")
    case "$n" in
      "2 ["*) if [ "$(.venv/bin/python -c "import json;print(sorted(json.load(open('$j'))['source_months'])==sorted([int(x) for x in '$pair'.split(',')]))")" = "True" ]; then VA="${j%.json}.parquet"; fi;;
      "10 ["*) if [ "$(.venv/bin/python -c "import json;m=set(json.load(open('$j'))['source_months']);print(m==set(range(1,13))-set(int(x) for x in '$pair'.split(',')))")" = "True" ]; then TR="${j%.json}.parquet"; fi;;
    esac
  done
  echo "$(date '+%H:%M:%S') pair $tag train=$TR valid=$VA"
  .venv/bin/python -u -B research_260/neighbour_sequence_screen.py --project . --core-train "$TR" --core-valid "$VA" --output .cache/rmse260/seq-screen --jobs 8 > "$LOG/seq_${tag}.log" 2>&1
  grep DECISION "$LOG/seq_${tag}.log"
done
echo ALL DONE
# Permutation control on pair 1_7 (pre-registered with the confirmation): neighbour tensors shuffled among
# departures within airport, separately per split. If the shuffled model scores like FLAT, the gain is the
# ordered neighbour information, not model capacity. One seed.
mkdir -p .cache/rmse260/seq-perm
.venv/bin/python -u -B research_260/neighbour_sequence_screen.py --project . --core-train .cache/rmse260/core_f758b1eff6740ec9.parquet --core-valid .cache/rmse260/core_7c63b099f4f3cfc0.parquet --output .cache/rmse260/seq-perm --seeds 13 --permute-seed 101 --jobs 8 > "$LOG/seq_perm_1_7.log" 2>&1
grep DECISION "$LOG/seq_perm_1_7.log"
echo PERMUTATION DONE
