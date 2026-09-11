#!/usr/bin/env bash
# Run the whole evening's work unattended, in the order that answers the open questions first:
#
#   1. serving matrix   every weights/server/MTP combination, speed plus a greedy sample
#   2. quant trial      what 4-bit, 8-bit and BF16 actually get right on hard problems
#   3. dungeon games    the remaining scaffolds; arms that already have a game are skipped
#
# Each stage manages its own model server and is resumable, so re-running this script picks up
# where it stopped. Between stages the finished stage's prompt caches are deleted: they are
# gigabytes of scaffolding on a disk with little room left.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
HERE="$(pwd)"
cd ../.. || exit 1                                     # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root
LOG="$ROOT/runs/tonight.log"
mkdir -p "$ROOT/runs"
echo $$ > "$ROOT/runs/tonight.pid"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
note() { stentor notify -l "${2:-info}" -s gvs5h "$1" "${3:-}" > /dev/null 2>&1 || true; }

caffeinate -i -w $$ &

free_gb() { df -g / | awk 'NR==2 {print $4}'; }

log "=== stage 1/3: serving matrix"
uv run --no-project --python 3.12 python escalation/serving_matrix.py \
  --out "$ROOT/runs/serving-matrix" >> "$ROOT/runs/serving-matrix.log" 2>&1
log "stage 1 done (exit $?), $(free_gb) GB free"
rm -rf "$ROOT"/runs/serving-matrix/omlx-*/cache
note "Serving matrix finished" info "$ROOT/runs/serving-matrix/matrix.md"

log "=== stage 2/3: quant trial (4-bit vs 8-bit vs BF16, graded)"
QUANTS="omlx:Qwen3.8-27B-8bit-MTP omlx:Qwen3.8-27B-4bit-MTP lms:qwen3.8-27b@bf16" \
  PROBLEMS=6 CAP=32000 "$HERE/run_quant_trial.sh" >> "$ROOT/runs/quant-trial.log" 2>&1
log "stage 2 done (exit $?), $(free_gb) GB free"
rm -rf "$ROOT"/runs/quant-trial/omlx-*/cache
note "Quant trial finished" info "$ROOT/runs/quant-trial/run.log"

log "=== stage 3/3: remaining dungeon scaffolds"
ENGINES="bestof3 refine multiagent multiagent-nocheck" "$HERE/run_dungeon.sh" >> "$ROOT/runs/dungeon-resume.log" 2>&1
log "stage 3 done (exit $?), $(free_gb) GB free"

uv run --no-project --python 3.12 --with playwright --with pillow \
  python escalation/game_report.py "$ROOT/runs/dungeon" >> "$LOG" 2>&1
log "all stages finished"
note "Tonight's runs all finished" info "matrix, quant trial and the dungeon games are done"
rm -f "$ROOT/runs/tonight.pid"
