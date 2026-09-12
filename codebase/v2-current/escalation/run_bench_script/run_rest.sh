#!/usr/bin/env bash
# The rest of the evening, in the order that answers the open questions soonest:
#
#   1. quant quality    4-bit vs 8-bit graded on hard problems, thinking off so answers finish
#   2. matrix gaps      the rows the first pass missed: 4-bit + MTP re-measured, BF16 at a
#                       context llama.cpp can actually reserve
#   3. dungeon games    the remaining scaffolds, now served with MTP (~31 tok/s, not 17)
#   4. report           browser checks and screenshots for every game produced
#
# Each stage is resumable and manages its own model server, so re-running this picks up where it
# stopped. Stage 3 is the long one and deliberately runs last.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
HERE="$(pwd)"
cd ../.. || exit 1                                     # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root
LOG="$ROOT/runs/rest.log"
mkdir -p "$ROOT/runs"
echo $$ > "$ROOT/runs/rest.pid"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
note() { stentor notify -l "${2:-info}" -s gvs5h "$1" "${3:-}" > /dev/null 2>&1 || true; }
free_gb() { df -g / | awk 'NR==2 {print $4}'; }

caffeinate -i -w $$ &

log "=== stage 1/4: quant quality (4-bit vs 8-bit, thinking off)"
# 32k, not 16k: with thinking off answers still run 10-16k tokens, and 16k cut one of the first two.
QUANTS="omlx:Qwen3.8-27B-8bit-MTP omlx:Qwen3.8-27B-4bit-MTP" PROBLEMS=8 CAP=32000 THINK=false \
  "$HERE/run_quant_trial.sh" >> "$ROOT/runs/quant-quality.log" 2>&1
log "stage 1 done (exit $?), $(free_gb) GB free"
note "Quant quality trial finished" info "$ROOT/runs/quant-quality/run.log"

log "=== stage 2/4: matrix gaps (4-bit + MTP, BF16)"
LONG_TOKENS=8000 LMS_CTX=32768 uv run --no-project --python 3.12 \
  python escalation/serving_matrix.py --out "$ROOT/runs/serving-matrix" \
  >> "$ROOT/runs/serving-matrix.log" 2>&1
log "stage 2 done (exit $?), $(free_gb) GB free"
rm -rf "$ROOT"/runs/serving-matrix/omlx-*/cache
note "Serving matrix complete" info "$ROOT/runs/serving-matrix/matrix.md"

log "=== stage 3/4: remaining dungeon scaffolds, served with MTP"
ENGINES="bestof3 refine multiagent multiagent-nocheck" "$HERE/run_dungeon.sh" \
  >> "$ROOT/runs/dungeon-resume.log" 2>&1
log "stage 3 done (exit $?), $(free_gb) GB free"

log "=== stage 4/4: report on every game produced"
uv run --no-project --python 3.12 --with playwright --with pillow \
  python escalation/game_report.py "$ROOT/runs/dungeon" >> "$LOG" 2>&1
log "all stages finished"
note "Everything finished" info "quant quality, serving matrix and the dungeon games are done"
rm -f "$ROOT/runs/rest.pid"
