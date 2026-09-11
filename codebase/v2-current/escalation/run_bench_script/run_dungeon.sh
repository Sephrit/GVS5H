#!/usr/bin/env bash
# Build the dungeon game twice from one brief with the local Qwen3.8-27B in LM Studio: once
# with a single call, once through the manager loop with a browser check between rounds.
# Both arms run at once (two of the model's four slots). When both finish, the two games are
# copied to runs/<tag>/play/ as dungeon-A.html and dungeon-B.html in a random order, with the
# answer in play/.key.json, so they can be judged blind. A finished arm is skipped on re-run.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root

MODEL_ID=${MODEL_ID:-qwen3.8-27b@4bit}
BASE=${LMSTUDIO_BASE:-http://localhost:1234/v1}
TAG=${TAG:-dungeon}
BRIEF=${BRIEF:-escalation/dungeon_brief.md}
UNLOAD=${UNLOAD:-1}
RUN="$ROOT/runs/$TAG"
mkdir -p "$RUN/play"

export MULTIAGENT_MAX_ITERS=${MULTIAGENT_MAX_ITERS:-10}
export MULTIAGENT_MAX_TASKS=12
export ESCALATION_OPENAI_BASE="$BASE/chat/completions"
export GROQ_API_KEY=lm-studio
export ESCALATION_GROQ_REASONING=""
export ESCALATION_CLOUD_MAX_TOKENS=${CAP:-128000}
export ESCALATION_CLOUD_TIMEOUT=${TIMEOUT:-21600}
export MULTIAGENT_MODEL="groq:$MODEL_ID"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/run.log"; }

if ! curl -sf -m 10 "$BASE/models" > /dev/null; then
  log "FATAL: LM Studio server not answering at $BASE"; exit 1
fi
if ! lms ps 2>/dev/null | grep -aq "$MODEL_ID"; then
  log "loading $MODEL_ID (262k context, 4 slots)"
  lms load "$MODEL_ID" -c 262144 --parallel 4 -y > /dev/null 2>&1 || { log "FATAL: lms load failed"; exit 1; }
fi

caffeinate -i -w $$ &   # no idle sleep while this script runs

{ echo "sha=$(git rev-parse HEAD)"
  echo "model=$MULTIAGENT_MODEL cap=$ESCALATION_CLOUD_MAX_TOKENS max_iters=$MULTIAGENT_MAX_ITERS"
  echo "brief=$BRIEF started=$(date '+%Y-%m-%d %H:%M')"
} > "$RUN/run_config.txt"

arm() {
  local eng=$1
  [ -s "$RUN/$eng/summary.json" ] && { log "skip done $eng"; return; }
  log "start $eng"
  uv run --no-project --python 3.12 --with playwright --with pillow \
    python escalation/build_game.py --engine "$eng" --brief "$BRIEF" --out "$RUN/$eng" \
    > "$RUN/$eng.log" 2>&1
  log "done  $eng (exit $?): $(tail -1 "$RUN/$eng.log")"
}

arm multiagent &
arm single &
wait

if [ -s "$RUN/single/game.html" ] && [ -s "$RUN/multiagent/game.html" ]; then
  python3 - "$RUN" <<'PY'
import json, os, random, shutil, sys
run = sys.argv[1]
arms = ["single", "multiagent"]
random.shuffle(arms)
for letter, arm in zip("AB", arms):
    shutil.copy(os.path.join(run, arm, "game.html"), os.path.join(run, "play", f"dungeon-{letter}.html"))
json.dump({"A": arms[0], "B": arms[1]}, open(os.path.join(run, "play", ".key.json"), "w"))
PY
  level=info; msg="Both games are ready to play blind: $RUN/play (dungeon-A.html, dungeon-B.html)"
else
  level=warn; msg="Finished, but at least one arm produced no game. See $RUN/single.log and $RUN/multiagent.log"
fi
log "$msg"
stentor notify -l "$level" -s gvs5h "Dungeon game build finished" "$msg" > /dev/null 2>&1 || true

[ "$UNLOAD" = 1 ] && lms unload "$MODEL_ID" > /dev/null 2>&1
log "finished"
