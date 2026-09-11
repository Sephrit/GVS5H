#!/usr/bin/env bash
# Local trial on a Mac: Qwen3.8-27B (MLX 4-bit) served by LM Studio, single call vs manager
# on a handful of LCB-100 problems (escalation/local_trial_ids.json).
#
# Every (arm, problem) pair is its own run_bench invocation with its own results file, so a
# crash, reboot or Ctrl-C loses at most the jobs in flight: re-run the script and finished
# jobs are skipped. Manager jobs are queued first because they are the long sequential chains.
#
#   TAG=smoke CAP=4000 MULTIAGENT_MAX_ITERS=1 IDS=<file> ./run_local_trial_lmstudio.sh   # plumbing check
#   ./run_local_trial_lmstudio.sh                                                          # the trial

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root

MODEL_ID=${MODEL_ID:-qwen3.8-27b@4bit}
BASE=${LMSTUDIO_BASE:-http://localhost:1234/v1}
CAP=${CAP:-128000}
JOBS=${JOBS:-4}        # concurrent requests; matches the model's --parallel slots
IDS=${IDS:-escalation/local_trial_ids.json}
TAG=${TAG:-local-trial}
UNLOAD=${UNLOAD:-1}    # unload the model when done so it doesn't sit in memory

export RUN_DIR="$ROOT/runs/$TAG"
mkdir -p "$RUN_DIR/results" "$RUN_DIR/ws" "$RUN_DIR/logs"

# test5+test6 hold every LCB-100 problem; skips 3.8 GB of older problems release_v6 would fetch
export LCB_RELEASE=${LCB_RELEASE:-v5_v6}
export MULTIAGENT_MAX_ITERS=${MULTIAGENT_MAX_ITERS:-10}
export MULTIAGENT_MAX_TASKS=12
# the groq: route is the harness's generic OpenAI-compatible path; the paper's vLLM arm used it too
export ESCALATION_OPENAI_BASE="$BASE/chat/completions"
export GROQ_API_KEY=lm-studio
export ESCALATION_GROQ_REASONING=""       # send no reasoning_effort; Qwen thinks by default
export ESCALATION_CLOUD_MAX_TOKENS="$CAP"
# a hard timeout abandons the call and retries it, so leave room: ~19 tok/s per stream with
# four in flight puts a full 128k generation near two hours
export ESCALATION_CLOUD_TIMEOUT=${TIMEOUT:-21600}
export MULTIAGENT_MODEL="groq:$MODEL_ID"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_DIR/run.log"; }

if ! curl -sf -m 10 "$BASE/models" > /dev/null; then
  log "FATAL: LM Studio server not answering at $BASE"; exit 1
fi
if ! lms ps 2>/dev/null | grep -aq "$MODEL_ID"; then
  log "loading $MODEL_ID (262k context, $JOBS slots)"
  lms load "$MODEL_ID" -c 262144 --parallel "$JOBS" -y > /dev/null 2>&1 || { log "FATAL: lms load failed"; exit 1; }
fi

caffeinate -i -w $$ &   # no idle sleep while this script runs

{ echo "sha=$(git rev-parse HEAD)"
  echo "model=$MULTIAGENT_MODEL base=$BASE cap=$CAP jobs=$JOBS max_iters=$MULTIAGENT_MAX_ITERS"
  echo "release=$LCB_RELEASE ids=$IDS started=$(date '+%Y-%m-%d %H:%M')"
} > "$RUN_DIR/run_config.txt"

one() {   # one <engine> <question_id>
  local eng=$1 qid=$2 out="$RUN_DIR/results/$1/$2.json" ids verdict
  [ -f "$out" ] && { echo "[$(date +%H:%M:%S)] skip done $eng $qid"; return 0; }
  mkdir -p "$RUN_DIR/results/$eng"
  ids=$(mktemp); printf '["%s"]\n' "$qid" > "$ids"
  echo "[$(date +%H:%M:%S)] start $eng $qid" | tee -a "$RUN_DIR/run.log"
  if MULTIAGENT_WS="$RUN_DIR/ws/$eng" \
     uv run --no-project --python 3.12 --with 'datasets<4' --with numpy --with anthropic \
       python escalation/run_bench.py --engine "$eng" --only lcb --lcb 1 \
       --ids-file "$ids" --parallel 1 --out "$out.tmp" > "$RUN_DIR/logs/${eng}_${qid}.log" 2>&1 \
     && [ -s "$out.tmp" ]; then
    mv "$out.tmp" "$out"
    verdict=$(python3 -c "import json,sys;print('PASS' if json.load(open(sys.argv[1]))['lcb']['records'][0]['passed'] else 'FAIL')" "$out")
  else
    verdict="ERROR (see logs/${eng}_${qid}.log)"
  fi
  rm -f "$ids"
  echo "[$(date +%H:%M:%S)] done  $eng $qid: $verdict" | tee -a "$RUN_DIR/run.log"
}
export -f one

log "trial $TAG: $(python3 -c "import json;print(len(json.load(open('$IDS'))))" ) problems x 2 arms, $JOBS at a time"
python3 -c "
import json
ids = json.load(open('$IDS'))
for eng in ('multiagent', 'single'):
    for q in ids:
        print(eng, q)
" | xargs -P "$JOBS" -n 2 bash -c 'one "$@"' _

python3 escalation/local_trial_summary.py "$RUN_DIR" > "$RUN_DIR/summary.md" 2>&1
cat "$RUN_DIR/summary.md" | tee -a "$RUN_DIR/run.log"

s=$(grep -a '^- Local single' "$RUN_DIR/summary.md" | sed 's/^- Local single: //')
m=$(grep -a '^- Local multiagent' "$RUN_DIR/summary.md" | sed 's/^- Local multiagent: //')
level=info; grep -aq 'ERROR' "$RUN_DIR/run.log" && level=warn
stentor notify -l "$level" -s gvs5h "GVS5H trial '$TAG' finished" \
  "Single call $s; manager $m. Table: $RUN_DIR/summary.md" > /dev/null 2>&1 || true

[ "$UNLOAD" = 1 ] && lms unload "$MODEL_ID" > /dev/null 2>&1
log "finished"
