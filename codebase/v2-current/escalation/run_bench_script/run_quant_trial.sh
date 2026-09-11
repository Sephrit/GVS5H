#!/usr/bin/env bash
# What each quantization of Qwen3.8-27B actually gets right, on the same hard problems.
#
# For every quant listed in QUANTS, the single-call arm runs over the first PROBLEMS ids of
# escalation/local_trial_ids.json and is graded by LiveCodeBench's own hidden tests (with the
# evaluator fix in codebase/livecodebench). One results file per (quant, problem), so an
# interrupted run resumes where it stopped.
#
#   QUANTS="omlx:Qwen3.8-27B-4bit-MTP omlx:Qwen3.8-27B-8bit-MTP lms:qwen3.8-27b@bf16"
#
# omlx: rows serve MLX weights through our own oMLX instance on a free port, with turboquant KV,
# specprefill and the thinking budget off, so only the quantization differs. lms: rows are GGUF
# through LM Studio's llama.cpp -- the only way to reach BF16 on this Mac, and a different engine,
# so read those against each other rather than against the MLX rows.
#
# CAP is 32k, not the paper's 128k: a 128k single call on these problems averaged ~74k output
# tokens, which is ~5 hours per quant on this machine. Pass rates here are therefore comparable
# across quants, not against the paper's numbers.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root

QUANTS=${QUANTS:-"omlx:Qwen3.8-27B-8bit-MTP omlx:Qwen3.8-27B-4bit-MTP"}
PROBLEMS=${PROBLEMS:-6}
CAP=${CAP:-32000}
MTP=${MTP:-true}                      # matched across quants; oMLX rows only
MLX_DIR=${MLX_DIR:-$HOME/AI/Models/oMLX/sephwa}
OMLX_KEY=${OMLX_KEY:-quant-trial}
TAG=${TAG:-quant-trial}
RUN="$ROOT/runs/$TAG"
mkdir -p "$RUN"
echo $$ > "$RUN/run.pid"

export LCB_RELEASE=${LCB_RELEASE:-v5_v6}
export ESCALATION_CLOUD_MAX_TOKENS="$CAP"
export ESCALATION_CLOUD_TIMEOUT=${TIMEOUT:-10800}
export ESCALATION_GROQ_REASONING=""

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/run.log"; }

IDS="$RUN/ids.json"
python3 - "$PROBLEMS" "$IDS" <<'PY'
import json, os, sys
n, out = int(sys.argv[1]), sys.argv[2]
ids = json.load(open(os.path.join("escalation", "local_trial_ids.json")))[:n]
json.dump(ids, open(out, "w"), indent=1)
print(f"problems: {', '.join(ids)}")
PY

caffeinate -i -w $$ &

start_omlx() {                        # start_omlx <model>; sets BASE and SERVE_PID
  local model=$1 port=8010 iso="$RUN/omlx-$1"
  while lsof -ti:"$port" > /dev/null 2>&1; do
    port=$((port + 1))
    [ "$port" -gt 8040 ] && { log "FATAL: no free port"; exit 1; }
  done
  mkdir -p "$iso/models"
  ln -sfn "$MLX_DIR/$model" "$iso/models/$model"
  python3 - "$iso" "$port" "$model" "$MTP" <<'PY'
import copy, json, os, sys
iso, port, model, mtp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] == "true"
s = json.load(open(os.path.expanduser("~/.omlx/settings.json")))
s["server"].update({"port": port, "auto_start_on_launch": False})
s["model"]["model_dirs"] = [os.path.join(iso, "models")]
json.dump(s, open(os.path.join(iso, "settings.json"), "w"), indent=2)
ms = json.load(open(os.path.expanduser("~/.omlx/model_settings.json")))["models"]
src = ms.get("Qwen3.8-27B-MTPLX-Optimized-Speed") or next(iter(ms.values()))
e = copy.deepcopy(src)
e.update({"max_context_window": 262144, "max_tokens": 131072, "ttl_seconds": 86400,
          "mtp_enabled": mtp, "mtp_num_draft_tokens": 3, "enable_thinking": True,
          "turboquant_kv_enabled": False, "specprefill_enabled": False,
          "thinking_budget_enabled": False})
e.pop("model_alias", None)
e["chat_template_kwargs"] = {}
json.dump({"version": 1, "models": {model: e}},
          open(os.path.join(iso, "model_settings.json"), "w"), indent=2)
PY
  OMLX_BASE_PATH="$iso" nohup /Applications/oMLX.app/Contents/MacOS/omlx-cli serve \
    --model-dir "$iso/models" --port "$port" --base-path "$iso" --api-key "$OMLX_KEY" \
    > "$iso/serve.log" 2>&1 &
  SERVE_PID=$!
  BASE="http://127.0.0.1:$port/v1"
  for i in $(seq 120); do
    curl -sf -m 2 -H "Authorization: Bearer $OMLX_KEY" "$BASE/models" > /dev/null && break
    sleep 2
  done
  curl -sf -m 5 -H "Authorization: Bearer $OMLX_KEY" "$BASE/models" \
    | MODEL="$model" python3 -c 'import json,os,sys; sys.exit(0 if os.environ["MODEL"] in [m["id"] for m in json.load(sys.stdin)["data"]] else 1)' \
    || { log "FATAL: oMLX on $BASE is not serving $model"; exit 1; }
}

stop_omlx() { [ -n "${SERVE_PID:-}" ] && { pkill -P "$SERVE_PID" 2>/dev/null; kill "$SERVE_PID" 2>/dev/null; sleep 5; }; SERVE_PID=""; }
trap 'stop_omlx; lms unload --all > /dev/null 2>&1' EXIT

one_problem() {                       # one_problem <label> <qid>
  local label=$1 qid=$2 out="$RUN/$label/$qid.json" ids
  [ -f "$out" ] && { echo "  skip done $qid"; return; }
  mkdir -p "$RUN/$label"
  ids=$(mktemp); printf '["%s"]\n' "$qid" > "$ids"
  MULTIAGENT_WS="$RUN/$label/ws" \
    uv run --no-project --python 3.12 --with 'datasets<4' --with numpy \
      python escalation/run_bench.py --engine single --only lcb --lcb 1 \
      --ids-file "$ids" --parallel 1 --out "$out.tmp" > "$RUN/$label/$qid.log" 2>&1 \
    && [ -s "$out.tmp" ] && mv "$out.tmp" "$out" \
    && echo "  $qid: $(python3 -c "import json,sys;r=json.load(open('$out'))['lcb']['records'][0];print(('PASS' if r['passed'] else 'FAIL'), r.get('status'), r.get('completion_tokens'), 'tok')")" \
    || echo "  $qid: ERROR (see $RUN/$label/$qid.log)"
  rm -f "$ids"
}

for spec in $QUANTS; do
  kind=${spec%%:*}; model=${spec#*:}
  label=$(echo "$model" | tr '/@' '--')
  log "=== $kind $model"
  if [ "$kind" = omlx ]; then
    start_omlx "$model"
    export GROQ_API_KEY="$OMLX_KEY" ESCALATION_OPENAI_BASE="$BASE/chat/completions"
    export MULTIAGENT_MODEL="groq:$model"
  else
    lms unload --all > /dev/null 2>&1
    lms load "$model" -c 262144 --parallel 1 -y > /dev/null 2>&1 \
      || { log "FATAL: lms load $model failed"; exit 1; }
    export GROQ_API_KEY=lm-studio ESCALATION_OPENAI_BASE="http://localhost:1234/v1/chat/completions"
    export MULTIAGENT_MODEL="groq:$model"
  fi
  while read -r qid; do one_problem "$label" "$qid"; done < <(python3 -c "import json;print('\n'.join(json.load(open('$IDS'))))")
  if [ "$kind" = omlx ]; then stop_omlx; else lms unload --all > /dev/null 2>&1; fi
  log "=== done $model"
done

python3 - "$RUN" $QUANTS <<'PY' | tee -a "$RUN/run.log"
import glob, json, os, sys
run, specs = sys.argv[1], sys.argv[2:]
ids = json.load(open(os.path.join(run, "ids.json")))
rows = []
for spec in specs:
    model = spec.split(":", 1)[1]
    label = model.replace("/", "-").replace("@", "-")
    cells, toks = {}, 0
    for qid in ids:
        f = os.path.join(run, label, f"{qid}.json")
        if not os.path.exists(f):
            cells[qid] = "-"
            continue
        r = json.load(open(f))["lcb"]["records"][0]
        cells[qid] = "PASS" if r["passed"] else ("trunc" if r.get("status") == "truncated" else "fail")
        toks += r.get("completion_tokens") or 0
    rows.append((model, cells, toks))
head = "| Quantization | " + " | ".join(ids) + " | Passed | Output tokens |"
print("\n# Pass rates by quantization (single call, same problems)\n")
print(head); print("|" + "---|" * (len(ids) + 3))
for model, cells, toks in rows:
    n = sum(1 for v in cells.values() if v == "PASS")
    print(f"| {model} | " + " | ".join(cells[q] for q in ids) + f" | {n}/{len(ids)} | {toks:,} |")
print("\n`trunc` = hit the output cap before finishing, which scores as a failure.")
PY
stentor notify -l info -s gvs5h "Quant trial finished" "Pass rates per quantization: $RUN/run.log" > /dev/null 2>&1 || true
rm -f "$RUN/run.pid"
log "finished"
