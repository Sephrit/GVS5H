#!/usr/bin/env bash
# What each quantization of Qwen3.8-27B actually gets right, on the same hard problems.
#
# For every quant in QUANTS, the single-call arm runs over the first PROBLEMS ids of
# escalation/local_trial_ids.json and is graded by LiveCodeBench's own hidden tests (with the
# evaluator fix in codebase/livecodebench). One results file per (quant, problem), so an
# interrupted run resumes where it stopped.
#
# THINK=false by design. With thinking on, these problems need ~74k output tokens (the paper's
# figure, and our own 57k game write agrees), so any cap short enough to finish in an evening
# truncates nearly every answer and scores it wrong: a first pass at CAP=32000 returned 0 of 6
# for both quants with four truncations each, measuring the cap rather than the weights. With
# thinking off the answers finish in a few thousand tokens, so a pass is a pass. It is a
# different setting from the one the scaffolds use -- read it as a comparison between quants,
# not as this model's ceiling.
#
#   QUANTS="omlx:Qwen3.8-27B-8bit-MTP omlx:Qwen3.8-27B-4bit-MTP" PROBLEMS=8 ./run_quant_trial.sh
#
# omlx: rows serve MLX weights through our own oMLX instance with turboquant KV, specprefill and
# the thinking budget off, so only the quantization differs. lms: rows are GGUF through LM Studio's
# llama.cpp -- the only way to reach BF16 on this Mac. llama.cpp reserves its whole KV cache up
# front, hence LMS_CTX rather than the 262k an MLX server will happily take.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root

QUANTS=${QUANTS:-"omlx:Qwen3.8-27B-8bit-MTP omlx:Qwen3.8-27B-4bit-MTP"}
PROBLEMS=${PROBLEMS:-8}
CAP=${CAP:-16000}
THINK=${THINK:-false}                 # see the note above
MTP=${MTP:-true}                      # matched across quants; oMLX rows only
LMS_CTX=${LMS_CTX:-32768}
MLX_DIR=${MLX_DIR:-$HOME/AI/Models/oMLX/sephwa}
OMLX_KEY=${OMLX_KEY:-quant-trial}
TAG=${TAG:-quant-quality}
RUN="$ROOT/runs/$TAG"
mkdir -p "$RUN"
echo $$ > "$RUN/run.pid"

export LCB_RELEASE=${LCB_RELEASE:-v5_v6}
export ESCALATION_CLOUD_MAX_TOKENS="$CAP"
export ESCALATION_CLOUD_TIMEOUT=${TIMEOUT:-10800}
export ESCALATION_GROQ_REASONING=""

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/run.log"; }

IDS="$RUN/ids.json"
python3 - "$PROBLEMS" "$IDS" <<'PYIDS'
import json, os, sys
n, out = int(sys.argv[1]), sys.argv[2]
ids = json.load(open(os.path.join("escalation", "local_trial_ids.json")))[:n]
json.dump(ids, open(out, "w"), indent=1)
print(f"problems: {', '.join(ids)}")
PYIDS

caffeinate -i -w $$ &

start_omlx() {                        # start_omlx <model>; sets BASE and SERVE_PID
  local model=$1 port=8010 iso="$RUN/omlx-$1"
  while lsof -ti:"$port" > /dev/null 2>&1; do
    port=$((port + 1))
    [ "$port" -gt 8040 ] && { log "FATAL: no free port"; exit 1; }
  done
  mkdir -p "$iso/models"
  ln -sfn "$MLX_DIR/$model" "$iso/models/$model"
  python3 - "$iso" "$port" "$model" "$MTP" "$THINK" <<'PYCONF'
import copy, json, os, sys
iso, port, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
mtp, think = sys.argv[4] == "true", sys.argv[5] == "true"
s = json.load(open(os.path.expanduser("~/.omlx/settings.json")))
s["server"].update({"port": port, "auto_start_on_launch": False})
s["model"]["model_dirs"] = [os.path.join(iso, "models")]
# The inherited cap is 50 GB of prompt cache per server, which filled the disk mid-run.
s.setdefault("cache", {})["ssd_cache_max_size"] = "2GB"
json.dump(s, open(os.path.join(iso, "settings.json"), "w"), indent=2)
ms = json.load(open(os.path.expanduser("~/.omlx/model_settings.json")))["models"]
src = ms.get("Qwen3.8-27B-MTPLX-Optimized-Speed") or next(iter(ms.values()))
e = copy.deepcopy(src)
e.update({"max_context_window": 262144, "max_tokens": 131072, "ttl_seconds": 86400,
          "mtp_enabled": mtp, "mtp_num_draft_tokens": 3, "enable_thinking": think,
          "turboquant_kv_enabled": False, "specprefill_enabled": False,
          "thinking_budget_enabled": False})
e.pop("model_alias", None)
e["chat_template_kwargs"] = {"enable_thinking": think}
json.dump({"version": 1, "models": {model: e}},
          open(os.path.join(iso, "model_settings.json"), "w"), indent=2)
PYCONF
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
    && echo "  $qid: $(python3 -c "import json;r=json.load(open('$out'))['lcb']['records'][0];print(('PASS' if r['passed'] else 'fail'), r.get('status'), r.get('completion_tokens'), 'tok')")" \
    || echo "  $qid: ERROR (see $RUN/$label/$qid.log)"
  rm -f "$ids"
}

for spec in $QUANTS; do
  kind=${spec%%:*}; model=${spec#*:}
  label=$(echo "$model" | tr '/@' '--')
  log "=== $kind $model (thinking=$THINK, cap=$CAP)"
  if [ "$kind" = omlx ]; then
    start_omlx "$model"
    export GROQ_API_KEY="$OMLX_KEY" ESCALATION_OPENAI_BASE="$BASE/chat/completions"
    export MULTIAGENT_MODEL="groq:$model"
  else
    lms unload --all > /dev/null 2>&1
    lms load "$model" -c "$LMS_CTX" --parallel 1 -y > /dev/null 2>&1 \
      || { log "FATAL: lms load $model failed"; exit 1; }
    export GROQ_API_KEY=lm-studio ESCALATION_OPENAI_BASE="http://localhost:1234/v1/chat/completions"
    export MULTIAGENT_MODEL="groq:$model"
  fi
  while read -r qid; do one_problem "$label" "$qid"; done < <(python3 -c "import json;print('\n'.join(json.load(open('$IDS'))))")
  if [ "$kind" = omlx ]; then
    stop_omlx
    rm -rf "$RUN/omlx-$model/cache"   # ~3 GB of prompt cache per server, and disk is tight
  else
    lms unload --all > /dev/null 2>&1
  fi
  log "=== done $model ($(df -g / | awk 'NR==2 {print $4}') GB free)"
done

python3 - "$RUN" $QUANTS <<'PYSUM' | tee -a "$RUN/run.log"
import json, os, sys
run, specs = sys.argv[1], sys.argv[2:]
ids = json.load(open(os.path.join(run, "ids.json")))
rows = []
for spec in specs:
    model = spec.split(":", 1)[1]
    label = model.replace("/", "-").replace("@", "-")
    cells, toks, trunc = {}, 0, 0
    for qid in ids:
        f = os.path.join(run, label, f"{qid}.json")
        if not os.path.exists(f):
            cells[qid] = "-"
            continue
        r = json.load(open(f))["lcb"]["records"][0]
        if r["passed"]:
            cells[qid] = "PASS"
        elif r.get("status") == "truncated":
            cells[qid] = "cut"
            trunc += 1
        else:
            cells[qid] = "wrong"
        toks += r.get("completion_tokens") or 0
    rows.append((model, cells, toks, trunc))
print("\n# Pass rates by quantization (single call, same problems, thinking off)\n")
print("| Quantization | " + " | ".join(ids) + " | Passed | Cut off | Output tokens |")
print("|" + "---|" * (len(ids) + 4))
for model, cells, toks, trunc in rows:
    n = sum(1 for v in cells.values() if v == "PASS")
    print(f"| {model} | " + " | ".join(cells[q] for q in ids)
          + f" | **{n}/{len(ids)}** | {trunc} | {toks:,} |")
print("\n`cut` = hit the output cap before finishing, which scores as a failure. "
      "`wrong` = finished but failed the hidden tests.")
PYSUM
stentor notify -l info -s gvs5h "Quant quality trial finished" "Pass rates per quantization: $RUN/run.log" > /dev/null 2>&1 || true
rm -f "$RUN/run.pid"
log "finished"
