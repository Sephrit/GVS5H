#!/usr/bin/env bash
# Build the dungeon game once per scaffold with a local Qwen3.8-27B, then shuffle the games for
# a blind comparison. Default scaffolds: the paper's manager loop, the same loop with no
# browser check, one worker looped against the check, three-samples-keep-the-best, and a plain
# single call.
#
# Serving: 8-bit MLX through oMLX, measured 2026-09-11 on an M5 Max with these same weights:
#   oMLX 8-bit                32.9 tok/s   <- chosen
#   LM Studio 4-bit MLX       28.6 tok/s
#   LM Studio Q8 GGUF + MTP   21.7 tok/s
#   LM Studio 8-bit MLX       17.6 tok/s
# Measured per-row with a fresh server (runs/serving-matrix), MTP nearly doubles 8-bit decode:
# 31.4 tok/s against 17.6. An earlier reading that showed no difference came from a server whose
# setting had not taken effect. MTP is on, and it is lossless: on and off produce identical
# greedy text on the same engine, so it buys speed at no cost to the result.
# The server runs on a private port with its own settings folder, leaving the oMLX config
# OpenCode uses untouched, and with turboquant KV, specprefill and the thinking budget off --
# all three trade quality for speed.
#
# Arms run one after another. Two concurrent streams on oMLX total 26.5 tok/s against 32.9 for
# one, so parallelism is a loss here; LM Studio's batched MLX engine crashed outright on
# Metal's buffer-count limit under the same load.
#
#   ENGINES="multiagent single" ./run_dungeon.sh        # a subset
#   TAG=dungeon2 ./run_dungeon.sh                       # a fresh run directory

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # codebase/v2-current
ROOT="$(cd ../.. && pwd)"                              # repo root

MODEL_ID=${MODEL_ID:-Qwen3.8-27B-8bit-MTP}          # the copy that carries the MTP head
MODEL_DIR=${MODEL_DIR:-$HOME/AI/Models/oMLX/sephwa}
OMLX_PORT=${OMLX_PORT:-8010}
OMLX_KEY=${OMLX_KEY:-local-test}
BASE="http://127.0.0.1:$OMLX_PORT/v1"
TAG=${TAG:-dungeon}
BRIEF=${BRIEF:-escalation/dungeon_brief.md}
ENGINES=${ENGINES:-"multiagent multiagent-nocheck refine bestof3 single"}
RUN="$ROOT/runs/$TAG"
ISO="$RUN/omlx"
mkdir -p "$RUN/play" "$ISO/models"
echo $$ > "$RUN/run.pid"

export MULTIAGENT_MAX_ITERS=${MULTIAGENT_MAX_ITERS:-10}
export MULTIAGENT_MAX_TASKS=12
export REFINE_MAX_ROUNDS=${REFINE_MAX_ROUNDS:-6}
export BESTOF_N=${BESTOF_N:-3}
export ESCALATION_OPENAI_BASE="$BASE/chat/completions"
export GROQ_API_KEY="$OMLX_KEY"          # the groq: route is the generic OpenAI-compatible path
export ESCALATION_GROQ_REASONING=""      # send no reasoning_effort; Qwen thinks by default
export ESCALATION_CLOUD_MAX_TOKENS=${CAP:-128000}
export ESCALATION_CLOUD_TIMEOUT=${TIMEOUT:-21600}
export MULTIAGENT_MODEL="groq:$MODEL_ID"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN/run.log"; }

# --- the model server: always ours, on a port nobody else holds ---------------------------
# An oMLX instance left behind on this port answers the probe happily and then 404s every
# request for a model it does not serve, so never adopt a server we did not start.
while lsof -ti:"$OMLX_PORT" > /dev/null 2>&1; do
  log "port $OMLX_PORT is taken, trying $((OMLX_PORT + 1))"
  OMLX_PORT=$((OMLX_PORT + 1))
  [ "$OMLX_PORT" -gt 8040 ] && { log "FATAL: no free port in 8010-8040"; exit 1; }
done
BASE="http://127.0.0.1:$OMLX_PORT/v1"
export ESCALATION_OPENAI_BASE="$BASE/chat/completions"
if true; then
  log "starting oMLX on port $OMLX_PORT (private settings in $ISO)"
  ln -sfn "$MODEL_DIR/$MODEL_ID" "$ISO/models/$MODEL_ID"
  python3 - "$ISO" "$OMLX_PORT" "$MODEL_ID" <<'PY'
import copy, json, os, sys
iso, port, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = json.load(open(os.path.expanduser("~/.omlx/settings.json")))
s["server"].update({"port": port, "auto_start_on_launch": False})
s["model"]["model_dirs"] = [os.path.join(iso, "models")]
json.dump(s, open(os.path.join(iso, "settings.json"), "w"), indent=2)
ms = json.load(open(os.path.expanduser("~/.omlx/model_settings.json")))["models"]
src = ms.get("Qwen3.8-27B-MTPLX-Optimized-Speed") or next(iter(ms.values()))
e = copy.deepcopy(src)
e.update({"max_context_window": 262144, "max_tokens": 131072, "ttl_seconds": 86400,
          # MTP nearly doubles decode here: 31.4 tok/s against 17.6 on the same weights and
          # prompt (runs/serving-matrix). It is lossless -- MTP on and off produce identical
          # greedy text on the same engine.
          "mtp_enabled": True, "mtp_num_draft_tokens": 3, "enable_thinking": True,
          "turboquant_kv_enabled": False,      # 4-bit KV cache: speed for quality
          "specprefill_enabled": False,        # draft-model prompt pruning: speed for quality
          "thinking_budget_enabled": False})   # let it think as long as the paper's runs did
e.pop("model_alias", None)
e["chat_template_kwargs"] = {}
json.dump({"version": 1, "models": {model: e}}, open(os.path.join(iso, "model_settings.json"), "w"), indent=2)
PY
  OMLX_BASE_PATH="$ISO" nohup /Applications/oMLX.app/Contents/MacOS/omlx-cli serve \
    --model-dir "$ISO/models" --port "$OMLX_PORT" --base-path "$ISO" --api-key "$OMLX_KEY" \
    > "$ISO/serve.log" 2>&1 &
  echo $! > "$ISO/serve.pid"
  cleanup() {                     # the server is ours; oMLX's worker is a child of the CLI
    local pid
    pid=$(cat "$ISO/serve.pid" 2>/dev/null) || return
    pkill -P "$pid" 2>/dev/null
    kill "$pid" 2>/dev/null
    sleep 3
    rm -rf "$ISO/cache"            # ~3 GB of prompt cache, and the disk has little room
  }
  trap cleanup EXIT               # never leave a server behind to confuse the next run
  for i in $(seq 120); do
    curl -sf -m 2 -H "Authorization: Bearer $OMLX_KEY" "$BASE/models" > /dev/null && break
    sleep 2
  done
fi
# exact id match: "Qwen3.8-27B-8bit" is also a prefix of "Qwen3.8-27B-8bit-MTP"
curl -sf -m 5 -H "Authorization: Bearer $OMLX_KEY" "$BASE/models" \
  | MODEL_ID="$MODEL_ID" python3 -c 'import json,os,sys; ids=[m["id"] for m in json.load(sys.stdin)["data"]]; sys.exit(0 if os.environ["MODEL_ID"] in ids else 1)' \
  || { log "FATAL: oMLX on $BASE is not serving exactly $MODEL_ID (see $ISO/serve.log)"; exit 1; }
log "model server ready: $MODEL_ID on $BASE"

caffeinate -i -w $$ &   # no idle sleep while this script runs

{ echo "sha=$(git rev-parse HEAD)"
  echo "model=$MULTIAGENT_MODEL base=$BASE cap=$ESCALATION_CLOUD_MAX_TOKENS max_iters=$MULTIAGENT_MAX_ITERS"
  echo "engines=$ENGINES brief=$BRIEF started=$(date '+%Y-%m-%d %H:%M')"
} > "$RUN/run_config.txt"

arm() {
  local eng=$1
  [ -s "$RUN/$eng/game.html" ] && { log "skip done $eng"; return; }
  log "start $eng"
  uv run --no-project --python 3.12 --with playwright --with pillow \
    python escalation/build_game.py --engine "$eng" --brief "$BRIEF" --out "$RUN/$eng" \
    > "$RUN/$eng.log" 2>&1
  log "done  $eng (exit $?): $(tail -1 "$RUN/$eng.log")"
}

for eng in $ENGINES; do arm "$eng"; done

# --- shuffle the finished games for blind judging ----------------------------------------
python3 - "$RUN" $ENGINES <<'PY'
import json, os, random, shutil, sys
run, engines = sys.argv[1], sys.argv[2:]
have = [e for e in engines
        if os.path.exists(os.path.join(run, e, "game.html"))
        and os.path.getsize(os.path.join(run, e, "game.html")) > 0]
random.shuffle(have)
key = {}
for letter, eng in zip("ABCDEFGH", have):
    shutil.copy(os.path.join(run, eng, "game.html"), os.path.join(run, "play", f"game-{letter}.html"))
    key[letter] = eng
json.dump(key, open(os.path.join(run, "play", ".key.json"), "w"), indent=1)
print(f"{len(have)} game(s) ready to play blind in {run}/play: " + ", ".join(sorted(key)))
PY
n=$(ls "$RUN/play"/game-*.html 2>/dev/null | wc -l | tr -d ' ')
if [ "$n" -ge 2 ]; then
  level=info; msg="$n dungeon games ready to play blind in $RUN/play"
else
  level=warn; msg="Finished with only $n game(s); see $RUN/*.log"
fi
log "$msg"
stentor notify -l "$level" -s gvs5h "Dungeon game builds finished" "$msg" > /dev/null 2>&1 || true
rm -f "$RUN/run.pid"
log "finished"
