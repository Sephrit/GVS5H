#!/usr/bin/env python3
"""Build a single-file browser game with one of five scaffolds, all on the same model.

    build_game.py --engine <engine> --brief escalation/dungeon_brief.md --out runs/dungeon/<engine>

    single            one call, no feedback -- the paper's baseline
    multiagent        the paper's manager loop, with the browser check between rounds
    multiagent-nocheck  the same loop with no check at all: decomposition and shared notes only
    refine            one worker looped against the check -- feedback without a manager
    bestof3           three independent single calls, keep whichever passes most checks

The loop is multiagent.py as the paper runs it, pointed at a different artifact: workers write
game.html instead of solution.py, and ground truth is game_check's headless-browser check
instead of the public stdin tests. Writes <out>/game.html, <out>/summary.json and the
workspace under <out>/ws/.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SOLVER_SYSTEM = (
    "an expert game developer who writes complete, polished browser games with three.js. "
    "Build exactly what the brief asks for, and make it work on the first load. "
    "Output EXACTLY ONE complete, self-contained HTML file inside a single ```html ...``` "
    "fenced block, and nothing else after it."
)


def _tokens(ws):
    tr = os.path.join(ws, "transcript.jsonl")
    calls = toks = 0
    if os.path.exists(tr):
        for line in open(tr):
            rec = json.loads(line)
            if not rec.get("_meta"):
                calls += 1
                toks += rec.get("completion_tokens") or 0
    return calls, toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True,
                    choices=["single", "multiagent", "multiagent-nocheck", "refine", "bestof3"])
    ap.add_argument("--brief", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    os.environ["MULTIAGENT_WS"] = os.path.join(out, "ws")   # multiagent reads it at import
    import multiagent
    import game_engines
    from game_check import check_game, NAME

    spec = {"kind": "code", "domain": "building browser games with three.js", "lang": "html",
            "artifact": "game.html", "check_name": NAME, "verify": check_game,
            # the worker prompt reads "You are a WORKER subagent, <solver_system>"
            "solver_system": "You are " + SOLVER_SYSTEM}
    if args.engine == "multiagent-nocheck":
        spec.pop("verify")

    solve = {"single": multiagent.single_solve,
             "multiagent": multiagent.multiagent_solve,
             "multiagent-nocheck": multiagent.multiagent_solve,
             "refine": game_engines.refine_solve,
             "bestof3": game_engines.bestofn_solve}[args.engine]
    brief = open(args.brief).read()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    status, t0 = {}, time.time()
    log(f"engine={args.engine} model={multiagent.MODEL} max_iters={multiagent.MAX_ITERS}")
    raw = solve(brief, spec, log=log, status_out=status)
    html = multiagent._extract_py(raw, "html")
    if not html.strip() and status.get("ws"):      # loop arms leave the artifact on disk
        html = multiagent._read(status["ws"], "game.html")
    if html.strip():
        with open(os.path.join(out, "game.html"), "w") as f:
            f.write(html)

    result = check_game(out)
    calls, toks = _tokens(status.get("ws", ""))
    summary = {"engine": args.engine, "model": multiagent.MODEL,
               "minutes": round((time.time() - t0) / 60, 1), "calls": calls,
               "completion_tokens": toks, "truncated_calls": status.get("truncated_calls"),
               "game_lines": html.count("\n") + 1 if html.strip() else 0,
               "check": {k: result.get(k) for k in ("ran", "passed", "total", "checks", "error")}}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    verdict = f"{result['passed']}/{result['total']} checks" if result.get("ran") else "no game produced"
    log(f"finished: {calls} calls, {toks} tokens, {summary['minutes']} min, {verdict}")


if __name__ == "__main__":
    main()
