#!/usr/bin/env python3
"""Measure every way this Mac can serve Qwen3.8-27B, at matched settings.

    python3 escalation/serving_matrix.py [--out runs/serving-matrix] [--only 8bit]

Each row is one (weights, server, MTP) combination. For each: decode tok/s on a fixed prompt,
prefill tok/s on a 24k-token prompt, and a greedy 400-token sample kept for comparison against
the others -- a quant that diverges early from the 16-bit sample is the interesting signal, and
MTP on vs off on the same weights should not diverge at all.

oMLX rows serve MLX weights with every quality-for-speed shortcut off (turboquant KV,
specprefill, thinking budget) so only the quantization differs. LM Studio rows run GGUF through
llama.cpp, which is the only way to reach the BF16 weights on this machine; that is a different
engine, so compare GGUF rows with each other, not against MLX rows.

Writes <out>/results.json and <out>/matrix.md. Needs the oMLX app and the lms CLI.
"""
import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OMLX_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"
MLX_DIR = os.path.expanduser("~/AI/Models/oMLX/sephwa")
KEY = "matrix-test"
GAME_PROMPT = "Write a complete three.js first-person maze game in one HTML file."
FIB_PROMPT = "Write a Python function fib(n) using fast doubling. Reply with only the code."

ROWS = [
    ("oMLX 4-bit MLX + MTP", "omlx", "Qwen3.8-27B-4bit-MTP", True),
    ("oMLX 4-bit MLX", "omlx", "Qwen3.8-27B-4bit-MTP", False),
    ("oMLX 8-bit MLX + MTP", "omlx", "Qwen3.8-27B-8bit-MTP", True),
    ("oMLX 8-bit MLX", "omlx", "Qwen3.8-27B-8bit-MTP", False),
    ("LM Studio Q8 GGUF + MTP", "lms", "qwen3.8-27b@q8_k_xl", True),
    ("LM Studio Q8 GGUF", "lms", "qwen3.8-27b@q8_k_xl", False),
    ("LM Studio BF16 GGUF + MTP", "lms", "qwen3.8-27b@bf16", True),
    ("LM Studio BF16 GGUF", "lms", "qwen3.8-27b@bf16", False),
]


def free_port(start=8010):
    for p in range(start, start + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise SystemExit("no free port in range")


def post(base, model, messages, max_tokens, temperature, key):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        base + "/chat/completions", json.dumps(body).encode(), headers), timeout=3600))
    return r, time.time() - t


def measure(base, model, key):
    out = {}
    post(base, model, [{"role": "user", "content": "hi"}], 8, 0.2, key)      # load / warm up
    r, dt = post(base, model, [{"role": "user", "content": GAME_PROMPT}], 1500, 0.2, key)
    out["decode_tok_s"] = round(r["usage"]["completion_tokens"] / dt, 1)
    out["decode_tokens"] = r["usage"]["completion_tokens"]
    filler = "Room notes: " + " ".join(
        f"corridor {i} joins room {i % 11} at tile ({i % 40},{i % 23})." for i in range(1400))
    r, dt = post(base, model, [{"role": "user", "content": filler + "\nReply with just OK."}], 1, 0.2, key)
    out["prefill_tok_s"] = round(r["usage"]["prompt_tokens"] / dt, 0)
    r, dt = post(base, model, [{"role": "user", "content": FIB_PROMPT}], 400, 0, key)
    m = r["choices"][0]["message"]
    out["greedy"] = ((m.get("reasoning_content") or "") + " " + (m.get("content") or "")).strip()
    return out


def omlx_row(model, mtp, out_dir):
    iso = os.path.join(out_dir, "omlx-" + model + ("-mtp" if mtp else ""))
    os.makedirs(os.path.join(iso, "models"), exist_ok=True)
    link = os.path.join(iso, "models", model)
    if not os.path.islink(link):
        os.symlink(os.path.join(MLX_DIR, model), link)
    port = free_port()
    s = json.load(open(os.path.expanduser("~/.omlx/settings.json")))
    s["server"].update({"port": port, "auto_start_on_launch": False})
    s["model"]["model_dirs"] = [os.path.join(iso, "models")]
    json.dump(s, open(os.path.join(iso, "settings.json"), "w"), indent=2)
    ms = json.load(open(os.path.expanduser("~/.omlx/model_settings.json")))["models"]
    src = ms.get("Qwen3.8-27B-MTPLX-Optimized-Speed") or next(iter(ms.values()))
    e = json.loads(json.dumps(src))
    e.update({"max_context_window": 262144, "max_tokens": 131072, "ttl_seconds": 86400,
              "mtp_enabled": bool(mtp), "mtp_num_draft_tokens": 3, "enable_thinking": True,
              "turboquant_kv_enabled": False, "specprefill_enabled": False,
              "thinking_budget_enabled": False})
    e.pop("model_alias", None)
    e["chat_template_kwargs"] = {}
    json.dump({"version": 1, "models": {model: e}},
              open(os.path.join(iso, "model_settings.json"), "w"), indent=2)
    env = dict(os.environ, OMLX_BASE_PATH=iso)
    proc = subprocess.Popen([OMLX_CLI, "serve", "--model-dir", os.path.join(iso, "models"),
                             "--port", str(port), "--base-path", iso, "--api-key", KEY],
                            env=env, stdout=open(os.path.join(iso, "serve.log"), "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{port}/v1"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(urllib.request.Request(
                    base + "/models", headers={"Authorization": "Bearer " + KEY}), timeout=2)
                break
            except Exception:
                time.sleep(2)
        return measure(base, model, KEY)
    finally:
        subprocess.run(["pkill", "-P", str(proc.pid)], capture_output=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        time.sleep(5)


def lms_row(model, mtp):
    subprocess.run(["lms", "unload", "--all"], capture_output=True)
    flag = "--speculative-draft-mtp" if mtp else "--no-speculative-draft-mtp"
    load = subprocess.run(["lms", "load", model, "-c", "262144", "--parallel", "1", flag, "-y"],
                          capture_output=True, text=True)
    if "successfully" not in (load.stdout + load.stderr):
        return {"error": "load failed: " + (load.stderr or load.stdout)[-200:]}
    try:
        return measure("http://localhost:1234/v1", model, "")
    finally:
        subprocess.run(["lms", "unload", "--all"], capture_output=True)
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "..", "runs", "serving-matrix"))
    ap.add_argument("--only", default="", help="substring filter on the row name")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    path = os.path.join(out_dir, "results.json")
    if os.path.exists(path):
        results = json.load(open(path))          # resume: finished rows are kept
    for name, kind, model, mtp in ROWS:
        if args.only and args.only not in name:
            continue
        if name in results and "error" not in results[name]:
            print(f"skip done: {name}", flush=True)
            continue
        print(f"\n=== {name}", flush=True)
        t0 = time.time()
        try:
            row = omlx_row(model, mtp, out_dir) if kind == "omlx" else lms_row(model, mtp)
        except Exception as e:                   # noqa: BLE001 - a dead row belongs in the table
            row = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        row.update({"weights": model, "server": kind, "mtp": bool(mtp),
                    "minutes": round((time.time() - t0) / 60, 1)})
        results[name] = row
        json.dump(results, open(path, "w"), indent=1)
        print("   " + json.dumps({k: v for k, v in row.items() if k != "greedy"}), flush=True)

    norm = lambda s: re.sub(r"\s+", " ", s or "").strip()
    ref_name = next((n for n in ("LM Studio BF16 GGUF", "oMLX 8-bit MLX") if norm(results.get(n, {}).get("greedy"))), None)
    ref = norm(results.get(ref_name, {}).get("greedy")) if ref_name else ""
    lines = ["# Serving matrix: Qwen3.8-27B on this Mac", "",
             f"Greedy agreement is measured against **{ref_name or 'n/a'}**: how far two answers "
             "match before the first differing character. MTP on vs off on the same weights "
             "should be identical; quants differing early is the quality signal.", "",
             "| Setup | Decode tok/s | Prefill tok/s | Agreement with reference | Minutes |",
             "|---|---:|---:|---:|---:|"]
    for name, _, _, _ in ROWS:
        r = results.get(name)
        if not r:
            continue
        if "error" in r:
            lines.append(f"| {name} | failed | | {r['error'][:60]} | {r.get('minutes', '')} |")
            continue
        got = norm(r.get("greedy"))
        if ref and got:
            i = next((n for n in range(min(len(got), len(ref))) if got[n] != ref[n]), min(len(got), len(ref)))
            agree = "identical" if got == ref else f"{100 * i / max(1, len(ref)):.0f}% of chars"
        else:
            agree = "-"
        lines.append(f"| {name} | {r['decode_tok_s']} | {r['prefill_tok_s']:.0f} | {agree} | {r['minutes']} |")
    open(os.path.join(out_dir, "matrix.md"), "w").write("\n".join(lines) + "\n")
    print(f"\nwrote {os.path.join(out_dir, 'matrix.md')}")


if __name__ == "__main__":
    main()
