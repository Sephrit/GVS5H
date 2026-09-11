#!/usr/bin/env python3
"""Summarize a local trial next to the paper's own Qwen3.8-27B runs on the same problems.

    python3 escalation/local_trial_summary.py [runs/<tag>]

Reads runs/<tag>/results/{single,multiagent}/<qid>.json (one file per problem, written by
run_bench_script/run_local_trial_lmstudio.sh) and the per-call token counts in each
workspace's transcript.jsonl, then prints a markdown table. The paper columns are pass counts
out of 5 from runs/firstparty-128k-reasoning-on-5pass (patched grader; the single arm is the
128k cap-matched replay), so "paper 0/5 -> 5/5" is what the loop did there.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PAPER = os.path.join(ROOT, "runs", "firstparty-128k-reasoning-on-5pass", "results")
ENGINES = ("single", "multiagent")


def paper_counts(pattern):
    counts = {}
    for f in sorted(glob.glob(os.path.join(PAPER, pattern))):
        for r in json.load(open(f))["lcb"]["records"]:
            counts[r["question_id"]] = counts.get(r["question_id"], 0) + bool(r["passed"])
    return counts


def tokens(ws):
    tr = os.path.join(ws or "", "transcript.jsonl")
    if not ws or not os.path.exists(tr):
        return None
    total = 0
    for line in open(tr):
        rec = json.loads(line)
        if not rec.get("_meta"):
            total += rec.get("completion_tokens") or 0
    return total


def main():
    run = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "local-trial"))
    ids = json.load(open(os.path.join(HERE, "local_trial_ids.json")))
    paper = {"single": paper_counts("q38_single_p?.cap128k.patched.json"),
             "multiagent": paper_counts("q38_multiagent_p?.patched.json")}
    rows, done = [], {e: [] for e in ENGINES}
    for qid in ids:
        cells = {}
        for eng in ENGINES:
            f = os.path.join(run, "results", eng, f"{qid}.json")
            if not os.path.exists(f):
                cells[eng] = None
                continue
            rec = json.load(open(f))["lcb"]["records"][0]
            cells[eng] = rec
            done[eng].append(bool(rec["passed"]))
        rows.append((qid, cells))

    def cell(rec):
        if rec is None:
            return "not run"
        tok = tokens(rec.get("ws"))
        extra = [f"{tok / 1000:.0f}k tok" if tok is not None else None,
                 f"{rec['n_calls']} calls" if rec.get("n_calls") else None,
                 rec.get("status") if rec.get("status") != "ok" else None]
        return ("PASS" if rec["passed"] else "FAIL") + " (" + ", ".join(x for x in extra if x) + ")"

    print(f"# Local trial: {os.path.basename(run)}\n")
    print("| Problem | Paper single | Paper manager | Local single | Local manager |")
    print("|---|---|---|---|---|")
    for qid, cells in rows:
        print(f"| {qid} | {paper['single'].get(qid, '?')}/5 | {paper['multiagent'].get(qid, '?')}/5 "
              f"| {cell(cells['single'])} | {cell(cells['multiagent'])} |")
    print()
    for eng in ENGINES:
        print(f"- Local {eng}: {sum(done[eng])}/{len(done[eng])} passed ({len(ids) - len(done[eng])} not run)")


if __name__ == "__main__":
    main()
