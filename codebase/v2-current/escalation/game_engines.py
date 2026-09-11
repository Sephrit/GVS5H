"""Cheaper scaffolds to compare against the paper's manager loop.

refine_solve   one worker looped against the browser check -- no manager, no notes, no task
               list. Separates "iterate on a failing check" from "decompose and coordinate".
bestofn_solve  N independent single calls, each checked once, keep the best. Separates
               "sample more" from "iterate at all".

Both reuse multiagent.py's workspace, transcript and extraction helpers, so every arm writes
the same artifacts and build_game.py can treat them alike.
"""
import json
import os
import time

import multiagent as M

REFINE_ROUNDS = int(os.environ.get("REFINE_MAX_ROUNDS", "6"))
BESTOF_N = int(os.environ.get("BESTOF_N", "3"))


def _fresh_ws(problem, spec, engine, extra=()):
    ws = os.path.join(M.WS_ROOT, M._slug(problem))
    os.makedirs(ws, exist_ok=True)
    for f in ("task.md", "transcript.jsonl", M._artifact(spec)) + tuple(extra):
        M._write(ws, f, "")
    M._write(ws, "task.md", problem)
    M._record(ws, {"_meta": True, "t": time.time(), "model": M.MODEL, "engine": engine,
                   "kind": spec["kind"], "problem": problem})
    return ws


def _finish(ws, spec, status_out):
    if status_out is None:
        return
    recs = [json.loads(l) for l in open(os.path.join(ws, "transcript.jsonl"))]
    recs = [r for r in recs if not r.get("_meta")]
    status_out["ws"] = ws
    status_out["n_calls"] = len(recs)
    status_out["finish_reason"] = recs[-1].get("finish_reason") if recs else None
    status_out["truncated_calls"] = sum(1 for r in recs if r.get("finish_reason") == "length")
    if not M._read(ws, M._artifact(spec)).strip() and any(r.get("infra_exhausted") for r in recs):
        status_out["infra_fail"] = True


def _code_out(spec, code):
    return f"```{spec.get('lang', 'python')}\n{code}\n```" if code else ""


def refine_solve(problem, spec, log=None, status_out=None, tests=None):
    """One worker, rewritten against the check's verdict, until it passes or rounds run out."""
    log = log or (lambda *a, **k: None)
    lang, artifact = spec.get("lang", "python"), M._artifact(spec)
    ws = _fresh_ws(problem, spec, "refine")
    verify = spec.get("verify")
    check_name = spec.get("check_name", "SAMPLE TESTS")
    sysmsg = (
        f"You are {spec['solver_system']} "
        "You may be shown your previous attempt and the verdict an automated check gave it. "
        "If so, fix what the verdict reports, keeping everything that already works. "
        f"Respond with EXACTLY these sections:\n### CODE\n```{lang}\n"
        "<the FULL updated self-contained program>\n```\n### STATUS\n<solved|continue>"
    )
    feedback, code = "", ""
    for rnd in range(1, REFINE_ROUNDS + 1):
        cur = M._read(ws, artifact)
        meta = {}
        reply = M._chat(ws, f"refine:{rnd}", [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": (
                f"PROBLEM:\n{problem}\n\nYOUR CURRENT WORK:\n{cur or '(none yet)'}\n\n"
                f"{check_name} ON THAT WORK: {feedback or '(not run yet)'}"
            )},
        ], temperature=0.2, meta=meta)
        sec = M._sections(reply)
        new = M._extract_py(sec.get("CODE", ""), lang) or M._extract_py(reply, lang)
        if new:
            code = new
            M._write(ws, artifact, code)
        elif meta.get("finish_reason") == "length":
            log(f"    [refine] round {rnd} cut off at the token limit, no program")
            continue
        res = verify(ws) if (verify and code) else {"ran": False}
        if res.get("ran"):
            feedback = res.get("feedback") or ""
            log(f"    [refine] round {rnd}: {res['passed']}/{res['total']} checks passed")
            if res["passed"] == res["total"]:
                break
        else:
            log(f"    [refine] round {rnd}: check did not run"
                + (f" ({res['error']})" if res.get("error") else ""))
            if not verify:
                break
    _finish(ws, spec, status_out)
    return _code_out(spec, M._read(ws, artifact))


def bestofn_solve(problem, spec, log=None, status_out=None, tests=None):
    """N independent single calls; the one that passes the most checks wins."""
    log = log or (lambda *a, **k: None)
    lang, artifact = spec.get("lang", "python"), M._artifact(spec)
    ws = _fresh_ws(problem, spec, f"bestof{BESTOF_N}")
    verify = spec.get("verify")
    best = (-1, 0, "")     # checks passed, length, code
    for i in range(1, BESTOF_N + 1):
        reply = M._chat(ws, f"candidate:{i}", [
            {"role": "system", "content": spec["solver_system"]},
            {"role": "user", "content": problem},
        ], temperature=0.2)
        code = M._extract_py(reply, lang)
        if not code:
            log(f"    [bestof] candidate {i}: no {lang} block")
            continue
        M._write(ws, f"candidate_{i}.{lang}", code)
        M._write(ws, artifact, code)
        res = verify(ws) if verify else {"ran": False}
        passed = res.get("passed", 0) if res.get("ran") else 0
        log(f"    [bestof] candidate {i}: {passed}/{res.get('total', 0)} checks passed"
            if res.get("ran") else f"    [bestof] candidate {i}: check did not run")
        if (passed, len(code)) > (best[0], best[1]):
            best = (passed, len(code), code)
    M._write(ws, artifact, best[2])
    log(f"    [bestof] kept the candidate with {max(best[0], 0)} checks passed")
    _finish(ws, spec, status_out)
    return _code_out(spec, best[2])
