#!/usr/bin/env python3
"""Score every game a run produced and write a comparison with screenshots.

    python3 escalation/game_report.py runs/dungeon

For each <run>/<engine>/game.html: re-runs the browser check, saves two screenshots (on load,
and after holding W for two seconds), and reads the arm's summary.json for calls, tokens and
wall time. Writes <run>/report/report.md plus the PNGs.

The checks are a floor, not a verdict: they say the page loads, draws something, exposes the
hook and responds to keys. Whether the dungeon is any fun is for a person to judge, which is
what <run>/play is for.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VIEW = {"width": 1024, "height": 640}


def shots(path, out_dir, name):
    """Save two screenshots; return their filenames and the lit fraction of each."""
    from playwright.sync_api import sync_playwright
    from PIL import Image
    made = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page(viewport=VIEW)
            page.goto("file://" + os.path.abspath(path), wait_until="load", timeout=30000)
            page.wait_for_timeout(4000)
            for tag in ("load", "moved"):
                if tag == "moved":
                    page.mouse.click(VIEW["width"] // 2, VIEW["height"] // 2)
                    page.keyboard.down("KeyW")
                    page.wait_for_timeout(2000)
                    page.keyboard.up("KeyW")
                    page.wait_for_timeout(200)
                png = page.screenshot()
                fn = f"{name}-{tag}.png"
                open(os.path.join(out_dir, fn), "wb").write(png)
                px = Image.open(io.BytesIO(png)).convert("L").tobytes()
                made.append((fn, sum(1 for v in px if v > 20) / len(px)))
        except Exception as e:  # noqa: BLE001 - a dead page still belongs in the report
            made.append((f"(screenshot failed: {type(e).__name__})", 0.0))
        finally:
            browser.close()
    return made


def main():
    from game_check import check_game
    run = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "runs/dungeon")
    out_dir = os.path.join(run, "report")
    os.makedirs(out_dir, exist_ok=True)
    engines = sorted(d for d in os.listdir(run)
                     if os.path.exists(os.path.join(run, d, "game.html")))
    rows = []
    for eng in engines:
        d = os.path.join(run, eng)
        res = check_game(d)
        summary = {}
        if os.path.exists(os.path.join(d, "summary.json")):
            summary = json.load(open(os.path.join(d, "summary.json")))
        pics = shots(os.path.join(d, "game.html"), out_dir, eng)
        rows.append((eng, summary, res, pics))

    lines = [f"# Dungeon game comparison: {os.path.basename(run)}", "",
             "| Scaffold | Checks | Calls | Tokens | Minutes | Lines |", "|---|---|---|---|---|---|"]
    for eng, s, res, _ in rows:
        checks = f"{res['passed']}/{res['total']}" if res.get("ran") else "did not run"
        lines.append(f"| {eng} | {checks} | {s.get('calls', '?')} | {s.get('completion_tokens', '?'):,} "
                     f"| {s.get('minutes', '?')} | {s.get('game_lines', '?')} |"
                     if isinstance(s.get("completion_tokens"), int) else
                     f"| {eng} | {checks} | {s.get('calls', '?')} | {s.get('completion_tokens', '?')} "
                     f"| {s.get('minutes', '?')} | {s.get('game_lines', '?')} |")
    for eng, s, res, pics in rows:
        lines += ["", f"## {eng}", ""]
        for c in (res.get("checks") or []):
            lines.append(f"- **{c['name']}**: {'pass' if c['ok'] else 'FAIL'} — {c['detail']}")
        if res.get("error"):
            lines.append(f"- check error: {res['error']}")
        for fn, lit in pics:
            lines.append("")
            lines.append(f"![{eng} {fn}]({fn}) — {lit:.1%} of the frame lit"
                         if fn.endswith(".png") else f"- {fn}")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(lines) + "\n")
    print(f"wrote {os.path.join(out_dir, 'report.md')} covering {len(rows)} arm(s)")


if __name__ == "__main__":
    main()
