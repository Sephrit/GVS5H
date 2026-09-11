#!/usr/bin/env python3
"""Browser check for a single-file three.js game -- the game task's stand-in for sample tests.

Opens <ws>/game.html in headless Chrome and runs four checks, returning the shape of
multiagent._run_samples plus a ready-made feedback line for the manager:

  1. loads    -- no uncaught errors or console errors in the first seconds
  2. hook     -- window.game exposes player {x,y,z}, rooms (8-12), hasKey/won false at start
  3. renders  -- a canvas is on the page and the screenshot is not blank
  4. moves    -- holding W, A, S or D changes window.game.player, without new errors

    python3 game_check.py path/to/game.html      # prints the result as JSON

Needs playwright and pillow (uv run --with playwright --with pillow) and Google Chrome.
"""
import io
import json
import os
import sys

NAME = "BROWSER CHECK"
SETTLE_MS = 4000
VIEW = {"width": 1024, "height": 640}
ENV_NOTE = ("The automated browser cannot lock the pointer, so it clicks the canvas and holds "
            "W/A/S/D without a lock; mouse look, collision, the key, the exit door and the "
            "minimap are not tested -- judge those from the code.")


def _benign(msg):
    m = msg.lower()
    return ("pointer" in m and "lock" in m) or "favicon" in m


def _numeric(v):
    return all(isinstance(x, (int, float)) for x in v)


def check_file(path):
    from playwright.sync_api import sync_playwright
    from PIL import Image

    checks, errors = [], []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        try:
            page = browser.new_page(viewport=VIEW)
            page.on("pageerror", lambda e: errors.append(f"uncaught {e}"))
            page.on("console", lambda m: m.type == "error" and errors.append(f"console.error: {m.text}"))
            try:
                page.goto("file://" + os.path.abspath(path), wait_until="load", timeout=30000)
            except Exception as e:  # noqa: BLE001 - reported to the manager as a load failure
                errors.append(f"page did not load: {e}")
            page.wait_for_timeout(SETTLE_MS)
            at_load = [e for e in errors if not _benign(e)]
            add("loads", not at_load,
                "no errors" if not at_load else "; ".join(dict.fromkeys(at_load))[:700])

            g = page.evaluate("""() => { const g = window.game; if (!g) return null;
                const p = g.player || {};
                return {player: [p.x, p.y, p.z], rooms: g.rooms, hasKey: g.hasKey, won: g.won}; }""")
            if g is None:
                add("hook", False, "window.game is not defined")
            else:
                probs = []
                if not _numeric(g["player"]):
                    probs.append(f"window.game.player is not a numeric {{x,y,z}} (got {g['player']})")
                if not isinstance(g["rooms"], (int, float)) or not 8 <= g["rooms"] <= 12:
                    probs.append(f"window.game.rooms should be 8-12 (got {g['rooms']!r})")
                if g["hasKey"] is not False:
                    probs.append(f"window.game.hasKey should start false (got {g['hasKey']!r})")
                if g["won"] is not False:
                    probs.append(f"window.game.won should start false (got {g['won']!r})")
                add("hook", not probs, "; ".join(probs) or "ok")

            has_canvas = page.evaluate("() => !!document.querySelector('canvas')")
            px = Image.open(io.BytesIO(page.screenshot())).convert("L").tobytes()
            lit = sum(1 for v in px if v > 20) / len(px)
            if not has_canvas:
                add("renders", False, "there is no <canvas> on the page")
            else:
                add("renders", lit >= 0.01, f"{lit:.1%} of the screen is visibly lit"
                    + ("" if lit >= 0.01 else " -- the view is black or blank"))

            if g is not None and _numeric(g["player"]):
                page.mouse.click(VIEW["width"] // 2, VIEW["height"] // 2)
                moved = {}
                for key in ("KeyW", "KeyA", "KeyS", "KeyD"):
                    before = page.evaluate("() => [window.game.player.x, window.game.player.z]")
                    page.keyboard.down(key)
                    page.wait_for_timeout(700)
                    page.keyboard.up(key)
                    page.wait_for_timeout(100)
                    after = page.evaluate("() => [window.game.player.x, window.game.player.z]")
                    moved[key[-1]] = round(((after[0] - before[0]) ** 2
                                            + (after[1] - before[1]) ** 2) ** 0.5, 3)
                late = [e for e in errors if not _benign(e) and e not in at_load]
                ok = any(d > 0.01 for d in moved.values()) and not late
                detail = f"distance moved per key: {moved}"
                if not any(d > 0.01 for d in moved.values()):
                    detail += " -- holding W/A/S/D did not move the player"
                if late:
                    detail += " -- errors while moving: " + "; ".join(dict.fromkeys(late))[:400]
                add("moves", ok, detail)
            else:
                add("moves", False, "not tested: window.game.player is missing")
        finally:
            browser.close()
    return checks


def check_game(ws, artifact="game.html"):
    path = os.path.join(ws, artifact)
    if not os.path.exists(path) or not open(path).read().strip():
        return {"ran": False}
    try:
        checks = check_file(path)
    except Exception as e:  # noqa: BLE001 - a broken check must not kill the loop
        return {"ran": False, "error": f"{type(e).__name__}: {e}"}
    passed = sum(c["ok"] for c in checks)
    fails = [c for c in checks if not c["ok"]]
    if not fails:
        feedback = (f"[{NAME}: PASSED all {len(checks)} checks -- "
                    + "; ".join(f"{c['name']}: {c['detail']}" for c in checks)
                    + f". {ENV_NOTE}] ")
    else:
        feedback = (f"[{NAME}: FAILED -- passed {passed}/{len(checks)}. The current game is BROKEN. "
                    + " ".join(f"({i}) {c['name']}: {c['detail']}." for i, c in enumerate(fails, 1))
                    + " Fix these first, or switch to a different approach if this keeps failing. "
                    + f"{ENV_NOTE}] ")
    return {"ran": True, "passed": passed, "total": len(checks), "fail": fails[0] if fails else None,
            "feedback": feedback, "checks": checks}


if __name__ == "__main__":
    target = os.path.abspath(sys.argv[1])
    print(json.dumps(check_game(os.path.dirname(target), os.path.basename(target)), indent=2))
