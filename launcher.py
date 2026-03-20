"""
Quantum Projects Hub -- Launcher
================================
Serves the dashboard and starts Pygame simulations.

Launch works via GET or POST: /launch/<id>
  schrodinger, bloch, cards, qa-gradient, kitaev  (or 1–5)

Usage:  python3 launcher.py
"""

import http.server
import json
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

PORT = 8000
PROJECT_DIR = Path(__file__).resolve().parent

# Primary slugs -> script
PROJECTS = {
    "schrodinger": {"script": "schrodinger_evolution.py", "name": "Schrödinger Evolution"},
    "bloch": {"script": "bloch_sphere.py", "name": "Bloch Sphere"},
    "cards": {"script": "cards_superposition_entanglement.py", "name": "Cards: Superposition & Entanglement"},
    "qa-gradient": {"script": "quantum_vs_gradient.py", "name": "QA vs Gradient Descent"},
    "kitaev": {"script": "kitaev_chain.py", "name": "Kitaev Chain"},
}

NUMERIC = {
    "1": "schrodinger",
    "2": "bloch",
    "3": "cards",
    "4": "qa-gradient",
    "5": "kitaev",
}

running_processes: dict[str, subprocess.Popen] = {}


# Extra URL aliases → canonical slug (covers bookmarks / old links)
SLUG_ALIASES = {
    "superposition": "cards",
    "entanglement": "cards",
    "deck": "cards",
    "playing-cards": "cards",
    "cards-superposition": "cards",
}


def _resolve_key(raw: str) -> str | None:
    raw = (raw or "").strip().lower()
    if not raw:
        return None
    if raw in NUMERIC:
        raw = NUMERIC[raw]
    raw = SLUG_ALIASES.get(raw, raw)
    return raw if raw in PROJECTS else None


def _launch_slug_from_path(path_only: str) -> str | None:
    """Extract slug from /launch/<slug> only — avoids /launcher.py matching /launch."""
    segs = [s for s in path_only.strip("/").split("/") if s]
    if len(segs) >= 2 and segs[0].lower() == "launch":
        return segs[1]
    return None


def _drain_request_body(handler):
    try:
        n = int(handler.headers.get("Content-Length", 0) or 0)
        if n > 0:
            handler.rfile.read(n)
    except Exception:
        pass


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def _try_launch(self) -> bool:
        path_only = urlparse(self.path).path
        slug = _launch_slug_from_path(path_only)
        if slug is None:
            return False
        key = _resolve_key(unquote(slug))
        if not key:
            valid = ", ".join(sorted(PROJECTS.keys()))
            self._json_response(
                404,
                {
                    "status": "error",
                    "error": f"Unknown launch target {slug!r}. Valid: {valid}",
                },
            )
            return True

        info = PROJECTS[key]
        script = PROJECT_DIR / info["script"]
        if not script.exists():
            self._json_response(
                404,
                {"status": "error", "error": f"Missing file: {info['script']}"},
            )
            return True

        old = running_processes.get(key)
        if old and old.poll() is None:
            self._json_response(
                200,
                {"status": "already_running", "name": info["name"]},
            )
            return True

        # Pygame needs the same environment as the GUI session. On macOS,
        # start_new_session + closed stdio often prevents a window from appearing.
        env = os.environ.copy()
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT_DIR),
                env=env,
                stdin=subprocess.DEVNULL,
                # Inherit stdout/stderr so crashes show in the terminal running launcher.py
            )
        except OSError as e:
            self._json_response(
                500,
                {"status": "error", "error": str(e)},
            )
            return True

        # If the script exits immediately (syntax error, pygame init), report it
        time.sleep(0.2)
        code = proc.poll()
        if code is not None:
            self._json_response(
                500,
                {
                    "status": "error",
                    "error": f"Simulation exited immediately (code {code}). Run in Terminal: python3 {info['script']}",
                },
            )
            return True

        running_processes[key] = proc
        print(f"  Launched {info['name']} -> {script} (pid {proc.pid})")
        self._json_response(
            200,
            {"status": "launched", "name": info["name"], "pid": proc.pid},
        )
        return True

    def do_GET(self):
        _drain_request_body(self)
        if self._try_launch():
            return
        super().do_GET()

    def do_POST(self):
        _drain_request_body(self)
        if self._try_launch():
            return
        self._json_response(404, {"status": "error", "error": "Not found"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "GET /favicon" not in msg:
            print(f"  {msg}")


def main():
    os.chdir(PROJECT_DIR)
    server = http.server.HTTPServer(("", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Quantum Projects Hub: {url}")
    print("Open that URL and click Launch — Pygame windows open separately.\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for _k, proc in running_processes.items():
            if proc.poll() is None:
                proc.terminate()
        server.server_close()


if __name__ == "__main__":
    main()
