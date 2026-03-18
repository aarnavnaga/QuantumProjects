"""
Quantum Projects Hub -- Launcher
================================
Serves the dashboard and starts Pygame simulations.

Launch works via GET or POST: /launch/<id>
  schrodinger, bloch, qa-gradient, kitaev  (or 1, 2, 3, 4)

Usage:  python3 launcher.py
"""

import http.server
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

PORT = 8000
PROJECT_DIR = Path(__file__).resolve().parent

# Primary slugs -> script
PROJECTS = {
    "schrodinger": {"script": "schrodinger_evolution.py", "name": "Schrödinger Evolution"},
    "bloch": {"script": "bloch_sphere.py", "name": "Bloch Sphere"},
    "qa-gradient": {"script": "quantum_vs_gradient.py", "name": "QA vs Gradient Descent"},
    "kitaev": {"script": "kitaev_chain.py", "name": "Kitaev Chain"},
}

# Numeric aliases (same order as dashboard: foundations first, then advanced)
NUMERIC = {"1": "schrodinger", "2": "bloch", "3": "qa-gradient", "4": "kitaev"}

running_processes: dict[str, subprocess.Popen] = {}


def _resolve_key(raw: str) -> str | None:
    raw = (raw or "").strip()
    if raw in NUMERIC:
        raw = NUMERIC[raw]
    return raw if raw in PROJECTS else None


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
        if not path_only.startswith("/launch"):
            return False
        # /launch/foo or /launch/foo/
        tail = path_only[len("/launch") :].lstrip("/")
        key = _resolve_key(unquote(tail.split("/")[0]))
        if not key:
            self._json_response(
                404,
                {
                    "status": "error",
                    "error": f"Unknown launch target: {tail!r}. Use schrodinger, bloch, qa-gradient, or kitaev.",
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

        try:
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            self._json_response(
                500,
                {"status": "error", "error": str(e)},
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
