"""
Quantum Projects Hub -- Launcher
================================
Lightweight HTTP server that serves the landing page and launches
Pygame projects on button click.

Usage:  python3 launcher.py
"""

import http.server
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

PORT = 8000
PROJECT_DIR = Path(__file__).resolve().parent

PROJECTS = {
    "1": {"script": "schrodinger_evolution.py", "name": "Schrödinger Evolution"},
    "2": {"script": "bloch_sphere.py", "name": "Bloch Sphere"},
    "3": {"script": "quantum_vs_gradient.py", "name": "QA vs Gradient Descent"},
    "4": {"script": "kitaev_chain.py", "name": "Kitaev Chain"},
}

running_processes: dict[str, subprocess.Popen] = {}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_DIR), **kwargs)

    def do_POST(self):
        if self.path.startswith("/launch/"):
            project_id = self.path.split("/")[-1]
            if project_id not in PROJECTS:
                self._json_response(404, {"error": "Unknown project"})
                return

            info = PROJECTS[project_id]
            script = PROJECT_DIR / info["script"]
            if not script.exists():
                self._json_response(404, {"error": f"{info['script']} not found"})
                return

            old = running_processes.get(project_id)
            if old and old.poll() is None:
                self._json_response(200, {"status": "already_running", "name": info["name"]})
                return

            proc = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=str(PROJECT_DIR),
            )
            running_processes[project_id] = proc
            self._json_response(200, {"status": "launched", "name": info["name"], "pid": proc.pid})
        else:
            self._json_response(404, {"error": "Not found"})

    def _json_response(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "GET /favicon" not in msg:
            print(f"  {msg}")


def main():
    os.chdir(PROJECT_DIR)
    server = http.server.HTTPServer(("", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Quantum Projects Hub running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        for pid, proc in running_processes.items():
            if proc.poll() is None:
                proc.terminate()
        server.server_close()


if __name__ == "__main__":
    main()
