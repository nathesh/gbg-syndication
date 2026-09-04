"""
Vercel Python function. GET / runs the syndication pipeline over the demo
fixtures and returns published reviews, the human-review queue, and the
per-stage log. The pipeline itself lives in ../gangnam_syndication_workflow.py
and has no dependencies beyond the standard library.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gangnam_syndication_workflow import run_demo  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(run_demo(), ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
