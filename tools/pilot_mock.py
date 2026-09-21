"""Loopback-only shared client fixtures. No real Google calls, accounts, corpus or PDFs."""

import argparse
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "contracts/fixtures"
WEB = ROOT / "backend/elderhelp/web"


def fixture(name):
    return json.loads((FIXTURES / f"{name}-v2.json").read_text())


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        files = {
            "/": "index.html",
            "/sw.js": "sw.js",
            **{
                "/assets/" + name: name
                for name in ["style.css", "app.mjs", "core.mjs", "storage.mjs"]
            },
        }
        if path in files:
            data = (WEB / files[path]).read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/javascript"
                if path.endswith(".mjs")
                else mimetypes.guess_type(files[path])[0],
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/healthz":
            return self.respond({"status": "mock"})
        if self.headers.get("Authorization") != "Bearer mock-local-pilot-token":
            return self.respond({"detail": "Mock invite required"}, 401)
        if path == "/v2/capabilities":
            return self.respond(fixture("capabilities"))
        if path in ["/v2/reports", "/v1/reports"]:
            return self.respond(fixture("reports"))
        if path.startswith("/v2/reports/"):
            return self.respond(fixture("report"))
        self.send_error(404)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/v2/demo/session":
            if data.get("invite_code") != "mock-invite-code-only":
                return self.respond({}, 401)
            return self.respond(
                {"token": "mock-local-pilot-token", "expires_at": int(time.time()) + 7200}
            )
        if self.headers.get("Authorization") != "Bearer mock-local-pilot-token":
            return self.respond({}, 401)
        if self.path == "/v2/search":
            return self.respond(fixture("search"))
        if self.path != "/v2/answers/stream":
            return self.send_error(404)
        if data.get("question") == "quota":
            return self.respond({}, 429)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        result = fixture("complete")
        result["request_id"] = str(uuid4())
        events = [
            ("start", {"request_id": result["request_id"]}),
            ("progress", fixture("progress")),
        ]
        if data.get("question") == "failure":
            events.append(("error", fixture("error")))
        else:
            events.extend([("delta", {"text": result["answer_markdown"]}), ("complete", result)])
        try:
            for name, value in events:
                if name == "delta":
                    time.sleep(1 if data.get("question") == "slow" else 0.1)
                frame = f"event: {name}\r\ndata: {json.dumps(value)}\r\n\r\n".encode()
                for start in range(0, len(frame), 37):
                    self.wfile.write(frame[start : start + 37])
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def respond(self, data, status=200):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        if status == 429:
            self.send_header("Retry-After", "60")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
