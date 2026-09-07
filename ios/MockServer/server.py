from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time


REPORT_ID = "1aa2870c-bf64-4f29-94c2-e4d670a36d2d"
REPORT = {
    "id": REPORT_ID,
    "slug": "age-friendly-nyc",
    "title": "Age-friendly NYC",
    "publisher": "City of New York",
    "community": "New York City",
    "publication_date": "2017-01-01",
    "source_url": "https://example.com/report.pdf",
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/reports":
            self._json({"items": [REPORT], "total": 1})
            return
        if self.path == f"/v1/reports/{REPORT_ID}" or self.path == f"/v1/reports/{REPORT_ID.upper()}":
            self._json({**REPORT, "description": "A city plan.", "suggested_questions": ["What supports aging in place?"], "page_count": 92})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/answers/stream":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            ("start", {"request_id": "9aa2870c-bf64-4f29-94c2-e4d670a36d2d"}),
            ("delta", {"text": "Communities can support older adults "}),
            ("delta", {"text": "with safe housing [S1]."}),
            ("complete", {
                "request_id": "9aa2870c-bf64-4f29-94c2-e4d670a36d2d",
                "answer_markdown": "Communities can support older adults with safe housing [S1].",
                "status": "grounded",
                "citations": [{
                    "id": "S1", "report_id": REPORT_ID, "report_title": REPORT["title"],
                    "publisher": REPORT["publisher"], "source_url": REPORT["source_url"],
                    "publication_date": REPORT["publication_date"], "page_number": 12,
                    "excerpt": "The plan describes safe housing support."
                }],
            }),
        ]
        for name, payload in events:
            data = f"event: {name}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
            midpoint = len(data) // 2
            self.wfile.write(data[:midpoint])
            self.wfile.flush()
            time.sleep(0.03)
            self.wfile.write(data[midpoint:])
            self.wfile.flush()

    def _json(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
