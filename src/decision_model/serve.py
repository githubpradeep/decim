"""Local /v1/systemone server. One state, many typed questions, pointer-head probabilities."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from decision_model.decide import decide, load_model
from decision_model.schema import DecisionError


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a pointer-head checkpoint on /v1/systemone.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/qwen/checkpoint.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()
    model, vocab, config = load_model(args.checkpoint)
    infer_lock = threading.Lock()

    pages = {
        "/": "snake_page.html",
        "/snake": "snake_page.html",
        "/demo": "snake_page.html",
        "/intents": "demo_page.html",
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            name = pages.get(self.path.split("?", 1)[0])
            if name is None:
                self._send(404, {"error": "not found"})
                return
            page = Path(__file__).with_name(name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                with infer_lock:
                    body = decide(model, vocab, payload, config.max_len, config.name)
            except DecisionError as exc:
                self._send(exc.status, {"error": str(exc)})
                return
            except json.JSONDecodeError:
                self._send(400, {"error": "request body is not JSON"})
                return
            self._send(200, body)

        def log_message(self, fmt: str, *log_args: object) -> None:
            return

        def _send(self, status: int, body: dict) -> None:
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"demo http://{args.host}:{args.port}/")
    print(f"pointer head listening on http://{args.host}:{args.port}/v1/systemone")
    server.serve_forever()


if __name__ == "__main__":
    main()
