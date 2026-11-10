#!/usr/bin/env python3
"""Local HTTP translation API backed by one persistent native model worker."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
from urllib.parse import urlsplit
try:
    from .model_catalog import add_model_argument, profile
    from .worker_client import APIError, Disconnected, PersistentWorker
except ImportError:
    from model_catalog import add_model_argument, profile
    from worker_client import APIError, Disconnected, PersistentWorker

MAX_BODY = 65536



def validate_request(body):
    if not isinstance(body, dict):
        raise APIError(400, "invalid_request", "request must be a JSON object")
    unknown = body.keys() - {"text", "source_lang", "target_lang", "max_tokens", "stream"}
    if unknown:
        raise APIError(400, "invalid_request", "unknown request field")
    text = body.get("text")
    if not isinstance(text, str) or not text or "\0" in text or len(text) > 4096:
        raise APIError(400, "invalid_text", "text must be nonempty, contain no NUL, and fit within 4096 Unicode scalars including the prompt")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise APIError(400, "invalid_text", "text contains invalid Unicode") from error
    if len(encoded) > 16384:
        raise APIError(400, "invalid_text", "text exceeds the UTF-8 byte limit")
    source, target = body.get("source_lang", "en"), body.get("target_lang", "ja")
    if (source, target) not in (("en", "ja"), ("ja", "en")):
        raise APIError(400, "invalid_language", "supported pairs are en to ja and ja to en")
    count, stream = body.get("max_tokens", 512), body.get("stream", False)
    if type(count) is not int or not 1 <= count <= 4096:
        raise APIError(400, "invalid_max_tokens", "max_tokens must be an integer in 1..4096")
    if type(stream) is not bool:
        raise APIError(400, "invalid_stream", "stream must be a boolean")
    return {"text": text, "source_lang": source, "target_lang": target, "max_tokens": count, "stream": stream}


class TranslationServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, worker, api_key=None):
        self.worker, self.api_key = worker, api_key
        super().__init__(address, TranslationHandler)


class TranslationHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "plamo-api/0.2"
    disable_nagle_algorithm = True  # Send each small SSE token without waiting for another packet.

    def setup(self):
        super().setup()
        self.connection.settimeout(15)
        self.streaming = False

    def log_message(self, *_args):
        pass  # Request text, credentials and URL query strings are never logged.

    def _authenticate(self):
        key = self.server.api_key
        if key:
            supplied = self.headers.get("Authorization", "").encode("utf-8")
            if not hmac.compare_digest(supplied, ("Bearer " + key).encode("utf-8")):
                raise APIError(401, "unauthorized", "a valid Bearer token is required")
        else:
            try:
                host = urlsplit("//" + self.headers.get("Host", "")).hostname
            except ValueError:
                host = None
            if host not in ("localhost", "127.0.0.1", "::1"):
                raise APIError(403, "invalid_host", "unauthenticated access requires a loopback Host")
            if self.headers.get("Origin"):
                raise APIError(403, "origin_not_allowed", "browser cross-origin requests are not enabled")

    def _json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        if status == 429:
            self.send_header("Retry-After", "1")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)

    def _error(self, error):
        value = {"error": {"code": error.code, "message": error.message}}
        if self.streaming:
            self._event("error", value)
        else:
            self._json(error.status, value)

    def _begin_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.streaming = self.close_connection = True
        self.wfile.flush()

    def _event(self, name, event):
        text = json.dumps(event, ensure_ascii=False, allow_nan=False)
        self.wfile.write(f"event: {name}\ndata: {text}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        try:
            self._authenticate()
            if self.path != "/health":
                raise APIError(404, "not_found", "use GET /health or POST /translate")
            health = self.server.worker.health()
            self._json(200 if health["status"] == "ready" else 503, health)
        except APIError as error:
            self._error(error)
        except OSError:
            self.close_connection = True

    def do_POST(self):
        try:
            self._authenticate()
            if self.path not in ("/translate", "/tokenize", "/v1/translate"):
                raise APIError(404, "not_found", "unknown API endpoint")
            if self.headers.get("Transfer-Encoding"):
                raise APIError(400, "invalid_framing", "chunked request bodies are not supported")
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                raise APIError(411, "length_required", "one valid Content-Length header is required")
            length = int(lengths[0])
            if not 0 < length <= MAX_BODY:
                raise APIError(413, "body_too_large", "request body must fit within 64 KiB")
            if self.headers.get_content_type() != "application/json":
                raise APIError(415, "content_type", "use Content-Type: application/json")
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise APIError(400, "incomplete_body", "incomplete request body")
            try:
                body = json.loads(payload.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as error:
                raise APIError(400, "invalid_json", "body must contain valid UTF-8 JSON") from error
            request = validate_request(body)
            tokenize = self.path == "/tokenize"
            stream = request["stream"] and not tokenize
            result = self.server.worker.run(request, tokenize=tokenize,
                                            begin=self._begin_stream if stream else None,
                                            emit=self._event if stream else None)
            if stream:
                self._event("done", result)
            else:
                self._json(200, result)
        except APIError as error:
            try:
                self._error(error)
            except OSError:
                pass
        except socket.timeout:
            try:
                self._error(APIError(408, "request_timeout", "request body timed out"))
            except OSError:
                pass
        except (OSError, Disconnected):
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, help="model data root (default: PLAMO_DATA_DIR or current directory)")
    parser.add_argument("--memory", choices=("auto", "resident", "stream"), default="auto",
                        help="auto preloads the BF16 cache within available memory, with a 2 GiB reserve")
    parser.add_argument("--threads", type=int, choices=range(1, 65), metavar="1..64",
                        help="CPU compute threads (default: performance cores); 1 disables parallelism")
    parser.add_argument("--backend", choices=("auto", "cpu", "metal", "metal-fp32"), default="auto",
                        help="auto prefers Metal FP32; metal-fp32 is an alias of metal; cpu uses assembly")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    try:
        p = profile(args.model)
    except ValueError as error:
        parser.error(str(error))
    api_key = os.environ.get("PLAMO_API_KEY") or None
    if args.host not in ("127.0.0.1", "localhost") and not api_key:
        parser.error("set PLAMO_API_KEY before binding to a non-loopback address")
    if not 1 <= args.port <= 65535 or not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        parser.error("port must be 1..65535 and startup-timeout must be positive")
    worker_path = args.worker
    if worker_path is None:
        worker_path = (args.worker_directory / ("plamo-worker-" + p.size) if args.worker_directory else
                       p.build_directory(Path(__file__).resolve().parents[1]) / "plamo-worker")
    worker_path, data_dir = worker_path.resolve(), (args.data_dir or p.data_root(args.cache)).resolve()
    if not worker_path.is_file() or not data_dir.is_dir():
        parser.error("worker or data directory is missing; build the worker and download model files first")
    marker = data_dir / p.directory / "REVISION"
    if marker.exists() and marker.read_text().strip() != p.revision:
        parser.error("model revision does not match this implementation; select the pinned model directory")
    worker = PersistentWorker(worker_path, data_dir, args.memory, args.startup_timeout, p, args.threads, args.backend)
    previous_term = None
    try:
        def stop(_signum, _frame):
            raise KeyboardInterrupt
        previous_term = signal.signal(signal.SIGTERM, stop)
        with TranslationServer((args.host, args.port), worker, api_key) as server:
            print(f"plamo-api: listening on http://{args.host}:{args.port}; check /health for model readiness", file=sys.stderr, flush=True)
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except OSError as error:
        parser.exit(1, f"plamo-api: {error}\n")
    finally:
        worker.close()
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
