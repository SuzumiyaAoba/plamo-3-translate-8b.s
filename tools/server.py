#!/usr/bin/env python3
"""Local HTTP translation API backed by one persistent native model worker."""
import argparse
import codecs
from contextlib import contextmanager
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
try:
    from .model_catalog import add_model_argument, profile
except ImportError:
    from model_catalog import add_model_argument, profile

MAX_BODY = 65536
MAX_EVENT = 262144


class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class Disconnected(Exception):
    pass


class PersistentWorker:
    """Own the process, protocol, request lock and cancellation in one place."""
    def __init__(self, executable, data_dir, memory="auto", startup_timeout=180, model_profile=None, threads=None, backend="auto"):
        self.executable, self.data_dir, self.memory = str(executable), str(data_dir), memory
        self.profile = model_profile or profile()
        self.startup_timeout = startup_timeout
        self.threads = threads
        self.backend = backend
        self._state_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._state, self._error, self._info = "loading", None, None
        self._process = None
        self._closed = False
        self._thread = threading.Thread(target=self._initialize, daemon=True)
        self._thread.start()

    def _read_event(self):
        line = self._process.stdout.readline(MAX_EVENT + 1)
        if not line or len(line) > MAX_EVENT or not line.endswith(b"\n"):
            raise RuntimeError("model worker stopped or returned an invalid event")
        event = json.loads(line)
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise RuntimeError("invalid model worker event")
        return event

    def _initialize(self):
        timer = None
        try:
            with self._state_lock:
                if self._closed:
                    return
                command = [self.executable, "--memory", self.memory]
                if self.threads is not None:
                    command.extend(["--threads", str(self.threads)])
                if self.backend != "auto":
                    command.extend(["--backend", self.backend])
                self._process = subprocess.Popen(
                    command, cwd=self.data_dir,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                )
            timer = threading.Timer(self.startup_timeout, self._process.kill)
            timer.daemon = True
            timer.start()
            event = self._read_event()
            if event["event"] == "error":
                raise RuntimeError(event.get("message", "model startup failed"))
            if event["event"] != "ready" or event.get("protocol") != 1:
                raise RuntimeError("unsupported model worker protocol")
            if event.get("model") != self.profile.name or event.get("revision") != self.profile.revision:
                raise RuntimeError("worker model/revision does not match --model; rebuild or select the matching worker")
            with self._state_lock:
                if self._closed:
                    return
                self._info, self._state = event, "ready"
            memory = event.get("memory", {})
            print(f"plamo-api: model ready; memory={memory.get('mode')}; "
                  f"preloaded={memory.get('preloaded_weight_bytes', 0):,} bytes; "
                  f"cache budget={memory.get('budget_bytes', 0):,} bytes; "
                  f"threads={event.get('threads', 1)}; "
                  f"backend={event.get('compute', {}).get('backend', 'cpu')}", file=sys.stderr, flush=True)
            if memory.get("note"):
                print(f"plamo-api: {memory['note']}", file=sys.stderr, flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            self._fail(str(error))
        finally:
            if timer:
                timer.cancel()

    def _fail(self, message):
        with self._state_lock:
            if self._closed:
                return
            self._state, self._error = "failed", message
        process = self._process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print(f"plamo-api: {message}", file=sys.stderr, flush=True)

    def health(self):
        with self._state_lock:
            state, info, error = self._state, self._info, self._error
        if state == "ready" and self._process.poll() is not None:
            self._fail("model worker exited; restart the server")
            state, error = "failed", "model worker exited; restart the server"
        result = {"status": state, "busy": self._request_lock.locked(), "model": self.profile.name,
                  "repository": self.profile.repository, "revision": self.profile.revision}
        if info:
            result["worker_pid"] = info["worker_pid"]
            result["load_seconds"] = info["load_seconds"]
            result["preload_seconds"] = info.get("preload_seconds", 0)
            result["threads"] = info.get("threads", 1)
            result["compute"] = info.get("compute", {"backend": "cpu"})
            result["memory"] = info["memory"]
            result["prompt_cache"] = info.get("prompt_cache", {})
        if error:
            result["error"] = error
        return result

    @contextmanager
    def _session(self):
        if self.health()["status"] != "ready":
            raise APIError(503, "model_not_ready", "model is loading or unavailable; inspect /health")
        if not self._request_lock.acquire(blocking=False):
            raise APIError(429, "model_busy", "one request is already running; retry when it finishes")
        try:
            if self.health()["status"] != "ready":
                raise APIError(503, "model_not_ready", "model worker is unavailable")
            yield
        finally:
            self._request_lock.release()

    def _send(self, op, direction=0, max_tokens=0, text=""):
        payload = text.encode("utf-8")
        self._process.stdin.write(struct.pack("<4I", op, direction, max_tokens, len(payload)) + payload)
        self._process.stdin.flush()

    def _cancel(self):
        try:
            self._send(3)
        except OSError:
            pass

    def run(self, request, tokenize=False, begin=None, emit=None):
        with self._session():
            if begin:
                begin()
            direction = 1 if request["source_lang"] == "en" else 2
            try:
                self._send(2 if tokenize else 1, direction, request["max_tokens"], request["text"])
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                parts = []
                disconnected = False

                def send_event(name, event):
                    nonlocal disconnected, emit
                    if emit:
                        try:
                            emit(name, event)
                        except OSError:
                            disconnected, emit = True, None
                            self._cancel()

                while True:
                    event = self._read_event()
                    kind = event["event"]
                    updates = {key: event[key] for key in ("memory", "compute") if key in event}
                    if updates:
                        with self._state_lock:
                            self._info = {**self._info, **updates}
                    if kind == "error":
                        status = 400 if event.get("code") == "invalid_input" else 500
                        raise APIError(status, event.get("code", "inference_failed"), event.get("message", "inference failed"))
                    if tokenize and kind == "tokenized":
                        return {key: value for key, value in event.items() if key != "event"}
                    if kind == "prefill":
                        send_event("prefill", {"prompt_tokens": event["prompt_tokens"]})
                    elif kind == "token":
                        text = decoder.decode(bytes.fromhex(event["hex"]))
                        parts.append(text)
                        send_event("token", {"text": text, "token_id": event["token_id"]})
                    elif kind == "done":
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            parts.append(tail)
                            send_event("token", {"text": tail})
                        if disconnected:
                            raise Disconnected()
                        result = {key: value for key, value in event.items() if key not in ("event", "verification")}
                        result.update(translation="".join(parts), model=self.profile.name,
                                      source_lang=request["source_lang"], target_lang=request["target_lang"],
                                      memory_mode=self.health()["memory"]["mode"])
                        return result
                    elif kind not in ("prefill", "token"):
                        raise RuntimeError("unexpected model worker response")
            except (OSError, ValueError, RuntimeError) as error:
                self._fail(str(error))
                raise APIError(503, "worker_failed", "model worker failed; inspect /health") from error

    def close(self):
        with self._state_lock:
            self._closed, self._state = True, "closed"
            process = self._process
        if process:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        self._thread.join(timeout=5)


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
