"""Persistent worker lifecycle, serialized requests and cancellation for the API."""
import codecs
from contextlib import contextmanager
import subprocess
import sys
import threading
try:
    from .model_catalog import profile
    from .worker_protocol import read_event, send
except ImportError:
    from model_catalog import profile
    from worker_protocol import read_event, send


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
            event = read_event(self._process)
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
            result["speculative"] = info.get("speculative", {"enabled": False, "block_tokens": 0})
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

    def _cancel(self):
        try:
            send(self._process, 3)
        except OSError:
            pass

    def run(self, request, tokenize=False, begin=None, emit=None):
        with self._session():
            if begin:
                begin()
            direction = 1 if request["source_lang"] == "en" else 2
            try:
                send(self._process, 2 if tokenize else 1, direction, request["max_tokens"], request["text"])
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
                    event = read_event(self._process)
                    kind = event["event"]
                    updates = {key: event[key] for key in ("memory", "compute", "speculative") if key in event}
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
