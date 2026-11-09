"""HTTP, worker lifecycle and streaming tests; no model files or network downloads."""
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
from tools import server as api
from tools.model_catalog import MODELS, profile

FAKE_WORKER = r'''
import json, os, select, struct, sys, time
def send(value):
    print(json.dumps(value), flush=True)
def read_exact(count):
    result=b''
    while len(result)<count:
        part=os.read(0,count-len(result))
        if not part: return None
        result+=part
    return result
backend=sys.argv[sys.argv.index('--backend')+1] if '--backend' in sys.argv else 'cpu'
compute={'backend':'metal' if backend.startswith('metal') else 'cpu','device':'Synthetic GPU','gpu_dispatches':0,'errors':0}
send({'event':'ready','protocol':1,'model':'plamo-3-translate-8b','revision':'19ecab3fb31424ee937b7503730318b50e55771c','worker_pid':os.getpid(),'load_seconds':0.01,'preload_seconds':0.005,
      'compute':compute,
      'threads':int(sys.argv[sys.argv.index('--threads')+1]) if '--threads' in sys.argv else 4,
      'memory':{'mode':'resident','weight_bytes':128,'preloaded_weight_bytes':128}})
while True:
    header=read_exact(16)
    if header is None: break
    op,direction,limit,length=struct.unpack('<4I',header)
    if op==3: continue
    raw=read_exact(length) if length else b''
    text=raw.decode()
    if text=='crash': sys.exit(2)
    if text=='invalid':
        send({'event':'error','code':'invalid_input','message':'synthetic invalid prompt'})
        continue
    if op==2:
        send({'event':'tokenized','prompt_tokens':len(text),'token_ids':list(range(len(text)))})
        continue
    memory = {'mode':'adaptive','weight_bytes':128,'preloaded_weight_bytes':0,
              'budget_bytes':64,'cached_weight_bytes':0,'staging_bytes':16} if text=='adaptive' else None
    send({'event':'prefill','prompt_tokens':len(text), **({'memory':memory} if memory else {})})
    output='こんにちは。'.encode()
    chunks=[output[:2],output[2:5],output[5:8],output[8:]]
    reason='stop'; generated=0
    for i,chunk in enumerate(chunks):
        if text=='slow': time.sleep(0.08)
        if select.select([0],[],[],0)[0]:
            cancel=read_exact(16)
            if cancel is None or struct.unpack('<4I',cancel)[0]==3:
                reason='cancelled'; break
        send({'event':'token','token_id':500+i,'hex':chunk.hex()})
        generated+=1
    if memory: memory.update(budget_bytes=48, cached_weight_bytes=32)
    if compute['backend']=='metal': compute['gpu_dispatches']+=4
    send({'event':'done','finish_reason':reason, 'compute':compute, **({'memory':memory} if memory else {}), 'usage':{'prompt_tokens':len(text),'completion_tokens':generated,'total_tokens':len(text)+generated},
          'timings':{'prefill_seconds':0.01,'decode_seconds':0.02,'total_seconds':0.03},
          'weight_io':{'read_calls':0,'staged_bytes':0},'verification':{'logits_hash':'private-test-field'}})
'''


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="plamo http ")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        executable = root / "fake-worker"
        executable.write_text(f"#!{sys.executable}\n" + FAKE_WORKER)
        executable.chmod(0o755)
        self.worker = api.PersistentWorker(executable, root, startup_timeout=5)
        self.addCleanup(self.worker.close)
        deadline = time.monotonic() + 5
        while self.worker.health()["status"] == "loading" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.worker.health()["status"], "ready")
        self.server = api.TranslationServer(("127.0.0.1", 0), self.worker)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)
        self.port = self.server.server_port

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def request(self, method="POST", path="/translate", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        payload = json.dumps(body if body is not None else {"text": "Hello."}, ensure_ascii=False).encode()
        connection.request(method, path, body=payload if method == "POST" else None,
                           headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        data = response.read()
        status, response_headers = response.status, dict(response.getheaders())
        connection.close()
        return status, response_headers, data

    def test_json_translation_reuses_the_same_worker(self):
        before = self.worker.health()["worker_pid"]
        for _ in range(2):
            status, headers, data = self.request(body={"text": "日本語 😀", "source_lang": "ja", "target_lang": "en"})
            result = json.loads(data)
            self.assertEqual(status, 200)
            self.assertEqual(int(headers["Content-Length"]), len(data))
            self.assertEqual(result["translation"], "こんにちは。")
            self.assertNotIn("verification", result)
            self.assertEqual(result["memory_mode"], "resident")
        self.assertEqual(self.worker.health()["worker_pid"], before)

    def test_compute_threads_are_forwarded_and_reported_in_health(self):
        root = Path(self.temp.name)
        worker = api.PersistentWorker(root / 'fake-worker', root, startup_timeout=5, threads=2)
        self.addCleanup(worker.close)
        deadline = time.monotonic() + 5
        while worker.health()['status'] == 'loading' and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(worker.health()['status'], 'ready')
        self.assertEqual(worker.health()['threads'], 2)
        status, _, payload = self.request('GET', '/health')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)['threads'], 4)
        self.assertEqual(json.loads(payload)['preload_seconds'], 0.005)
        self.assertEqual(json.loads(payload)['memory']['preloaded_weight_bytes'], 128)

    def test_metal_selection_and_live_compute_counters(self):
        root = Path(self.temp.name)
        worker = api.PersistentWorker(root / 'fake-worker', root, startup_timeout=5, backend='metal')
        self.addCleanup(worker.close)
        deadline = time.monotonic() + 5
        while worker.health()['status'] == 'loading' and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(worker.health()['status'], 'ready')
        self.assertEqual(worker.health()['compute']['backend'], 'metal')
        worker.run(api.validate_request({'text': 'Hello.'}))
        self.assertEqual(worker.health()['compute']['gpu_dispatches'], 4)

    def test_sse_reassembles_utf8_split_between_tokens(self):
        status, headers, payload = self.request(body={"text": "Hello.", "stream": True})
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/event-stream"))
        events = []
        for frame in payload.decode().strip().split("\n\n"):
            name, value = frame.split("\n", 1)
            events.append((name[7:], json.loads(value[6:])))
        self.assertEqual(events[0][0], "prefill")
        self.assertEqual("".join(value["text"] for name, value in events if name == "token"), "こんにちは。")
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[-1][1]["translation"], "こんにちは。")

    def test_health_and_translation_reflect_adaptive_memory_updates(self):
        status, _, payload = self.request(body={"text": "adaptive"})
        result = json.loads(payload)
        self.assertEqual(status, 200)
        self.assertEqual(result["memory_mode"], "adaptive")
        self.assertEqual(result["memory"]["budget_bytes"], 48)
        status, _, payload = self.request("GET", "/health")
        self.assertEqual(status, 200)
        memory = json.loads(payload)["memory"]
        self.assertEqual(memory["mode"], "adaptive")
        self.assertEqual(memory["cached_weight_bytes"], 32)
        self.assertEqual(memory["budget_bytes"], 48)

    def test_health_remains_responsive_and_busy_requests_get_429(self):
        responses = []
        thread = threading.Thread(target=lambda: responses.append(self.request(body={"text": "slow"})))
        thread.start()
        deadline = time.monotonic() + 3
        while not self.worker.health()["busy"] and time.monotonic() < deadline:
            time.sleep(0.005)
        status, _, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(data)["busy"])
        status, headers, _ = self.request()
        self.assertEqual(status, 429)
        self.assertEqual(headers["Retry-After"], "1")
        thread.join(timeout=5)
        self.assertEqual(responses[0][0], 200)

    def test_token_count_endpoint(self):
        status, _, data = self.request(path="/tokenize", body={"text": "Hello."})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["prompt_tokens"], 6)

    def test_loading_health_and_request_do_not_wait_for_the_model(self):
        with self.worker._state_lock:
            self.worker._state = "loading"
        status, _, data = self.request("GET", "/health")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(data)["status"], "loading")
        self.assertEqual(self.request()[0], 503)
        with self.worker._state_lock:
            self.worker._state = "ready"

    def test_request_validation(self):
        bad = [None, [], {}, {"text": ""}, {"text": "x\0y"}, {"text": "\ud800"},
               {"text": "x" * 4097}, {"text": "x", "source_lang": "fr"},
               {"text": "x", "max_tokens": True}, {"text": "x", "max_tokens": 0},
               {"text": "x", "max_tokens": 4097}, {"text": "x", "stream": "false"},
               {"text": "x", "unexpected": 1}]
        for body in bad:
            with self.subTest(body=repr(body)[:70]):
                with self.assertRaises(api.APIError):
                    api.validate_request(body)

    def test_input_failure_does_not_poison_the_worker(self):
        self.assertEqual(self.request(body={"text": "invalid"})[0], 400)
        self.assertEqual(self.request()[0], 200)

    def test_crashed_worker_is_unavailable_without_request_restarts(self):
        self.assertEqual(self.request(body={"text": "crash"})[0], 503)
        status, _, payload = self.request("GET", "/health")
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(payload)["status"], "failed")
        self.assertEqual(self.request()[0], 503)

    def test_cancellation_drains_the_old_response_before_the_next_request(self):
        def disconnected(_name, _event):
            raise BrokenPipeError("test client disconnected")
        request = api.validate_request({"text": "slow"})
        with self.assertRaises(api.Disconnected):
            self.worker.run(request, emit=disconnected)
        result = self.worker.run(api.validate_request({"text": "Hello."}))
        self.assertEqual(result["translation"], "こんにちは。")

    def test_host_and_origin_checks(self):
        self.assertEqual(self.request(headers={"Host": "example.invalid"})[0], 403)
        self.assertEqual(self.request(headers={"Origin": "https://example.invalid"})[0], 403)
        self.server.api_key = "test-key"
        self.assertEqual(self.request()[0], 401)
        self.assertEqual(self.request(headers={"Authorization": "Bearer test-key"})[0], 200)

    def test_request_framing_and_json_errors(self):
        requests = [
            (b'Content-Length: 2\r\nContent-Type: application/json\r\n', b'{x', 400),
            (b'Content-Length: 2\r\nContent-Type: text/plain\r\n', b'{}', 415),
            (b'Content-Length: 70000\r\nContent-Type: application/json\r\n', b'', 413),
            (b'Content-Type: application/json\r\n', b'', 411),
            (b'Transfer-Encoding: chunked\r\n', b'', 400),
            (b'Content-Length: 2\r\nContent-Length: 2\r\n', b'{}', 411),
        ]
        for headers, payload, expected in requests:
            with self.subTest(expected=expected, headers=headers):
                with socket.create_connection(("127.0.0.1", self.port), timeout=5) as connection:
                    connection.sendall(b'POST /translate HTTP/1.1\r\nHost: 127.0.0.1\r\n' + headers + b'\r\n' + payload)
                    response = http.client.HTTPResponse(connection)
                    response.begin()
                    self.assertEqual(response.status, expected)
                    response.read()


class NativeWorkerArgumentsTests(unittest.TestCase):
    def test_missing_model_explains_the_selected_download(self):
        import subprocess
        for size in MODELS:
            worker = profile(size).build_directory(ROOT) / 'plamo-worker'
            if not worker.exists():
                self.skipTest('run make models first')
            with self.subTest(model=size), tempfile.TemporaryDirectory() as folder:
                result = subprocess.run([worker], cwd=folder, capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                event = json.loads(result.stdout)
                self.assertEqual(event['event'], 'error')
                self.assertIn('--model ' + size, event['message'])

    def test_wrong_model_worker_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            executable = root / 'wrong-worker'
            executable.write_text(f'#!{sys.executable}\n' + FAKE_WORKER)
            executable.chmod(0o755)
            worker = api.PersistentWorker(executable, root, startup_timeout=5, model_profile=profile('31b'))
            try:
                deadline = time.monotonic() + 5
                while worker.health()['status'] == 'loading' and time.monotonic() < deadline:
                    time.sleep(0.01)
                health = worker.health()
                self.assertEqual(health['status'], 'failed')
                self.assertEqual(health['model'], 'plamo-3-translate-31b')
                self.assertIn('worker model/revision', health['error'])
            finally:
                worker.close()

    def test_native_help_needs_no_model(self):
        import subprocess
        worker = ROOT / "build/plamo-worker"
        if not worker.exists():
            self.skipTest("build the native worker first")
        result = subprocess.run([worker, "--help"], cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("resident", result.stdout)

    def test_server_rejects_a_different_revision_before_starting_worker(self):
        import subprocess
        for size in MODELS:
            with self.subTest(model=size), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                marker = root / profile(size).directory / "REVISION"
                marker.parent.mkdir(parents=True)
                marker.write_text("another-revision\n")
                result = subprocess.run([sys.executable, ROOT / "tools/server.py", "--model", size,
                                         "--worker", sys.executable, "--data-dir", root], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("revision does not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
