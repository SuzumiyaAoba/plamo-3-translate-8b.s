"""The API and verification tools must agree on framing and bounded events."""
import io
import json
import struct
from types import SimpleNamespace
import unittest
from tools.worker_protocol import MAX_EVENT, read_event, receive, send


class WorkerProtocolTests(unittest.TestCase):
    def test_request_length_counts_utf8_bytes_and_cancel_has_no_body(self):
        stream = io.BytesIO()
        process = SimpleNamespace(stdin=stream)
        send(process, 1, 2, 17, "日本語😀")
        send(process, 3)
        frame = stream.getvalue()
        payload = "日本語😀".encode()
        self.assertEqual(struct.unpack("<4I", frame[:16]), (1, 2, 17, len(payload)))
        self.assertEqual(frame[16:16 + len(payload)], payload)
        self.assertEqual(frame[16 + len(payload):], struct.pack("<4I", 3, 0, 0, 0))

    def test_error_events_can_be_handled_by_api_or_raised_by_verifiers(self):
        event = {"event": "error", "code": "invalid_input", "message": "invalid prompt"}
        frame = json.dumps(event).encode() + b"\n"
        self.assertEqual(read_event(SimpleNamespace(stdout=io.BytesIO(frame))), event)
        with self.assertRaisesRegex(RuntimeError, "invalid prompt"):
            receive(SimpleNamespace(stdout=io.BytesIO(frame)))

    def test_invalid_or_incomplete_events_are_rejected(self):
        for frame in (b"", b'{}\n', b'[]\n', b'{"event":2}\n', b'{"event":"done"}', b'invalid\n'):
            with self.subTest(frame=frame):
                with self.assertRaises((RuntimeError, ValueError)):
                    read_event(SimpleNamespace(stdout=io.BytesIO(frame)))

    def test_large_events_never_consume_an_unbounded_line(self):
        stream = io.BytesIO(b"x" * (MAX_EVENT + 100))
        with self.assertRaises(RuntimeError):
            read_event(SimpleNamespace(stdout=stream))
        self.assertEqual(stream.tell(), MAX_EVENT + 1)

    def test_valid_event_leaves_next_event_for_next_read(self):
        process = SimpleNamespace(stdout=io.BytesIO(b'{"event":"ready"}\n{"event":"done"}\n'))
        self.assertEqual(receive(process)["event"], "ready")
        self.assertEqual(receive(process)["event"], "done")
