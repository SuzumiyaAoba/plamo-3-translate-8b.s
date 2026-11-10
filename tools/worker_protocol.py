"""Bounded JSONL events and binary requests for the private native worker pipe."""
import json
import struct
import subprocess

MAX_EVENT = 262144
HEADER = struct.Struct("<4I")


def read_event(process):
    line = process.stdout.readline(MAX_EVENT + 1)
    if not line or len(line) > MAX_EVENT or not line.endswith(b"\n"):
        raise RuntimeError("model worker stopped or returned an invalid event")
    event = json.loads(line)
    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
        raise RuntimeError("invalid model worker event")
    return event


def receive(process):
    """Verification tools fail immediately on a worker error event."""
    event = read_event(process)
    if event["event"] == "error":
        raise RuntimeError(event["message"])
    return event


def send(process, op, direction=0, limit=0, text=""):
    payload = text.encode("utf-8")
    process.stdin.write(HEADER.pack(op, direction, limit, len(payload)) + payload)
    process.stdin.flush()


def close_worker(process, timeout=5):
    """Finish an idle worker through EOF, and reap an unresponsive worker too."""
    try:
        process.stdin.close()
    except OSError:
        pass  # An exited worker has already closed its end of the pipe.
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        process.stdout.close()


def check_memory(memory):
    if memory["mode"] == "adaptive":
        assert 0 <= memory["preloaded_weight_bytes"] <= memory["weight_bytes"], memory
        assert 0 < memory["staging_bytes"] <= 64 * 1024 * 1024, memory
        assert memory["staging_bytes"] <= memory["budget_bytes"] <= memory["max_budget_bytes"], memory
        assert memory.get("cached_storage_bytes", memory["cached_weight_bytes"]) + memory["staging_bytes"] <= memory["budget_bytes"], memory
