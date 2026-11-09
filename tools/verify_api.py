#!/usr/bin/env python3
"""Verify persistent worker IDs and optional real-model request isolation."""
import argparse
import json
from pathlib import Path
import random
import struct
import subprocess
from model_catalog import add_model_argument, profile

ROOT = Path(__file__).resolve().parents[1]


def receive(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("worker exited unexpectedly")
    event = json.loads(line)
    if event["event"] == "error":
        raise RuntimeError(event["message"])
    return event


def send(process, op, direction, limit, text):
    payload = text.encode()
    process.stdin.write(struct.pack("<4I", op, direction, limit, len(payload)) + payload)
    process.stdin.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--report', type=Path, help='write real-model readiness, timings and verification hashes as JSON')
    parser.add_argument("--translate", action="store_true", help="run Hello. twice; compare request hashes and the fixed 8B baseline")
    parser.add_argument("--memory", choices=("auto", "resident", "stream"), default="auto")
    args = parser.parse_args()
    p = profile(args.model)
    directory = p.build_directory(ROOT)
    data_root = p.data_root()
    command = [str(directory / "plamo-worker"), "--memory", args.memory]
    if not args.translate:
        command.append("--tokenizer-only")
    process = subprocess.Popen(command, cwd=data_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    report = {'model': p.name, 'requests': []}
    try:
        ready = receive(process)
        assert ready["event"] == "ready", ready
        assert ready["model"] == p.name and ready['revision'] == p.revision, ready
        report['ready'] = ready
        cases = ["", "Hello", "こんにちは😀", "a  b<|plamo:bos|>日本語", "aaaaa", "日日日日", "\n\n\n\n", "𠮷野家", "<|plamo:unknown|>", "<|plamo:|>", "a\uee00b"]
        rng = random.Random(4)
        for _ in range(20):
            cases.append("".join(rng.choice("abc日東京 \n😀") * rng.randrange(1, 7) for _ in range(10)))
        for text in cases:
            expected = subprocess.check_output([str(directory / "plamo"), text], cwd=data_root, text=True)
            send(process, 2, 0, 0, text)
            actual = receive(process)
            assert actual["event"] == "tokenized" and actual["token_ids"] == list(map(int, expected.split())), text
        print(f"persistent tokenizer vs assembly CLI: PASS ({len(cases)} cases)", flush=True)
        if args.translate:
            expected = {"prefill_logits_hash": "c9e54cc1de2684c9", "prefill_state_hash": "b370bb3890a4400e",
                        "logits_hash": "68d5318fc63d9103", "state_hash": "a44087018da49b9d"}
            send(process, 1, 1, 2, "Hello.")
            assert receive(process)["event"] == "prefill"
            send(process, 3, 0, 0, "")
            cancelled = receive(process)
            assert cancelled["event"] == "done" and cancelled["finish_reason"] == "cancelled", cancelled
            assert cancelled["usage"]["completion_tokens"] == 0, cancelled
            print("cancel at token boundary: PASS", flush=True)
            first = None
            for i in range(2):
                send(process, 1, 1, 2, "Hello.")
                output = bytearray()
                token_ids = []
                while True:
                    event = receive(process)
                    if event["event"] == "token":
                        output.extend(bytes.fromhex(event["hex"]))
                        token_ids.append(event['token_id'])
                    elif event["event"] == "done":
                        if p.size == '8b':
                            assert output.decode() == "こんにちは。" and event["finish_reason"] == "stop", event
                            assert event["verification"] == expected, event
                        actual = (bytes(output), event['verification'])
                        if first is None:first = actual
                        assert actual == first, event
                        report['requests'].append(dict(translation=output.decode(), token_ids=token_ids, **event))
                        print(f"{p.name} request {i + 1}: repeated request hashes PASS ({ready['memory']['mode']}); {output.decode()!r}", flush=True)
                        break
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        process.stdout.close()


if __name__ == "__main__":
    main()
