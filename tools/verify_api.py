#!/usr/bin/env python3
"""Verify persistent worker IDs and optional real-model request isolation."""
import argparse
import json
from pathlib import Path
import random
import statistics
import subprocess
from model_catalog import add_model_argument, profile
from worker_protocol import check_memory, close_worker, receive, send

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--report', type=Path, help='write real-model readiness, timings and verification hashes as JSON')
    parser.add_argument('--worker', type=Path, help='use a saved worker executable for a controlled comparison')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--text', default='Hello.', help='en->ja translation probe')
    parser.add_argument('--direction', choices=('en-ja', 'ja-en'), default='en-ja')
    parser.add_argument('--max-tokens', type=int, default=2)
    parser.add_argument("--translate", action="store_true", help="repeat the translation probe; compare request hashes and the fixed 8B Hello. baseline")
    parser.add_argument("--memory", choices=("auto", "resident", "stream"), default="auto")
    parser.add_argument("--threads", type=int, choices=range(1, 65), metavar="1..64")
    parser.add_argument("--backend", choices=("auto", "cpu", "metal", "metal-fp32"))
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.max_tokens <= 4096:
        parser.error('--repeats must be positive and --max-tokens must be in 1..4096')
    p = profile(args.model)
    directory = p.build_directory(ROOT)
    data_root = p.data_root()
    command = [str(args.worker.resolve() if args.worker else directory / "plamo-worker"), "--memory", args.memory]
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    if args.backend is not None:
        command.extend(["--backend", args.backend])
    if not args.translate:
        command.append("--tokenizer-only")
    process = subprocess.Popen(command, cwd=data_root, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    report = {'model': p.name, 'text': args.text, 'direction': args.direction, 'max_tokens': args.max_tokens, 'requests': []}
    try:
        ready = receive(process)
        assert ready["event"] == "ready", ready
        assert ready["model"] == p.name and ready['revision'] == p.revision, ready
        check_memory(ready["memory"])
        metal = ready.get('compute', {}).get('backend') == 'metal'
        if args.translate and args.backend in ('metal', 'metal-fp32'):
            assert metal, ready
        print(f"worker ready: {ready['memory']}", flush=True)
        report['ready'] = ready
        cases = ["", "Hello", "こんにちは😀", "a  b<|plamo:bos|>日本語", "aaaaa", "日日日日", "\n\n\n\n", "𠮷野家", "<|plamo:unknown|>", "<|plamo:|>", "a\uee00b"]
        rng = random.Random(4)
        for _ in range(20):
            cases.append("".join(rng.choice("abc日東京 \n😀") * rng.randrange(1, 7) for _ in range(10)))
        for text in cases:
            expected = subprocess.check_output([str(directory / "plamo-asm"), text], cwd=data_root, text=True)
            native = subprocess.check_output([str(directory / "plamo"), "--", text], cwd=data_root, text=True)
            assert native == expected, text
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
            check_memory(cancelled["memory"])
            print("cancel at token boundary: PASS", flush=True)
            first = None
            for i in range(args.repeats):
                send(process, 1, 1 if args.direction == 'en-ja' else 2, args.max_tokens, args.text)
                output = bytearray()
                token_ids = []
                while True:
                    event = receive(process)
                    if event["event"] == "token":
                        output.extend(bytes.fromhex(event["hex"]))
                        token_ids.append(event['token_id'])
                    elif event["event"] == "done":
                        check_memory(event["memory"])
                        if event["memory"]["mode"] == "adaptive":
                            assert event["memory"]["max_budget_bytes"] == ready["memory"]["max_budget_bytes"], event
                        if p.size == '8b' and args.text == 'Hello.' and args.direction == 'en-ja' and args.max_tokens >= 2:
                            assert output.decode() == "こんにちは。" and event["finish_reason"] == "stop", event
                            if not metal:
                                assert event["verification"] == expected, event
                        if metal:
                            assert event['compute']['gpu_dispatches'] > 0, event
                            assert not event['compute']['errors'] and not event['compute']['cpu_fallbacks'], event
                        actual = (bytes(output), token_ids, event['verification'])
                        if first is None:first = actual
                        assert actual == first, event
                        report['requests'].append(dict(translation=output.decode(), token_ids=token_ids, **event))
                        print(f"{p.name} request {i + 1}: repeated request hashes PASS ({ready['memory']['mode']}); {output.decode()!r}", flush=True)
                        break
            report['seconds_median'] = {
                key: statistics.median(r['timings'][key] for r in report['requests'])
                for key in ('prefill_seconds', 'decode_seconds', 'first_token_seconds', 'total_seconds')
            }
            print(f"median seconds: {report['seconds_median']}", flush=True)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    finally:
        close_worker(process)


if __name__ == "__main__":
    main()
