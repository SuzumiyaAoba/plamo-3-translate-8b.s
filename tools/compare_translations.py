#!/usr/bin/env python3
"""Compare complete greedy CPU/Metal translations, including every token ID."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
from model_catalog import add_model_argument, profile
from verify_api import receive, send

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (1, 'Hello.'),
    (1, 'The meeting starts at ten tomorrow morning.'),
    (1, 'Please save this document before closing the window.'),
    (2, 'こんにちは。'),
    (2, '会議は明日の午前10時に始まります。'),
    (2, 'この文書を保存してから、ウィンドウを閉じてください。'),
)


def run(p, backend, memory):
    process = subprocess.Popen([str(p.build_directory(ROOT) / 'plamo-worker'), '--backend', backend,
                                '--memory', memory, '--threads', '4'], cwd=p.data_root(),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    results = []
    try:
        ready = receive(process)
        assert ready['event'] == 'ready', ready
        assert ready['compute']['backend'] == backend, ready
        for direction, text in CASES:
            send(process, 1, direction, 128, text)
            tokens, output = [], bytearray()
            while True:
                event = receive(process)
                if event['event'] == 'prefill':
                    continue
                if event['event'] == 'token':
                    tokens.append(event['token_id'])
                    output.extend(bytes.fromhex(event['hex']))
                    continue
                assert event['event'] == 'done' and event['finish_reason'] == 'stop', event
                assert not event['compute']['errors'] and not event['compute']['cpu_fallbacks'], event
                break
            results.append({'direction': 'en-ja' if direction == 1 else 'ja-en', 'input': text,
                            'token_ids': tokens, 'text': output.decode('utf-8'), 'result': event})
            print(f'{p.size} {backend}: {text!r} -> {results[-1]["text"]!r}', flush=True)
        return {'ready': ready, 'translations': results}
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--memory', choices=('auto', 'resident', 'stream'), default='auto')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/translation-equivalence.json')
    args = parser.parse_args()
    p = profile(args.model)
    runs = {backend: run(p, backend, args.memory) for backend in ('cpu', 'metal')}
    for cpu, gpu in zip(runs['cpu']['translations'], runs['metal']['translations']):
        assert cpu['token_ids'] == gpu['token_ids'] and cpu['text'] == gpu['text'], (cpu, gpu)
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model': p.name,
              'memory': args.memory, 'runs': runs, 'exact_token_match': True,
              'limits': 'Six fixed completed translations; not a broad translation-quality evaluation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'CPU/Metal FP32: all token IDs and text match in {len(CASES)} completed translations.')


if __name__ == '__main__':
    main()
