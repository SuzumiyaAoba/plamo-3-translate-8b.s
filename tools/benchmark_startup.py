#!/usr/bin/env python3
"""Measure readiness and immediate first/second requests, without a warm-up."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from model_catalog import add_model_argument, profile
from verify_api import check_memory, receive, send

ROOT = Path(__file__).resolve().parents[1]


def measure(worker, p, backend, memory, text, direction, max_tokens):
    start = time.monotonic()
    process = subprocess.Popen([str(worker), '--backend', backend, '--memory', memory, '--threads', '4'],
                               cwd=p.data_root(), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        ready = receive(process)
        startup = time.monotonic() - start
        assert ready['event'] == 'ready' and ready['model'] == p.name and ready['revision'] == p.revision, ready
        check_memory(ready['memory'])
        requests = []
        for _ in range(2):
            sent = time.monotonic()
            send(process, 1, direction, max_tokens, text)
            first_token, tokens, output = None, [], bytearray()
            while True:
                event = receive(process)
                if event['event'] == 'prefill':
                    continue
                if event['event'] == 'token':
                    if first_token is None:
                        first_token = time.monotonic() - sent
                    tokens.append(event['token_id'])
                    output.extend(bytes.fromhex(event['hex']))
                    continue
                assert event['event'] == 'done', event
                check_memory(event['memory'])
                assert not event.get('compute', {}).get('errors', 0), event
                assert not event.get('compute', {}).get('cpu_fallbacks', 0), event
                break
            requests.append({'wall_first_token_seconds': first_token,
                             'wall_total_seconds': time.monotonic() - sent,
                             'token_ids': tokens, 'translation': output.decode(), 'result': event})
        assert requests[0]['token_ids'] == requests[1]['token_ids'] and requests[0]['translation'] == requests[1]['translation']
        return {'startup_wall_seconds': startup, 'ready': ready, 'requests': requests}
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        # Let the previous process's GPU allocations be released before the
        # next worker samples available memory for its fixed startup ceiling.
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--baseline-worker', type=Path)
    parser.add_argument('--backend', choices=('auto', 'cpu', 'metal', 'metal-fp32'), default='auto')
    parser.add_argument('--memory', choices=('auto', 'resident', 'stream'), default='auto')
    parser.add_argument('--text', default='Hello.')
    parser.add_argument('--direction', choices=('en-ja', 'ja-en'), default='en-ja')
    parser.add_argument('--max-tokens', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path, default=ROOT / 'build/startup-benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.max_tokens <= 4096:
        parser.error('repeats must be positive and max-tokens must be 1..4096')
    p = profile(args.model)
    workers = {'after': (args.worker or p.build_directory(ROOT) / 'plamo-worker').resolve()}
    if args.baseline_worker:
        workers = {'before': args.baseline_worker.resolve(), **workers}
    runs = {key: [] for key in workers}
    reference = None
    for i in range(args.repeats):
        for label, worker in workers.items():
            run = measure(worker, p, args.backend, args.memory, args.text,
                          1 if args.direction == 'en-ja' else 2, args.max_tokens)
            current = (run['requests'][0]['token_ids'], run['requests'][0]['translation'])
            if reference is None:
                reference = current
            assert current == reference, (label, current, reference)
            runs[label].append(run)
            print(f'{p.size} {label} {i+1}: startup={run["startup_wall_seconds"]:.3f}s, '
                  f'first token={run["requests"][0]["wall_first_token_seconds"]:.3f}s, '
                  f'first request={run["requests"][0]["wall_total_seconds"]:.3f}s, '
                  f'preloaded={run["ready"]["memory"]["preloaded_weight_bytes"]}', flush=True)
    medians = {}
    for label, samples in runs.items():
        medians[label] = {'startup_wall_seconds': statistics.median(s['startup_wall_seconds'] for s in samples),
                          'preload_seconds': statistics.median(s['ready'].get('preload_seconds', 0) for s in samples)}
        for i, name in enumerate(('first', 'second')):
            for field in ('wall_first_token_seconds', 'wall_total_seconds'):
                medians[label][name + '_' + field] = statistics.median(s['requests'][i][field] for s in samples)
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model': p.name, 'backend': args.backend, 'memory': args.memory,
              'text': args.text, 'direction': args.direction, 'max_tokens': args.max_tokens,
              'worker_sha256': {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in workers.items()},
              'runs': runs, 'seconds_median': medians,
              'limits': 'Startup cost is reported separately; preloading moves I/O before readiness. '
                        'No primer, cancelled request or tokenizer subprocess precedes the first request. '
                        'A one-second gap separates worker processes to allow resource cleanup. '
                        'System memory pressure and file caches can differ between runs.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(medians, indent=2), flush=True)


if __name__ == '__main__':
    main()
