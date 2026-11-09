#!/usr/bin/env python3
"""Measure generation throughput separately from loading and prompt processing."""
import argparse
import datetime
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from model_catalog import add_model_argument, profile
from verify_api import check_memory, receive, send

ROOT = Path(__file__).resolve().parents[1]
TEXT = ('Translation quality depends on context, terminology, and the structure of the original document. '
        'A useful system should preserve numbers, dates, and product names while producing natural sentences. '
        'This benchmark compares the same full precision model on the CPU and the GPU. '
        'It measures the delay before the first output token and the time spent generating subsequent tokens, '
        'while checking the translated text and the numerical differences between the two implementations.')


def request(process, text, direction, limit):
    send(process, 1, direction, limit, text)
    output, ids, arrivals = bytearray(), [], []
    while True:
        event = receive(process)
        if event['event'] == 'prefill':
            continue
        if event['event'] == 'token':
            arrivals.append(time.monotonic())
            ids.append(event['token_id'])
            output.extend(bytes.fromhex(event['hex']))
            continue
        assert event['event'] == 'done', event
        check_memory(event['memory'])
        assert not event.get('compute', {}).get('errors', 0), event
        assert not event.get('compute', {}).get('cpu_fallbacks', 0), event
        break
    timing = event['timings']
    steps = timing['decode_steps']
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {'translation': output.decode(), 'token_ids': ids, 'decode_steps': steps,
            'decode_tokens_per_second': steps / timing['decode_seconds'] if steps else None,
            'seconds_per_decode_step': timing['decode_seconds'] / steps if steps else None,
            'inter_token_seconds': intervals, 'result': event}


def measure(worker, storage, p, args):
    environment = dict(os.environ, PLAMO_WEIGHT_STORAGE=storage)
    process = subprocess.Popen([str(worker), '--backend', args.backend, '--memory', args.memory, '--threads', '4'],
                               cwd=p.data_root(), env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        ready = receive(process)
        assert ready['event'] == 'ready' and ready['model'] == p.name and ready['revision'] == p.revision, ready
        check_memory(ready['memory'])
        warmup = request(process, args.text, 1 if args.direction == 'en-ja' else 2, 2)
        results = []
        for i in range(args.repeats):
            sample = request(process, args.text, 1 if args.direction == 'en-ja' else 2, args.max_tokens)
            assert sample['decode_steps'] > 0, 'The prompt produced no decode steps; use a longer text.'
            if results:
                assert sample['token_ids'] == results[0]['token_ids'], sample
            results.append(sample)
            print(f'{p.size} {storage} {i+1}: {sample["decode_tokens_per_second"]:.3f} tok/s, '
                  f'{sample["seconds_per_decode_step"]:.6f} s/step, '
                  f'decode I/O={sample["result"]["weight_io"].get("decode_staged_bytes", "not reported")}', flush=True)
        intervals = [x for r in results for x in r['inter_token_seconds']]
        summary = {'decode_tokens_per_second': statistics.median(r['decode_tokens_per_second'] for r in results),
                   'seconds_per_decode_step': statistics.median(r['seconds_per_decode_step'] for r in results),
                   'prefill_seconds': statistics.median(r['result']['timings']['prefill_seconds'] for r in results),
                   'total_seconds': statistics.median(r['result']['timings']['total_seconds'] for r in results),
                   'inter_token_median_seconds': statistics.median(intervals) if intervals else None}
        for key in ('decode_gpu_seconds', 'decode_command_buffers', 'decode_gpu_dispatches'):
            if key in results[0]['result']['timings']:
                summary[key] = statistics.median(r['result']['timings'][key] / r['decode_steps'] for r in results)
        return {'worker': str(worker), 'sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
                'requested_storage': storage, 'ready': ready, 'warmup': warmup, 'requests': results, 'median': summary}
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--baseline-worker', type=Path)
    parser.add_argument('--before-storage', choices=('auto', 'raw', 'lossless'), default='raw')
    parser.add_argument('--storage', choices=('auto', 'raw', 'lossless'), default='auto')
    parser.add_argument('--backend', choices=('auto', 'metal', 'metal-fp32', 'cpu'), default='metal')
    parser.add_argument('--memory', choices=('auto', 'resident', 'stream'), default='auto')
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--text', default=TEXT)
    parser.add_argument('--direction', choices=('en-ja', 'ja-en'), default='en-ja')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/decode-benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1 or not 2 <= args.max_tokens <= 4096:
        parser.error('repeats must be positive and max-tokens must be 2..4096')
    p = profile(args.model)
    worker = (args.worker or p.build_directory(ROOT) / 'plamo-worker').resolve()
    sources = [ROOT / 'Makefile', *sorted((ROOT / 'src').glob('*.s')),
               *sorted((ROOT / 'runtime').glob('*.[chm]')), ROOT / 'runtime/kernels.metal']
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    before = measure((args.baseline_worker or worker).resolve(), args.before_storage, p, args)
    after = measure(worker, args.storage, p, args)
    assert before['requests'][0]['token_ids'] == after['requests'][0]['token_ids'], 'Generation token IDs differ.'
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value for path, value in source_hashes.items()), 'Sources changed during measurement.'
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'model': p.name,
              'backend': args.backend, 'memory': args.memory, 'text': args.text, 'direction': args.direction,
              'host': {'system': platform.system(), 'release': platform.release(), 'machine': platform.machine()},
              'source_sha256': source_hashes,
              'max_tokens': args.max_tokens, 'before': before, 'after': after,
              'speedup': after['median']['decode_tokens_per_second'] / before['median']['decode_tokens_per_second'],
              'exact_token_match': True,
              'limits': 'Loading and prefill are excluded from decode throughput. A two-token warmup precedes measurements. '
                        'Cache ceilings are sampled independently and reported; available memory can change between workers.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('model', 'speedup', 'exact_token_match')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
