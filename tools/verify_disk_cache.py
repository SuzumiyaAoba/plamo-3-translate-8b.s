#!/usr/bin/env python3
"""Compare disk-cache loading with original loading, including complete output hashes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from benchmark_decode import request
from model_catalog import add_model_argument, profile
from verify_api import receive
from verify_prefix_cache import CASES

ROOT = Path(__file__).resolve().parents[1]


def run(worker, p, memory, storage, cache, limit):
    start = time.monotonic()
    process = subprocess.Popen([str(worker), '--backend', 'metal', '--memory', memory], cwd=p.data_root(),
                               env=dict(os.environ, PLAMO_WEIGHT_STORAGE=storage, PLAMO_METAL_CACHE=cache),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        ready = receive(process)
        assert ready['event'] == 'ready' and ready['model'] == p.name and ready['revision'] == p.revision, ready
        startup = time.monotonic() - start
        assert bool(ready['compute']['disk_cache_hits']) == (cache != 'off'), ready
        samples = [dict(direction=direction, input=text, **request(process, text, direction, limit)) for direction, text in CASES]
        return dict(startup_wall_seconds=startup, ready=ready, requests=samples)
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--storage', choices=('raw', 'lossless', 'auto'), default='auto')
    parser.add_argument('--memory', choices=('auto', 'resident'), default='auto')
    parser.add_argument('--max-tokens', type=int, default=12)
    parser.add_argument('--report', type=Path, default=ROOT / 'build/disk-cache-verification.json')
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 4096:
        parser.error('max-tokens must be 1..4096')
    p = profile(args.model)
    worker = (args.worker or p.build_directory(ROOT) / 'plamo-worker').resolve()
    path = args.file.resolve()
    stat = path.stat()
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    runs = {}
    for label, cache in [('original', 'off'), ('disk_cache', str(path))]:
        runs[label] = run(worker, p, args.memory, args.storage, cache, args.max_tokens)
        print(f'{p.size} {label}: completed {len(CASES)} requests', flush=True)
    for before, after in zip(runs['original']['requests'], runs['disk_cache']['requests']):
        assert before['token_ids'] == after['token_ids'] and before['translation'] == after['translation']
        assert before['result']['verification'] == after['result']['verification']
    before, after = (runs[label]['ready']['memory'] for label in ('original', 'disk_cache'))
    assert before['cached_weight_bytes'] == after['cached_weight_bytes']
    assert before['cached_storage_bytes'] == after['cached_storage_bytes']
    assert (stat.st_size, stat.st_mtime_ns, stat.st_ino) == (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino)
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == digest
    with path.open('rb') as stream:
        directory_digest = hashlib.sha256(stream.read(16384)).hexdigest()
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), model=p.name, revision=p.revision,
                  memory=args.memory, storage=args.storage, worker_sha256=digest, cache_file_bytes=stat.st_size,
                  cache_first_page_sha256=directory_digest, runs=runs, bitwise_outputs_equal=True,
                  max_tokens=args.max_tokens,
                  limits='Ten fixed requests, alternating directions, long input, reserved tokens and raw generation. '
                         'Compares full-vector prefill/final hashes and generated tokens. Requires equal retained weights/storage.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'{p.size} disk cache: PASS (all hashes, token IDs, text and retained storage match)', flush=True)


if __name__ == '__main__':
    main()
