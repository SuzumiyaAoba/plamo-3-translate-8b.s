#!/usr/bin/env python3
"""Compare fixed-prefix reuse against fresh GPU prefill, including all hashes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from benchmark_decode import request, TEXT
from model_catalog import add_model_argument, profile
from worker_protocol import close_worker, receive

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    (1, 'Hello.'),
    (2, 'こんにちは。'),
    (1, 'The meeting starts at ten tomorrow morning.'),
    (2, 'この文書を保存してから、ウィンドウを閉じてください。'),
    (1, 'Preserve product names, dates, and numbers such as 42.5 and 2026-11-10.'),
    (1, TEXT),
    (1, '<|plamo:tag|>input'),
    (0, 'Hello'),
    (2, 'こんにちは。'),
    (1, 'Hello.'),
)


def run(worker, p, memory, enabled):
    start = time.monotonic()
    process = subprocess.Popen([str(worker), '--backend', 'metal', '--memory', memory],
                               cwd=p.data_root(), env=dict(os.environ, PLAMO_PREFIX_CACHE=str(enabled)),
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        ready = receive(process)
        startup = time.monotonic() - start
        assert ready['event'] == 'ready' and ready['model'] == p.name, ready
        cache = ready['prompt_cache']
        assert bool(cache['en_ja_tokens']) == bool(enabled) and bool(cache['ja_en_tokens']) == bool(enabled), ready
        samples = []
        for direction, text in CASES:
            sample = request(process, text, direction, 12)
            reused = sample['result']['usage']['cached_prompt_tokens']
            assert reused == (cache['en_ja_tokens' if direction == 1 else 'ja_en_tokens'] if enabled and direction else 0), sample
            assert sample['result']['timings']['prefill_command_buffers'] == 1, sample
            samples.append(dict(direction=direction, input=text, **sample))
        return {'startup_wall_seconds': startup, 'ready': ready, 'requests': samples}
    finally:
        close_worker(process)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--memory', choices=('auto', 'resident'), default='auto')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/prefix-cache-verification.json')
    args = parser.parse_args()
    p = profile(args.model)
    worker = (args.worker or p.build_directory(ROOT) / 'plamo-worker').resolve()
    digest = hashlib.sha256(worker.read_bytes()).hexdigest()
    runs = {'fresh': run(worker, p, args.memory, 0), 'cached': run(worker, p, args.memory, 1)}
    for fresh, cached in zip(runs['fresh']['requests'], runs['cached']['requests']):
        assert fresh['token_ids'] == cached['token_ids'] and fresh['translation'] == cached['translation'], (fresh, cached)
        assert fresh['result']['verification'] == cached['result']['verification'], (fresh, cached)
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == digest
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model': p.name, 'revision': p.revision, 'memory': args.memory, 'worker_sha256': digest,
              'runs': runs, 'bitwise_prefill_and_decode_match': True, 'exact_token_match': True,
              'limits': f'{len(CASES)} fixed requests, including alternating directions, an 81-token prompt, reserved input tokens and raw generation. '
                        'Checks full-vector prefill/final hashes and up to 12 output tokens per request. '
                        'Requires every projection to fit in the GPU cache; not a broad translation-quality evaluation.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'{p.size} fixed prefix reuse: PASS (all hashes, token IDs and text match in {len(CASES)} requests)', flush=True)


if __name__ == '__main__':
    main()
