#!/usr/bin/env python3
"""Compare every 8B state/logit vector and token with ordinary FP32 decoding.

Requires real weights and `make build/generation-probe build/0.8b/plamo-worker`.
Run separately from throughput benchmarks; capturing all vectors adds I/O.
"""
import argparse
import array
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from benchmark_decode import TEXT
from verify_generation_target import JAPANESE

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(name=f'length-{n}', direction=1, text=TEXT, limit=n) for n in (1, 2, 3, 8, 9)
] + [
    dict(name='english-64', direction=1, text=TEXT, limit=64),
    dict(name='japanese-64', direction=2, text=JAPANESE, limit=64),
    dict(name='english-eos', direction=1, text='Hello.', limit=32),
    dict(name='japanese-eos', direction=2, text='本日は晴天です。', limit=32),
    dict(name='long-prompt', direction=2, text=JAPANESE * 3, limit=16),
] + [
    dict(name=f'cancel-{n}', direction=1, text=TEXT, limit=64, cancel=n) for n in (0, 1, 5, 9)
] + [dict(name='request-after-cancel', direction=1, text=TEXT, limit=16),
     dict(name='sliding-kv', direction=2, text='東京の天気は晴れです。\n' * 300, limit=4)]


def exact(stream, count):
    data = bytearray()
    while len(data) < count:
        part = stream.read(count - len(data))
        if not part:
            raise RuntimeError('generation probe exited unexpectedly')
        data.extend(part)
    return bytes(data)


def capture(probe, data, enabled, cases):
    p = subprocess.Popen([str(probe)], cwd=data, env=dict(os.environ, PLAMO_SPECULATIVE=str(enabled)),
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        magic, hidden, vocab = struct.unpack('<3I', exact(p.stdout, 12))
        assert (magic, hidden, vocab) == (0x504c5632, 4096, 107520)
        runs = []
        for case in cases:
            text = case['text'].encode()
            p.stdin.write(struct.pack('<4I', case['direction'], case['limit'], len(text), case.get('cancel', 0xffffffff)) + text)
            p.stdin.flush()
            ids, vectors = [], []
            while True:
                token, = struct.unpack('<I', exact(p.stdout, 4))
                if token == 0xffffffff:
                    summary = struct.unpack('<8Q', exact(p.stdout, 64))
                    break
                ids.append(token)
                vectors.append(exact(p.stdout, (hidden + vocab) * 4))
            vectors.append(exact(p.stdout, (hidden + vocab) * 4))
            assert summary[0] == len(ids)
            runs.append(dict(ids=ids, summary=summary, vectors=vectors))
            print(f'{"speculative" if enabled else "ordinary"} {case["name"]}: {len(ids)} tokens, {summary[3]} verification batches', flush=True)
        return runs
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
        p.stdout.close()
        time.sleep(2)


def error(actual, reference):
    a, b = array.array('f'), array.array('f')
    a.frombytes(actual); b.frombytes(reference)
    squared, norm, maximum, magnitude = 0.0, 0.0, 0.0, 0.0
    for x, y in zip(a, b):
        assert math.isfinite(x) and math.isfinite(y), 'nonfinite target state/logits'
        delta = x - y
        squared += delta * delta; norm += y * y
        maximum = max(maximum, abs(delta)); magnitude = max(magnitude, abs(y))
    relative = math.sqrt(squared / max(norm, 1e-30))
    assert relative <= 1e-4 and maximum <= 2e-4 * (1 + magnitude), (relative, maximum, magnitude)
    return relative, maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, default=ROOT / 'build/generation-probe')
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=ROOT / 'build/speculative-accuracy.json')
    parser.add_argument('--case', action='append', choices=[c['name'] for c in CASES], help='run selected cases only')
    args = parser.parse_args()
    probe = args.probe.resolve()
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    cases = [c for c in CASES if not args.case or c['name'] in args.case]
    before = capture(probe, args.data_dir, 0, cases)
    after = capture(probe, args.data_dir, 1, cases)
    comparisons = []
    for case, ref, candidate in zip(cases, before, after):
        assert ref['ids'] == candidate['ids'], f'{case["name"]}: generated tokens changed'
        assert ref['summary'][:3] == candidate['summary'][:3], f'{case["name"]}: finish/cancellation changed'
        state, logits = [0., 0.], [0., 0.]
        for actual, expected in zip(candidate['vectors'], ref['vectors']):
            for start, end, metrics in ((0, 4096 * 4, state), (4096 * 4, len(actual), logits)):
                relative, maximum = error(actual[start:end], expected[start:end])
                metrics[0] = max(metrics[0], relative); metrics[1] = max(metrics[1], maximum)
        comparisons.append(dict(**case, token_ids=candidate['ids'], exact_tokens=True,
                                finish_reason=('length', 'stop', 'cancelled')[candidate['summary'][2]],
                                decode_steps=candidate['summary'][1], verification_batches=candidate['summary'][3],
                                consumed_tokens=candidate['summary'][4], proposed_tokens=candidate['summary'][5],
                                accepted_tokens=candidate['summary'][6], captured_vectors=len(candidate['vectors']),
                                prompt_tokens=candidate['summary'][7],
                                state_relative_l2=state[0], state_max_abs=state[1],
                                logits_relative_l2=logits[0], logits_max_abs=logits[1]))
        print(case['name'], 'exact tokens/finish; max relative L2', max(state[0], logits[0]), flush=True)
    if any(c['limit'] > 1 and c.get('cancel', 1) for c in cases):
        assert any(x['verification_batches'] for x in comparisons), 'No speculative verification was exercised'
    assert all(x['finish_reason'] == 'stop' for x in comparisons if x['name'].endswith('-eos'))
    assert all(x['finish_reason'] == 'cancelled' for x in comparisons if x['name'].startswith('cancel-'))
    assert all(x['prompt_tokens'] > 2048 for x in comparisons if x['name'] == 'sliding-kv')
    assert hashlib.sha256(probe.read_bytes()).hexdigest() == digest
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  target_precision='original BF16 weights, FP32 arithmetic', auxiliary_storage='Q4 proposals only',
                  probe_sha256=digest, all_token_sequences_and_finish_states_match=True,
                  relative_l2_tolerance=1e-4, max_abs_tolerance='2e-4 * (1 + maximum reference magnitude)',
                  cases=comparisons)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
