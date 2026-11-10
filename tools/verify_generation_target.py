#!/usr/bin/env python3
"""Verify sustained generation throughput against full-GPU reference outputs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from benchmark_decode import request, TEXT
from model_catalog import add_model_argument, profile
from worker_protocol import close_worker, receive

ROOT = Path(__file__).resolve().parents[1]
JAPANESE = (
    '翻訳システムの性能を評価するときは、最初の応答までの時間と、その後の文章を生成する速度を区別する必要があります。'
    'この文書では、複数の言語で同じ内容を伝えながら、人名、製品名、日付、数値を正確に保つことを求めます。'
    '担当者は2026年11月10日の会議までに報告書を作成し、42.5パーセントという測定値が変わっていないことを確認してください。'
    '測定では同じ入力を繰り返して、モデルの読み込みと入力処理を除いたトークン生成の時間を調べます。'
    '結果には利用したメモリと計算精度も記録し、実行条件が違う数値を直接比較しないようにしてください。'
)
CASES = ((1, TEXT), (2, JAPANESE))


def run(worker, p, data, repeats, limit):
    process = subprocess.Popen([str(worker), '--backend', 'metal', '--memory', 'auto'], cwd=data,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        ready = receive(process)
        assert ready['event'] == 'ready' and ready['model'] == p.name and ready['revision'] == p.revision, ready
        warmup = request(process, TEXT, 1, 2)
        samples = []
        for trial in range(repeats):
            for direction, text in CASES:
                sample = request(process, text, direction, limit)
                assert sample['decode_steps'] >= 32, 'Use at least 33 output tokens for sustained throughput.'
                timing = sample['result']['timings']
                speculative = sample['result'].get('speculative', {})
                if speculative.get('batches', 0):
                    assert speculative['consumed_tokens'] == sample['decode_steps']
                    assert timing['decode_command_buffers'] >= speculative['batches']
                else:
                    assert timing['decode_command_buffers'] == sample['decode_steps'], 'Both workers must use complete GPU graphs.'
                samples.append(dict(trial=trial+1, direction=direction, input=text, **sample))
                print(f'{worker.name} {trial+1} {direction}: {sample["decode_tokens_per_second"]:.3f} tokens/s '
                      f'({sample["decode_steps"]} decode steps)', flush=True)
        return dict(ready=ready, warmup=warmup, samples=samples)
    finally:
        close_worker(process)
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--reference-worker', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--target', type=float, default=4.0)
    parser.add_argument('--allow-fp32-rounding', action='store_true',
                        help='compare exact tokens/text and prefill hashes; use verify_speculative.py for complete FP32 vector errors')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/generation-target.json')
    args = parser.parse_args()
    if args.repeats < 1 or not 33 <= args.max_tokens <= 4096 or not 0 < args.target < 100000:
        parser.error('invalid repeats, max-tokens or target')
    p = profile(args.model)
    data = (args.data_dir or p.data_root(True)).resolve()
    binaries = dict(reference=args.reference_worker.resolve(), candidate=args.worker.resolve())
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}
    paths = [*sorted((ROOT / 'runtime').glob('*.[chm]')), ROOT / 'runtime/kernels.metal', Path(__file__).resolve()]
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    results = {name: run(binary, p, data, args.repeats, args.max_tokens) for name, binary in binaries.items()}
    references = {}
    for sample in results['reference']['samples']:
        value = (sample['token_ids'], sample['translation'], sample['result']['verification'])
        if sample['direction'] in references:
            assert references[sample['direction']] == value
        references[sample['direction']] = value
    for sample in results['candidate']['samples']:
        reference = references[sample['direction']]
        assert reference[:2] == (sample['token_ids'], sample['translation']), 'Generated tokens/text changed.'
        verification = sample['result']['verification']
        if args.allow_fp32_rounding:
            assert all(reference[2][key] == verification[key] for key in ('prefill_logits_hash', 'prefill_state_hash'))
        else:
            assert reference[2] == verification, 'Full-vector reference changed.'
    rates = [s['decode_tokens_per_second'] for s in results['candidate']['samples']]
    passed = all(rate > args.target for rate in rates)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digests[name] for name, path in binaries.items())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in sources.items())
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), platform=platform.platform(),
                  model=p.name, revision=p.revision, data_dir=str(data), repeats=args.repeats, max_tokens=args.max_tokens,
                  target_tokens_per_second=args.target, all_samples_exceed_target=passed,
                  median_tokens_per_second=statistics.median(rates), minimum_tokens_per_second=min(rates),
                  worker_sha256=digests, source_sha256=sources,
                  environment={name: os.environ.get(name, 'auto') for name in ('PLAMO_METAL_CACHE', 'PLAMO_WEIGHT_STORAGE')},
                  exact_token_text_and_full_vector_hashes=not args.allow_fp32_rounding,
                  exact_tokens_and_text=True, fp32_rounding_allowed=args.allow_fp32_rounding, runs=results,
                  limits='Sequential reference/candidate workers; one two-token warmup, then repeated English/Japanese requests. '
                         'Decode excludes startup and prefill. Cache sizes may differ and are reported; file-view storage is '
                         'additional to private-cache budgets. OS caching and background memory pressure are uncontrolled.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(f'Generation target: median={statistics.median(rates):.3f}, minimum={min(rates):.3f}, exact tokens=True', flush=True)
    assert passed, 'A candidate sample did not exceed the target; see report.'


if __name__ == '__main__':
    main()
