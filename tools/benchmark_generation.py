#!/usr/bin/env python3
"""Measure paired API-worker generation runs, with equal retained-weight checks."""
import argparse
import datetime
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
from types import SimpleNamespace
from benchmark_decode import measure, TEXT
from benchmark_weight_cache import worker
from model_catalog import add_model_argument, profile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--baseline-worker', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--data-dir', type=Path, help='default: the installed API data directory')
    parser.add_argument('--storage', choices=('auto', 'raw', 'lossless'), default='lossless')
    parser.add_argument('--budget-mib', type=int, default=0, help='benchmark-only memory ceiling for the candidate worker')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=12)
    parser.add_argument('--text', default=TEXT)
    parser.add_argument('--report', type=Path, default=ROOT / 'build/generation-benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1 or not 2 <= args.max_tokens <= 4096 or (args.budget_mib and not 64 <= args.budget_mib <= 65536):
        parser.error('invalid repeats, max-tokens or budget-mib')
    p = profile(args.model)
    before = args.baseline_worker.resolve()
    after, resident = (args.worker.resolve(), None) if args.worker else worker(p, args.budget_mib)
    binaries = {'before': before, 'after': after}
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in binaries.items()}
    sources = [*sorted((ROOT / 'runtime').glob('*.[chm]')), ROOT / 'runtime/kernels.metal', Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    data_dir = (args.data_dir or p.data_root(True)).resolve()
    saved = os.environ.get('PLAMO_DATA_DIR')
    config = SimpleNamespace(backend='metal', memory='auto', text=args.text, direction='en-ja', repeats=1, max_tokens=args.max_tokens)
    runs = {'before': [], 'after': []}
    comparable = []
    reference = None
    try:
        os.environ['PLAMO_DATA_DIR'] = str(data_dir)
        for trial in range(args.repeats):
            for name in (('before', 'after') if trial % 2 == 0 else ('after', 'before')):
                run = measure(binaries[name], args.storage, p, config)
                sample = run['requests'][0]
                value = (sample['token_ids'], sample['translation'], sample['result']['verification'])
                if reference is None:
                    reference = value
                assert value == reference, f'{name}: output or full-vector hashes changed'
                runs[name].append(run)
            equal = True
            for phase in ('ready', 'warmup', 'request'):
                states = []
                for name in ('before', 'after'):
                    run = runs[name][trial]
                    event = run['ready'] if phase == 'ready' else run['warmup']['result'] if phase == 'warmup' else run['requests'][0]['result']
                    states.append((event['memory']['cached_weight_bytes'], event['memory']['cached_storage_bytes'], event['compute']['weight_storage']))
                equal &= states[0] == states[1]
            comparable.append(equal)
            print(f'pair {trial+1}: equal retained weights/storage={equal}', flush=True)
    finally:
        if saved is None:
            os.environ.pop('PLAMO_DATA_DIR', None)
        else:
            os.environ['PLAMO_DATA_DIR'] = saved
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digests[name] for name, path in binaries.items())
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in source_hashes.items())
    summary = {}
    for name in ('before', 'after'):
        samples = [r['requests'][0] for i, r in enumerate(runs[name]) if comparable[i]]
        summary[name] = None if not samples else dict(
            tokens_per_second=statistics.median(s['decode_tokens_per_second'] for s in samples),
            seconds_per_step=statistics.median(s['seconds_per_decode_step'] for s in samples),
            io_bytes_per_step=statistics.median(s['result']['weight_io']['decode_staged_bytes'] / s['decode_steps'] for s in samples),
            mapped_bytes_per_step=statistics.median(s['result']['weight_io'].get('decode_mapped_bytes', 0) / s['decode_steps'] for s in samples),
            gpu_seconds_per_step=statistics.median(s['result']['timings']['decode_gpu_seconds'] / s['decode_steps'] for s in samples),
            commands_per_step=statistics.median(s['result']['timings']['decode_command_buffers'] / s['decode_steps'] for s in samples))
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), model=p.name, revision=p.revision,
                  platform=platform.platform(), text=args.text, data_dir=str(data_dir), budget_mib=args.budget_mib,
                  max_tokens=args.max_tokens, storage=args.storage, weight_cache_environment=os.environ.get('PLAMO_METAL_CACHE', 'auto'),
                  budgeted_resident_sha256=hashlib.sha256(resident.read_bytes()).hexdigest() if resident else None,
                  source_sha256=source_hashes, worker_sha256=digests, comparable_pairs=comparable, summary=summary,
                  runs=runs, exact_outputs_and_hashes=True,
                  method='Alternating before/after workers; one two-token warmup then one measured request each. '
                         'Decode time excludes startup and prefill. Medians include only pairs with equal retained weights/storage '
                         'at ready, after warmup and after the measured request.',
                  limits='OS caching and background memory pressure are uncontrolled. A saved baseline worker is required; '
                         'the same benchmark-only memory cap must be used for both workers when specified. '
                         'I/O bytes count explicit pread only. Mapped bytes count borrowed file ranges, not physical disk reads '
                         'or current residency; mapped file pages are reclaimable OS storage outside the private weight budget.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    assert any(comparable), 'no equal-retention pairs; see report'


if __name__ == '__main__':
    main()
