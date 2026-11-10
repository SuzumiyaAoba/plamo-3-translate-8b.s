#!/usr/bin/env python3
"""Compare original loading and prepared weight files with identical worker code."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from benchmark_startup import measure
from model_catalog import add_model_argument, profile

ROOT = Path(__file__).resolve().parents[1]


def worker(p, budget):
    if not budget:
        directory = p.build_directory(ROOT)
        subprocess.run(['make', 'MODEL=' + p.size, str(directory.relative_to(ROOT) / 'plamo-worker')], cwd=ROOT, check=True)
        return directory / 'plamo-worker', None
    directory = ROOT / 'build/disk-cache-study' / ('budget-' + str(budget)) / p.size
    directory.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'runtime/resident.c').read_text()
    needle = 'return ((uint64_t)info.free_count + file_pages - overlap) * (uint64_t)page;'
    assert source.count(needle) == 1
    ceiling = (budget + 2048) * 1024 * 1024  # Production's 2 GiB reserve.
    source = source.replace(needle, 'uint64_t observed = ((uint64_t)info.free_count + file_pages - overlap) * (uint64_t)page;\n'
                           f'    return observed < UINT64_C({ceiling}) ? observed : UINT64_C({ceiling});')
    resident = directory / 'resident.c'
    if not resident.exists() or resident.read_text() != source:
        resident.write_text(source)
    subprocess.run(['make', 'MODEL=' + p.size, 'BUILD_DIR=' + str(directory.relative_to(ROOT)),
                    'RESIDENT_SOURCE=' + str(resident.relative_to(ROOT)),
                    'CFLAGS=-O2 -arch arm64 -Wall -Wextra -Iruntime', str(directory.relative_to(ROOT) / 'plamo-worker')], cwd=ROOT, check=True)
    return directory / 'plamo-worker', resident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--storage', choices=('raw', 'lossless'), default='lossless')
    parser.add_argument('--budget-mib', type=int, default=0, help='benchmark-only weight/staging ceiling; 0 keeps ordinary memory sizing')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--text', default='Hello.')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/weight-cache-benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1 or (args.budget_mib and not 64 <= args.budget_mib <= 65536):
        parser.error('repeats must be positive and budget-mib must be 0 or 64..65536')
    p = profile(args.model)
    binary, resident = worker(p, args.budget_mib)
    path = args.file.resolve()
    stamp = path.stat()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    sources = list((ROOT / 'runtime').glob('*.[chm]')) + [ROOT / 'runtime/kernels.metal', ROOT / 'tools/pack_weights.m',
               ROOT / 'tools/prepare_weights.py', Path(__file__).resolve()]
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    runs = {'original': [], 'prepared': []}
    comparable = []
    reference = None
    saved = os.environ.get('PLAMO_METAL_CACHE')
    try:
        for i in range(args.repeats):
            for label in (('original', 'prepared') if i % 2 == 0 else ('prepared', 'original')):
                os.environ['PLAMO_METAL_CACHE'] = 'off' if label == 'original' else str(path)
                run = measure(binary, p, 'metal', 'auto', args.text, 1, 2, args.storage)
                assert bool(run['ready']['compute']['disk_cache_hits']) == (label == 'prepared'), run['ready']
                value = (run['requests'][0]['token_ids'], run['requests'][0]['translation'], run['requests'][0]['result']['verification'])
                if reference is None:
                    reference = value
                assert value == reference
                runs[label].append(run)
                print(f'{p.size} {label} {i+1}: ready={run["startup_wall_seconds"]:.3f}s, '
                      f'first={run["requests"][0]["wall_first_token_seconds"]:.3f}s, '
                      f'held={run["ready"]["memory"]["cached_storage_bytes"]}', flush=True)
            a, b = (runs[label][i]['ready']['memory'] for label in ('original', 'prepared'))
            comparable.append(a['cached_weight_bytes'] == b['cached_weight_bytes'] and a['cached_storage_bytes'] == b['cached_storage_bytes'])
    finally:
        if saved is None:
            os.environ.pop('PLAMO_METAL_CACHE', None)
        else:
            os.environ['PLAMO_METAL_CACHE'] = saved
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == digest
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h for name, h in hashes.items())
    assert (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ino) == (stamp.st_size, stamp.st_mtime_ns, stamp.st_ino)
    medians = {}
    for label, samples in runs.items():
        accepted = [s for i, s in enumerate(samples) if comparable[i]]
        medians[label] = None if not accepted else {
            'startup_seconds': statistics.median(s['startup_wall_seconds'] for s in accepted),
            'first_token_seconds': statistics.median(s['requests'][0]['wall_first_token_seconds'] for s in accepted),
            'second_token_seconds': statistics.median(s['requests'][1]['wall_first_token_seconds'] for s in accepted)}
    report = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), model=p.name, revision=p.revision,
                  budget_mib=args.budget_mib, storage=args.storage, text=args.text, cache_file_bytes=stamp.st_size,
                  worker=str(binary.relative_to(ROOT)), worker_sha256=digest, source_sha256=hashes,
                  budgeted_resident_sha256=hashlib.sha256(resident.read_bytes()).hexdigest() if resident else None,
                  comparable_pairs=comparable, seconds_median=medians, runs=runs,
                  method='Identical worker, cache off/on, alternating order, two requests after ready, no user warmup. '
                         'Only pairs with equal retained weights and storage contribute to medians. '
                         'Nonzero budget uses a benchmark-only capped reclaimable-memory function and retains production pressure handling.',
                  limits='Filesystem cache, OS pressure and other application activity are not controlled. '
                         'Capped 8B measurements may require weight I/O after ready; they are not fully resident measurements.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(medians, indent=2), flush=True)
    assert any(comparable), 'no equal-retention pairs; see the saved report'


if __name__ == '__main__':
    main()
