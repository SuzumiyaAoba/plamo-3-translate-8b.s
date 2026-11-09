#!/usr/bin/env python3
"""Measure one-shot CLI time to first output byte, including model loading."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from model_catalog import add_model_argument, profile

ROOT = Path(__file__).resolve().parents[1]


def measure(binary, p, args):
    start = time.monotonic()
    process = subprocess.Popen([str(binary), '--backend', args.backend, '--memory', args.memory,
                                '--max-tokens', str(args.max_tokens),
                                '-j' if args.direction == 'en-ja' else '-e', args.text],
                               cwd=p.data_root(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    try:
        first = process.stdout.read(1)
        first_seconds = time.monotonic() - start if first else None
        tail, error = process.communicate(timeout=300)
        elapsed = time.monotonic() - start
        assert process.returncode == 0 and first, (process.returncode, error.decode())
        output = first + tail
        return {'first_output_seconds': first_seconds, 'total_seconds': elapsed,
                'stdout_hex': output.hex(), 'translation': output.decode()}
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        process.stdout.close()
        process.stderr.close()
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--baseline-binary', type=Path)
    parser.add_argument('--backend', choices=('auto', 'cpu', 'metal'), default='metal')
    parser.add_argument('--memory', choices=('auto', 'resident', 'stream'), default='auto')
    parser.add_argument('--text', default='Hello.')
    parser.add_argument('--direction', choices=('en-ja', 'ja-en'), default='en-ja')
    parser.add_argument('--max-tokens', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--report', type=Path, default=ROOT / 'build/cli-startup-benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.max_tokens <= 4096:
        parser.error('repeats must be positive and max-tokens must be 1..4096')
    p = profile(args.model)
    binaries = {'after': (args.binary or p.build_directory(ROOT) / 'plamo').resolve()}
    if args.baseline_binary:
        binaries = {'before': args.baseline_binary.resolve(), **binaries}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in binaries.items()}
    runs = {key: [] for key in binaries}
    reference = None
    for i in range(args.repeats):
        order = list(binaries) if not i % 2 else list(reversed(binaries))
        for label in order:
            run = measure(binaries[label], p, args)
            if reference is None:
                reference = run['stdout_hex']
            assert run['stdout_hex'] == reference, (label, run)
            runs[label].append(run)
            print(f'{p.size} {label} {i + 1}: first output={run["first_output_seconds"]:.3f}s, '
                  f'total={run["total_seconds"]:.3f}s', flush=True)
    assert hashes == {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in binaries.items()}
    medians = {label: {field: statistics.median(run[field] for run in samples)
                       for field in ('first_output_seconds', 'total_seconds')} for label, samples in runs.items()}
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model': p.name, 'revision': p.revision, 'backend': args.backend, 'memory': args.memory,
              'text': args.text, 'direction': args.direction, 'max_tokens': args.max_tokens,
              'binary_sha256': hashes, 'runs': runs, 'seconds_median': medians,
              'exact_output_match': True,
              'limits': 'Each invocation starts a new process and includes loading, GPU setup and prefill. '
                        'No warm-up requests. Measures first output byte, not a decoded token event. '
                        'Before/after order alternates; memory pressure and OS caches can still differ.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(medians, indent=2), flush=True)


if __name__ == '__main__':
    main()
