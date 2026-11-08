#!/usr/bin/env python3
"""Measure the real assembly engine and check unchanged state/logits hashes.

The fixed probe is the official en->ja prompt for "Hello." (21 input tokens),
followed by one decode step: token 51871 then stop token 16. No Python inference.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--policy', type=int, choices=(0, 1, 2), default=2,
                        help='0: mapped, 1: mapped/read-ahead, 2: bounded bulk reads')
    parser.add_argument('--baseline', type=Path,
                        default=ROOT / 'docs/benchmarks/optimization-baseline.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    baseline = json.loads(args.baseline.read_text())
    expected = {p['phase']: p for p in baseline['phases']}
    subprocess.run(['make', 'build/bench-model'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    runs = []
    for repetition in range(args.repeats):
        result = subprocess.run([str(ROOT / 'build/bench-model'), str(args.policy)],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        phases = [json.loads(line) for line in result.stdout.splitlines()]
        assert [p['phase'] for p in phases] == list(expected), result.stdout
        for phase in phases:
            reference = expected[phase['phase']]
            for key in ('token', 'logits_hash', 'state_hash'):
                assert phase[key] == reference[key], (phase['phase'], key, phase, reference)
            assert phase.get('read_failures', 0) == 0, phase
        if args.policy == 2:
            assert phases[-1]['staged_bytes'] == 32363249664, phases[-1]
        runs.append(phases)
        print(f"run {repetition + 1}: prefill {phases[1]['seconds']:.3f}s, "
              f"decode {phases[2]['seconds']:.3f}s; exact hashes PASS", flush=True)
    summary = {}
    for phase in expected:
        elapsed = statistics.median(next(p['seconds'] for p in run if p['phase'] == phase)
                                    for run in runs)
        summary[phase] = {'seconds_median': elapsed,
                          'baseline_seconds': expected[phase]['seconds'],
                          'speedup': expected[phase]['seconds'] / elapsed}
    summary['inference_total'] = {
        'seconds_median': statistics.median(sum(p['seconds'] for p in run[1:]) for run in runs),
        'baseline_seconds': sum(p['seconds'] for p in baseline['phases'][1:])}
    summary['inference_total']['speedup'] = (summary['inference_total']['baseline_seconds'] /
                                           summary['inference_total']['seconds_median'])
    summary['peak_rss_bytes'] = max(p['max_rss_bytes'] for run in runs for p in run)
    memory_bytes = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'environment': {'machine': platform.machine(), 'macos': platform.mac_ver()[0],
                              'memory_bytes': memory_bytes},
              'revision': (ROOT / 'models/plamo/REVISION').read_text().strip(),
              'policy': args.policy, 'runs': runs, 'summary': summary,
              'baseline_file': str(args.baseline.relative_to(ROOT) if args.baseline.is_relative_to(ROOT) else args.baseline)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if report['environment'] != baseline['environment']:
        print('Environment differs from the recorded baseline; speedups are not a controlled comparison.')


if __name__ == '__main__':
    main()
