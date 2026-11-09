#!/usr/bin/env python3
"""Compare real CPU/Metal FP32 logits, hidden states and selected tokens."""
import argparse
from array import array
import datetime
import json
import math
from pathlib import Path
import subprocess
from model_catalog import add_model_argument, profile

ROOT = Path(__file__).resolve().parents[1]


def compare(reference, candidate, limit):
    before, after = array('f'), array('f')
    before.frombytes(reference.read_bytes())
    after.frombytes(candidate.read_bytes())
    assert before and len(before) == len(after)
    assert all(math.isfinite(x) for x in before) and all(math.isfinite(x) for x in after)
    error = math.fsum((a - b) ** 2 for a, b in zip(before, after))
    norm = math.fsum(a * a for a in before)
    relative = math.sqrt(error / norm) if norm else math.sqrt(error)
    result = {'values': len(before), 'relative_l2': relative,
              'max_absolute_error': max(abs(a - b) for a, b in zip(before, after)),
              'rms_error': math.sqrt(error / len(before)), 'relative_l2_limit': limit}
    assert relative <= limit, (candidate.name, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--memory', choices=('auto', 'resident', 'stream'), default='auto')
    parser.add_argument('--text', default='Hello.')
    parser.add_argument('--direction', choices=('en-ja', 'ja-en'), default='en-ja')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/metal-equivalence.json')
    args = parser.parse_args()
    p = profile(args.model)
    subprocess.run(['make', 'MODEL=' + p.size, str(p.build_directory(ROOT).relative_to(ROOT) / 'bench-backend')], cwd=ROOT, check=True)
    work = ROOT / 'build/metal-equivalence' / p.size
    runs, errors = {}, {}
    reference_tokens = None
    for backend in ('cpu', 'metal-fp32', 'metal'):
        directory = work / backend
        directory.mkdir(parents=True, exist_ok=True)
        print(f'Running {p.size} {backend}...', flush=True)
        result = subprocess.run([str(p.build_directory(ROOT) / 'bench-backend'), backend, str(directory), args.memory,
                                 args.text, '1' if args.direction == 'en-ja' else '2'],
                                cwd=p.data_root(), check=True, capture_output=True, text=True)
        phases = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(phases) == 3 and phases[0]['phase'] == 'open' and phases[1]['phase'].startswith('prefill_') and phases[2]['phase'] == 'decode', phases
        tokens = [x['token'] for x in phases[1:]]
        if reference_tokens is None: reference_tokens = tokens
        assert tokens == reference_tokens, phases
        if args.text == 'Hello.' and args.direction == 'en-ja': assert tokens == [51871, 16], phases
        assert all(not x['read_failures'] and not x['gpu_errors'] for x in phases), phases
        if backend != 'cpu':
            assert phases[-1]['backend'] == 'metal' and phases[-1]['gpu_dispatches'] > 0, phases
        runs[backend] = phases
        if backend != 'cpu':
            errors[backend] = {}
            for name in (phases[1]['phase'] + '-logits.bin', phases[1]['phase'] + '-state.bin', 'decode-logits.bin', 'decode-state.bin'):
                errors[backend][name] = compare(work / 'cpu' / name, directory / name, 0.0001)
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model': p.name, 'memory': args.memory, 'text': args.text, 'direction': args.direction,
              'runs': runs, 'numerical_errors': errors,
              'limits': 'One prompt and one subsequent decode; numerical checks are not a broad translation-quality evaluation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(errors, indent=2), flush=True)


if __name__ == '__main__':
    main()
