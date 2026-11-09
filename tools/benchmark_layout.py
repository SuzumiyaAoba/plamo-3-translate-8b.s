#!/usr/bin/env python3
"""Compare exact BF16 GPU layouts with co-designed FP32 matvec/matmul kernels."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
from model_catalog import add_model_argument, profile

ROOT = Path(__file__).resolve().parents[1]


def sample(p, tensor, maximum, directory):
    for name in p.weights:
        path = p.data_root() / p.directory / name
        with path.open('rb', buffering=0) as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            assert 2 <= length <= 16 * 1024 * 1024
            header = json.loads(stream.read(length))
            if tensor not in header:
                continue
            record = header[tensor]
            assert record['dtype'] == 'BF16' and len(record['shape']) == 2
            rows, cols = record['shape']
            rows = min(rows, maximum // (cols * 2)) // 32 * 32
            assert rows and cols % 256 == 0
            stream.seek(8 + length + record['data_offsets'][0])
            data = stream.read(rows * cols * 2)
            assert len(data) == rows * cols * 2
        output = directory / (tensor.split('.')[-2] + '.bf16')
        output.write_bytes(data)
        return output, {'tensor': tensor, 'shape': record['shape'], 'sample_rows': rows,
                        'columns': cols, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    raise ValueError(f'missing tensor: {tensor}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--sample-mib', type=int, default=64)
    parser.add_argument('--validate', action='store_true', help='enable GPU API/shader validation; skip timing')
    parser.add_argument('--report', type=Path, default=ROOT / 'build/gpu-layout-benchmark.json')
    args = parser.parse_args()
    if not 1 <= args.sample_mib <= 256:
        parser.error('sample-mib must be 1..256')
    p = profile(args.model)
    subprocess.run(['make', 'build/bench-metal-layout'], cwd=ROOT, check=True)
    directory = ROOT / 'build/gpu-layout-study' / p.size
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / 'runtime/kernels.metal', ROOT / 'runtime/packing.h', ROOT / 'tests/metal_layout.metal',
               ROOT / 'tests/bench_metal_layout.m', ROOT / 'tools/embed_metal_layout.py', Path(__file__).resolve()]
    hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    results = []
    environment = dict(os.environ)
    environment['MTL_DEBUG_LAYER'] = environment['MTL_SHADER_VALIDATION'] = '1' if args.validate else '0'
    for projection in ('gate_up_proj', 'down_proj'):
        tensor = f'model.layers.layers.0.mlp.{projection}.weight'
        path, metadata = sample(p, tensor, args.sample_mib * 1024 * 1024, directory)
        command = [str(ROOT / 'build/bench-metal-layout'), str(path), str(metadata['columns'])]
        if args.validate:
            command.append('--validate')
        run = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        (directory / (projection + '.log')).write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(run.stderr)
        measurements = [json.loads(line) for line in run.stdout.splitlines()]
        for item in measurements:
            baseline = next(r for r in measurements if r['layout'] == 0 and r['lossless'] == item['lossless'] and r['tokens'] == item['tokens'])
            item['gpu_speedup'] = baseline['gpu_seconds'] / item['gpu_seconds']
            item['storage_ratio'] = item['storage_bytes'] / baseline['storage_bytes']
        results.append({'sample': metadata, 'validation_passed': True, 'bf16_patterns_tested': 65536,
                        'tokens_tested': [1, 8, 9, 21, 69, 81], 'measurements': measurements})
        print(f'{p.size} {projection}: exact BF16/FP32 parity passed; {len(measurements)} measurements', flush=True)
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    report = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model': p.name, 'revision': p.revision, 'system': platform.system(), 'machine': platform.machine(),
              'os_version': platform.mac_ver()[0],
              'gpu_validation': args.validate, 'source_sha256': hashes,
              'layouts': {'0': 'row-major baseline', '1': '4 rows x 8 columns; coalesced SIMD reads',
                          '2': '8 rows x 32 columns; coalesced SIMD reads',
                          '3': '4 rows x 8 columns; one row per SIMD group',
                          '4': '8 rows x 32 columns; one row per SIMD group',
                          '5': '32 rows x 32 columns; one row per SIMD group'},
              'results': results,
              'method': 'Three warmup rounds, then seven samples with rotating layout/storage order; '
                        'each sample batches 5..128 projections in one command, targeting at least 30 ms based on the baseline pilot. '
                        'Records median GPU and host wall time and GPU quartiles; layout preparation and lossless encoding excluded from execution time. '
                        'All 65536 BF16 patterns and all real sample values are reconstructed bitwise. '
                        'FP32 outputs for 1/8/9/21/69/81 tokens must match the production row-major kernels bitwise.',
              'limits': 'Aligned real-weight submatrices, not complete model/API measurements. '
                        'New layouts are benchmark-only; no production default is changed.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
