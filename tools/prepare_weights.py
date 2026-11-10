#!/usr/bin/env python3
"""Build a page-aligned, exact BF16 GPU cache once, without changing the originals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from model_catalog import add_model_argument, profile
from weight_schema import schema

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 64 * 1024 * 1024


def plan(p, directory, packed):
    tensors, sources, stamps = {}, [], []
    for index, name in enumerate(p.weights):
        path = (directory / name).resolve()
        st = path.stat()
        if st.st_size != p.manifest['files'][name]['size']:
            raise ValueError(f'wrong original file size: {path}')
        stamps.append((path, st.st_size, st.st_mtime_ns))
        print(f'verify original: {name}', flush=True)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
            stream.seek(0)
            length = struct.unpack('<Q', stream.read(8))[0]
            if not 2 <= length <= 16 * 1024 * 1024:
                raise ValueError(f'invalid safetensors header: {path}')
            header = json.loads(stream.read(length))
        if digest.hexdigest() != p.manifest['files'][name]['sha256']:
            raise ValueError(f'original SHA-256 mismatch: {path}')
        sources.append({'path': str(path), 'bytes': st.st_size})
        for name, item in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise ValueError(f'duplicate tensor: {name}')
            begin, end = item['data_offsets']
            if not 0 <= begin <= end <= st.st_size - 8 - length:
                raise ValueError(f'invalid tensor range: {name}')
            tensors[name] = (index, 8 + length + begin, end - begin, item)
    names = []
    for layer in range(p.config['num_hidden_layers']):
        prefix = f'model.layers.layers.{layer}.'
        names += [prefix + suffix + '.weight' for suffix in ('mixer.qkv_proj', 'mixer.o_proj', 'mlp.gate_up_proj', 'mlp.down_proj')]
    names += ['model.embed_tokens.weight' if p.config['tie_word_embeddings'] else 'lm_head.weight']
    expected = dict(schema(p.config))
    chunks = []
    for name in names:
        source, offset, size, item = tensors[name]
        rows, columns = expected[name]
        if item['dtype'] != 'BF16' or item['shape'] != [rows, columns] or size != rows * columns * 2:
            raise ValueError(f'invalid BF16 projection: {name}')
        window = LIMIT // (columns * 2)
        if not window:
            raise ValueError(f'row wider than conversion window: {name}')
        if window >= 8:
            window = window // 8 * 8
        for first in range(0, rows, window):
            chunks.append({'source': source, 'offset': offset + first * columns * 2,
                           'rows': min(rows - first, window), 'columns': columns})
    return {'revision': p.revision, 'sources': sources, 'chunks': chunks, 'packed': packed}, stamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_argument(parser)
    parser.add_argument('--directory', type=Path, help='original model directory')
    parser.add_argument('--cache', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--storage', choices=('raw', 'lossless'), default='lossless')
    parser.add_argument('--output', type=Path, help='default: MODEL_DIRECTORY/metal-cache-v1.bin')
    args = parser.parse_args()
    p = profile(args.model)
    directory = args.directory or p.data_root(args.cache) / p.directory
    output = (args.output or directory / 'metal-cache-v1.bin').resolve()
    report_path = output.with_suffix(output.suffix + '.json').resolve()
    originals = {(directory / name).resolve() for name in p.manifest['files']}
    if output in originals or report_path in originals:
        parser.error('output and report must differ from all original model files')
    output.parent.mkdir(parents=True, exist_ok=True)
    packer = Path(os.environ.get('PLAMO_PACK_TOOL', ROOT / 'build/pack-metal-weights'))
    if 'PLAMO_PACK_TOOL' not in os.environ:
        subprocess.run(['make', 'build/pack-metal-weights'], cwd=ROOT, check=True)
    start = time.monotonic()
    try:
        contents, stamps = plan(p, directory, args.storage == 'lossless')
        with tempfile.TemporaryDirectory(prefix='.metal-cache-', dir=output.parent) as temporary:
            temporary = Path(temporary)
            path = temporary / 'plan.json'
            path.write_text(json.dumps(contents))
            run = subprocess.run([str(packer), str(path), str(temporary / 'weights.bin')], text=True, stdout=subprocess.PIPE, check=True)
            report = json.loads(run.stdout)
            for path, size, mtime in stamps:
                st = path.stat()
                if (st.st_size, st.st_mtime_ns) != (size, mtime):
                    raise ValueError(f'original changed during conversion: {path}')
            os.replace(temporary / 'weights.bin', output)
        report.update(model=p.name, revision=p.revision, storage=args.storage, prepare_seconds=time.monotonic() - start)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(f'ready: {output} ({report["file_bytes"]:,} bytes)', flush=True)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'prepare_weights: {error}\n')


if __name__ == '__main__':
    main()
