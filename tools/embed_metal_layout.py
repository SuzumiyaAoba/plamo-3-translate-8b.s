#!/usr/bin/env python3
"""Reuse the production FP32 matrix tile with benchmark-only weight readers."""
import json
from pathlib import Path
import sys

runtime, probe, output = map(Path, sys.argv[1:])
base = runtime.read_text()
extra = probe.read_text()
begin = extra.index('// BEGIN_LAYOUT_READERS')
end = extra.index('// END_LAYOUT_READERS') + len('// END_LAYOUT_READERS')
readers = extra[begin:end]
extra = extra[:begin] + extra[end:]
start = base.index('// 32 output rows x 8/16/24/32 prompt tokens.')
finish = base.index('#define VECTOR_KERNEL', start)
tile = base[start:finish]
signature = 'template<bool Packed, uint N>'
assert tile.count(signature) == 1
tile = tile.replace(signature, 'template<bool Packed, uint N, uint R = 0, uint K = 0>')
accumulators = '    simdgroup_float8x8 total0(0.0f), total1(0.0f), total2(0.0f), total3(0.0f);'
assert tile.count(accumulators) == 1
tile = tile.replace(accumulators, accumulators + '\n    ulong layout_first = layout_matrix_index<R, K>(group.x * 32 + tid / 8, (tid % 8) * 4, p.columns);')
for old, new in (('weight4<Packed>(w, ulong(row) * p.columns + col, p)',
                  'layout_matrix_weight4<Packed, R, K>(w, row, col, layout_first + ulong(i / 128) * (R > 16 ? 16 * K : 16 * p.columns) + ulong(base) * R, p)'),
                 ('weight_bits<Packed>(w, ulong(row) * p.columns + col + j, p)',
                  'layout_matrix_bits<Packed, R, K>(w, row, col + j, p)')):
    assert tile.count(old) == 1, old
    tile = tile.replace(old, new)
source = base[:start] + readers + '\n' + tile + base[finish:] + '\n' + extra
output.write_text('static const char plamo_metal_source[] =\n' + json.dumps(source, ensure_ascii=True) + ';\n')
