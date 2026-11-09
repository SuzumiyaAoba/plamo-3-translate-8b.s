#!/usr/bin/env python3
"""Embed the readable Metal source for runtime compilation without an SDK tool."""
import json
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:])
output.write_text('static const char plamo_metal_source[] =\n' + json.dumps(source.read_text(), ensure_ascii=True) + ';\n')
