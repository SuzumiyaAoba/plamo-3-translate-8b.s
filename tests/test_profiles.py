"""Pinned configurations and real native loading for all four model shapes."""
import json
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from tools.model_catalog import MODELS, profile
from tools.generate_model import render_profile
from tools.generate_launchers import render
from tools import model_layout, rope_constants, weight_schema

ROOT = Path(__file__).resolve().parents[1]


class ProfileTests(unittest.TestCase):
    def test_generated_default_is_unchanged(self):
        p = profile()
        self.assertEqual(p.config, json.loads((ROOT / "docs/config.json").read_text()))
        self.assertEqual(p.manifest, json.loads((ROOT / "tools/model-manifest.json").read_text()))
        for name, text in (("weight_schema.s", weight_schema.render(p.config)),
                           ("model_layout.s", model_layout.render(p.config)),
                           ("rope_constants.s", rope_constants.render(p.config)),
                           ("model_profile.s", render_profile(p))):
            self.assertEqual((ROOT / "src" / name).read_text(), text, name)
        self.assertEqual((ROOT / "nix/model-profiles.sh").read_text(), render())

    def test_profile_shapes_and_original_tokenizer(self):
        expected = {"0.8b": (1536, 16, 16, 1, 1), "2b": (2560, 24, 24, 4, 2),
                    "8b": (4096, 32, 32, 4, 4), "31b": (5120, 64, 64, 8, 14)}
        directories = set()
        for size in MODELS:
            p = profile(size); c = p.config
            self.assertEqual(hashlib.sha256((ROOT / 'tools/profiles' / size / 'config.json').read_bytes()).hexdigest(),
                             p.manifest['files']['config.json']['sha256'])
            self.assertEqual((c['hidden_size'], c['num_hidden_layers'], c['num_attention_heads'],
                              c['num_key_value_heads'], len(p.weights)), expected[size])
            for name in ('tokenizer.jsonl', 'modeling_plamo.py', 'tokenizer_config.json',
                         'chat_template.jinja', 'generation_config.json', 'LICENSE/en', 'LICENSE/ja'):
                self.assertEqual(p.manifest['files'][name]['sha256'], profile().manifest['files'][name]['sha256'])
            self.assertNotIn(p.directory, directories)
            directories.add(p.directory)
            shapes = dict(weight_schema.schema(c))
            self.assertEqual(shapes['model.layers.layers.0.mixer.o_proj.weight'],
                             [c['hidden_size'], c['num_attention_heads'] * c['head_dim']])
            self.assertEqual('lm_head.weight' not in shapes, c['tie_word_embeddings'])

    def test_native_loader_for_each_profile(self):
        for size in MODELS:
            with self.subTest(model=size):
                p = profile(size); c = p.config
                probe = p.build_directory(ROOT) / 'test-profile'
                if not probe.exists():
                    self.skipTest('run make test-tools to build profile probes')
                rows = weight_schema.schema(c)
                # Sparse holes represent original BF16 tensors without allocating RAM/disk payloads.
                groups = [[] for _ in p.weights]
                for i, row in enumerate(rows):
                    groups[i % len(groups)].append(row)
                with tempfile.TemporaryDirectory(prefix='plamo profile ') as folder:
                    model_dir = Path(folder) / p.directory
                    model_dir.mkdir(parents=True)
                    for filename, group in zip(p.weights, groups):
                        header = {}; offset = 0
                        for name, shape in group:
                            end = offset + math.prod(shape) * 2
                            header[name] = dict(dtype='BF16', shape=shape, data_offsets=[offset, end])
                            offset = end
                        data = json.dumps(header, separators=(',', ':')).encode()
                        data += b' ' * (-len(data) % 8)
                        with (model_dir / filename).open('wb') as target:
                            target.write(struct.pack('<Q', len(data)) + data)
                            target.truncate(8 + len(data) + offset)
                    last = model_dir / p.weights[-1]
                    last.rename(last.with_name(last.name + '.withheld'))
                    result = subprocess.run([probe, str(c['hidden_size']), str(c['num_hidden_layers']),
                                             str(c['num_attention_heads']), str(c['num_key_value_heads']),
                                             str(int(c['tie_word_embeddings'])), str(len(rows))], cwd=folder, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
