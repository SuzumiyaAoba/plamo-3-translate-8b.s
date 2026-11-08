#!/usr/bin/env python3
"""Compare assembly segment encoding against independent exhaustive-prefix DP.

This verifies the Unigram core with original vocabulary, not boundary preprocessing.
"""
import ctypes as C
import json
from pathlib import Path
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
class Trie(C.Structure):
    _fields_ = [('nodes', C.c_void_p), ('capacity', C.c_uint64), ('used', C.c_uint64), ('roots', C.c_void_p)]
class Descriptor(C.Structure):
    _fields_ = [('trie', C.POINTER(Trie)), ('cp', C.c_void_p), ('n', C.c_uint64),
                ('bytes', C.c_void_p), ('costs', C.c_void_p), ('paths', C.c_void_p),
                ('candidates', C.c_void_p), ('capacity', C.c_uint64),
                ('out', C.c_void_p), ('outcap', C.c_uint64)]

def main():
    sources = ['encode', 'trie', 'unigram', 'utf8', 'vocab_load', 'vocab', 'string', 'score', 'json', 'load', 'marked', 'preprocess', 'boundary']
    target = ROOT / 'build/segments.dylib'
    target.parent.mkdir(exist_ok=True)
    subprocess.run(['clang', '-arch', 'arm64', '-dynamiclib', *[str(ROOT / f'src/{s}.s') for s in sources], '-o', str(target)], check=True)
    lib = C.CDLL(str(target))
    load = lib.plamo_vocab_load
    load.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Trie), C.c_void_p, C.c_uint64, C.c_void_p]
    load.restype = C.c_uint64
    encode = lib.plamo_encode_segment
    encode.argtypes = [C.POINTER(Descriptor)]
    encode.restype = C.c_uint64
    raw = (ROOT / 'models/plamo/tokenizer.jsonl').read_bytes()
    data = C.create_string_buffer(raw)
    nodes = C.create_string_buffer(24 * 1000000)
    roots = (C.c_uint32 * 0x110000)()
    trie = Trie(C.addressof(nodes), 1000000, 1, C.addressof(roots))
    scratch = (C.c_uint32 * 4096)()
    byte_ids = (C.c_uint32 * 256)()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert load(C.addressof(data), C.addressof(data)+len(raw), C.byref(trie), scratch, 4096, byte_ids) == len(rows)
    # Independent Python prefix tree and backward DP; no assembly intermediates.
    tree = {}
    fallback = {}
    for token_id, (piece, score, kind, *_) in enumerate(rows):
        if kind == 'BYTE':
            fallback[int(piece[3:5], 16)] = token_id
            continue
        node = tree
        for char in piece:
            node = node.setdefault(char, {})
        node[None] = (token_id, round(score * 10000))
    def reference(text):
        n = len(text)
        costs = [0] * (n+1)
        paths = [None] * n
        for i in range(n-1, -1, -1):
            options = [(costs[i+1]+10000000, -1, -1)]
            node = tree
            for j in range(i, n):
                node = node.get(text[j])
                if node is None:
                    break
                if None in node:
                    token_id, score = node[None]
                    options.append((costs[j+1]-score, -(j+1-i), token_id))
            cost, negative_length, token_id = min(options)
            costs[i] = cost
            paths[i] = (-negative_length, token_id)
        result = []
        pos = 0
        while pos < n:
            length, token_id = paths[pos]
            result.extend([token_id] if token_id >= 0 else [fallback[b] for b in text[pos].encode()])
            pos += length
        return result
    cases = ['', 'Hello, world!', '日本語を英語に翻訳してください。', 'こんにちは😀', '𠮷野家', '\x00\U0010ffff', 'aaaaa', '    ', 'The quick brown fox jumps over the lazy dog.', '<|plamo:bos|>']
    rng = random.Random(42)
    for _ in range(200):
        cases.append(''.join(rng.choice('日本語東京abcXYZ0123、。😀\n \t') for _ in range(rng.randrange(1, 60))))
    for text in cases:
        n = len(text)
        cp = (C.c_uint32 * max(n, 1))(*map(ord, text))
        costs = (C.c_int64 * (n+1))()
        paths = C.create_string_buffer(max(n, 1)*24)
        candidates = C.create_string_buffer((n+1)*16)
        output = (C.c_uint32 * (4*n+1))()
        d = Descriptor(C.pointer(trie), C.addressof(cp), n, C.addressof(byte_ids), C.addressof(costs), C.addressof(paths), C.addressof(candidates), n+1, C.addressof(output), 4*n+1)
        count = encode(C.byref(d))
        assert count <= 4*n, (text, count)
        actual = list(output[:count])
        expected = reference(text)
        assert actual == expected, (text, actual, expected)
    class Prep(C.Structure):
        _fields_ = [('input',C.c_void_p),('n',C.c_uint64),('temp',C.c_void_p),('tcap',C.c_uint64),('output',C.c_void_p),('ocap',C.c_uint64),('flags',C.c_void_p)]
    prep=lib.plamo_preprocess
    prep.argtypes=[C.POINTER(Prep)];prep.restype=C.c_uint64
    marked=lib.plamo_encode_marked
    marked.argtypes=[C.POINTER(Descriptor)];marked.restype=C.c_uint64
    for text in cases + ['a  b', 'x<|plamo:bos|>aaaa', '<|plamo:x|><|plamo:y|>', '日日日日東京', '\uee00a\uee00']:
        n=len(text);cap=9*n+10
        cp=(C.c_uint32*max(n,1))(*map(ord,text));temp=(C.c_uint32*cap)();expanded=(C.c_uint32*cap)();flags=(C.c_uint8*(cap+1))()
        pd=Prep(C.addressof(cp),n,C.addressof(temp),cap,C.addressof(expanded),cap,C.addressof(flags))
        k=prep(C.byref(pd));assert k<cap
        costs=(C.c_int64*(k+1))();paths=C.create_string_buffer(max(k,1)*24);candidates=C.create_string_buffer((k+1)*16);output=(C.c_uint32*(4*k+1))()
        d=Descriptor(C.pointer(trie),C.addressof(expanded),k,C.addressof(byte_ids),C.addressof(costs),C.addressof(paths),C.addressof(candidates),k+1,C.addressof(output),4*k+1)
        count=marked(C.byref(d));assert count<=4*k
        transformed=re.sub(r'(<\|plamo:[^|\s]{,64}\|>)',lambda m:'\uee00'+m[0]+'\uee00',text)
        transformed=re.sub(r'((.)\2{3,}| {2,})',lambda m:'\uee00'+m[0]+'\uee00',transformed)
        expected=[]
        for part in transformed.split('\uee00'):expected.extend(reference(part))
        assert list(output[:count])==expected,(text,list(output[:count]),expected)
    print('preprocessing + marked encoding: PASS')
    print(f'official-vocabulary Unigram segments: PASS ({len(cases)} cases; {trie.used} nodes)')

if __name__ == '__main__':
    main()
