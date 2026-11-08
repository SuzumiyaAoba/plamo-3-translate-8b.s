#!/usr/bin/env python3
"""Compare assembly boundary markers with upstream Python regex semantics."""
import ctypes as C
from pathlib import Path
import random
import re
import subprocess
ROOT = Path(__file__).resolve().parents[1]

def main():
    target = ROOT / 'build/boundary.dylib'
    target.parent.mkdir(exist_ok=True)
    subprocess.run(['clang', '-arch', 'arm64', '-dynamiclib', str(ROOT/'src/boundary.s'), '-o', str(target)], check=True)
    lib = C.CDLL(str(target))
    rng = random.Random(3)
    cases = ['', '<|plamo:|>', '<|plamo:bos|>', '<|plamo:'+'a'*64+'|>', '<|plamo:'+'a'*65+'|>', '<|plamo:x|><|plamo:y|>']
    whitespace = [chr(i) for i in range(0x110000) if chr(i).isspace()]
    cases += ['<|plamo:a'+c+'b|>' for c in whitespace]
    for _ in range(1000):
        s = ''.join(rng.choice('日a \n\t')*rng.randrange(1,8) for _ in range(rng.randrange(30)))
        cases += [s, s+'<|plamo:'+rng.choice(['x','日','a b','a|b',''])+'|>'+s]
    for name, pattern in [('repeat', r'((.)\2{3,}| {2,})'), ('special', r'(<\|plamo:[^|\s]{,64}\|>)')]:
        f = getattr(lib, f'plamo_{name}_boundaries')
        f.argtypes = [C.c_void_p,C.c_uint64,C.c_void_p]
        for s in cases:
            cp = (C.c_uint32*max(1,len(s)))(*map(ord,s))
            flags = (C.c_uint8*(len(s)+1))()
            f(cp,len(s),flags)
            expected = set()
            for m in re.finditer(pattern,s):
                expected.update((m.start(),m.end()))
            assert {i for i,v in enumerate(flags) if v} == expected, (name,s)
    print(f'boundary regex equivalence: PASS ({len(cases)} cases per stage)')

if __name__ == '__main__':
    main()
