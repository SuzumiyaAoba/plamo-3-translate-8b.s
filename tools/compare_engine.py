#!/usr/bin/env python3
"""Compare every output byte of two assembly engines on the real 8B model.

Both benchmark executables must support the optional dump-directory argument.
Save build/bench-model before optimizing, then pass it as --baseline.
"""
import argparse
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline',type=Path)
    group.add_argument('--reference-dumps',type=Path,help='reuse byte dumps from a previous baseline run')
    p.add_argument('--output',type=Path,default=ROOT/'build/equivalence.json')
    args=p.parse_args()
    subprocess.run(['make','build/bench-model'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    work=ROOT/'build/equivalence';work.mkdir(exist_ok=True)
    runs={}
    programs=[('optimized',ROOT/'build/bench-model',2)]
    if args.baseline:programs.insert(0,('baseline',args.baseline.resolve(),0))
    for name,binary,policy in programs:
        folder=work/name;folder.mkdir(exist_ok=True)
        print(f'Running {name} engine...',flush=True)
        result=subprocess.run([str(binary),str(policy),str(folder)],cwd=ROOT,check=True,capture_output=True,text=True)
        runs[name]=[json.loads(line) for line in result.stdout.splitlines()]
    reference=args.reference_dumps.resolve() if args.reference_dumps else work/'baseline'
    files=['prefill_21-logits.bin','prefill_21-state.bin','decode-logits.bin','decode-state.bin']
    compared=0
    for name in files:
        before=(reference/name).read_bytes();after=(work/'optimized'/name).read_bytes()
        assert before and before==after,name
        compared+=len(before)
    report={'bitwise_equal':True,'reference_directory':str(reference.relative_to(ROOT) if reference.is_relative_to(ROOT) else reference),'files':files,'compared_bytes':compared,'runs':runs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Engine byte-for-byte comparison: PASS ({compared:,} bytes; prefill and cached decode)')
if __name__=='__main__':main()
