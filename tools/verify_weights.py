#!/usr/bin/env python3
"""Check assembly lookup/shape validation against original shard JSON headers."""
import ctypes as C
import json
import mmap
from pathlib import Path
import struct
import subprocess
import argparse
import tempfile
from weight_schema import schema
from model_catalog import add_model_argument, profile
from safetensors_headers import sparse_header_files
ROOT=Path(__file__).resolve().parents[1]
class Shard(C.Structure):
    _fields_=[('base',C.c_void_p),('size',C.c_uint64)]
def verify(p, directory):
    target=ROOT/f'build/weights-{p.size}.dylib';target.parent.mkdir(exist_ok=True)
    subprocess.run(['clang','-arch','arm64','-dynamiclib',*[str(ROOT/f'src/{s}.s') for s in ['bind','resolve','find','tensor','json','load']],'-o',str(target)],check=True)
    lib=C.CDLL(str(target));resolve=lib.plamo_resolve_weight
    resolve.argtypes=[C.POINTER(Shard),C.c_uint64,C.c_char_p,C.c_uint64,C.c_void_p];resolve.restype=C.c_void_p
    files=[directory/name for name in p.weights]
    if not all(f.is_file() for f in files):raise SystemExit(f'No complete weight files; run tools/download.py --model {p.size} --weights')
    mappings=[];shards=(Shard*len(files))();records=[]
    for i,path in enumerate(files):
        with path.open('rb') as f:
            m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY)
        mappings.append(m);base=C.addressof(C.c_char.from_buffer(m));shards[i]=Shard(base,len(m))
        size=struct.unpack_from('<Q',m)[0];header=json.loads(m[8:8+size])
        for name,t in header.items():
            if name=='__metadata__':continue
            assert t['dtype']=='BF16'
            records.append((name,t,base+8+size+t['data_offsets'][0]))
    for name,t,expected in records:
        dims=(C.c_uint64*len(t['shape']))(*t['shape'])
        assert resolve(shards,len(files),name.encode(),len(dims),dims)==expected,name
        if len(dims):
            dims[0]+=1
            assert not resolve(shards,len(files),name.encode(),len(dims),dims),name
    assert not resolve(shards,len(files),b'no.such.weight',0,None)
    expected_shapes=dict(schema(p.config))
    assert set(expected_shapes)=={name for name,_,_ in records}
    for name,t,_ in records:assert expected_shapes[name]==t['shape'],name
    class Binding(C.Structure):
        _fields_=[('name',C.c_char_p),('rank',C.c_uint64),('dims',C.c_void_p),('slot',C.c_void_p)]
    bindings=(Binding*(len(records)+1))();slots=(C.c_void_p*len(records))();keep=[]
    for i,(name,t,_) in enumerate(records):
        dims=(C.c_uint64*len(t['shape']))(*expected_shapes[name]);keep.append(dims)
        bindings[i]=Binding(name.encode(),len(dims),C.addressof(dims),C.addressof(slots)+i*C.sizeof(C.c_void_p))
    bind=lib.plamo_bind_weights;bind.argtypes=[C.POINTER(Shard),C.c_uint64,C.POINTER(Binding)];bind.restype=C.c_int
    assert bind(shards,len(files),bindings)==1
    assert list(slots)==[r[2] for r in records]
    bindings[0].name=b'no.such.weight'
    assert bind(shards,len(files),bindings)==0 and not any(slots)
    print('model schema / transactional binding: PASS')
    print(f'{p.name} original BF16 weight resolution: PASS ({len(files)} files, {len(records)} tensors; shape mismatches rejected)')
    # Drop ctypes exports before closing mappings.
    for m in mappings:m.close()
def main():
    parser=argparse.ArgumentParser(description=__doc__);add_model_argument(parser);parser.add_argument('--directory',type=Path)
    parser.add_argument('--headers-only',action='store_true',help='fetch only original headers and check native binding using temporary sparse files; no weight payload download')
    parser.add_argument('--header-cache',type=Path,help='optional cache of pinned headers for repeated offline checks')
    args=parser.parse_args();p=profile(args.model)
    if args.headers_only:
        with tempfile.TemporaryDirectory(prefix='plamo headers ') as folder:
            sparse_header_files(p,Path(folder),args.header_cache)
            verify(p,Path(folder))
        print('Header-only check: weight payloads were not downloaded or verified.')
    else:verify(p,args.directory or p.data_root()/p.directory)
if __name__=='__main__':main()
