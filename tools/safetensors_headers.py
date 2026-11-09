"""Fetch bounded safetensors headers, refusing a server that ignores Range."""
import concurrent.futures
import json
from pathlib import Path
import struct
import urllib.request

MAX_HEADER = 16 * 1024 * 1024


def fetch_header(model_profile, filename):
    p = model_profile
    size = p.manifest['files'][filename]['size']
    url = f'https://huggingface.co/{p.repository}/resolve/{p.revision}/{filename}?download=true'

    def read_range(end):
        request = urllib.request.Request(url + f'&header_range={end}',
                                         headers={'Range': f'bytes=0-{end}', 'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes 0-{end}/{size}':
                raise ValueError(f'{filename}: server did not honor the bounded byte range')
            data = response.read(end + 2)
            if len(data) != end + 1:
                raise ValueError(f'{filename}: incomplete header response')
            return data

    length = struct.unpack('<Q', read_range(7))[0]
    if not 2 <= length <= MAX_HEADER or length + 8 > size:
        raise ValueError(f'{filename}: invalid safetensors header length')
    data = read_range(length + 7)
    if data[:8] != struct.pack('<Q', length):
        raise ValueError(f'{filename}: inconsistent header responses')
    json.loads(data[8:])
    return data


def sparse_header_files(p, directory, cache=None):
    """Create sparse files with real headers; tensor payloads remain holes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache) / p.size if cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def one(filename):
        cached = cache_dir / (filename + '.header') if cache_dir else None
        data = cached.read_bytes() if cached and cached.exists() else fetch_header(p, filename)
        if len(data) < 10 or struct.unpack('<Q', data[:8])[0] != len(data) - 8 or len(data) > MAX_HEADER + 8:
            raise ValueError(f'{filename}: invalid cached header')
        json.loads(data[8:])
        if cached and not cached.exists():
            cached.write_bytes(data)
        with (directory / filename).open('wb') as output:
            output.write(data)
            output.truncate(p.manifest['files'][filename]['size'])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, p.weights))
