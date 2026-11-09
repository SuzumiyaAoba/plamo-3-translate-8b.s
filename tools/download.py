#!/usr/bin/env python3
"""Download and verify original PLaMo files without converting them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request
try:
    from .model_catalog import add_model_argument, profile
except ImportError:
    from model_catalog import add_model_argument, profile

MANIFEST = profile().manifest
REPO = MANIFEST["repository"]
REVISION = MANIFEST["revision"]
SMALL = tuple(name for name in MANIFEST["files"] if not name.endswith(".safetensors"))
WEIGHTS = tuple(name for name in MANIFEST["files"] if name.endswith(".safetensors"))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_file(name, directory, expected, verify_existing=False, model_profile=None):
    """Publish a file atomically only after its pinned size and SHA-256 match."""
    target = directory / name
    if target.is_file() and target.stat().st_size == expected["size"]:
        if not verify_existing and name.endswith(".safetensors"):
            print(f"exists (size checked): {target}", flush=True)
            return
        if sha256_file(target) == expected["sha256"]:
            print(f"exists (SHA-256 verified): {target}", flush=True)
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    p = model_profile or profile()
    url = f"https://huggingface.co/{p.repository}/resolve/{p.revision}/{name}?download=true"
    print(f"download: {name} ({expected['size']:,} bytes)", flush=True)
    try:
        digest = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                digest.update(block)
                total += len(block)
        if total != expected["size"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"size or SHA-256 mismatch: {name}; rerun to download again")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Built with PLaMo. Original checkpoints use the PLaMo Community License.",
    )
    add_model_argument(parser)
    parser.add_argument("--directory", type=Path, help="override the selected model's storage directory")
    parser.add_argument("--cache", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--weights", action="store_true", help="also download original BF16 weights (0.8b: 1.78 GB; 2b: 5.76 GB; 8b: 17.06 GB; 31b: 63.51 GB)")
    parser.add_argument("--verify", action="store_true", help="also SHA-256-check cached weight shards (all new downloads are always checked)")
    args = parser.parse_args()
    try:
        p = profile(args.model)
        args.directory = args.directory or p.data_root(args.cache) / p.directory
        marker = args.directory / "REVISION"
        previous_revision = marker.read_text().strip() if marker.exists() else None
        if previous_revision not in (None, p.revision):
            raise ValueError("this directory contains another model revision; use a separate --directory")
        args.directory.mkdir(parents=True, exist_ok=True)
        small = tuple(n for n in p.manifest["files"] if not n.endswith(".safetensors"))
        print(f"{p.name} @ {p.revision} -> {args.directory}", flush=True)
        for name in small + (p.weights if args.weights else ()):
            download_file(name, args.directory, p.manifest["files"][name],
                          verify_existing=args.verify or previous_revision is None, model_profile=p)
        marker.write_text(p.revision + "\n")
        print(f"Model license: {args.directory / 'LICENSE/ja'} (English: LICENSE/en)", flush=True)
    except (OSError, ValueError) as error:
        parser.exit(1, f"plamo-download: {error}\n")


if __name__ == "__main__":
    main()
