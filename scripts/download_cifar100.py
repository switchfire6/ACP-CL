"""Download canonical CIFAR-100 in ranges, then verify torchvision's checksum.

Useful when the Toronto server limits throughput on a single connection. No
mirror or transformed dataset is used. Run before starting an experiment.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import urllib.request

from torchvision.datasets import CIFAR100


def digest(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def download(root: Path, workers: int) -> Path:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1, 8]")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / CIFAR100.filename
    if destination.exists() and digest(destination) == CIFAR100.tgz_md5:
        print(f"Already verified: {destination}", flush=True)
        return destination
    request = urllib.request.Request(CIFAR100.url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        length = int(response.headers["Content-Length"])
        canonical_url = response.url
    parts_dir = root / ".cifar100-download"
    parts_dir.mkdir(exist_ok=True)
    chunk_size = (length+workers-1)//workers
    ranges = [(start, min(start+chunk_size, length)-1) for start in range(0, length, chunk_size)]

    def part(index, start, end):
        path = parts_dir / f"{start}-{end}.part"
        if path.exists() and path.stat().st_size == end-start+1:
            return index, path
        request = urllib.request.Request(canonical_url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{length}":
                raise RuntimeError("server did not honor the requested byte range")
            shutil.copyfileobj(response, handle, length=1024*1024)
        if path.stat().st_size != end-start+1:
            raise RuntimeError(f"truncated download: {path}")
        return index, path

    paths = {}
    print(f"Downloading {length:,} bytes from {canonical_url} with {workers} connections", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(part, i, start, end) for i, (start, end) in enumerate(ranges)]
        for future in as_completed(futures):
            i, path = future.result()
            paths[i] = path
            print(f"Verified range length {i+1}/{len(ranges)}", flush=True)
    temporary = root / (CIFAR100.filename+".verified-download")
    with temporary.open("wb") as output:
        for i in sorted(paths):
            with paths[i].open("rb") as source:
                shutil.copyfileobj(source, output)
    if digest(temporary) != CIFAR100.tgz_md5:
        raise RuntimeError("canonical CIFAR checksum mismatch; dataset was not installed")
    temporary.replace(destination)
    # Only remove the exact part files created by this download, never recurse.
    for path in paths.values():
        path.unlink()
    print(f"Verified canonical MD5 {CIFAR100.tgz_md5}: {destination}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    options = parser.parse_args()
    download(options.root, options.workers)
