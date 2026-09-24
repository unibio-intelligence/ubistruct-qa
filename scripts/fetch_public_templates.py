#!/usr/bin/env python3
"""Download fixed public structural references from RCSB PDB."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

FILES = {
    "6FDB-assembly1.cif.gz": "https://files.rcsb.org/download/6FDB-assembly1.cif.gz",
    "7QV6.pdb": "https://files.rcsb.org/download/7QV6.pdb",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in FILES.items():
        destination = args.output_dir / filename
        if destination.exists() and not args.force:
            print(f"kept {destination} sha256={sha256(destination)}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                temporary.write_bytes(response.read())
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"downloaded {destination} sha256={sha256(destination)} source={url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
