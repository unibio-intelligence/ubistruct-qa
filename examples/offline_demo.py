#!/usr/bin/env python3
"""Exercise installed scoring workflows on synthetic coordinates, entirely offline."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import casp17_cage_qa as cage
import casp17_fibril_qa as peptide
import casp17_qa as immune

AA1 = {v: k for k, v in cage.AA3.items()}


def pdb(chains: list[tuple[str, str]], spacing: float, phase: float = 0.0) -> str:
    lines = []
    serial = 0
    for c, (chain, sequence) in enumerate(chains):
        for i, residue in enumerate(sequence, 1):
            angle = i * 1.74 + phase
            point = (2.3 * math.cos(angle), 2.3 * math.sin(angle) + c * spacing, i * 1.5)
            for atom, shift in (("N", -0.6), ("CA", 0.0), ("C", 0.6), ("O", 0.9), ("CB", 1.2)):
                serial += 1
                x, y, z = point
                lines.append(
                    f"ATOM  {serial:5d} {atom:^4s} {AA1[residue]:>3s} {chain}{i:4d}    "
                    f"{x + shift:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}"
                    f"{65 + (i % 20):6.2f}          {atom[0]:>2s}\n"
                )
    return "".join(lines) + "END\n"


def archive(path: Path, members: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, contents in members.items():
            raw = contents.encode()
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


def run(module: str, args: list[str], output: Path) -> None:
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.run([sys.executable, "-m", module, *args], cwd=output, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo-output"))
    out = parser.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for directory in ("incoming", "results", "work", "evidence"):
        (out / directory).mkdir()
    author = "0000-0000-0000"
    immune_args = []
    for target, config in immune.TARGETS.items():
        directory = out / "data/raw/casp" / target
        directory.mkdir(parents=True)
        chains = [(chain, "A" * length) for chain, length in config["expected_lengths"].items()]
        (directory / f"{target}TS001_1.pdb").write_text(pdb(chains, 7.0))
        mf = out / "incoming" / f"{target}.tar.gz"
        archive(mf, {"Model_1_af3_demo.pdb": pdb(chains, 10.0)})
        immune_args.extend([f"--{target.lower()}-casp", str(directory), f"--{target.lower()}-mf", str(mf)])
    run("casp17_qa", ["run-all", *immune_args, "--author", author], out)
    for target, config in peptide.TARGETS.items():
        directory = out / "data/raw/casp" / (target + "o")
        directory.mkdir(parents=True)
        chains = [(chr(65 + i), config["sequence"]) for i in range(6)]
        members = {}
        for i in range(1, 13):
            text = pdb(chains, 5.5 + i * 0.17, i * 0.02)
            (directory / f"{target}TS{i:03d}_{1 if i <= 6 else 6}o.pdb").write_text(text)
            members[f"Model_{i}_af3_demo.pdb"] = pdb(chains, 6.0 + i * 0.21)
        archive(out / "incoming" / f"{target}_all_pdbs_MassiveFold.tar.gz", members)
        if target == "T2464":
            (out / "evidence/synthetic-peptide-reference.pdb").write_text(pdb(chains, 5.0))
    run("casp17_fibril_qa", ["--author", author, "--template-7qv6", "evidence/synthetic-peptide-reference.pdb"], out)
    directory = out / "data/raw/casp/T2461o"
    directory.mkdir()
    for i, spacing in enumerate((8.0, 12.0), 1):
        (directory / f"T2461TS001_{i}o.pdb").write_text(pdb([("A", cage.SEQUENCE), ("B", cage.SEQUENCE)], spacing))
    run("casp17_cage_qa", ["--author", author], out)
    counts = {"H2441": 2, "H2444": 2, "T2463": 24, "T2464": 24, "T2461": 2}
    report = {"synthetic_inputs": True, "network_required": False, "targets": {}}
    for target, count in counts.items():
        path = out / "results" / f"{target}_QA.txt"
        rows = path.read_text().split("MODEL 1\n", 1)[1].split("END", 1)[0].strip().splitlines()
        assert len(rows) == count, (target, len(rows))
        assert all(0 <= float(row.split()[1]) <= 1 for row in rows)
        report["targets"][target] = {"models": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "demo_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
