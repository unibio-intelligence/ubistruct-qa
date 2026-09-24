#!/usr/bin/env python3
"""Leave-one-family/source-out diagnostics for contact-consensus rankings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from casp17_qa import (
    TARGETS,
    _independent_unit,
    read_features,
    score_records,
)


def ranks(scored: list[dict]) -> dict[str, int]:
    return {
        model: index + 1
        for index, model in enumerate(record["model"] for record in scored)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = read_features(args.features)
    full_scored, _ = score_records(records, args.target)
    full_top = [record["model"] for record in full_scored]
    exclusions = [
        "massivefold:af3",
        "massivefold:afm",
        "massivefold:cf",
        "massivefold:esmf2",
        "casp:*",
        "massivefold:*",
    ]
    variants = {}
    for exclusion in exclusions:
        prefix = exclusion.removesuffix("*")
        reference = [
            record
            for record in records
            if not (
                _independent_unit(record) == exclusion
                or (exclusion.endswith("*") and _independent_unit(record).startswith(prefix))
            )
        ]
        scored, _ = score_records(
            records,
            args.target,
            consensus_reference_records=reference,
        )
        variant_ranks = ranks(scored)
        variant_top = [record["model"] for record in scored]
        variants[exclusion] = {
            "reference_models": len(reference),
            "full_top_model_rank": variant_ranks[full_top[0]],
            "top10_overlap": len(set(full_top[:10]) & set(variant_top[:10])),
            "top25_overlap": len(set(full_top[:25]) & set(variant_top[:25])),
            "top_model": variant_top[0],
        }

    report = {
        "target": args.target,
        "model_count": len(records),
        "full_top_model": full_top[0],
        "full_top25": full_top[:25],
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["variants"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
