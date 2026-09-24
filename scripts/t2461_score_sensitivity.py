#!/usr/bin/env python3
"""Quantify T2461 rank sensitivity to target-specific scoring assumptions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def rank_map(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda model: (-values[model], model))
    return {model: index + 1 for index, model in enumerate(ordered)}


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results" / "T2461_cage_scores.csv").open()))
    components = {
        row["model"]: {key: float(row[key]) for key in (
            "topology", "interface", "subunit_repeat", "clash_score", "completeness",
            "template_geometry", "confidence",
        )}
        for row in rows
    }
    variants: dict[str, dict[str, float]] = {}
    weights = {
        "no_template": (0.28, 0.267, 0.173, 0.147, 0.133, 0.0),
        "template_15pct": (0.238, 0.227, 0.147, 0.125, 0.113, 0.15),
        "template_25pct_baseline_physics": (0.21, 0.20, 0.13, 0.11, 0.10, 0.25),
        "template_35pct": (0.182, 0.173, 0.113, 0.095, 0.087, 0.35),
        "interface_heavy": (0.18, 0.32, 0.10, 0.10, 0.08, 0.22),
    }
    for name, weight in weights.items():
        variants[name] = {}
        for model, item in components.items():
            physics = sum(
                value * coefficient
                for value, coefficient in zip(
                    (
                        item["topology"], item["interface"], item["subunit_repeat"],
                        item["clash_score"], item["completeness"], item["template_geometry"],
                    ),
                    weight,
                )
            )
            variants[name][model] = 0.84 * physics + 0.16 * item["confidence"]
    ranks = {name: rank_map(values) for name, values in variants.items()}
    baseline_order = [row["model"] for row in rows]
    baseline_rank = {model: index + 1 for index, model in enumerate(baseline_order)}
    output = {
        "target": "T2461",
        "purpose": "rank sensitivity; no variant is an experimental calibration",
        "variants": {},
    }
    for name, values in variants.items():
        ordered = sorted(values, key=lambda model: (-values[model], model))
        correlation = spearmanr(
            [baseline_rank[model] for model in baseline_order],
            [ranks[name][model] for model in baseline_order],
        ).statistic
        output["variants"][name] = {
            "spearman_vs_submission": correlation,
            "top20_overlap_with_submission": len(set(ordered[:20]) & set(baseline_order[:20])),
            "top_models": [
                {"model": model, "variant_rank": ranks[name][model], "submission_rank": baseline_rank[model]}
                for model in ordered[:20]
            ],
        }
    path = ROOT / "results" / "T2461_score_sensitivity.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
