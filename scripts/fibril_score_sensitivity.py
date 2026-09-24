#!/usr/bin/env python3
"""Ablation and weight sensitivity audit for fibril QA rankings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

WEIGHTS = {
    "T2463": {
        "baseline": (0.75, 0.15, 0.10),
        "no_consensus": (0.82, 0.18, 0.00),
        "physics_high": (0.85, 0.10, 0.05),
        "confidence_high": (0.65, 0.25, 0.10),
        "physics_only": (1.00, 0.00, 0.00),
    },
    "T2464": {
        "baseline": (0.80, 0.15, 0.05),
        "no_consensus": (0.82, 0.18, 0.00),
        "physics_high": (0.90, 0.10, 0.00),
        "confidence_high": (0.70, 0.25, 0.05),
        "physics_only": (1.00, 0.00, 0.00),
    },
}


def percentiles(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 1) / max(1, len(values) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(WEIGHTS), required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.scores.open(newline="", encoding="utf-8")))
    physical_raw = np.asarray([float(row["physical"]) for row in rows])
    confidence_raw = np.asarray([float(row["combined_confidence"]) for row in rows])
    consensus_raw = np.asarray([float(row["cluster_support"]) for row in rows])
    physical = 0.65 * physical_raw + 0.35 * percentiles(physical_raw)
    confidence = 0.60 * confidence_raw + 0.40 * percentiles(confidence_raw)
    consensus = percentiles(consensus_raw)
    scenarios = {}
    baseline_scores = None
    for name, (wp, wc, ws) in WEIGHTS[args.target].items():
        scores = wp * physical + wc * confidence + ws * consensus
        if baseline_scores is None:
            baseline_scores = scores
        order = np.argsort(-scores)
        state_winners = {}
        for state in ("v1", "v2", "unassigned"):
            candidates = [idx for idx in order if rows[idx]["state"] == state]
            state_winners[state] = [
                {"model": rows[idx]["model"], "score": round(float(scores[idx]), 8)}
                for idx in candidates[:10]
            ]
        top_baseline = set(np.argsort(-baseline_scores)[:100])
        scenarios[name] = {
            "weights": {"physics": wp, "confidence": wc, "consensus": ws},
            "spearman_vs_baseline": float(spearmanr(baseline_scores, scores).statistic),
            "top100_overlap_vs_baseline": len(top_baseline & set(order[:100])) / 100,
            "state_top10": state_winners,
        }
    output = {
        "target": args.target,
        "purpose": "ranking robustness; no fitted native labels are available",
        "model_count": len(rows),
        "scenarios": scenarios,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
