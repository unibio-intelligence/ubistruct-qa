#!/usr/bin/env python3
"""CASP17 T2461 designed-protein-cage quality assessment.

T2461 is a 191-residue de novo protein submitted as an A24 cage.  This scorer
does not attempt to predict or refine the 24-mer and deliberately excludes
cross-model consensus.  It ranks the supplied CASP assemblies using internal
physical and symmetry evidence: sequence/completeness, subunit repeatability,
contact-graph closure and regularity, repeated interface geometry, hollow-shell
geometry, steric clashes, and informative predictor confidence.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree

TARGET = "T2461"
SEQUENCE = (
    "MHHHHHHGSMAIGIVELSSIAMGLKLADEMLKAADVKLLVSRPILPGKFLIILGGETEAIRKAI"
    "AVATEAAGSKLVRSALIEDIHPSVLPAISGINPVEERQAVGIVETESLEAAILAANAAVKGSNV"
    "TLVRIRMLSGITGKCYIVVAGDVDDVALAVVVAAEVAASRGKLIYAALIPRPHPAIWPLIVEG"
)
EXPECTED_CHAINS = 24
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "evidence" / "6FDB-assembly1.cif.gz"

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _lddt_distances(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape or not candidate.size:
        return 0.0
    delta = np.abs(candidate - reference)
    return float(np.mean([(delta < threshold).mean() for threshold in (0.5, 1.0, 2.0, 4.0)]))


def percentile_ranks(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [1.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + end - 1) / 2) / (len(values) - 1)
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def parse_structure(text: str, _name: str) -> dict[str, list[dict]]:
    """Parse the standard one-character-chain PDB records in the CASP archive."""
    raw: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        if line[16:17] not in (" ", "A", "1"):
            continue
        chain = line[21:22].strip()
        if not chain:
            continue
        key = (line[22:26].strip(), line[26:27].strip())
        atom = line[12:16].strip()
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            bfactor = float(line[60:66]) if len(line) >= 66 else 50.0
        except ValueError:
            continue
        residue = raw[chain].setdefault(
            key,
            {"resname": line[17:20].strip().upper(), "atoms": {}, "bfactors": []},
        )
        residue["atoms"][atom] = xyz
        residue["bfactors"].append(bfactor)
    return {chain: list(residues.values()) for chain, residues in raw.items()}


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _residue_points(residues: list[dict]) -> np.ndarray | None:
    points = []
    for residue in residues[: len(SEQUENCE)]:
        point = residue["atoms"].get("CB", residue["atoms"].get("CA"))
        if point is None:
            return None
        points.append(point)
    if len(points) < int(0.75 * len(SEQUENCE)):
        return None
    return np.asarray(points, dtype=float)


def _sequence_identity(sequence: str) -> float:
    """Best ungapped identity, allowing omission of a terminal expression tag."""
    if not sequence:
        return 0.0
    if len(sequence) > len(SEQUENCE):
        sequence = sequence[: len(SEQUENCE)]
    width = len(sequence)
    return max(
        sum(left == right for left, right in zip(sequence, SEQUENCE[offset : offset + width])) / width
        for offset in range(len(SEQUENCE) - width + 1)
    )


def _distance_signature(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    matrix = np.sqrt(np.sum(delta * delta, axis=2))
    return matrix[np.triu_indices(len(points), 1)]


@functools.lru_cache(maxsize=1)
def _template_centroid_spectrum(path: Path | None = None) -> np.ndarray:
    """Sorted CA-centroid distances for the experimental A24 6FDB assembly."""
    path = path or TEMPLATE_PATH
    chains: dict[str, list[list[float]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM "):
                continue
            fields = line.split()
            # RCSB assembly mmCIF atom_site layout: atom, xyz and auth_asym_id.
            if len(fields) < 26 or fields[3] != "CA":
                continue
            chains[fields[23]].append([float(fields[10]), float(fields[11]), float(fields[12])])
    if len(chains) != EXPECTED_CHAINS:
        raise ValueError(f"6FDB assembly has {len(chains)} parsed chains, expected 24")
    centroids = np.asarray([np.mean(points, axis=0) for points in chains.values()])
    return np.sort(_distance_signature(centroids))


def _spectrum_agreement(candidate: np.ndarray, reference: np.ndarray) -> float:
    if candidate.shape != reference.shape or not candidate.size:
        return 0.0
    delta = np.abs(np.sort(candidate) - np.sort(reference))
    return float(np.mean([(delta < threshold).mean() for threshold in (1.0, 2.0, 4.0, 8.0)]))


def _connected_fraction(n_chains: int, edge_pairs: Iterable[tuple[int, int]]) -> float:
    if n_chains == 0:
        return 0.0
    graph: dict[int, set[int]] = defaultdict(set)
    for left, right in edge_pairs:
        graph[left].add(right)
        graph[right].add(left)
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbour in graph[node] - seen:
            seen.add(neighbour)
            queue.append(neighbour)
    return len(seen) / n_chains


def _jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _interface_repeat(edges: list[dict]) -> float:
    """Best other-edge contact-map agreement, invariant to swapping homomers."""
    if len(edges) < 2:
        return 0.0
    scores = []
    for index, edge in enumerate(edges):
        contacts = edge["contact_set"]
        best = 0.0
        for other_index, other in enumerate(edges):
            if index == other_index:
                continue
            direct = _jaccard(contacts, other["contact_set"])
            swapped = _jaccard(contacts, {(right, left) for left, right in other["contact_set"]})
            best = max(best, direct, swapped)
        scores.append(best)
    return statistics.fmean(scores)


def _distinct_interfaces(edges: list[dict], limit: int = 2) -> list[dict]:
    selected: list[dict] = []
    for edge in sorted(edges, key=lambda item: (-item["local_score"], item["chains"])):
        if all(
            max(
                _jaccard(edge["contact_set"], prior["contact_set"]),
                _jaccard(
                    edge["contact_set"],
                    {(right, left) for left, right in prior["contact_set"]},
                ),
            ) < 0.70
            for prior in selected
        ):
            selected.append(edge)
            if len(selected) == limit:
                break
    if not selected and edges:
        selected.append(max(edges, key=lambda item: item["local_score"]))
    return selected


def compute_features(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_structure(text, path.name)
    selected: dict[str, tuple[list[dict], np.ndarray]] = {}
    confidences: list[float] = []
    identities: list[float] = []
    completions: list[float] = []
    for chain_id, residues in parsed.items():
        if len(residues) < int(0.75 * len(SEQUENCE)):
            continue
        points = _residue_points(residues)
        if points is None:
            continue
        selected[chain_id] = (residues[: len(SEQUENCE)], points)
        sequence = "".join(AA3.get(residue["resname"], "X") for residue in residues[: len(SEQUENCE)])
        identities.append(_sequence_identity(sequence))
        completions.append(min(1.0, len(residues) / len(SEQUENCE)))
        confidences.extend(
            statistics.fmean(residue["bfactors"])
            for residue in residues[: len(SEQUENCE)]
            if residue["bfactors"]
        )

    # Release preparation: reject empty/too-short structures before modal lookup.
    if not selected:
        return {"model": path.name, "n_chains": 0, "invalid": True, "interfaces": []}

    # Distance signatures must have equal length.  Retain the modal subunit
    # length; outlying partial chains remain reflected in chain completeness.
    length_counts = Counter(len(points) for _, points in selected.values())
    modal_length = length_counts.most_common(1)[0][0]
    selected = {
        chain_id: value for chain_id, value in selected.items() if len(value[1]) == modal_length
    }
    chain_ids = list(selected)
    n_chains = len(chain_ids)
    if not selected:
        return {"model": path.name, "n_chains": 0, "invalid": True, "interfaces": []}

    signatures = np.stack([_distance_signature(points) for _, points in selected.values()])
    median_signature = np.median(signatures, axis=0)
    subunit_repeat = statistics.fmean(
        _lddt_distances(signature, median_signature) for signature in signatures
    )

    rep_points: list[np.ndarray] = []
    rep_meta: list[tuple[int, int]] = []
    heavy_points: list[np.ndarray] = []
    heavy_meta: list[tuple[int, str]] = []
    centroids = []
    for chain_index, chain_id in enumerate(chain_ids):
        residues, points = selected[chain_id]
        ca_points = [residue["atoms"].get("CA") for residue in residues]
        ca_points = [point for point in ca_points if point is not None]
        centroids.append(np.mean(np.asarray(ca_points), axis=0))
        for residue_index, point in enumerate(points):
            rep_points.append(point)
            rep_meta.append((chain_index, residue_index))
        for residue in residues:
            for atom_name, point in residue["atoms"].items():
                if not atom_name.upper().startswith("H"):
                    heavy_points.append(point)
                    heavy_meta.append((chain_index, atom_name))

    contact_sets: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    tight_sets: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    points = np.asarray(rep_points)
    tree = cKDTree(points)
    for left, right in tree.query_pairs(8.0):
        ci, ri = rep_meta[left]
        cj, rj = rep_meta[right]
        if ci == cj:
            continue
        distance = float(np.linalg.norm(points[left] - points[right]))
        if ci > cj:
            ci, cj, ri, rj = cj, ci, rj, ri
        contact_sets[(ci, cj)].add((ri, rj))
        if distance <= 5.0:
            tight_sets[(ci, cj)].add((ri, rj))

    pair_clashes: Counter[tuple[int, int]] = Counter()
    if heavy_points:
        heavy = np.asarray(heavy_points)
        for left, right in cKDTree(heavy).query_pairs(1.9):
            ci, _ = heavy_meta[left]
            cj, _ = heavy_meta[right]
            if ci == cj:
                continue
            pair_clashes[tuple(sorted((ci, cj)))] += 1

    edges = []
    for pair, contacts in contact_sets.items():
        if len(contacts) < 8:
            continue
        tight = tight_sets[pair]
        contact_score = 1.0 - math.exp(-len(contacts) / 55.0)
        tight_score = 1.0 - math.exp(-len(tight) / 18.0)
        clash_score = math.exp(-pair_clashes[pair] / 8.0)
        local_score = 0.47 * contact_score + 0.33 * tight_score + 0.20 * clash_score
        edges.append({
            "chains": [chain_ids[pair[0]], chain_ids[pair[1]]],
            "pair": pair,
            "contacts": len(contacts),
            "tight_contacts": len(tight),
            "clashes": pair_clashes[pair],
            "contact_set": contacts,
            "local_score": local_score,
        })

    # Edges with at least a modest protein-protein contact patch define the cage graph.
    strong_edges = [edge for edge in edges if edge["contacts"] >= 20]
    degrees = Counter(node for edge in strong_edges for node in edge["pair"])
    degree_values = [degrees.get(index, 0) for index in range(n_chains)]
    mean_degree = statistics.fmean(degree_values)
    degree_cv = statistics.pstdev(degree_values) / mean_degree if mean_degree else 9.0
    graph_regularity = math.exp(-degree_cv)
    connected_fraction = _connected_fraction(n_chains, (edge["pair"] for edge in strong_edges))
    # A finite, closed polyhedral shell normally gives every subunit >=2 neighbours.
    closure = sum(degree >= 2 for degree in degree_values) / max(1, n_chains)
    plausible_degree = math.exp(-0.5 * ((mean_degree - 3.5) / 2.0) ** 2)

    centroid_array = np.asarray(centroids)
    cage_center = centroid_array.mean(axis=0)
    radii = np.linalg.norm(centroid_array - cage_center, axis=1)
    radius_mean = float(np.mean(radii))
    radius_cv = float(np.std(radii) / max(radius_mean, 1e-6))
    radial_regularity = math.exp(-4.0 * radius_cv)
    covariance = np.cov((centroid_array - cage_center).T) if n_chains >= 3 else np.eye(3)
    eigenvalues = np.linalg.eigvalsh(covariance)
    isotropy = float(max(0.0, eigenvalues[0] / max(eigenvalues[-1], 1e-6)))
    atom_radii = np.linalg.norm(points - cage_center, axis=1)
    void_ratio = float(np.percentile(atom_radii, 2) / max(np.median(atom_radii), 1e-6))
    hollow_shell = _bounded((void_ratio - 0.10) / 0.45)
    centroid_spectrum = np.sort(_distance_signature(centroid_array))
    template_geometry = (
        _spectrum_agreement(centroid_spectrum, _template_centroid_spectrum())
        if n_chains == EXPECTED_CHAINS else 0.0
    )

    interface_repeat = _interface_repeat(strong_edges)
    interface_quality = (
        statistics.fmean(edge["local_score"] for edge in strong_edges)
        if strong_edges else 0.0
    )
    completeness = (
        min(1.0, n_chains / EXPECTED_CHAINS)
        * statistics.fmean(completions)
        * statistics.fmean(identities)
    )
    total_clashes = sum(pair_clashes.values())
    clash_score = math.exp(-total_clashes / max(1.0, n_chains * 12.0))
    confidence_mean = statistics.fmean(confidences) if confidences else 50.0
    confidence_std = statistics.pstdev(confidences) if len(confidences) > 1 else 0.0
    confidence = _bounded(confidence_mean / 100.0) if confidence_std >= 0.5 else 0.5

    return {
        "model": path.name,
        "n_chains": n_chains,
        "invalid": False,
        "completeness": completeness,
        "sequence_identity": statistics.fmean(identities),
        "subunit_repeat": subunit_repeat,
        "connected_fraction": connected_fraction,
        "closure": closure,
        "graph_regularity": graph_regularity,
        "mean_degree": mean_degree,
        "plausible_degree": plausible_degree,
        "radial_regularity": radial_regularity,
        "isotropy": isotropy,
        "void_ratio": void_ratio,
        "hollow_shell": hollow_shell,
        "template_geometry": template_geometry,
        "interface_repeat": interface_repeat,
        "interface_quality": interface_quality,
        "clashes": total_clashes,
        "clash_score": clash_score,
        "confidence": confidence,
        "confidence_mean": confidence_mean,
        "confidence_std": confidence_std,
        "interfaces": _distinct_interfaces(edges),
    }


def score_records(records: list[dict]) -> list[dict]:
    valid = [record for record in records if not record.get("invalid")]
    for record in valid:
        topology = (
            0.20 * record["connected_fraction"]
            + 0.20 * record["closure"]
            + 0.18 * record["graph_regularity"]
            + 0.12 * record["plausible_degree"]
            + 0.16 * record["radial_regularity"]
            + 0.08 * record["isotropy"]
            + 0.06 * record["hollow_shell"]
        )
        interface = 0.58 * record["interface_quality"] + 0.42 * record["interface_repeat"]
        physical = (
            0.21 * topology
            + 0.20 * interface
            + 0.13 * record["subunit_repeat"]
            + 0.11 * record["clash_score"]
            + 0.10 * record["completeness"]
            + 0.25 * record["template_geometry"]
        )
        record.update(topology=topology, interface=interface, physical=physical)

    # Percentile blending compares physical measurements, not model topology votes.
    for key in ("physical", "topology", "interface", "confidence"):
        ranks = percentile_ranks([record[key] for record in valid])
        for record, rank in zip(valid, ranks):
            record[key + "_percentile"] = rank

    for record in valid:
        physics = 0.72 * record["physical"] + 0.28 * record["physical_percentile"]
        confidence = 0.60 * record["confidence"] + 0.40 * record["confidence_percentile"]
        raw = 0.84 * physics + 0.16 * confidence
        # Keep relative QA scores conservative because there is no target-specific calibration.
        record["overall"] = min(0.94, max(0.03, 0.04 + 0.88 * raw))
        for edge in record["interfaces"]:
            edge["score"] = min(
                0.94,
                max(0.01, 0.48 * record["overall"] + 0.52 * edge["local_score"]),
            )
    for record in records:
        if record.get("invalid"):
            record["overall"] = 0.01
            record.setdefault("interfaces", [])
    return sorted(records, key=lambda item: (-item["overall"], item["model"]))


def write_scores(records: list[dict], path: Path) -> None:
    fields = [
        "model", "overall", "n_chains", "physical", "topology", "interface",
        "confidence", "completeness", "sequence_identity", "subunit_repeat",
        "connected_fraction", "closure", "graph_regularity", "mean_degree",
        "radial_regularity", "isotropy", "void_ratio", "hollow_shell",
        "template_geometry", "interface_repeat", "interface_quality", "clashes",
        "clash_score", "interfaces",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["interfaces"] = ",".join(
                f"{''.join(edge['chains'])}:{edge.get('score', 0.0):.4f}"
                for edge in record["interfaces"]
            )
            writer.writerow(row)


def write_submission(records: list[dict], author: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("PFRMAT QA\n")
        handle.write(f"TARGET {TARGET}\n")
        handle.write(f"AUTHOR {author}\n")
        handle.write("METHOD A24 cage completeness, subunit repeat and steric-clash assessment.\n")
        handle.write("METHOD Closed contact graph, repeated interfaces and hollow-shell geometry.\n")
        handle.write("METHOD Family-level A24 geometry is compared with experimental O3-33/6FDB.\n")
        handle.write("METHOD Predictor confidence included; cross-model consensus excluded.\n")
        handle.write("MODEL 1\n")
        for record in sorted(records, key=lambda item: item["model"]):
            interfaces = record["interfaces"][:2]
            labels = [
                f"{''.join(edge['chains'])}:{edge.get('score', 0.0):.4f}"
                for edge in interfaces if len("".join(edge["chains"])) == 2
            ]
            if not labels:
                labels = ["AB:0.0000"]
            handle.write(f"{record['model']} {record['overall']:.4f} {', '.join(labels)}\n")
        handle.write("END\n")


def validate_submission(path: Path, expected: Iterable[str]) -> dict:
    expected_set = set(expected)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "PFRMAT QA" or f"TARGET {TARGET}" not in lines[:5]:
        raise ValueError("invalid QA header")
    start, end = lines.index("MODEL 1"), lines.index("END")
    seen = set()
    for line_number, line in enumerate(lines[start + 1 : end], start + 2):
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: malformed record")
        model, score_text, interface_text = fields
        score = float(score_text)
        if model in seen or not 0.0 <= score <= 1.0:
            raise ValueError(f"{path}:{line_number}: duplicate/invalid record")
        seen.add(model)
        for item in interface_text.replace(" ", "").split(","):
            label, value = item.split(":", 1)
            if len(label) != 2 or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{path}:{line_number}: invalid interface")
    if seen != expected_set:
        raise ValueError(
            f"coverage mismatch: missing={sorted(expected_set - seen)[:5]} "
            f"unexpected={sorted(seen - expected_set)[:5]}"
        )
    return {"valid": True, "model_count": len(seen)}


def main() -> None:
    global TEMPLATE_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("data/raw/casp/T2461o"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--template-6fdb",
        type=Path,
        default=Path("evidence/6FDB-assembly1.cif.gz"),
    )
    parser.add_argument("--author", default="REPLACE-WITH-CASP-REGISTRATION-ID")
    parser.add_argument("--limit", type=int, default=0, help="Development-only model limit")
    args = parser.parse_args()
    TEMPLATE_PATH = args.template_6fdb

    paths = sorted(path for path in args.models.iterdir() if path.is_file())
    if args.limit:
        paths = paths[: args.limit]
    records = []
    for index, path in enumerate(paths, 1):
        records.append(compute_features(path))
        if index % 10 == 0 or index == len(paths):
            print(f"scored {index}/{len(paths)}", flush=True)
    records = score_records(records)
    scores_path = args.results / "T2461_cage_scores.csv"
    submission_path = args.results / "T2461_QA.txt"
    diagnostics_path = args.results / "T2461_cage_diagnostics.json"
    write_scores(records, scores_path)
    write_submission(records, args.author, submission_path)
    validation = validate_submission(submission_path, (path.name for path in paths))
    diagnostics = {
        "target": TARGET,
        "description": "191-residue de novo A24 designed protein cage (865)",
        "model_count": len(records),
        "consensus_used": False,
        "validation": validation,
        "top_models": [
            {key: record.get(key) for key in (
                "model", "overall", "physical", "topology", "interface", "confidence",
                "n_chains", "subunit_repeat", "connected_fraction", "closure",
                "graph_regularity", "mean_degree", "template_geometry", "interface_repeat", "clashes",
            )}
            for record in records[:30]
        ],
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation": validation,
        "top_model": records[0]["model"] if records else None,
        "scores": str(scores_path),
        "submission": str(submission_path),
    }, indent=2))


if __name__ == "__main__":
    main()
