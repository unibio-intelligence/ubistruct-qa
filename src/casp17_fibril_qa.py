#!/usr/bin/env python3
"""CASP17 fibril/peptide assembly quality assessment for T2463 and T2464.

This pipeline is deliberately separate from the immune-complex scorer.  It
uses label-invariant peptide geometry, cross-chain hydrogen-bond and packing
features, method-family-balanced cluster support, and native predictor
confidence. For T2464 it retains comparison with experimental fibril PDB 7QV6
as a zero-weight alternate-polymorph diagnostic. Consensus is a small
tie-breaking term, not a claim that the majority must be correct.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

TARGETS = {
    "T2463": {
        "sequence": "FLGAIAQALTSLLGKL",
        "minimum_chains": 24,
        "template": None,
        "description": "16-residue amyloid peptide; two solvent-dependent conformations",
    },
    "T2464": {
        "sequence": "GLFDIVKKIAGHIVSSI",
        "minimum_chains": 8,
        "template": "7QV6",
        "description": "aurein 3.3; two requested agglomeration folds",
    },
}

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _gaussian(value: float, center: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - center) / sigma) ** 2)


def _lddt_distances(candidate: np.ndarray, reference: np.ndarray, mask=None) -> float:
    """lDDT-like agreement between equally shaped distance arrays."""
    if candidate.shape != reference.shape or candidate.size == 0:
        return 0.0
    valid = np.isfinite(candidate) & np.isfinite(reference)
    if mask is not None:
        valid &= mask
    delta = np.abs(candidate[valid] - reference[valid])
    if not delta.size:
        return 0.0
    return float(np.mean([(delta < threshold).mean() for threshold in (0.5, 1, 2, 4)]))


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


def _model_state(name: str) -> str:
    match = re.search(r"_(\d+)o(?:\.cif|\.pdb)?$", name)
    if not match:
        return "unassigned"
    return "v1" if int(match.group(1)) <= 5 else "v2"


def _independent_unit(record: dict) -> str:
    name = record["model"].lower()
    if record["source"] == "casp":
        match = re.search(r"ts(\d+)", name)
        return f"casp:{match.group(1) if match else name}"
    for family in ("esmf2", "af3", "afm", "cf"):
        if f"_{family}_" in name:
            return f"massivefold:{family}"
    return f"massivefold:{name}"


def _clean_model_name(name: str, source: str) -> str:
    base = Path(name).name
    if source == "casp" and base.endswith((".cif", ".pdb")):
        return str(Path(base).with_suffix(""))
    return base


def _qa_chain_id(chain_id: str) -> str:
    """Map extended-CIF IDs such as Axp/Bxp to CASP's one-character QA IDs."""
    if len(chain_id) == 1:
        return chain_id
    # CASP extended submissions observed here preserve the canonical chain as
    # the first character and add a shared suffix (e.g. Axp ... Xxp).
    return chain_id[0]


def _parse_pdb(text: str) -> dict[str, list[dict]]:
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
        resname = line[17:20].strip().upper()
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            bfactor = float(line[60:66]) if len(line) >= 66 else 50.0
        except ValueError:
            continue
        residue = raw[chain].setdefault(
            key, {"resname": resname, "atoms": {}, "bfactors": []}
        )
        residue["atoms"][atom] = xyz
        residue["bfactors"].append(bfactor)
    return {chain: list(residues.values()) for chain, residues in raw.items()}


def _parse_mmcif(text: str) -> dict[str, list[dict]]:
    raw: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    headers: list[str] = []
    indices: dict[str, int] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "loop_":
            headers, indices = [], None
            continue
        if stripped.startswith("_atom_site."):
            headers.append(stripped)
            continue
        if headers and stripped.startswith("_"):
            headers, indices = [], None
            continue
        if not headers or not stripped or stripped == "#":
            if stripped == "#" and indices is not None:
                break
            continue
        if indices is None:
            indices = {name: idx for idx, name in enumerate(headers)}
        if not stripped.startswith(("ATOM ", "HETATM ")):
            continue
        fields = stripped.split()
        if len(fields) < len(headers):
            continue

        def field(*names: str, default: str = "") -> str:
            for name in names:
                idx = indices.get(name)
                if idx is not None and idx < len(fields) and fields[idx] not in (".", "?"):
                    return fields[idx].strip("'\"")
            return default

        if field("_atom_site.label_alt_id", default=".") not in (".", "?", "A", "1"):
            continue
        chain = field("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        atom = field("_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        resname = field("_atom_site.auth_comp_id", "_atom_site.label_comp_id").upper()
        key = (
            field("_atom_site.auth_seq_id", "_atom_site.label_seq_id"),
            field("_atom_site.pdbx_PDB_ins_code"),
        )
        try:
            xyz = [
                float(field("_atom_site.Cartn_x")),
                float(field("_atom_site.Cartn_y")),
                float(field("_atom_site.Cartn_z")),
            ]
            bfactor = float(field("_atom_site.B_iso_or_equiv", default="50"))
        except ValueError:
            continue
        residue = raw[chain].setdefault(
            key, {"resname": resname, "atoms": {}, "bfactors": []}
        )
        residue["atoms"][atom] = xyz
        residue["bfactors"].append(bfactor)
    return {chain: list(residues.values()) for chain, residues in raw.items()}


def parse_structure(text: str, name: str) -> dict[str, list[dict]]:
    return _parse_mmcif(text) if name.lower().endswith(".cif") else _parse_pdb(text)


def _chain_distance_matrix(chain: list[dict], length: int) -> np.ndarray | None:
    points = []
    for residue in chain[:length]:
        atom = residue["atoms"].get("CA")
        if atom is None:
            return None
        points.append(atom)
    if len(points) != length:
        return None
    xyz = np.asarray(points, dtype=float)
    delta = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _secondary_scores(matrix: np.ndarray) -> tuple[float, float]:
    length = matrix.shape[0]
    beta_values = []
    alpha_values = []
    for offset, center, sigma in ((2, 6.6, 1.4), (3, 9.7, 2.0)):
        beta_values.extend(
            _gaussian(float(matrix[i, i + offset]), center, sigma)
            for i in range(length - offset)
        )
    for offset, center, sigma in ((3, 5.2, 0.9), (4, 6.2, 1.1)):
        alpha_values.extend(
            _gaussian(float(matrix[i, i + offset]), center, sigma)
            for i in range(length - offset)
        )
    return statistics.fmean(beta_values), statistics.fmean(alpha_values)


def _interface_matrix_rmsd(left: dict, right: dict) -> float:
    """Compare homomer interfaces independent of which chain is listed first."""
    left_flat = np.asarray(left.get("matrix", []), dtype=float)
    right_flat = np.asarray(right.get("matrix", []), dtype=float)
    length = math.isqrt(left_flat.size)
    if (
        length == 0
        or length * length != left_flat.size
        or right_flat.size != left_flat.size
    ):
        return math.inf
    left_matrix = left_flat.reshape(length, length)
    right_matrix = right_flat.reshape(length, length)
    direct = float(np.sqrt(np.mean((left_matrix - right_matrix) ** 2)))
    swapped = float(np.sqrt(np.mean((left_matrix - right_matrix.T) ** 2)))
    return min(direct, swapped)


def select_distinct_interfaces(
    edges: Iterable[dict], limit: int = 3, rmsd_threshold: float = 1.0,
) -> list[dict]:
    """Retain quality-ranked representatives of geometrically distinct interfaces."""
    selected: list[dict] = []
    for edge in edges:
        if all(
            _interface_matrix_rmsd(edge, representative) >= rmsd_threshold
            for representative in selected
        ):
            selected.append(edge)
            if len(selected) >= limit:
                break
    return selected


def compute_features(
    chains: dict[str, list[dict]], model: str, source: str, target: str,
    native: dict | None = None, keep_edges: int = 3,
) -> dict:
    config = TARGETS[target]
    length = len(config["sequence"])
    # Ignore waters/tiny ligands and require enough residues for meaningful geometry.
    selected = {
        chain_id: residues[:length]
        for chain_id, residues in chains.items()
        if len(residues) >= max(4, int(0.75 * length))
    }
    chain_ids = list(selected)
    matrices: list[np.ndarray] = []
    valid_ids: list[str] = []
    sequences = []
    confidences: list[float] = []
    for chain_id, residues in selected.items():
        matrix = _chain_distance_matrix(residues, length)
        if matrix is None:
            continue
        matrices.append(matrix)
        valid_ids.append(chain_id)
        sequences.append("".join(AA3.get(r["resname"], "X") for r in residues[:length]))
        confidences.extend(
            statistics.fmean(r["bfactors"])
            for r in residues[:length]
            if r["atoms"] and r["bfactors"]
        )
    selected = {chain: selected[chain] for chain in valid_ids}
    chain_ids = valid_ids
    n_chains = len(chain_ids)
    if not matrices:
        return {
            "model": model, "source": source, "target": target, "state": _model_state(model),
            "n_chains": 0, "residue_completeness": 0.0, "sequence_identity": 0.0,
            "confidence": 0.0, "confidence_std": 0.0, "confidence_informative": False,
            "beta": 0.0, "alpha": 0.0, "repeat": 0.0,
            "edge_quality": 0.0, "contact_quality": 0.0, "hbond_score": 0.0, "graph_regularity": 0.0,
            "packing": 0.0, "clashes": 9999, "interfaces": [], "fingerprint": [],
            **(native or {}),
        }

    chain_stack = np.stack(matrices)
    median_matrix = np.median(chain_stack, axis=0)
    tri = np.triu_indices(length, 1)
    repeat = statistics.fmean(
        _lddt_distances(matrix[tri], median_matrix[tri]) for matrix in matrices
    )
    beta, alpha = zip(*(_secondary_scores(matrix) for matrix in matrices))
    sequence_identity = statistics.fmean(
        sum(a == b for a, b in zip(sequence, config["sequence"])) / length
        for sequence in sequences
    )

    rep_points, rep_meta = [], []
    heavy_points, heavy_chain = [], []
    donor_acceptor_points, donor_acceptor_meta = [], []
    centroids = []
    for chain_index, chain_id in enumerate(chain_ids):
        chain_reps = []
        for residue_index, residue in enumerate(selected[chain_id]):
            atoms = residue["atoms"]
            representative = atoms.get("CB", atoms.get("CA"))
            if representative is not None:
                rep_points.append(representative)
                rep_meta.append((chain_index, residue_index))
                chain_reps.append(representative)
            for atom_name, xyz in atoms.items():
                if not atom_name.upper().startswith("H"):
                    heavy_points.append(xyz)
                    heavy_chain.append(chain_index)
                if atom_name in ("N", "O"):
                    donor_acceptor_points.append(xyz)
                    donor_acceptor_meta.append((chain_index, atom_name))
        centroids.append(np.mean(np.asarray(chain_reps), axis=0))

    contacts: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    min_distances: dict[tuple[int, int], np.ndarray] = {}
    if len(rep_points) >= 2:
        points = np.asarray(rep_points)
        for left, right in cKDTree(points).query_pairs(8.0):
            ci, ri = rep_meta[left]
            cj, rj = rep_meta[right]
            if ci == cj:
                continue
            if ci > cj:
                ci, cj, ri, rj = cj, ci, rj, ri
            contacts[(ci, cj)].add((ri, rj))
    clashes = 0
    if len(heavy_points) >= 2:
        for left, right in cKDTree(np.asarray(heavy_points)).query_pairs(1.9):
            if heavy_chain[left] != heavy_chain[right]:
                clashes += 1
    hbonds: Counter = Counter()
    if len(donor_acceptor_points) >= 2:
        da = np.asarray(donor_acceptor_points)
        for left, right in cKDTree(da).query_pairs(3.6):
            ci, ai = donor_acceptor_meta[left]
            cj, aj = donor_acceptor_meta[right]
            if ci == cj or ai == aj or np.linalg.norm(da[left] - da[right]) < 2.2:
                continue
            if ci > cj:
                ci, cj = cj, ci
            hbonds[(ci, cj)] += 1

    edges = []
    for (ci, cj), residue_contacts in contacts.items():
        if len(residue_contacts) < 3:
            continue
        reps_i = np.asarray([
            r["atoms"].get("CB", r["atoms"].get("CA")) for r in selected[chain_ids[ci]]
        ])
        reps_j = np.asarray([
            r["atoms"].get("CB", r["atoms"].get("CA")) for r in selected[chain_ids[cj]]
        ])
        matrix = np.sqrt(np.sum((reps_i[:, None, :] - reps_j[None, :, :]) ** 2, axis=2))
        same_index = np.diag(matrix)
        axial = _gaussian(float(np.median(same_index)), 4.8, 1.4)
        contact_score = 1.0 - math.exp(-len(residue_contacts) / max(1.0, 1.5 * length))
        hbond_score = 1.0 - math.exp(-hbonds[(ci, cj)] / max(1.0, 0.55 * length))
        quality = 0.45 * contact_score + 0.35 * hbond_score + 0.20 * axial
        edges.append({
            "chains": [chain_ids[ci], chain_ids[cj]],
            "contacts": len(residue_contacts), "hbonds": hbonds[(ci, cj)],
            "axial": round(axial, 6), "quality": round(quality, 6),
            "contact_quality": round(contact_score, 6),
            "matrix": np.round(matrix, 3).ravel().tolist(),
        })
        min_distances[(ci, cj)] = matrix
    edges.sort(key=lambda edge: (-edge["quality"], -edge["contacts"], edge["chains"]))

    strong_edges = [edge for edge in edges if edge["contacts"] >= max(4, length // 2)]
    degrees = Counter(chain for edge in strong_edges for chain in edge["chains"])
    degree_values = [degrees.get(chain, 0) for chain in chain_ids]
    mean_degree = statistics.fmean(degree_values) if degree_values else 0.0
    degree_cv = statistics.pstdev(degree_values) / max(1e-6, mean_degree) if mean_degree else 9.0
    graph_regularity = math.exp(-degree_cv)
    packing = _gaussian(mean_degree, 3.5, 2.5)
    edge_quality = statistics.fmean(edge["quality"] for edge in strong_edges[: max(1, n_chains)]) if strong_edges else 0.0
    contact_quality = statistics.fmean(
        1.0 - math.exp(-edge["contacts"] / max(1.0, 1.5 * length))
        for edge in strong_edges[: max(1, n_chains)]
    ) if strong_edges else 0.0
    total_hbonds = sum(edge["hbonds"] for edge in strong_edges)
    hbond_score = 1.0 - math.exp(-total_hbonds / max(1.0, n_chains * length * 0.35))

    centroid_hist = np.zeros(12, dtype=float)
    if len(centroids) > 1:
        centroid_array = np.asarray(centroids)
        upper = np.triu_indices(len(centroids), 1)
        distances = np.sqrt(np.sum((centroid_array[:, None, :] - centroid_array[None, :, :]) ** 2, axis=2))[upper]
        centroid_hist, _ = np.histogram(distances, bins=np.linspace(0, 60, 13))
        centroid_hist = centroid_hist / max(1, centroid_hist.sum())
    best_matrix = (
        np.asarray(edges[0]["matrix"], dtype=float) / 20.0 if edges
        else np.ones(length * length, dtype=float)
    )
    fingerprint = np.concatenate([
        np.clip(median_matrix[tri] / 30.0, 0, 1),
        np.clip(best_matrix, 0, 1),
        centroid_hist,
        np.asarray([statistics.fmean(beta), statistics.fmean(alpha), repeat, graph_regularity]),
    ])
    residue_completeness = statistics.fmean(
        min(1.0, len(residues) / length) for residues in selected.values()
    )
    confidence = min(1.0, max(0.0, statistics.fmean(confidences) / 100.0)) if confidences else 0.5
    confidence_std = statistics.pstdev(confidences) / 100.0 if len(confidences) > 1 else 0.0
    confidence_informative = confidence_std >= 0.005
    return {
        "model": model, "source": source, "target": target, "state": _model_state(model),
        "n_chains": n_chains, "residue_completeness": round(residue_completeness, 6),
        "sequence_identity": round(sequence_identity, 6), "confidence": round(confidence, 6),
        "confidence_std": round(confidence_std, 6),
        "confidence_informative": confidence_informative,
        "beta": round(statistics.fmean(beta), 6), "alpha": round(statistics.fmean(alpha), 6),
        "repeat": round(repeat, 6), "edge_quality": round(edge_quality, 6),
        "contact_quality": round(contact_quality, 6),
        "hbond_score": round(hbond_score, 6), "graph_regularity": round(graph_regularity, 6),
        "packing": round(packing, 6), "mean_degree": round(mean_degree, 6),
        "clashes": clashes,
        "interfaces": select_distinct_interfaces(edges, limit=keep_edges),
        "chain_distance_matrix": np.round(median_matrix, 3).ravel().tolist(),
        "fingerprint": np.round(fingerprint, 5).tolist(),
        **(native or {}),
    }


def _read_mf_ranking(archive: Path, target: str) -> dict[str, dict]:
    with tarfile.open(archive, "r:gz") as bundle:
        candidates = [
            member for member in bundle.getmembers()
            if member.name.endswith(f"ranking_{target}_all_pdbs.csv")
        ]
        if not candidates:
            return {}
        handle = bundle.extractfile(candidates[0])
        if handle is None:
            return {}
        rows = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
        result = {}
        for row in rows:
            name = row.get("mapped_name", "")
            if not name:
                continue
            def number(key: str):
                try:
                    return float(row[key]) if row.get(key) not in (None, "") else None
                except ValueError:
                    return None
            result[name] = {
                "native_iptm": number("iptm"), "native_ptm": number("ptm"),
                "native_ranking_score": number("af3_ranking_score"),
            }
        return result


def iter_casp_models(directory: Path, target: str) -> Iterator[dict]:
    for path in sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")):
        text = path.read_text(encoding="utf-8", errors="replace")
        name = _clean_model_name(path.name, "casp")
        yield compute_features(parse_structure(text, path.name), name, "casp", target)


def iter_mf_models(archive: Path, target: str) -> Iterator[dict]:
    ranking = _read_mf_ranking(archive, target)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            suffix = Path(member.name).suffix.lower()
            if not member.isfile() or suffix not in (".pdb", ".cif"):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            name = Path(member.name).name
            text = extracted.read().decode("utf-8", errors="replace")
            yield compute_features(
                parse_structure(text, name), name, "massivefold", target,
                ranking.get(name, {}),
            )


def write_features(records: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"features {path.stem}: {count}", flush=True)
    return count


def read_features(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def template_reference(path: Path, target: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return compute_features(
        parse_structure(text, path.name), path.stem, "experimental-template", target,
        keep_edges=100,
    )


def add_template_scores(records: list[dict], reference: dict | None) -> None:
    if reference is None:
        for record in records:
            record["template_chain"] = record["template_interface"] = record["template"] = 0.0
        return
    length = int(math.sqrt(len(reference["chain_distance_matrix"])))
    ref_chain = np.asarray(reference["chain_distance_matrix"]).reshape(length, length)
    ref_edges = [
        np.asarray(edge["matrix"]).reshape(length, length)
        for edge in reference["interfaces"]
    ]
    tri = np.triu_indices(length, 1)
    for record in records:
        try:
            chain = np.asarray(record["chain_distance_matrix"]).reshape(length, length)
            chain_score = _lddt_distances(chain[tri], ref_chain[tri])
        except (ValueError, KeyError):
            chain_score = 0.0
        edge_scores = []
        for edge in record.get("interfaces", []):
            try:
                matrix = np.asarray(edge["matrix"]).reshape(length, length)
            except ValueError:
                continue
            edge_scores.append(max(
                (_lddt_distances(matrix, ref, mask=ref < 15.0) for ref in ref_edges),
                default=0.0,
            ))
        interface_score = statistics.fmean(edge_scores[:2]) if edge_scores else 0.0
        record["template_chain"] = round(chain_score, 6)
        record["template_interface"] = round(interface_score, 6)
        record["template"] = round(0.45 * chain_score + 0.55 * interface_score, 6)


def add_balanced_cluster_support(records: list[dict], random_seed: int = 2463) -> dict:
    usable = [record for record in records if record.get("fingerprint")]
    matrix = np.asarray([record["fingerprint"] for record in usable], dtype=np.float32)
    if len(usable) < 20:
        for record in records:
            record["cluster_support"] = 0.0
        return {"clusters": 0}
    rng = np.random.default_rng(random_seed)
    sample = matrix if len(matrix) <= 20000 else matrix[rng.choice(len(matrix), 20000, replace=False)]
    clusterer = MiniBatchKMeans(n_clusters=10, random_state=random_seed, batch_size=1024, n_init=5)
    clusterer.fit(sample)
    labels = clusterer.predict(matrix)
    distances = clusterer.transform(matrix)[np.arange(len(matrix)), labels]
    distance_scale = max(1e-6, float(np.median(distances)))
    for record, label, distance in zip(usable, labels, distances):
        record["cluster"] = int(label)
        record["cluster_centrality"] = math.exp(-float(distance) / distance_scale)

    state_support = {}
    for state in ("v1", "v2", "unassigned"):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in usable:
            if record["source"] == "casp" and record["state"] != state:
                continue
            grouped[_independent_unit(record)].append(record)
        by_source = {}
        for source in ("casp", "massivefold"):
            units = [members for unit, members in grouped.items() if unit.startswith(source + ":")]
            values = np.zeros(10, dtype=float)
            seen_distributions = set()
            unique_units = 0
            for members in units:
                counts = Counter(member["cluster"] for member in members)
                distribution = np.asarray([
                    counts.get(index, 0) / len(members) for index in range(10)
                ])
                fingerprint = tuple(np.round(distribution, 8))
                if fingerprint in seen_distributions:
                    continue
                seen_distributions.add(fingerprint)
                values += distribution
                unique_units += 1
            by_source[source] = values / max(1, unique_units)
        combined = 0.5 * np.sqrt(by_source["casp"] * by_source["massivefold"]) + 0.25 * (
            by_source["casp"] + by_source["massivefold"]
        )
        state_support[state] = combined
    for record in usable:
        state = record["state"] if record["state"] in ("v1", "v2") else "unassigned"
        record["cluster_support"] = float(
            state_support[state][record["cluster"]] * record["cluster_centrality"]
        )
    for record in records:
        record.setdefault("cluster_support", 0.0)
    return {
        "clusters": 10,
        "distance_scale": distance_scale,
        "state_support": {state: values.round(6).tolist() for state, values in state_support.items()},
    }


def score_records(records: list[dict], target: str) -> tuple[list[dict], dict]:
    config = TARGETS[target]
    add_balanced_cluster_support(records, int(target[1:]))
    for record in records:
        minimum = config["minimum_chains"]
        chain_coverage = min(1.0, record["n_chains"] / minimum)
        completeness = chain_coverage * record["residue_completeness"] * record["sequence_identity"]
        clash_score = math.exp(-record["clashes"] / max(1.0, record["n_chains"] * len(config["sequence"]) * 0.10))
        if target == "T2463":
            # CASP explicitly calls this target an amyloid: cross-beta geometry
            # and repeated inter-chain backbone H-bonds are therefore direct
            # physical priors, not ensemble-vote assumptions.
            physical = (
                0.28 * record["beta"]
                + 0.14 * record["repeat"]
                + 0.18 * record["edge_quality"]
                + 0.14 * record["hbond_score"]
                + 0.10 * record["graph_regularity"]
                + 0.06 * record["packing"]
                + 0.06 * completeness
                + 0.04 * clash_score
            )
        else:
            # The held wwPDB target title identifies the X-ray form as helical.
            # Do not reward the older 7QV6 cross-beta polymorph as if it were
            # either of these new crystal agglomeration folds.
            physical = (
                0.32 * record["alpha"]
                + 0.18 * record["repeat"]
                + 0.20 * record["contact_quality"]
                + 0.10 * record["graph_regularity"]
                + 0.10 * record["packing"]
                + 0.06 * completeness
                + 0.04 * clash_score
            )
        native_values = [
            value for value in (record.get("native_iptm"), record.get("native_ptm"), record.get("native_ranking_score"))
            if value is not None
        ]
        native = statistics.fmean(native_values) if native_values else record["confidence"]
        parsed_confidence = record["confidence"] if record.get("confidence_informative", False) else 0.5
        confidence = 0.55 * parsed_confidence + 0.45 * native
        record.update({
            "completeness": completeness, "clash_score": clash_score,
            "physical": physical, "combined_confidence": confidence,
        })

    # Percentiles temper incompatible confidence scales and are blended with absolute physics.
    for key in ("physical", "combined_confidence", "template", "cluster_support"):
        ranks = percentile_ranks([float(record[key]) for record in records])
        for record, rank in zip(records, ranks):
            record[key + "_percentile"] = rank

    template_state = None
    state_template = {}
    if target == "T2464":
        for state in ("v1", "v2"):
            values = sorted(
                (record["template"] for record in records if record["source"] == "casp" and record["state"] == state),
                reverse=True,
            )
            state_template[state] = statistics.fmean(values[: max(1, len(values) // 10)])
        # Diagnostic only: indicates which submitted track resembles the
        # already-published beta polymorph, not which track is correct.
        template_state = max(state_template, key=state_template.get)

    for record in records:
        physical = 0.65 * record["physical"] + 0.35 * record["physical_percentile"]
        confidence = 0.60 * record["combined_confidence"] + 0.40 * record["combined_confidence_percentile"]
        consensus = record["cluster_support_percentile"]
        if target == "T2464":
            raw = 0.80 * physical + 0.15 * confidence + 0.05 * consensus
        else:
            raw = 0.75 * physical + 0.15 * confidence + 0.10 * consensus
        record["overall"] = min(0.98, max(0.02, 0.04 + 0.92 * raw))
        for edge in record["interfaces"]:
            template_edge = 0.0
            if target == "T2464":
                # Retained for reporting only; the held target is a different,
                # helical X-ray polymorph.
                template_edge = 0.0
            local_quality = edge["quality"] if target == "T2463" else edge["contact_quality"]
            edge["score"] = min(0.98, max(0.01,
                0.45 * record["overall"] + 0.40 * local_quality + 0.15 * template_edge
            ))

    records.sort(key=lambda record: (-record["overall"], record["model"]))
    diagnostics = {
        "target": target, "model_count": len(records),
        "casp_count": sum(r["source"] == "casp" for r in records),
        "massivefold_count": sum(r["source"] == "massivefold" for r in records),
        "template_state": template_state, "state_top_decile_template": state_template,
        "score_definition": {
            "T2463": "75% beta-amyloid physics, 15% confidence, 10% family-balanced cluster support",
            "T2464": "80% helical packing physics, 15% confidence, 5% cluster support; 7QV6 diagnostic only",
        }[target],
        "top_models": [
            {key: record.get(key) for key in (
                "model", "source", "state", "overall", "physical", "combined_confidence",
                "template", "cluster_support", "n_chains", "clashes",
            )}
            for record in records[:30]
        ],
    }
    return records, diagnostics


def write_scores(records: list[dict], path: Path) -> None:
    fields = [
        "model", "source", "state", "overall", "physical", "combined_confidence",
        "template", "template_chain", "template_interface", "cluster_support",
        "n_chains", "completeness", "beta", "alpha", "repeat", "edge_quality", "contact_quality",
        "hbond_score", "graph_regularity", "packing", "clashes", "interfaces",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            row["interfaces"] = ",".join(
                f"{''.join(_qa_chain_id(c) for c in edge['chains'])}:{edge['score']:.4f}"
                for edge in select_distinct_interfaces(record["interfaces"], limit=2)
            )
            writer.writerow(row)


def write_submission(records: list[dict], target: str, author: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("PFRMAT QA\n")
        handle.write(f"TARGET {target}\n")
        handle.write(f"AUTHOR {author}\n")
        handle.write("METHOD Fibril-specific geometry, repeat, hydrogen-bond, packing and clash assessment.\n")
        if target == "T2464":
            handle.write("METHOD Exact-sequence PDB 7QV6 is retained only as a distinct-polymorph diagnostic.\n")
        handle.write("METHOD Predictor confidence and low-weight method-family-balanced cluster support are included.\n")
        handle.write("METHOD Symmetry-equivalent homomer interfaces are collapsed by distance-matrix similarity.\n")
        handle.write("MODEL 1\n")
        for record in sorted(records, key=lambda item: item["model"]):
            edges = select_distinct_interfaces(record["interfaces"], limit=2)
            if not edges:
                edges = [{"chains": ["A", "B"], "score": 0.0}]
            labels = []
            seen = set()
            for edge in edges:
                label = "".join(_qa_chain_id(chain) for chain in edge["chains"])
                if len(label) != 2 or label in seen:
                    continue
                seen.add(label)
                labels.append(f"{label}:{edge['score']:.4f}")
            if not labels:
                labels = ["AB:0.0000"]
            handle.write(f"{record['model']} {record['overall']:.4f} {', '.join(labels)}\n")
        handle.write("END\n")


def validate_submission(path: Path, target: str, expected: Iterable[str]) -> dict:
    expected_set = set(expected)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "PFRMAT QA" or f"TARGET {target}" not in lines[:5]:
        raise ValueError(f"{path}: invalid QA header")
    start, end = lines.index("MODEL 1"), lines.index("END")
    seen = set()
    for line_number, line in enumerate(lines[start + 1:end], start + 2):
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: malformed record")
        name, overall_text, interface_text = fields
        overall = float(overall_text)
        if not 0 <= overall <= 1 or name in seen:
            raise ValueError(f"{path}:{line_number}: invalid/duplicate model")
        seen.add(name)
        for item in interface_text.replace(" ", "").split(","):
            label, score_text = item.split(":", 1)
            if len(label) != 2 or not 0 <= float(score_text) <= 1:
                raise ValueError(f"{path}:{line_number}: invalid interface {item}")
    if seen != expected_set:
        raise ValueError(
            f"{path}: coverage mismatch missing={sorted(expected_set-seen)[:5]} "
            f"unexpected={sorted(seen-expected_set)[:5]}"
        )
    return {"valid": True, "model_count": len(seen)}


def run_target(args, target: str) -> dict:
    feature_path = args.work_dir / f"{target}_fibril_features.jsonl.gz"
    if args.reuse_features and feature_path.exists():
        records = read_features(feature_path)
        reused = True
    else:
        casp_dir = args.casp_root / f"{target}o"
        mf_archive = args.incoming / f"{target}_all_pdbs_MassiveFold.tar.gz"
        write_features(
            chain(iter_casp_models(casp_dir, target), iter_mf_models(mf_archive, target)),
            feature_path,
        )
        records = read_features(feature_path)
        reused = False
    reference = None
    if target == "T2464":
        reference = template_reference(args.template_7qv6, target)
    add_template_scores(records, reference)
    scored, diagnostics = score_records(records, target)
    score_path = args.results_dir / f"{target}_fibril_scores.csv"
    submission_path = args.results_dir / f"{target}_QA.txt"
    write_scores(scored, score_path)
    write_submission(scored, target, args.author, submission_path)
    diagnostics["validation"] = validate_submission(
        submission_path, target, (record["model"] for record in records)
    )
    diagnostics.update({
        "feature_file": str(feature_path), "feature_reused": reused,
        "score_file": str(score_path), "submission_file": str(submission_path),
        "template": None if reference is None else {
            "pdb": "7QV6", "path": str(args.template_7qv6),
            "modelled_chains": reference["n_chains"],
        },
    })
    diagnostic_path = args.results_dir / f"{target}_fibril_diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    print(f"wrote {submission_path} ({len(records)} models)", flush=True)
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), action="append")
    parser.add_argument("--casp-root", type=Path, default=Path("data/raw/casp"))
    parser.add_argument("--incoming", type=Path, default=Path("incoming"))
    parser.add_argument("--template-7qv6", type=Path, default=Path("evidence/7QV6.pdb"))
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--author", default="REPLACE-WITH-CASP-REGISTRATION-ID")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    diagnostics = [run_target(args, target) for target in (args.target or sorted(TARGETS))]
    return 0 if all(item["validation"]["valid"] for item in diagnostics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
