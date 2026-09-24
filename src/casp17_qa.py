#!/usr/bin/env python3
"""CASP17 immune-complex QA using cross-source structural consensus."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import itertools
import json
import math
import re
import statistics
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from antibody_template_pairs import ANTIBODY_TEMPLATE_DISTANCE_PAIRS
from template_pairs import TEMPLATE_DISTANCE_PAIRS

TARGETS = {
    "H2441": {
        "expected_lengths": {"A": 243, "B": 134},
        "interfaces": ["AB"],
        "interface_weights": {"AB": 1.0},
    },
    "H2444": {
        "expected_lengths": {"A": 148, "B": 226, "C": 214},
        "interfaces": ["AB", "AC", "BC"],
        # B is Fab heavy and C is Fab light. The two antigen interfaces drive
        # complex placement; the covalently associated Fab heavy/light packing
        # is comparatively conserved and therefore a lower-weight QA signal.
        # Its absolute consensus is used below because percentile-ranking an
        # almost invariant interface magnifies negligible differences.
        "interface_weights": {"AB": 0.50, "AC": 0.33, "BC": 0.17},
        "absolute_consensus_interfaces": ["BC"],
    },
}

CONTACT_CUTOFF = 10.0
TIGHT_CONTACT_CUTOFF = 6.0
CLASH_CUTOFF = 3.2
ANTIBODY_TEMPLATE_WEIGHT = 0.03


@dataclass
class Residue:
    order: int
    representative: tuple[float, float, float] | None = None
    ca: tuple[float, float, float] | None = None
    confidence_sum: float = 0.0
    confidence_n: int = 0

    @property
    def confidence(self) -> float:
        if not self.confidence_n:
            return 0.5
        return max(0.0, min(1.0, self.confidence_sum / self.confidence_n / 100.0))


def _distance_sq(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _grid_contacts(
    left: list[Residue], right: list[Residue], cutoff: float
) -> set[tuple[int, int]]:
    """Return representative-atom contacts using a small spatial hash."""
    cell = cutoff
    cutoff_sq = cutoff * cutoff
    grid: dict[tuple[int, int, int], list[Residue]] = defaultdict(list)
    for residue in right:
        point = residue.representative or residue.ca
        if point is None:
            continue
        key = tuple(math.floor(v / cell) for v in point)
        grid[key].append(residue)

    contacts: set[tuple[int, int]] = set()
    for lres in left:
        point = lres.representative or lres.ca
        if point is None:
            continue
        base = tuple(math.floor(v / cell) for v in point)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for rres in grid.get((base[0] + dx, base[1] + dy, base[2] + dz), ()):
                        other = rres.representative or rres.ca
                        if other is not None and _distance_sq(point, other) <= cutoff_sq:
                            contacts.add((lres.order, rres.order))
    return contacts


def _ca_clashes(left: list[Residue], right: list[Residue]) -> int:
    left_ca = [Residue(r.order, representative=r.ca, ca=r.ca) for r in left if r.ca]
    right_ca = [Residue(r.order, representative=r.ca, ca=r.ca) for r in right if r.ca]
    return len(_grid_contacts(left_ca, right_ca, CLASH_CUTOFF))


def _template_lddt(chains: dict[str, list[Residue]], target: str) -> float:
    model = {residue.order: residue.ca for residue in chains.get("A", ()) if residue.ca}
    values = []
    for left, right, reference_distance in TEMPLATE_DISTANCE_PAIRS[target]:
        if left not in model or right not in model:
            continue
        error = abs(math.dist(model[left], model[right]) - reference_distance)
        values.append(
            sum(error < threshold for threshold in (0.5, 1.0, 2.0, 4.0)) / 4.0
        )
    return statistics.fmean(values) if values else 0.0


def _antibody_template_lddt(chains: dict[str, list[Residue]], target: str) -> float:
    coordinates = {
        chain: {residue.order: residue.ca for residue in residues if residue.ca}
        for chain, residues in chains.items()
    }
    values = []
    for label, pairs in ANTIBODY_TEMPLATE_DISTANCE_PAIRS[target].items():
        left_chain, right_chain = (label, label) if len(label) == 1 else label
        left_model = coordinates.get(left_chain, {})
        right_model = coordinates.get(right_chain, {})
        for left, right, reference_distance in pairs:
            if left not in left_model or right not in right_model:
                continue
            error = abs(
                math.dist(left_model[left], right_model[right]) - reference_distance
            )
            values.append(
                sum(error < threshold for threshold in (0.5, 1.0, 2.0, 4.0)) / 4.0
            )
    return statistics.fmean(values) if values else 0.0


def _finalize_features(
    raw: dict[str, dict[tuple[str, str], Residue]],
    model_name: str,
    source: str,
    target: str,
    chain_mapping: dict[str, str] | None = None,
) -> dict:
    config = TARGETS[target]
    chains = {chain: list(residues.values()) for chain, residues in raw.items()}
    all_residues = [residue for residues in chains.values() for residue in residues]
    confidence = statistics.fmean(r.confidence for r in all_residues) if all_residues else 0.0
    template_lddt = _template_lddt(chains, target)
    antibody_template_lddt = _antibody_template_lddt(chains, target)
    completeness_parts = [
        min(1.0, len(chains.get(chain, ())) / expected)
        for chain, expected in config["expected_lengths"].items()
    ]
    completeness = statistics.fmean(completeness_parts)

    contacts: dict[str, list[str]] = {}
    tight_contacts: dict[str, list[str]] = {}
    interface_confidence: dict[str, float] = {}
    clashes = 0
    for interface in config["interfaces"]:
        left_chain, right_chain = interface
        left = chains.get(left_chain, [])
        right = chains.get(right_chain, [])
        pairs = _grid_contacts(left, right, CONTACT_CUTOFF)
        tight_pairs = _grid_contacts(left, right, TIGHT_CONTACT_CUTOFF)
        contacts[interface] = [f"{i}:{j}" for i, j in sorted(pairs)]
        tight_contacts[interface] = [f"{i}:{j}" for i, j in sorted(tight_pairs)]
        involved_left = {i for i, _ in pairs}
        involved_right = {j for _, j in pairs}
        involved = [r.confidence for r in left if r.order in involved_left]
        involved.extend(r.confidence for r in right if r.order in involved_right)
        interface_confidence[interface] = (
            statistics.fmean(involved) if involved else confidence * 0.5
        )
        clashes += _ca_clashes(left, right)

    return {
        "model": model_name,
        "source": source,
        "target": target,
        "chain_mapping": chain_mapping or {chain: chain for chain in chains},
        "chain_counts": {chain: len(residues) for chain, residues in chains.items()},
        "confidence": round(confidence, 6),
        "template_lddt": round(template_lddt, 6),
        "antibody_template_lddt": round(antibody_template_lddt, 6),
        "completeness": round(completeness, 6),
        "interface_confidence": {
            interface: round(value, 6) for interface, value in interface_confidence.items()
        },
        "contacts": contacts,
        "tight_contacts": tight_contacts,
        "clashes": clashes,
    }


def _canonicalize_chains(
    raw: dict[str, dict[tuple[str, str], Residue]],
    target: str,
) -> tuple[dict[str, dict[tuple[str, str], Residue]], dict[str, str]]:
    """Map participant chain IDs onto target chain IDs using residue counts."""
    expected = TARGETS[target]["expected_lengths"]
    if all(chain in raw for chain in expected):
        mapping = {chain: chain for chain in expected}
        return {chain: raw[chain] for chain in expected}, mapping

    observed = [chain for chain, residues in raw.items() if residues]
    if len(observed) < len(expected):
        mapping = {chain: chain for chain in expected if chain in raw}
        return {chain: raw[chain] for chain in mapping}, mapping

    # Ignore tiny solvent/ligand chains before considering permutations. Eight
    # candidates still permit exhaustive matching for the three-chain targets.
    observed = sorted(observed, key=lambda chain: len(raw[chain]), reverse=True)[:8]
    canonical = list(expected)
    best_score = math.inf
    best_mapping: dict[str, str] = {}
    for assignment in itertools.permutations(observed, len(canonical)):
        mapping = dict(zip(canonical, assignment))
        score = sum(
            abs(len(raw[source_chain]) - expected[target_chain]) / expected[target_chain]
            for target_chain, source_chain in mapping.items()
        )
        if score < best_score:
            best_score = score
            best_mapping = mapping
    return (
        {target_chain: raw[source_chain] for target_chain, source_chain in best_mapping.items()},
        best_mapping,
    )


def _add_atom(
    raw: dict[str, dict[tuple[str, str], Residue]],
    chain: str,
    residue_key: tuple[str, str],
    atom: str,
    point: tuple[float, float, float],
    bfactor: float,
) -> None:
    if residue_key not in raw[chain]:
        raw[chain][residue_key] = Residue(order=len(raw[chain]) + 1)
    residue = raw[chain][residue_key]
    residue.confidence_sum += bfactor
    residue.confidence_n += 1
    if atom == "CA":
        residue.ca = point
        if residue.representative is None:
            residue.representative = point
    elif atom == "CB":
        residue.representative = point


def parse_pdb(lines: Iterable[str], model_name: str, source: str, target: str) -> dict:
    """Extract compact, numbering-independent features from one PDB model."""
    raw: dict[str, dict[tuple[str, str], Residue]] = defaultdict(dict)

    for line in lines:
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        if len(line) < 54:
            continue
        altloc = line[16:17]
        if altloc not in (" ", "A", "1"):
            continue
        atom = line[12:16].strip()
        chain = line[21:22].strip()
        if not chain:
            continue
        residue_key = (line[22:26].strip(), line[26:27].strip())
        try:
            point = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        try:
            bfactor = float(line[60:66])
        except (ValueError, IndexError):
            bfactor = 50.0

        _add_atom(raw, chain, residue_key, atom, point, bfactor)

    canonical, mapping = _canonicalize_chains(raw, target)
    return _finalize_features(canonical, model_name, source, target, mapping)


def parse_mmcif(lines: Iterable[str], model_name: str, source: str, target: str) -> dict:
    """Parse the atom_site loop from AlphaFold/ModelCIF mmCIF output."""
    raw: dict[str, dict[tuple[str, str], Residue]] = defaultdict(dict)
    headers: list[str] = []
    indices: dict[str, int] | None = None

    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            headers = []
            indices = None
            continue
        if stripped.startswith("_atom_site."):
            headers.append(stripped)
            continue
        if headers and stripped.startswith("_"):
            headers = []
            indices = None
            continue
        if not headers or not stripped or stripped == "#":
            if stripped == "#" and indices is not None:
                break
            continue
        if indices is None:
            indices = {name: i for i, name in enumerate(headers)}
        if not (stripped.startswith("ATOM ") or stripped.startswith("HETATM ")):
            continue
        fields = stripped.split()
        if len(fields) < len(headers):
            continue

        def field(*names: str, default: str = "") -> str:
            for name in names:
                index = indices.get(name)
                if index is not None and index < len(fields):
                    value = fields[index]
                    if value not in (".", "?"):
                        return value
            return default

        altloc = field("_atom_site.label_alt_id", default=".")
        if altloc not in (".", "?", "A", "1"):
            continue
        chain = field("_atom_site.auth_asym_id", "_atom_site.label_asym_id")
        if not chain:
            continue
        atom = field("_atom_site.auth_atom_id", "_atom_site.label_atom_id")
        if atom not in ("CA", "CB"):
            continue
        residue_number = field("_atom_site.auth_seq_id", "_atom_site.label_seq_id")
        insertion_code = field("_atom_site.pdbx_PDB_ins_code")
        try:
            point = (
                float(field("_atom_site.Cartn_x")),
                float(field("_atom_site.Cartn_y")),
                float(field("_atom_site.Cartn_z")),
            )
            bfactor = float(field("_atom_site.B_iso_or_equiv", default="50"))
        except ValueError:
            continue
        _add_atom(raw, chain, (residue_number, insertion_code), atom, point, bfactor)

    canonical, mapping = _canonicalize_chains(raw, target)
    return _finalize_features(canonical, model_name, source, target, mapping)


def iter_directory_models(directory: Path, source: str, target: str) -> Iterator[dict]:
    paths = (p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    for path in sorted(paths):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            feature = parse_pdb(handle, path.name.removesuffix(".pdb"), source, target)
        if any(feature["chain_counts"].values()):
            yield feature


def iter_tar_models(archive: Path, source: str, target: str) -> Iterator[dict]:
    records: dict[str, dict] = {}
    confidence: dict[str, dict] = {}
    interfaces = TARGETS[target]["interfaces"]
    chain_order = list(TARGETS[target]["expected_lengths"])

    def compact_confidence(payload: dict) -> dict:
        def interface_matrix(key: str) -> dict[str, float]:
            result: dict[str, float] = {}
            matrix = payload.get(key)
            if not isinstance(matrix, list):
                return result
            for interface in interfaces:
                try:
                    i = chain_order.index(interface[0])
                    j = chain_order.index(interface[1])
                    result[interface] = float(matrix[i][j])
                except (ValueError, IndexError, TypeError):
                    pass
            return result

        # pair_iPTM is the general chain-pair confidence. actiF-pTM is a
        # different, active-interface-focused statistic and may legitimately
        # be zero for antigen interfaces even when pair_iPTM is informative.
        # Preserve both instead of silently substituting one for the other.
        result = {
            "native_interface_confidence": interface_matrix("chain_pair_iptm"),
            "native_interface_actifptm": interface_matrix("chain_pair_actifptm"),
            "native_confidence": None,
            "native_actifptm": None,
            "native_ptm": None,
            "native_ranking_score": None,
            "native_has_clash": bool(payload.get("has_clash", False)),
        }
        for source_key, result_key in (
            ("iptm", "native_confidence"),
            ("actifptm", "native_actifptm"),
            ("ptm", "native_ptm"),
            ("ranking_score", "native_ranking_score"),
        ):
            try:
                if payload.get(source_key) is not None:
                    result[result_key] = float(payload[source_key])
            except (TypeError, ValueError):
                pass
        return result

    with tarfile.open(archive, mode="r:gz") as bundle:
        for member in bundle:
            suffix = Path(member.name).suffix.lower()
            if not member.isfile() or member.size == 0:
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            artifact_name = Path(member.name).name
            model_key = Path(member.name).stem
            if suffix == ".json" and "/confidences/" in member.name:
                try:
                    with io.TextIOWrapper(extracted, encoding="utf-8", errors="replace") as handle:
                        payload = json.load(handle)
                    if isinstance(payload, dict):
                        confidence[model_key] = compact_confidence(payload)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                continue
            if suffix not in (".pdb", ".cif"):
                continue
            with io.TextIOWrapper(extracted, encoding="utf-8", errors="replace") as handle:
                parser = parse_mmcif if suffix == ".cif" else parse_pdb
                feature = parser(
                    handle,
                    artifact_name,
                    source,
                    target,
                )
            if any(feature["chain_counts"].values()):
                records[artifact_name] = feature
    for model_name, feature in records.items():
        feature.update(
            confidence.get(
                Path(model_name).stem,
                {
                    "native_interface_confidence": {},
                    "native_interface_actifptm": {},
                    "native_confidence": None,
                    "native_actifptm": None,
                    "native_ptm": None,
                    "native_ranking_score": None,
                    "native_has_clash": False,
                },
            )
        )
        yield feature


def write_features(records: Iterable[dict], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_features(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile_ranks(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        percentile = average_rank / (len(values) - 1)
        for position in order[start:end]:
            ranks[position] = percentile
        start = end
    return ranks


def _independent_unit(record: dict) -> str:
    """Return a method-level unit so prolific generators do not dominate."""
    model = record["model"].lower()
    if record["source"] == "casp":
        match = re.search(r"ts(\d+)", model)
        return f"casp:{match.group(1) if match else model}"
    for family in ("esmf2", "af3", "afm", "cf"):
        if f"_{family}_" in model:
            return f"massivefold:{family}"
    return f"massivefold:{model}"


def _balanced_contact_support(
    records: list[dict], interface: str
) -> tuple[dict[str, float], dict[str, int]]:
    """Estimate contact support with equal weight per group/model family."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_independent_unit(record)].append(record)

    source_units: dict[str, list[dict[str, float]]] = defaultdict(list)
    for unit, members in grouped.items():
        counts: Counter = Counter()
        for member in members:
            counts.update(set(member["contacts"].get(interface, ())))
        denominator = max(1, len(members))
        source = unit.split(":", 1)[0]
        source_units[source].append(
            {contact: count / denominator for contact, count in counts.items()}
        )

    source_support: dict[str, dict[str, float]] = {}
    for source in ("casp", "massivefold"):
        units = source_units.get(source, [])
        # Paired server/human groups sometimes resubmit the exact same model
        # ensemble under different group IDs. Collapse only byte-equivalent
        # support fingerprints so copied predictions do not receive extra
        # votes, while independently convergent but non-identical models stay.
        unique_units: list[dict[str, float]] = []
        seen_fingerprints: set[tuple[tuple[str, float], ...]] = set()
        for unit_support in units:
            fingerprint = tuple(sorted(unit_support.items()))
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_units.append(unit_support)
        units = unique_units
        totals: Counter = Counter()
        for unit_support in units:
            totals.update(unit_support)
        source_support[source] = {
            contact: total / max(1, len(units)) for contact, total in totals.items()
        }

    contacts = set(source_support["casp"]) | set(source_support["massivefold"])
    combined: dict[str, float] = {}
    for contact in contacts:
        pc = source_support["casp"].get(contact, 0.0)
        pm = source_support["massivefold"].get(contact, 0.0)
        # Cross-source agreement is strongest, but source-specific signals are
        # retained because CASP groups and the four MF families are not fully
        # independent observations of the same search space.
        combined[contact] = 0.5 * math.sqrt(pc * pm) + 0.25 * pc + 0.25 * pm
    return combined, {
        source: len({tuple(sorted(unit.items())) for unit in units})
        for source, units in source_units.items()
    }


def _contact_consensus(
    contact_set: set[str],
    support: dict[str, float],
    reference_contacts: set[str],
) -> tuple[float, float, float]:
    """Return support-weighted precision, recall and harmonic consensus."""
    if not contact_set or not reference_contacts:
        return 0.0, 0.0, 0.0
    reference_scale = statistics.fmean(support[c] for c in reference_contacts)
    precision = statistics.fmean(support.get(c, 0.0) for c in contact_set)
    precision = min(1.0, precision / max(1e-12, reference_scale))
    reference_mass = sum(support[c] for c in reference_contacts)
    recall = sum(support[c] for c in contact_set & reference_contacts) / max(
        1e-12, reference_mass
    )
    consensus = 2.0 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, consensus




def score_records(
    records: list[dict],
    target: str,
    consensus_reference_records: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    config = TARGETS[target]
    reference_records = consensus_reference_records or records
    by_source = Counter(record["source"] for record in records)

    raw_consensus: dict[str, list[float]] = {interface: [] for interface in config["interfaces"]}
    raw_precision: dict[str, list[float]] = {interface: [] for interface in config["interfaces"]}
    raw_recall: dict[str, list[float]] = {interface: [] for interface in config["interfaces"]}
    median_contacts: dict[str, float] = {}
    consensus_units: dict[str, dict[str, int]] = {}
    reference_contact_counts: dict[str, int] = {}
    reference_tight_contact_counts: dict[str, int] = {}
    for interface in config["interfaces"]:
        sizes = [
            len(record["contacts"].get(interface, ())) for record in reference_records
        ]
        positive_sizes = [size for size in sizes if size > 0]
        median_contacts[interface] = statistics.median(positive_sizes) if positive_sizes else 1.0
        support, unit_counts = _balanced_contact_support(reference_records, interface)
        consensus_units[interface] = unit_counts
        # A median-sized, highest-support contact set defines recall. This
        # prevents the union of many low-frequency docking poses from becoming
        # an impossible and noise-dominated reference interface.
        reference_n = max(1, round(median_contacts[interface]))
        reference = {
            contact
            for contact, _ in sorted(
                support.items(), key=lambda item: (-item[1], item[0])
            )[:reference_n]
        }
        reference_contact_counts[interface] = len(reference)
        tight_support, _ = _balanced_contact_support(
            [
                {**record, "contacts": record.get("tight_contacts", {})}
                for record in reference_records
            ],
            interface,
        )
        tight_sizes = [
            len(record.get("tight_contacts", {}).get(interface, ()))
            for record in reference_records
        ]
        positive_tight_sizes = [size for size in tight_sizes if size > 0]
        tight_reference_n = max(
            1, round(statistics.median(positive_tight_sizes or [1]))
        )
        tight_reference = {
            contact
            for contact, _ in sorted(
                tight_support.items(), key=lambda item: (-item[1], item[0])
            )[:tight_reference_n]
        }
        reference_tight_contact_counts[interface] = len(tight_reference)
        for record in records:
            precision, recall, consensus = _contact_consensus(
                set(record["contacts"].get(interface, ())), support, reference
            )
            tight_precision, tight_recall, tight_consensus = _contact_consensus(
                set(record.get("tight_contacts", {}).get(interface, ())),
                tight_support,
                tight_reference,
            )
            raw_precision[interface].append(0.7 * precision + 0.3 * tight_precision)
            raw_recall[interface].append(0.7 * recall + 0.3 * tight_recall)
            raw_consensus[interface].append(0.7 * consensus + 0.3 * tight_consensus)
    ranks = {interface: percentile_ranks(values) for interface, values in raw_consensus.items()}

    scored: list[dict] = []
    for index, record in enumerate(records):
        interface_scores: dict[str, float] = {}
        for interface in config["interfaces"]:
            contact_n = len(record["contacts"].get(interface, ()))
            if contact_n == 0:
                interface_scores[interface] = 0.0
                continue
            median_n = max(1.0, median_contacts[interface])
            sparse_penalty = min(1.0, contact_n / max(6.0, median_n * 0.2))
            dense_penalty = min(1.0, (median_n * 3.0) / contact_n)
            size_plausibility = sparse_penalty * dense_penalty
            clash_penalty = math.exp(-0.35 * record["clashes"])
            native_interface = record.get("native_interface_confidence", {}).get(interface)
            if native_interface is None:
                native_interface = record["interface_confidence"].get(interface, 0.0)
            active_interface = record.get("native_interface_actifptm", {}).get(interface)
            if active_interface is None:
                active_interface = 0.0
            consensus_signal = (
                raw_consensus[interface][index]
                if interface in config.get("absolute_consensus_interfaces", ())
                else ranks[interface][index]
            )
            core = (
                0.48 * consensus_signal
                + 0.18 * record["interface_confidence"].get(interface, 0.0)
                + 0.17 * native_interface
                + 0.03 * active_interface
                + 0.08 * record["completeness"]
                + 0.06 * size_plausibility
            )
            if record.get("native_has_clash"):
                clash_penalty *= 0.85
            interface_scores[interface] = max(
                0.0, min(1.0, (0.03 + 0.94 * core) * clash_penalty)
            )

        weighted_interface = sum(
            config["interface_weights"][interface] * interface_scores[interface]
            for interface in config["interfaces"]
        )
        overall = (
            0.72 * weighted_interface
            + 0.10 * record["confidence"]
            + 0.08 * record["completeness"]
            + 0.10 * record.get("template_lddt", 0.0)
        )
        if "antibody_template_lddt" in record:
            overall = (
                (1.0 - ANTIBODY_TEMPLATE_WEIGHT) * overall
                + ANTIBODY_TEMPLATE_WEIGHT * record["antibody_template_lddt"]
            )
        scored.append(
            {
                **record,
                "raw_consensus": {
                    interface: raw_consensus[interface][index]
                    for interface in config["interfaces"]
                },
                "consensus_precision": {
                    interface: raw_precision[interface][index]
                    for interface in config["interfaces"]
                },
                "consensus_recall": {
                    interface: raw_recall[interface][index]
                    for interface in config["interfaces"]
                },
                "interface_scores": interface_scores,
                "overall": max(0.0, min(1.0, overall)),
            }
        )

    scored.sort(key=lambda item: (-item["overall"], item["model"]))
    diagnostics = {
        "target": target,
        "model_count": len(scored),
        "source_counts": dict(by_source),
        "consensus_units": consensus_units,
        "median_contact_counts": median_contacts,
        "reference_contact_counts": reference_contact_counts,
        "reference_tight_contact_counts": reference_tight_contact_counts,
        "consensus_scaling": {
            interface: (
                "absolute"
                if interface in config.get("absolute_consensus_interfaces", ())
                else "percentile"
            )
            for interface in config["interfaces"]
        },
        "overall_summary": {
            "min": min((r["overall"] for r in scored), default=0.0),
            "median": statistics.median((r["overall"] for r in scored)) if scored else 0.0,
            "max": max((r["overall"] for r in scored), default=0.0),
        },
        "antibody_template": {
            "source": "ImmuneBuilder 2.0.0",
            "weight": ANTIBODY_TEMPLATE_WEIGHT,
            "records": sum("antibody_template_lddt" in record for record in records),
        },
        "top_models": [
            {
                "model": record["model"],
                "source": record["source"],
                "overall": record["overall"],
                "interfaces": record["interface_scores"],
            }
            for record in scored[:25]
        ],
    }
    return scored, diagnostics


def write_score_csv(records: list[dict], target: str, output: Path) -> None:
    interfaces = TARGETS[target]["interfaces"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "source",
                "overall",
                "confidence",
                "template_lddt",
                "antibody_template_lddt",
                "native_confidence",
                "native_ptm",
                "completeness",
                "clashes",
            ]
            + [f"{interface}_score" for interface in interfaces]
            + [f"{interface}_consensus" for interface in interfaces]
            + [f"{interface}_contacts" for interface in interfaces]
            + [f"{interface}_tight_contacts" for interface in interfaces]
        )
        for record in records:
            writer.writerow(
                [
                    record["model"],
                    record["source"],
                    f"{record['overall']:.6f}",
                    f"{record['confidence']:.6f}",
                    f"{record.get('template_lddt', 0.0):.6f}",
                    f"{record.get('antibody_template_lddt', 0.0):.6f}",
                    ""
                    if record.get("native_confidence") is None
                    else f"{record['native_confidence']:.6f}",
                    "" if record.get("native_ptm") is None else f"{record['native_ptm']:.6f}",
                    f"{record['completeness']:.6f}",
                    record["clashes"],
                ]
                + [f"{record['interface_scores'][interface]:.6f}" for interface in interfaces]
                + [f"{record['raw_consensus'][interface]:.8f}" for interface in interfaces]
                + [len(record["contacts"].get(interface, ())) for interface in interfaces]
                + [
                    len(record.get("tight_contacts", {}).get(interface, ()))
                    for interface in interfaces
                ]
            )


def write_submission(records: list[dict], target: str, author: str, output: Path) -> None:
    interfaces = TARGETS[target]["interfaces"]
    by_name = sorted(records, key=lambda record: record["model"])
    with output.open("w", encoding="utf-8") as handle:
        handle.write("PFRMAT QA\n")
        handle.write(f"TARGET {target}\n")
        handle.write(f"AUTHOR {author}\n")
        handle.write(
            "METHOD Cross-source interface-contact consensus over CASP and MassiveFold models.\n"
        )
        handle.write(
            "METHOD Scores combine contact consensus, native confidence, completeness, "
            "clash checks, exact public antigen-template lDDT, and low-weight Ubi "
            "ImmuneBuilder antibody lDDT.\n"
        )
        handle.write("MODEL 1\n")
        for record in by_name:
            contacting = [
                interface
                for interface in interfaces
                if record["contacts"].get(interface)
            ]
            if not contacting:
                contacting = [interfaces[0]]
            interface_text = ", ".join(
                f"{interface}:{record['interface_scores'][interface]:.4f}"
                for interface in contacting
            )
            handle.write(f"{record['model']} {record['overall']:.4f} {interface_text}\n")
        handle.write("END\n")


def validate_submission(
    output: Path,
    target: str,
    expected_models: Iterable[str],
) -> dict:
    """Fail closed on malformed or incomplete CASP17 QA output."""
    expected = set(expected_models)
    seen: set[str] = set()
    lines = output.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "PFRMAT QA":
        raise ValueError(f"{output}: missing PFRMAT QA header")
    if f"TARGET {target}" not in lines[:5]:
        raise ValueError(f"{output}: incorrect TARGET header")
    try:
        model_start = lines.index("MODEL 1")
        model_end = lines.index("END", model_start + 1)
    except ValueError as exc:
        raise ValueError(f"{output}: missing MODEL 1 / END block") from exc
    for line_number, line in enumerate(lines[model_start + 1 : model_end], model_start + 2):
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            raise ValueError(f"{output}:{line_number}: malformed QA record")
        name, overall_text, interface_text = fields
        if name in seen:
            raise ValueError(f"{output}:{line_number}: duplicate model {name}")
        seen.add(name)
        try:
            overall = float(overall_text)
        except ValueError as exc:
            raise ValueError(f"{output}:{line_number}: invalid overall score") from exc
        if not 0.0 <= overall <= 1.0:
            raise ValueError(f"{output}:{line_number}: overall score outside [0, 1]")
        for item in interface_text.replace(" ", "").split(","):
            try:
                interface, score_text = item.split(":", 1)
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(f"{output}:{line_number}: invalid interface score") from exc
            if interface not in TARGETS[target]["interfaces"] or not 0.0 <= score <= 1.0:
                raise ValueError(f"{output}:{line_number}: unsupported interface score {item}")
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise ValueError(
            f"{output}: model coverage mismatch; missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return {"valid": True, "model_count": len(seen)}


def run_target(
    target: str,
    casp_dir: Path,
    massivefold_archive: Path,
    work_dir: Path,
    results_dir: Path,
    author: str,
    reuse_features: bool = False,
) -> dict:
    feature_path = work_dir / f"{target}_features.jsonl.gz"
    feature_reused = reuse_features and feature_path.exists()
    if feature_reused:
        records = read_features(feature_path)
        count = len(records)
    else:
        casp_records = list(iter_directory_models(casp_dir, "casp", target))
        mf_records = list(iter_tar_models(massivefold_archive, "massivefold", target))
        count = write_features(iter(casp_records + mf_records), feature_path)
        records = read_features(feature_path)
    scored, diagnostics = score_records(records, target)
    write_score_csv(scored, target, results_dir / f"{target}_scores.csv")
    submission_path = results_dir / f"{target}_QA.txt"
    write_submission(scored, target, author, submission_path)
    diagnostics["submission"] = validate_submission(
        submission_path,
        target,
        (record["model"] for record in records),
    )
    diagnostics["feature_file"] = str(feature_path)
    diagnostics["feature_reused"] = feature_reused
    diagnostics["written_model_count"] = count
    with (results_dir / f"{target}_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-all", help="Score both immune-complex targets")
    run.add_argument("--h2441-casp", type=Path, required=True)
    run.add_argument("--h2441-mf", type=Path, required=True)
    run.add_argument("--h2444-casp", type=Path, required=True)
    run.add_argument("--h2444-mf", type=Path, required=True)
    run.add_argument("--work-dir", type=Path, default=Path("work"))
    run.add_argument("--results-dir", type=Path, default=Path("results"))
    run.add_argument("--author", default="REPLACE-WITH-REGISTRATION-CODE")
    run.add_argument("--reuse-features", action="store_true",
                     help="Reuse existing feature files instead of reparsing inputs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    jobs = (("H2441", args.h2441_casp, args.h2441_mf),
            ("H2444", args.h2444_casp, args.h2444_mf))
    for target, casp_dir, mf_archive in jobs:
        diagnostics = run_target(target, casp_dir, mf_archive, args.work_dir,
                                 args.results_dir, args.author,
                                 reuse_features=args.reuse_features)
        print(f"[{target}] wrote {diagnostics['model_count']} models", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
