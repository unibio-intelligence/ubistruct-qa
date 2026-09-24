import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


from antibody_template_pairs import ANTIBODY_TEMPLATE_DISTANCE_PAIRS
from casp17_qa import (
    Residue,
    _antibody_template_lddt,
    _template_lddt,
    iter_tar_models,
    parse_mmcif,
    parse_pdb,
    percentile_ranks,
    score_records,
    validate_submission,
    write_submission,
)
from template_pairs import TEMPLATE_DISTANCE_PAIRS


def atom(serial, atom, residue, chain, residue_number, x, y, z, bfactor=80.0):
    element = atom.strip()[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfactor:6.2f}          {element:>2s}\n"
    )


class ParserTests(unittest.TestCase):
    def test_contacts_and_completeness(self):
        pdb = "".join(
            [
                atom(1, "CA", "ALA", "A", 1, 0, 0, 0, 90),
                atom(2, "CB", "ALA", "A", 1, 0, 1, 0, 90),
                atom(3, "CA", "ALA", "B", 1, 0, 0, 8, 70),
                atom(4, "CB", "ALA", "B", 1, 0, 1, 8, 70),
            ]
        )
        feature = parse_pdb(io.StringIO(pdb), "model_1", "casp", "H2441")
        self.assertEqual(feature["contacts"]["AB"], ["1:1"])
        self.assertEqual(feature["tight_contacts"]["AB"], [])
        self.assertEqual(feature["chain_counts"], {"A": 1, "B": 1})
        self.assertGreater(feature["confidence"], 0.7)
        self.assertLess(feature["completeness"], 0.01)

    def test_alternate_chain_ids_are_mapped_by_length(self):
        lines = []
        serial = 1
        for chain, length, z in (("M", 148, 0), ("H", 226, 8), ("L", 214, 25)):
            for residue_number in range(1, length + 1):
                lines.append(atom(serial, "CA", "ALA", chain, residue_number, residue_number, 0, z))
                serial += 1
        feature = parse_pdb(lines, "alternate", "casp", "H2444")
        self.assertEqual(feature["chain_mapping"], {"A": "M", "B": "H", "C": "L"})
        self.assertEqual(feature["chain_counts"], {"A": 148, "B": 226, "C": 214})

    def test_split_h2444_antigen_light_keeps_ac_interface(self):
        lines = []
        serial = 1
        for chain, length, z in (("A", 148, 0), ("C", 214, 8)):
            for residue_number in range(1, length + 1):
                lines.append(
                    atom(serial, "CA", "ALA", chain, residue_number, residue_number, 0, z)
                )
                serial += 1
        feature = parse_pdb(lines, "split_ac", "casp", "H2444")
        self.assertEqual(feature["chain_mapping"], {"A": "A", "C": "C"})
        self.assertEqual(feature["chain_counts"], {"A": 148, "C": 214})
        self.assertTrue(feature["contacts"]["AC"])
        self.assertFalse(feature["contacts"]["AB"])

    def test_percentile_ties(self):
        self.assertEqual(percentile_ranks([1.0, 2.0, 2.0, 3.0]), [0.0, 0.5, 0.5, 1.0])

    def test_template_lddt_uses_exact_ca_distances(self):
        original = TEMPLATE_DISTANCE_PAIRS["H2441"]
        try:
            TEMPLATE_DISTANCE_PAIRS["H2441"] = ((1, 2, 5.0),)
            chains = {
                "A": [
                    Residue(1, ca=(0.0, 0.0, 0.0)),
                    Residue(2, ca=(5.0, 0.0, 0.0)),
                ]
            }
            self.assertEqual(_template_lddt(chains, "H2441"), 1.0)
            chains["A"][1].ca = (10.0, 0.0, 0.0)
            self.assertEqual(_template_lddt(chains, "H2441"), 0.0)
        finally:
            TEMPLATE_DISTANCE_PAIRS["H2441"] = original

    def test_antibody_template_lddt_uses_canonical_antibody_chain(self):
        original = ANTIBODY_TEMPLATE_DISTANCE_PAIRS["H2441"]
        try:
            ANTIBODY_TEMPLATE_DISTANCE_PAIRS["H2441"] = {"B": ((1, 2, 5.0),)}
            chains = {
                "B": [
                    Residue(1, ca=(0.0, 0.0, 0.0)),
                    Residue(2, ca=(5.0, 0.0, 0.0)),
                ]
            }
            self.assertEqual(_antibody_template_lddt(chains, "H2441"), 1.0)
            chains["B"][1].ca = (10.0, 0.0, 0.0)
            self.assertEqual(_antibody_template_lddt(chains, "H2441"), 0.0)
        finally:
            ANTIBODY_TEMPLATE_DISTANCE_PAIRS["H2441"] = original

    def test_mmcif_contacts(self):
        cif = """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
ATOM 1 CA . A 1 0.0 0.0 0.0 90.0 1 A
ATOM 2 CB . A 1 0.0 1.0 0.0 90.0 1 A
ATOM 3 CA . B 1 0.0 0.0 8.0 70.0 1 B
ATOM 4 CB . B 1 0.0 1.0 8.0 70.0 1 B
#
"""
        feature = parse_mmcif(io.StringIO(cif), "model_cif", "massivefold", "H2441")
        self.assertEqual(feature["contacts"]["AB"], ["1:1"])
        self.assertEqual(feature["chain_counts"], {"A": 1, "B": 1})

    def test_massivefold_model_name_keeps_extension(self):
        pdb = "".join(
            [
                atom(1, "CA", "ALA", "A", 1, 0, 0, 0),
                atom(2, "CA", "ALA", "B", 1, 0, 0, 8),
            ]
        ).encode()
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "models.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("target/Model_1.pdb")
                info.size = len(pdb)
                bundle.addfile(info, io.BytesIO(pdb))
            records = list(iter_tar_models(archive, "massivefold", "H2441"))
        self.assertEqual(records[0]["model"], "Model_1.pdb")

    def test_pair_iptm_is_not_replaced_by_active_interface_ptm(self):
        pdb = "".join(
            [
                atom(1, "CA", "ALA", "A", 1, 0, 0, 0),
                atom(2, "CA", "ALA", "B", 1, 0, 0, 8),
            ]
        ).encode()
        confidence = json.dumps(
            {
                "iptm": 0.61,
                "actifptm": 0.0,
                "ptm": 0.78,
                "chain_pair_iptm": [[0.8, 0.62], [0.62, 0.7]],
                "chain_pair_actifptm": [[0.8, 0.0], [0.0, 0.7]],
            }
        ).encode()
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "models.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, body in (
                    ("target/Model_1.pdb", pdb),
                    ("target/confidences/Model_1.json", confidence),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    bundle.addfile(info, io.BytesIO(body))
            record = list(iter_tar_models(archive, "massivefold", "H2441"))[0]
        self.assertEqual(record["native_confidence"], 0.61)
        self.assertEqual(record["native_actifptm"], 0.0)
        self.assertEqual(record["native_interface_confidence"]["AB"], 0.62)
        self.assertEqual(record["native_interface_actifptm"]["AB"], 0.0)

    def test_massivefold_families_are_balanced_in_consensus(self):
        base = {
            "source": "massivefold",
            "target": "H2441",
            "chain_counts": {"A": 243, "B": 134},
            "confidence": 0.8,
            "completeness": 1.0,
            "interface_confidence": {"AB": 0.8},
            "clashes": 0,
        }
        casp = {
            **base,
            "model": "H2441TS001_1",
            "source": "casp",
            "contacts": {"AB": ["1:1"]},
        }
        copied_casp_group = {
            **casp,
            "model": "H2441TS002_1",
        }
        records = [casp, copied_casp_group]
        records.extend(
            {
                **base,
                "model": f"Model_{index}_af3_basic_seed_1.cif",
                "contacts": {"AB": ["1:1"]},
            }
            for index in range(20)
        )
        records.append(
            {
                **base,
                "model": "Model_99_esmf2_basic_seed_1.cif",
                "contacts": {"AB": ["2:2"]},
            }
        )
        _, diagnostics = score_records(records, "H2441")
        self.assertEqual(diagnostics["consensus_units"]["AB"]["casp"], 1)
        self.assertEqual(diagnostics["consensus_units"]["AB"]["massivefold"], 2)


    def test_scores_are_bounded(self):
        records = []
        for source in ("casp", "massivefold"):
            records.append(
                {
                    "model": f"{source}_1",
                    "source": source,
                    "target": "H2441",
                    "chain_counts": {"A": 243, "B": 134},
                    "confidence": 0.8,
                    "completeness": 1.0,
                    "interface_confidence": {"AB": 0.8},
                    "contacts": {"AB": ["1:1", "2:2"]},
                    "clashes": 0,
                }
            )
        scored, diagnostics = score_records(records, "H2441")
        self.assertEqual(diagnostics["model_count"], 2)
        self.assertTrue(all(0.0 <= record["overall"] <= 1.0 for record in scored))
        self.assertTrue(all(0.0 <= record["interface_scores"]["AB"] <= 1.0 for record in scored))

    def test_h2444_bc_uses_absolute_consensus_not_percentile(self):
        base = {
            "source": "casp",
            "target": "H2444",
            "chain_counts": {"A": 148, "B": 226, "C": 214},
            "confidence": 0.8,
            "completeness": 1.0,
            "interface_confidence": {"AB": 0.8, "AC": 0.8, "BC": 0.8},
            "clashes": 0,
        }
        records = [
            {
                **base,
                "model": "H2444TS001_1",
                "contacts": {"AB": ["1:1"], "AC": ["1:1"], "BC": ["1:1"]},
            },
            {
                **base,
                "model": "H2444TS002_1",
                "contacts": {
                    "AB": ["1:1"],
                    "AC": ["1:1"],
                    "BC": ["1:1", "2:2"],
                },
            },
        ]
        scored, diagnostics = score_records(records, "H2444")
        self.assertEqual(
            diagnostics["consensus_scaling"],
            {"AB": "percentile", "AC": "percentile", "BC": "absolute"},
        )
        for record in scored:
            consensus = record["raw_consensus"]["BC"]
            contact_count = len(record["contacts"]["BC"])
            size_plausibility = contact_count / 6.0
            expected_core = (
                0.48 * consensus
                + 0.18 * 0.8
                + 0.17 * 0.8
                + 0.08
                + 0.06 * size_plausibility
            )
            self.assertAlmostEqual(
                record["interface_scores"]["BC"], 0.03 + 0.94 * expected_core
            )

    def test_zero_contact_interface_is_zero(self):
        records = []
        for source in ("casp", "massivefold"):
            records.append(
                {
                    "model": f"{source}_1",
                    "source": source,
                    "target": "H2441",
                    "chain_counts": {"A": 243, "B": 134},
                    "confidence": 0.9,
                    "completeness": 1.0,
                    "interface_confidence": {"AB": 0.9},
                    "contacts": {"AB": []},
                    "clashes": 0,
                }
            )
        scored, _ = score_records(records, "H2441")
        self.assertTrue(all(record["interface_scores"]["AB"] == 0.0 for record in scored))


    def test_submission_round_trip(self):
        record = {
            "model": "model_1",
            "source": "casp",
            "target": "H2441",
            "chain_counts": {"A": 243, "B": 134},
            "confidence": 0.8,
            "completeness": 1.0,
            "interface_confidence": {"AB": 0.8},
            "contacts": {"AB": ["1:1"]},
            "clashes": 0,
        }
        scored, _ = score_records([record], "H2441")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            submission = directory / "H2441_QA.txt"
            write_submission(scored, "H2441", "0000-0000-0000", submission)
            report = validate_submission(submission, "H2441", ["model_1"])
            self.assertEqual(report, {"valid": True, "model_count": 1})



if __name__ == "__main__":
    unittest.main()
