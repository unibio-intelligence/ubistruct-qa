import tempfile
import unittest
from pathlib import Path

import numpy as np

from casp17_fibril_qa import (
    _lddt_distances,
    _model_state,
    select_distinct_interfaces,
    validate_submission,
    write_submission,
)


class FibrilQATests(unittest.TestCase):
    def test_distinct_interface_selection_handles_chain_swap(self):
        first = {
            "chains": ["A", "B"],
            "matrix": [1.0, 2.0, 3.0, 4.0],
        }
        symmetry_copy_with_swapped_roles = {
            "chains": ["C", "D"],
            "matrix": [1.0, 3.0, 2.0, 4.0],
        }
        distinct = {
            "chains": ["E", "F"],
            "matrix": [7.0, 8.0, 9.0, 10.0],
        }
        selected = select_distinct_interfaces(
            [first, symmetry_copy_with_swapped_roles, distinct], limit=2,
        )
        self.assertEqual([edge["chains"] for edge in selected], [["A", "B"], ["E", "F"]])

    def test_distance_agreement(self):
        reference = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(_lddt_distances(reference, reference), 1.0)
        self.assertLess(_lddt_distances(reference + 10, reference), 0.1)

    def test_requested_conformation_slots(self):
        self.assertEqual(_model_state("T2463TS028_5o"), "v1")
        self.assertEqual(_model_state("T2463TS028_6o"), "v2")
        self.assertEqual(_model_state("T2463TS028_10o.cif"), "v2")
        self.assertEqual(_model_state("Model_1_af3_basic.cif"), "unassigned")

    def test_submission_round_trip_and_exact_coverage(self):
        records = [
            {
                "model": "T2463TS001_1o",
                "overall": 0.75,
                "interfaces": [{"chains": ["A", "B"], "score": 0.70}],
            },
            {
                "model": "Model_1_af3.cif",
                "overall": 0.55,
                "interfaces": [],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "T2463_QA.txt"
            write_submission(records, "T2463", "0000-0000-0000", path)
            result = validate_submission(path, "T2463", (r["model"] for r in records))
            submission = path.read_text()
        self.assertEqual(result, {"valid": True, "model_count": 2})
        self.assertIn("Model_1_af3.cif 0.5500 AB:0.0000", submission)


if __name__ == "__main__":
    unittest.main()
