import tempfile
import unittest
from pathlib import Path

from casp17_cage_qa import (
    SEQUENCE,
    _connected_fraction,
    _interface_repeat,
    _sequence_identity,
    _spectrum_agreement,
    compute_features,
    score_records,
    validate_submission,
    write_submission,
)


class CageQATests(unittest.TestCase):
    def test_empty_structure_is_reported_invalid_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pdb"
            path.write_text("END\n")
            record = compute_features(path)
        self.assertTrue(record["invalid"])
        self.assertEqual(score_records([record])[0]["overall"], 0.01)

    def test_connected_fraction(self):
        self.assertEqual(_connected_fraction(4, [(0, 1), (1, 2)]), 0.75)
        self.assertEqual(_connected_fraction(4, [(0, 1), (1, 2), (2, 3)]), 1.0)

    def test_interface_repeat_is_swap_invariant(self):
        edges = [
            {"contact_set": {(1, 2), (3, 4)}},
            {"contact_set": {(2, 1), (4, 3)}},
        ]
        self.assertEqual(_interface_repeat(edges), 1.0)

    def test_terminal_tag_omission_is_aligned(self):
        self.assertEqual(_sequence_identity(SEQUENCE[8:]), 1.0)

    def test_centroid_spectrum_agreement(self):
        import numpy as np

        reference = np.asarray([1.0, 2.0, 4.0])
        self.assertEqual(_spectrum_agreement(reference, reference), 1.0)
        self.assertLess(_spectrum_agreement(reference + 20.0, reference), 0.1)

    def test_submission_round_trip(self):
        records = [{
            "model": "T2461TS001_1o",
            "overall": 0.7,
            "interfaces": [{"chains": ["A", "B"], "score": 0.6}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "T2461_QA.txt"
            write_submission(records, "0000-0000-0000", path)
            result = validate_submission(path, ["T2461TS001_1o"])
        self.assertEqual(result, {"valid": True, "model_count": 1})


if __name__ == "__main__":
    unittest.main()
