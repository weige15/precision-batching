import sys
import unittest

sys.path.insert(0, "scripts")

from structured_precision_experiment import (  # noqa: E402
    MatrixShape,
    Profile,
    Unit,
    manual_storage_cases,
)


class StorageAccountingTests(unittest.TestCase):
    def test_manual_representation_cases(self):
        cases = manual_storage_cases()
        self.assertTrue(all(cases["checks"].values()))
        self.assertEqual(cases["fp16_no_scale"]["scale_bits"], 0)
        self.assertEqual(cases["w4_divisible"]["total_bits"], 128 * 2 * 4 + 2 * 16)
        self.assertGreater(cases["w4_nondivisible"]["padding_bits"], 0)

    def test_profile_cost_is_sum_of_unit_costs(self):
        units = [
            Unit(0, "q", (MatrixShape("q", 2, 128),)),
            Unit(0, "ffn", (MatrixShape("ffn_gate", 4, 128), MatrixShape("ffn_down", 2, 4))),
        ]
        profile = Profile("mixed", "test", {unit.key: bits for unit, bits in zip(units, (4, 16))})
        storage = profile.storage(units)
        expected = units[0].storage(4)["total_bits"] + units[1].storage(16)["total_bits"]
        self.assertEqual(storage["total_bits"], expected)
        self.assertEqual(storage["scale_bits"], units[0].storage(4)["scale_bits"])


if __name__ == "__main__":
    unittest.main()
