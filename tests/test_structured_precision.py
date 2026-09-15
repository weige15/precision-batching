import sys
import unittest

sys.path.insert(0, "scripts")

from structured_precision_experiment import (  # noqa: E402
    MatrixShape,
    Profile,
    Unit,
    calibration_stability,
    manual_storage_cases,
    neighbor_profiles,
    projection_profiles,
    storage_delta,
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

    def test_storage_delta_uses_exact_integer_bytes(self):
        unit = Unit(0, "q", (MatrixShape("q", 2, 128),))
        base = Profile("w8", "test", {unit.key: 8}).storage([unit])
        candidate = Profile("w4", "test", {unit.key: 4}).storage([unit])
        delta = storage_delta(candidate, base)
        self.assertEqual(delta["total_bits"], -((128 * 2 * 8 + 2 * 16) - (128 * 2 * 4 + 2 * 16)))
        self.assertEqual(delta["total_bytes"] * 8, delta["total_bits"])

    def test_stability_requires_two_of_three_shards_and_tolerance(self):
        stable = calibration_stability({"shards": [{"summary": {"nll_delta": -0.01}}, {"summary": {"nll_delta": -0.002}}, {"summary": {"nll_delta": 0.001}}]}, 3, 0.002)
        unstable = calibration_stability({"shards": [{"summary": {"nll_delta": -0.01}}, {"summary": {"nll_delta": 0.001}}, {"summary": {"nll_delta": 0.003}}]}, 3, 0.002)
        self.assertTrue(stable["stable_improvement"])
        self.assertFalse(unstable["stable_improvement"])

    def test_projection_enumeration_is_three_to_the_fifth(self):
        units = [Unit(0, name, (MatrixShape(name, 2, 128),)) for name in ("q", "k", "v", "o", "ffn")]
        profiles = projection_profiles(units, 0)
        self.assertEqual(len(profiles), 243)
        self.assertEqual(len({profile.signature(units) for profile in profiles}), 243)
        self.assertIn("3^5", profiles[0].optimizer)

    def test_neighbor_generation_contains_compensating_pair(self):
        units = [
            Unit(0, "q", (MatrixShape("q", 8, 128),)),
            Unit(0, "k", (MatrixShape("k", 1, 128),)),
        ]
        anchor = Profile("w8", "uniform", {unit.key: 8 for unit in units})
        target = anchor.storage(units)["total_bits"]
        proposals = neighbor_profiles(anchor, units, target, 0, units, list(reversed(units)), 2)
        self.assertTrue(any(move.startswith("paired:") for _, move in proposals))


if __name__ == "__main__":
    unittest.main()
