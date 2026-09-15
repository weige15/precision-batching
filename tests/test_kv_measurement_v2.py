import unittest

import torch

from scripts.kv_measurement_v2 import dense_page_layout, select_request_local_blocks, storage_ledger
from scripts.reproduce_kivi import cache_bytes, cache_inventory


class KVMeasurementV2Tests(unittest.TestCase):
    def test_dynamic_cache_accounting_includes_k_and_v_and_deduplicates_aliases(self):
        backing = torch.arange(24, dtype=torch.float16)
        key = backing[:12].reshape(1, 1, 3, 4)
        value = backing[12:].reshape(1, 1, 3, 4)
        dynamic = type("DynamicCacheFixture", (), {"key_cache": [key], "value_cache": [value]})()
        self.assertEqual(cache_bytes(dynamic), 2 * key.numel() * key.element_size())

        alias = backing[:12].view(2, 2, 3)
        inventory = cache_inventory(type("DynamicCacheFixture", (), {"key_cache": [key], "value_cache": [alias]})())
        self.assertEqual(inventory["unique_live_storage_bytes"], backing.untyped_storage().nbytes())
        self.assertEqual(inventory["logical_payload_bytes"], key.numel() * 2 * key.element_size())

    def test_dense_layout_permutes_token_and_head_axes_instead_of_reshape(self):
        # Deliberately non-symmetric values make a token/head permutation visible.
        page = torch.arange(2 * 3 * 4, dtype=torch.float16).reshape(2, 3, 4)
        dense_k, dense_v = dense_page_layout([page], [page + 100])
        for token in range(2):
            for head in range(3):
                self.assertTrue(torch.equal(dense_k[0, 0, head, token], page[token, head]))
                self.assertTrue(torch.equal(dense_v[0, 0, head, token], (page + 100)[token, head]))

    def test_request_local_selection_follows_nonidentity_physical_table(self):
        table = torch.tensor([[5, 1, 9], [7, 3, 8]], dtype=torch.int32)
        self.assertEqual(select_request_local_blocks(table, 1 / 3, "old"), [5, 7])
        self.assertEqual(select_request_local_blocks(table, 1 / 3, "recent"), [9, 8])

    def test_storage_ledger_counts_an_aliased_batched_storage_once(self):
        backing = torch.zeros(16, dtype=torch.float16)
        ledger = storage_ledger([("first", backing[:8]), ("alias", backing[2:10])])
        self.assertEqual(ledger["logical_bytes"], 16 * backing.element_size())
        self.assertEqual(ledger["unique_live_storage_bytes"], backing.untyped_storage().nbytes())

    def test_v2_performance_boundary_is_separate_from_quality_scoring(self):
        with open("scripts/reproduce_kivi.py", encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('"performance": performance', source)
        self.assertIn('"quality": quality', source)
        self.assertIn("model prefill/decode calls only", source)
        self.assertIn("run_quality(model", source)


if __name__ == "__main__":
    unittest.main()
