import argparse
import types
import unittest

import torch

from swiftllm.engine_config import EngineConfig
from swiftllm.worker.kv_cache import PagedKVCache, page_attention_for_layer


class KVPageTests(unittest.TestCase):
    def make_cache(self, default_format="fp16"):
        return PagedKVCache(
            num_blocks=4,
            num_layers=2,
            num_kv_heads=2,
            block_size=4,
            head_dim=4,
            device="cpu",
            default_format=default_format,
            group_size=4,
        )

    def page_values(self, offset=0):
        k = (torch.arange(32, dtype=torch.float16).reshape(4, 2, 4) - 16) / 8 + offset
        v = -k / 3
        return k, v

    def test_fp16_int8_int4_round_trip_and_exact_reclamation(self):
        cache = self.make_cache()
        k, v = self.page_values()
        cache.write(0, 0, k, v)
        before = cache.logical_bytes()
        int8 = cache.demote_page(0, 0, "int8")
        self.assertEqual(int8.before_bytes, before)
        self.assertEqual(int8.after_bytes, cache.logical_bytes())
        self.assertLess(int8.after_bytes, before)
        self.assertFalse(cache.has_unreclaimed_shadow())
        self.assertFalse(cache.page(0, 0).has_fp16_payload)
        got_k, got_v = cache.read(0, 0)
        self.assertTrue(torch.allclose(got_k, k, atol=0.2, rtol=0))
        self.assertTrue(torch.allclose(got_v, v, atol=0.1, rtol=0))

        int4 = cache.demote_page(0, 0, "int4")
        self.assertLess(int4.after_bytes, int8.after_bytes)
        got_k, got_v = cache.read(0, 0)
        self.assertTrue(torch.allclose(got_k, k, atol=0.2, rtol=0))
        self.assertTrue(torch.allclose(got_v, v, atol=0.2, rtol=0))
        self.assertEqual(cache.metadata_counts(), {"k": {"int4": 1}, "v": {"int4": 1}})

    def test_k_only_and_v_only_formats_are_explicit(self):
        cache = self.make_cache()
        k, v = self.page_values()
        cache.write(0, 0, k, v)
        cache.demote_page(0, 0, "int8", components=("k",))
        page = cache.page(0, 0)
        self.assertEqual((page.k_format, page.v_format), ("int8", "fp16"))
        self.assertEqual(cache.metadata_codes[0, 0].tolist(), [1, 0])
        cache.demote_page(0, 0, "int4", components=("v",))
        page = cache.page(0, 0)
        self.assertEqual((page.k_format, page.v_format), ("int8", "int4"))
        self.assertEqual(cache.metadata_codes[0, 0].tolist(), [1, 2])

    def test_mixed_format_attention_matches_explicit_page_reference(self):
        cache = self.make_cache()
        k0, v0 = self.page_values()
        k1, v1 = self.page_values(100)
        cache.write(0, 0, k0, v0)
        cache.write(1, 0, k1, v1)
        cache.demote_page(0, 0, "int4")
        cache.demote_page(1, 0, "int8")

        model = types.SimpleNamespace(num_q_heads=4, num_kv_heads=2, head_dim=4)
        engine = types.SimpleNamespace(block_size=4)
        q = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4) / 7
        block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        seq_ids = torch.tensor([0], dtype=torch.int32)
        lengths = torch.tensor([8], dtype=torch.int32)
        out = torch.empty(1, 16, dtype=torch.float16)
        page_attention_for_layer(q, cache, block_table, seq_ids, lengths, model, engine, 0, out)

        k0r, v0r = cache.read(0, 0)
        k1r, v1r = cache.read(1, 0)
        k = torch.cat((k0r, k1r)).repeat_interleave(2, dim=1).float()
        v = torch.cat((v0r, v1r)).repeat_interleave(2, dim=1).float()
        weights = torch.softmax(torch.einsum("hd,thd->ht", q[0].float(), k) / 2, dim=-1)
        expected = torch.einsum("ht,thd->hd", weights, v).reshape(-1).to(torch.float16)
        self.assertTrue(torch.allclose(out[0], expected, atol=1e-3, rtol=0))

    def test_append_requantizes_in_place_without_changing_format(self):
        cache = self.make_cache()
        k, v = self.page_values()
        cache.write(0, 0, k, v)
        cache.demote_page(0, 0, "int8")
        replacement_k = torch.full((1, 2, 4), 0.75, dtype=torch.float16)
        replacement_v = torch.full((1, 2, 4), -0.5, dtype=torch.float16)
        cache.write(0, 0, replacement_k, replacement_v, token_offset=2)
        page = cache.page(0, 0)
        self.assertEqual((page.k_format, page.v_format), ("int8", "int8"))
        got_k, got_v = cache.read(0, 0)
        self.assertTrue(torch.allclose(got_k[2], replacement_k[0], atol=0.02, rtol=0))
        self.assertTrue(torch.allclose(got_v[2], replacement_v[0], atol=0.02, rtol=0))
        self.assertFalse(cache.has_unreclaimed_shadow())

    def test_batched_int8_conversion_reclaims_once_and_preserves_metadata(self):
        cache = self.make_cache()
        for block_id in range(3):
            k, v = self.page_values(offset=block_id)
            cache.write(block_id, 0, k, v)
        result = cache.demote_pages_batch([(0, 0), (1, 0), (2, 0)], "int8")
        self.assertEqual(result.target_format, "int8")
        self.assertEqual(len(result.per_page), 3)
        self.assertEqual(result.before_bytes - result.after_bytes, result.reclaimed_bytes)
        self.assertEqual(result.reclaimed_bytes, sum(row.before_bytes - row.after_bytes for row in result.per_page))
        self.assertEqual(cache.metadata_counts(), {"k": {"int8": 3}, "v": {"int8": 3}})
        self.assertFalse(cache.has_unreclaimed_shadow())

    def test_batched_conversion_rejects_non_int8_target(self):
        cache = self.make_cache()
        k, v = self.page_values()
        cache.write(0, 0, k, v)
        with self.assertRaises(ValueError):
            cache.demote_pages_batch([(0, 0)], "int4")

    def test_runtime_cli_exposes_only_lossless_dense_cache_mode(self):
        parser = argparse.ArgumentParser()
        EngineConfig.add_cli_args(parser)
        action = next(action for action in parser._actions if action.dest == "kv_page_format")
        self.assertEqual(action.choices, ("dense_fp16",))

    def test_batched_promotion_restores_fp16_storage(self):
        cache = self.make_cache()
        for block_id in range(2):
            k, v = self.page_values(offset=block_id)
            cache.write(block_id, 0, k, v)
        cache.demote_pages_batch([(0, 0), (1, 0)], "int8")
        result = cache.promote_pages_batch([(0, 0), (1, 0)], "fp16")
        self.assertEqual(result.target_format, "fp16")
        self.assertEqual(cache.metadata_counts(), {"k": {"fp16": 2}, "v": {"fp16": 2}})
        self.assertEqual(cache.logical_bytes(), 2 * 32 * 2 * 2)
        self.assertFalse(cache.has_unreclaimed_shadow())

    def test_invalid_async_components_are_rejected(self):
        cache = self.make_cache()
        k, v = self.page_values()
        cache.write(0, 0, k, v)
        with self.assertRaises(ValueError):
            cache.demote_page_async(0, 0, "int8", components=("x",))

    def test_compressed_default_writes_without_fp16_shadow(self):
        cache = self.make_cache("int4")
        k, v = self.page_values()
        cache.write(0, 0, k, v)
        page = cache.page(0, 0)
        self.assertEqual((page.k_format, page.v_format), ("int4", "int4"))
        self.assertFalse(page.has_fp16_payload)
        self.assertEqual(cache.storage_summary()["pending_old_page_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
