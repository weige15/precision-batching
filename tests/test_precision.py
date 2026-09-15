import unittest

import fastapi
import torch

from swiftllm.precision import PrecisionProfile, all_fp16, quantize_dequantize
from swiftllm.server.scheduler import Scheduler
from swiftllm.server.structs import RawRequest, Request
from swiftllm.engine_config import EngineConfig
from swiftllm.model_config import LlamaModelConfig
from swiftllm.worker.kernels.linear import linear
from swiftllm.worker.weight import infer_model_version
from swiftllm.server.api_server import _raw_request_from_dict


class PrecisionMetadataTests(unittest.TestCase):
    def test_default_profile_is_all_fp16(self):
        profile = PrecisionProfile.default()
        self.assertEqual(profile.as_dict(), {
            "q_bits": 16,
            "k_bits": 16,
            "v_bits": 16,
            "o_bits": 16,
            "ffn_bits": 16,
        })
        self.assertTrue(all_fp16([profile]))
        self.assertTrue(all_fp16(None))

    def test_mapping_rejects_unknown_fields(self):
        with self.assertRaises(ValueError):
            PrecisionProfile.from_mapping({"q_bits": 4, "unknown": 8})
        with self.assertRaises(ValueError):
            PrecisionProfile(q_bits=3)

    def test_request_carries_profile(self):
        request = Request(RawRequest("prompt", 2, {"q_bits": 4, "k_bits": 8}))
        self.assertEqual(request.precision_profile.q_bits, 4)
        self.assertEqual(request.precision_profile.k_bits, 8)
        self.assertEqual(request.precision_profile.v_bits, 16)

    def test_loader_distinguishes_llama31_from_llama32_by_tied_embeddings(self):
        self.assertEqual(infer_model_version({"rope_scaling": {"rope_type": "llama3"}, "tie_word_embeddings": False}), "llama")
        self.assertEqual(infer_model_version({"rope_scaling": {"rope_type": "llama3"}, "tie_word_embeddings": True}), "llama3.2")

    def test_fp16_linear_matches_legacy_operation(self):
        a = torch.arange(12, dtype=torch.float16).reshape(3, 4)
        w = torch.arange(20, dtype=torch.float16).reshape(5, 4)
        profile = PrecisionProfile.default()
        expected = torch.nn.functional.linear(a, w)
        actual = linear(
            a,
            w,
            projection="q",
            precision_profiles=[profile] * 3,
            token_request_ids=[0, 1, 2],
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_invalid_api_profile_is_a_422(self):
        with self.assertRaises(fastapi.HTTPException) as context:
            _raw_request_from_dict({
                "prompt": "x",
                "output_len": 1,
                "precision_profile": {"q_bits": 3},
            })
        self.assertEqual(context.exception.status_code, 422)

    def test_eager_proxy_supports_request_specific_bits(self):
        a = torch.arange(24, dtype=torch.float16).reshape(3, 8)
        w = torch.linspace(-1, 1, 32, dtype=torch.float16).reshape(4, 8)
        profiles = [PrecisionProfile(q_bits=4), PrecisionProfile(q_bits=8), PrecisionProfile()]
        actual = linear(
            a,
            w,
            projection="q",
            precision_profiles=profiles,
            token_request_ids=[0, 1, 2],
        )
        self.assertEqual(actual.shape, (3, 4))
        self.assertFalse(torch.equal(actual[:2], torch.nn.functional.linear(a[:2], w)))
        self.assertTrue(torch.equal(actual[2:], torch.nn.functional.linear(a[2:], w)))
        self.assertTrue(torch.allclose(quantize_dequantize(w, 16), w))


class SchedulerSemanticsTests(unittest.TestCase):
    def test_precision_metadata_does_not_change_fcfs_admission(self):
        model_config = LlamaModelConfig({
            "model_type": "llama",
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_size": 8,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "intermediate_size": 16,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
        })
        engine_config = EngineConfig(
            model_path="unused",
            use_dummy=True,
            block_size=4,
            gpu_mem_utilization=0.9,
            num_cpu_blocks=0,
            max_seqs_in_block_table=8,
            max_blocks_per_seq=8,
            max_batch_size=8,
            max_tokens_in_batch=64,
        )
        plain = [Request(RawRequest("a", 4)), Request(RawRequest("b", 4))]
        profiled = [
            Request(RawRequest("a", 4, {"q_bits": 4})),
            Request(RawRequest("b", 4, {"v_bits": 4})),
        ]
        for request in plain + profiled:
            request.prompt_len = 4
        first = Scheduler(model_config, engine_config, num_gpu_blocks=8)
        second = Scheduler(model_config, engine_config, num_gpu_blocks=8)
        first.on_requests_arrival(plain)
        second.on_requests_arrival(profiled)
        first_batch = first.get_next_batch()[0]
        second_batch = second.get_next_batch()[0]
        self.assertEqual([request.prompt_len for request in first_batch], [request.prompt_len for request in second_batch])
        self.assertEqual(first.num_decoding_gpu_blocks, second.num_decoding_gpu_blocks)
        self.assertEqual([request.request_id for request in first_batch], [0, 1])
        self.assertEqual([request.request_id for request in second_batch], [0, 1])


if __name__ == "__main__":
    unittest.main()
