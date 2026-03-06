#!/usr/bin/env python3
"""Unit tests for current FairPro mixin behavior."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from fairpro.fairpro_mixin import FairProMixin


class _TokenizerOutput(dict):
    """Simple mapping with .to() like HF tokenizer outputs."""

    def to(self, _device):
        return self


class FakeTokenizer:
    """Minimal tokenizer stub for FairPro mixin tests."""

    eos_token_id = 0

    def __init__(self, decoded_outputs: list[str]):
        self._decoded_outputs = iter(decoded_outputs)

    def __call__(
        self,
        text,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 2048,
    ):
        del return_tensors, padding, truncation, max_length
        batch = len(text) if isinstance(text, list) else 1
        seq_len = 4
        input_ids = torch.ones((batch, seq_len), dtype=torch.long)
        attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
        return _TokenizerOutput(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    def decode(self, _tokens, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return next(self._decoded_outputs)


def _build_mock_model_and_tokenizer(*decoded_outputs: str):
    model = MagicMock()
    model.device = torch.device("cpu")

    def _generate(**kwargs):
        batch = int(kwargs["input_ids"].shape[0])
        num_return_sequences = int(kwargs.get("num_return_sequences", 1))
        seq_len = int(kwargs["input_ids"].shape[1]) + 2
        return torch.ones((batch * num_return_sequences, seq_len), dtype=torch.long)

    model.generate.side_effect = _generate
    tokenizer = FakeTokenizer(list(decoded_outputs))
    return model, tokenizer


class DummyFairProMixin(FairProMixin):
    """Concrete test double for FairProMixin."""

    def enable_fairpro(
        self,
        model_name: str | None = None,
        device: str | None = None,
        model=None,
        tokenizer=None,
        quantization: str | None = None,
    ) -> None:
        del model_name, quantization
        self._fairpro_model = model or MagicMock()
        self._fairpro_tokenizer = tokenizer or FakeTokenizer([])
        self._fairpro_enabled = True
        self._fairpro_device = device or "cpu"
        self.clear_fairpro_cache()

    def disable_fairpro(self) -> None:
        self._fairpro_enabled = False
        self._fairpro_model = None
        self._fairpro_tokenizer = None
        self.clear_fairpro_cache()

    def get_default_system_prompt(self) -> str:
        return "DEFAULT SYSTEM PROMPT"

    def get_meta_prompt(self) -> str:
        return "<system_prompt>{user_prompt}</system_prompt>"


class TestFairProMixin(unittest.TestCase):
    def test_parse_system_prompt_with_tags(self):
        mixin = DummyFairProMixin()
        parsed = mixin._parse_system_prompt(
            "analysis\n<system_prompt>fair prompt</system_prompt>\nnotes"
        )
        self.assertEqual(parsed, "fair prompt")

    def test_generate_fairpro_requires_enable(self):
        mixin = DummyFairProMixin()
        with self.assertRaises(RuntimeError):
            mixin.generate_fairpro_system_prompt("A doctor in a clinic")

    def test_check_prompt_for_bias_is_deterministic_by_default(self):
        model, tokenizer = _build_mock_model_and_tokenizer("no")
        mixin = DummyFairProMixin()
        mixin.enable_fairpro(model=model, tokenizer=tokenizer, device="cpu")

        result = mixin.check_prompt_for_bias("A mountain landscape")

        self.assertFalse(result)
        kwargs = model.generate.call_args.kwargs
        self.assertFalse(kwargs["do_sample"])
        self.assertNotIn("temperature", kwargs)

    def test_batch_generation_uses_default_prompt_for_non_bias(self):
        model, tokenizer = _build_mock_model_and_tokenizer("no")
        mixin = DummyFairProMixin()
        mixin.enable_fairpro(model=model, tokenizer=tokenizer, device="cpu")

        outputs = mixin.generate_fairpro_system_prompts_batch(
            ["A landscape"],
            max_new_tokens=4,
        )

        self.assertEqual(outputs, ["DEFAULT SYSTEM PROMPT"])
        self.assertEqual(model.generate.call_count, 1)

    def test_true_batch_generation_single_generate_call(self):
        model, tokenizer = _build_mock_model_and_tokenizer(
            "<system_prompt>p1</system_prompt>",
            "<system_prompt>p2</system_prompt>",
            "<system_prompt>p3</system_prompt>",
        )
        mixin = DummyFairProMixin()
        mixin.enable_fairpro(model=model, tokenizer=tokenizer, device="cpu")

        outputs = mixin.generate_fairpro_system_prompts_batch(
            ["A doctor", "An engineer", "A teacher"],
            skip_bias_check=True,
            batch_size=8,
            max_new_tokens=4,
        )

        self.assertEqual(outputs, ["p1", "p2", "p3"])
        self.assertEqual(model.generate.call_count, 1)

    def test_prompt_generation_uses_cache(self):
        model, tokenizer = _build_mock_model_and_tokenizer(
            "yes",
            "<system_prompt>cached prompt</system_prompt>",
        )
        mixin = DummyFairProMixin()
        mixin.enable_fairpro(model=model, tokenizer=tokenizer, device="cpu")

        first = mixin.generate_fairpro_system_prompts_batch(
            ["A doctor"], use_cache=True
        )
        second = mixin.generate_fairpro_system_prompts_batch(
            ["A doctor"], use_cache=True
        )

        self.assertEqual(first, ["cached prompt"])
        self.assertEqual(second, ["cached prompt"])
        # First call: 1 bias check + 1 prompt generation. Second call: cache hit.
        self.assertEqual(model.generate.call_count, 2)

    def test_candidate_selector_picks_best_weighted_candidate(self):
        model, tokenizer = _build_mock_model_and_tokenizer(
            "yes",  # bias check
            "<system_prompt>cand_a</system_prompt>",
            "<system_prompt>cand_b</system_prompt>",
            "<system_prompt>cand_c</system_prompt>",
            "fairness=2;faithfulness=2",
            "fairness=5;faithfulness=4",
            "fairness=1;faithfulness=5",
        )
        mixin = DummyFairProMixin()
        mixin.enable_fairpro(model=model, tokenizer=tokenizer, device="cpu")

        outputs = mixin.generate_fairpro_system_prompts_batch(
            ["A doctor"],
            num_candidates=3,
            select_best=True,
            fairness_weight=0.6,
            faithfulness_weight=0.4,
            use_cache=False,
        )

        self.assertEqual(outputs, ["cand_b"])
        # 1 bias check + 1 candidate generation + 3 candidate scoring calls
        self.assertEqual(model.generate.call_count, 5)


if __name__ == "__main__":
    unittest.main()
