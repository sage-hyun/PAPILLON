import unittest

from papillon.dspy_compat import (
    build_local_lm,
    normalize_openai_compatible_model_name,
    normalize_openai_model_name,
)


class DspyCompatTests(unittest.TestCase):
    def test_normalize_openai_model_name_adds_provider_prefix(self):
        self.assertEqual(normalize_openai_model_name("gpt-4o-mini"), "openai/gpt-4o-mini")

    def test_normalize_openai_model_name_preserves_qualified_name(self):
        self.assertEqual(normalize_openai_model_name("openai/gpt-4o-mini"), "openai/gpt-4o-mini")

    def test_normalize_openai_compatible_model_name_for_openrouter_openai_model(self):
        self.assertEqual(
            normalize_openai_compatible_model_name("gpt-4o-mini"),
            "openai/openai/gpt-4o-mini",
        )

    def test_normalize_openai_compatible_model_name_for_provider_scoped_model(self):
        self.assertEqual(
            normalize_openai_compatible_model_name("anthropic/claude-3.5-sonnet"),
            "openai/anthropic/claude-3.5-sonnet",
        )

    def test_normalize_openai_compatible_model_name_preserves_double_prefixed_name(self):
        self.assertEqual(
            normalize_openai_compatible_model_name("openai/openai/gpt-4o-mini"),
            "openai/openai/gpt-4o-mini",
        )

    def test_build_local_lm_uses_localhost_style_api_base(self):
        lm = build_local_lm("meta-llama/Llama-3.1-8B-Instruct", host="127.0.0.1", port=3012, api_key="x", max_tokens=10)
        self.assertEqual(lm.model, "openai/meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(lm.kwargs["api_base"], "http://127.0.0.1:3012/v1")
        self.assertEqual(lm.kwargs["api_key"], "x")


if __name__ == "__main__":
    unittest.main()
