import unittest

from papillon.dspy_compat import normalize_openai_model_name


class DspyCompatTests(unittest.TestCase):
    def test_normalize_openai_model_name_adds_provider_prefix(self):
        self.assertEqual(normalize_openai_model_name("gpt-4o-mini"), "openai/gpt-4o-mini")

    def test_normalize_openai_model_name_preserves_qualified_name(self):
        self.assertEqual(normalize_openai_model_name("openai/gpt-4o-mini"), "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
