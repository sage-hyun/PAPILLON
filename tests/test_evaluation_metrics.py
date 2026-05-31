import unittest

from papillon.evaluation_metrics import (
    entity_retention_rate,
    exposed_token_count,
    parse_pii_units,
    parse_structured_prompt_sections,
    schema_valid,
    weighted_leakage_from_level_counts,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_parse_pii_units_deduplicates_and_ignores_empty_values(self):
        self.assertEqual(parse_pii_units("Alice||alice|| ||OpenAI"), ["Alice", "OpenAI"])

    def test_exposed_token_count_counts_unique_matches(self):
        pii_str = "Alice||OpenAI||Seoul"
        prompt = "Task:\nWrite a note for Alice applying to OpenAI.\n\nContext:\nGeneric\n\nStyle:\nFormal"
        self.assertEqual(exposed_token_count(pii_str, prompt), 2)

    def test_entity_retention_rate_only_scores_entities_present_in_target(self):
        pii_str = "Alice||OpenAI||Seoul"
        target = "Alice should mention OpenAI in the response."
        output = "Alice can mention OpenAI and keep the rest generic."
        self.assertEqual(entity_retention_rate(pii_str, target, output), 1.0)

    def test_parse_structured_prompt_sections_extracts_required_fields(self):
        prompt = "Task:\nWrite an email\n\nContext:\nUse placeholders only\n\nStyle:\nProfessional"
        self.assertEqual(
            parse_structured_prompt_sections(prompt),
            {
                "task": "Write an email",
                "safe_context": "Use placeholders only",
                "style_constraints": "Professional",
            },
        )

    def test_schema_valid_requires_structured_sections_on_protected_route(self):
        structured_fields = {
            "task": "Write an email",
            "safe_context": "Use placeholders only",
            "style_constraints": "Professional",
        }
        prompt = "Task:\nWrite an email\n\nContext:\nUse placeholders only\n\nStyle:\nProfessional"
        self.assertTrue(schema_valid("protected", structured_fields, prompt))
        self.assertFalse(schema_valid("protected", structured_fields, "Write an email"))
        self.assertTrue(schema_valid("direct", {}, "Anything"))

    def test_weighted_leakage_prioritizes_l1_over_l2_and_l3(self):
        scores = weighted_leakage_from_level_counts(
            leaked_l1=1,
            total_l1=1,
            leaked_l2=0,
            total_l2=1,
            leaked_l3=0,
            total_l3=1,
        )
        self.assertGreater(scores["weighted_leakage"], 0.5)


if __name__ == "__main__":
    unittest.main()
