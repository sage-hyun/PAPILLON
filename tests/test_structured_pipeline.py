import unittest
from types import SimpleNamespace

from papillon.pipeline_types import DetectedEntity, PrivacyFilterResult
from papillon.structured_pipeline import StructuredPAPILLON


class FakePrivacyFilter:
    def __init__(self, result):
        self.result = result

    def analyze(self, text):
        return self.result


class FakePromptCreator:
    def __call__(self, **kwargs):
        return SimpleNamespace(
            task="Write a professional application email.",
            safe_context="Candidate [PERSON_1] is applying to [ORGANIZATION_1].",
            style_constraints="Professional, concise, and persuasive.",
        )


class FakeAggregator:
    def __call__(self, **kwargs):
        return SimpleNamespace(finalOutput="Final locally synthesized answer")


class FakeRemoteModel:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return ["Cloud draft"]


class StructuredPipelineTests(unittest.TestCase):
    def test_protected_route_renders_schema_and_avoids_raw_detected_pii(self):
        result = PrivacyFilterResult(
            entities=[
                DetectedEntity("PERSON", "Alice Kim", 0, 9, score=0.95, source="presidio"),
                DetectedEntity("ORGANIZATION", "OpenAI", 24, 30, score=0.91, source="presidio"),
            ],
            redacted_query="[PERSON_1] is applying to [ORGANIZATION_1].",
            placeholder_map={
                "[PERSON_1]": "Alice Kim",
                "[ORGANIZATION_1]": "OpenAI",
            },
            has_pii=True,
            detector_available=True,
            uncertain=False,
        )
        remote_model = FakeRemoteModel()
        pipeline = StructuredPAPILLON(
            untrusted_model=remote_model,
            privacy_filter=FakePrivacyFilter(result),
        )
        pipeline.structured_prompt_creator = FakePromptCreator()
        pipeline.info_aggregator = FakeAggregator()

        prediction = pipeline("Alice Kim is applying to OpenAI.")

        self.assertEqual(prediction.route, "protected")
        self.assertIn("Task:\nWrite a professional application email.", prediction.cloud_prompt)
        self.assertIn("Context:\nCandidate [PERSON_1] is applying to [ORGANIZATION_1].", prediction.cloud_prompt)
        self.assertIn("Style:\nProfessional, concise, and persuasive.", prediction.cloud_prompt)
        self.assertNotIn("Alice Kim", prediction.cloud_prompt)
        self.assertNotIn("OpenAI", prediction.cloud_prompt)
        self.assertEqual(prediction.output, "Final locally synthesized answer")
        self.assertEqual(remote_model.prompts[-1], prediction.cloud_prompt)

    def test_direct_route_bypasses_local_aggregation(self):
        result = PrivacyFilterResult(
            entities=[],
            redacted_query="Summarize the article.",
            placeholder_map={},
            has_pii=False,
            detector_available=True,
            uncertain=False,
        )
        remote_model = FakeRemoteModel()
        pipeline = StructuredPAPILLON(
            untrusted_model=remote_model,
            privacy_filter=FakePrivacyFilter(result),
        )
        pipeline.info_aggregator = FakeAggregator()

        prediction = pipeline("Summarize the article.")

        self.assertEqual(prediction.route, "direct")
        self.assertEqual(prediction.cloud_prompt, "Summarize the article.")
        self.assertEqual(prediction.output, "Cloud draft")
        self.assertEqual(remote_model.prompts[-1], "Summarize the article.")


if __name__ == "__main__":
    unittest.main()
