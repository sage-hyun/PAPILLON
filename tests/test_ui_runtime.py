import unittest
from types import SimpleNamespace

from papillon.pipeline_types import DetectedEntity, PrivacyFilterResult
from papillon_ui.app import FinalInput, PipelineRuntime


class FakeStructuredPipeline:
    def preview(self, user_query):
        return {
            "route": "protected",
            "route_reason": "pii_detected",
            "cloud_prompt": "Task:\nWrite an email.\n\nContext:\nCandidate [PERSON_1] applies.\n\nStyle:\nProfessional.",
            "structured_fields": {
                "task": "Write an email.",
                "safe_context": "Candidate [PERSON_1] applies.",
                "style_constraints": "Professional.",
            },
            "detected_pii": [{"entity_type": "PERSON", "text": "Alice Kim"}],
            "redacted_query": "[PERSON_1] applies.",
            "placeholder_map": {"[PERSON_1]": "Alice Kim"},
            "detector_available": True,
            "detector_uncertain": False,
            "detector_error": None,
        }

    def run_with_prompt(self, original_query, cloud_prompt=None):
        return SimpleNamespace(
            output="Final output",
            route="protected",
            cloud_prompt=cloud_prompt or "Task:\nWrite an email.",
            structured_fields={"task": "Write an email."},
            detected_pii=[{"entity_type": "PERSON", "text": "Alice Kim"}],
        )


class FakeBrokenPipeline:
    def analyze_query(self, user_query):
        filter_result = PrivacyFilterResult(
            entities=[
                DetectedEntity("PERSON", "Duc", 26, 29, score=0.85, source="presidio"),
                DetectedEntity("LOCATION", "Vietnam", 33, 40, score=0.85, source="presidio"),
            ],
            redacted_query="write me an email send to [PERSON_1] at [LOCATION_1]",
            placeholder_map={
                "[PERSON_1]": "Duc",
                "[LOCATION_1]": "Vietnam",
            },
            has_pii=True,
            detector_available=True,
            uncertain=False,
            error=None,
        )
        route_decision = SimpleNamespace(route="protected", reason="pii_detected")
        return filter_result, route_decision

    def preview(self, user_query):
        raise RuntimeError("local model unavailable")


class PipelineRuntimeTests(unittest.TestCase):
    def test_preview_query_exposes_explicit_cloud_prompt_metadata(self):
        runtime = PipelineRuntime()
        runtime.configure(FakeStructuredPipeline())

        preview = runtime.preview_query("Write a professional application email.")

        self.assertEqual(preview["prompt"], preview["cloud_prompt"])
        self.assertEqual(preview["prompt_title"], "Protected Cloud Prompt")
        self.assertEqual(preview["cloud_prompt_format"], "structured_task_context_style")
        self.assertTrue(preview["prompt_editable"])
        self.assertIn("exact text sent to the cloud model", preview["prompt_explanation"])
        self.assertIsNone(preview["preview_error"])

    def test_preview_query_returns_structured_fallback_on_preview_error(self):
        runtime = PipelineRuntime()
        runtime.configure(FakeBrokenPipeline())

        preview = runtime.preview_query("write me an email send to Duc at Vietnam")

        self.assertEqual(preview["route"], "protected")
        self.assertEqual(preview["route_reason"], "pii_detected")
        self.assertIn("Task:\nFulfill the user's request", preview["cloud_prompt"])
        self.assertIn("Redacted user query: write me an email send to [PERSON_1] at [LOCATION_1]", preview["cloud_prompt"])
        self.assertTrue(preview["prompt_editable"])
        self.assertIn("safe fallback prompt", preview["prompt_hint"])
        self.assertEqual(preview["preview_error"], "local model unavailable")
        self.assertEqual(
            preview["detected_pii"],
            [
                {
                    "entity_type": "PERSON",
                    "text": "Duc",
                    "start": 26,
                    "end": 29,
                    "score": 0.85,
                    "source": "presidio",
                },
                {
                    "entity_type": "LOCATION",
                    "text": "Vietnam",
                    "start": 33,
                    "end": 40,
                    "score": 0.85,
                    "source": "presidio",
                },
            ],
        )
        self.assertEqual(preview["redacted_query"], "write me an email send to [PERSON_1] at [LOCATION_1]")
        self.assertTrue(preview["detector_available"])
        self.assertFalse(preview["detector_uncertain"])

    def test_process_query_returns_prompt_aliases_and_metadata(self):
        runtime = PipelineRuntime()
        runtime.configure(FakeStructuredPipeline())

        result = runtime.process_query(
            FinalInput(
                original_query="Write a professional application email.",
                original_prompt="Task:\nWrite an email.",
                edited_prompt="Task:\nWrite a polished email.",
                route="protected",
            )
        )

        self.assertEqual(result["prompt"], result["cloud_prompt"])
        self.assertEqual(result["prompt_title"], "Protected Cloud Prompt")
        self.assertEqual(result["cloud_prompt"], "Task:\nWrite a polished email.")
        self.assertIsNotNone(result["edit_record"])


if __name__ == "__main__":
    unittest.main()
