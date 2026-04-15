import unittest

from papillon.pipeline_types import decide_route
from papillon.privacy_filter import PrivacyFilter


class FakeRecognizerResult:
    def __init__(self, entity_type, start, end, score):
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score


class FakeAnalyzer:
    def __init__(self, results=None, should_raise=False):
        self.results = results or []
        self.should_raise = should_raise

    def analyze(self, **kwargs):
        if self.should_raise:
            raise RuntimeError("analyzer failure")
        return list(self.results)


class PrivacyFilterTests(unittest.TestCase):
    def test_regex_detection_redacts_common_pii(self):
        def broken_factory():
            raise RuntimeError("missing analyzer")

        detector = PrivacyFilter(analyzer_factory=broken_factory)
        text = (
            "Email sam@example.com, call +1 (555) 123-4567, visit https://papillon.ai "
            "on 03/14/2026 with ticket AB-123456."
        )

        result = detector.analyze(text)

        self.assertTrue(result.has_pii)
        self.assertIn("[EMAIL_ADDRESS_1]", result.redacted_query)
        self.assertIn("[PHONE_NUMBER_1]", result.redacted_query)
        self.assertIn("[URL_1]", result.redacted_query)
        self.assertIn("[DATE_TIME_1]", result.redacted_query)
        self.assertIn("[ID_LIKE_1]", result.redacted_query)
        self.assertFalse(result.detector_available)
        self.assertTrue(result.uncertain)
        self.assertEqual(decide_route(result).route, "protected")

    def test_presidio_entities_support_person_org_and_location(self):
        text = "John Doe is interviewing at OpenAI in San Francisco."
        analyzer = FakeAnalyzer(
            results=[
                FakeRecognizerResult("PERSON", 0, 8, 0.95),
                FakeRecognizerResult("ORGANIZATION", 28, 34, 0.92),
                FakeRecognizerResult("LOCATION", 38, 51, 0.91),
            ]
        )
        detector = PrivacyFilter(analyzer=analyzer)

        result = detector.analyze(text)
        types = [entity.entity_type for entity in result.entities]

        self.assertEqual(types, ["PERSON", "ORGANIZATION", "LOCATION"])
        self.assertTrue(result.detector_available)
        self.assertFalse(result.uncertain)
        self.assertEqual(decide_route(result).route, "protected")

    def test_overlap_merges_regex_and_analyzer_spans(self):
        text = "Visit https://papillon.ai for details."
        start = text.index("https://papillon.ai")
        end = start + len("https://papillon.ai")
        detector = PrivacyFilter(
            analyzer=FakeAnalyzer([FakeRecognizerResult("URL", start, end, 0.88)])
        )

        result = detector.analyze(text)

        self.assertEqual(len(result.entities), 1)
        self.assertEqual(result.entities[0].entity_type, "URL")
        self.assertIn("presidio", result.entities[0].source)
        self.assertIn("regex", result.entities[0].source)

    def test_low_confidence_detection_blocks_direct_bypass(self):
        detector = PrivacyFilter(
            analyzer=FakeAnalyzer([FakeRecognizerResult("PERSON", 0, 4, 0.3)]),
            score_threshold=0.5,
        )

        result = detector.analyze("Alex wrote a generic status update.")
        route = decide_route(result)

        self.assertFalse(result.has_pii)
        self.assertTrue(result.detector_available)
        self.assertTrue(result.uncertain)
        self.assertEqual(route.route, "protected")

    def test_direct_route_requires_confident_zero_pii_result(self):
        detector = PrivacyFilter(analyzer=FakeAnalyzer([]))

        result = detector.analyze("Summarize the attached article in three bullet points.")
        route = decide_route(result)

        self.assertFalse(result.has_pii)
        self.assertTrue(result.detector_available)
        self.assertFalse(result.uncertain)
        self.assertEqual(route.route, "direct")


if __name__ == "__main__":
    unittest.main()
