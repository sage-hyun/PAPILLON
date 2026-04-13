import os

import dspy

try:
    from .pipeline_types import decide_route
except ImportError:
    from pipeline_types import decide_route


os.environ["LITELLM_LOG"] = "ERROR"


class CreateStructuredDelegation(dspy.Signature):
    """
    You are a trusted local privacy planner. Given the original user query plus a locally redacted version,
    create a structured cloud prompt that preserves utility without revealing raw personally identifiable
    information. Never copy raw names, organizations, locations, dates, phone numbers, email addresses, URLs,
    IDs, or other identifying details into any output field. Use abstract placeholder-based descriptions instead.
    """

    userQuery = dspy.InputField(desc="The full user request. This stays local.")
    redactedQuery = dspy.InputField(desc="A locally redacted version of the user request.")
    placeholderHints = dspy.InputField(desc="The detected placeholders available in the redacted query.")
    task = dspy.OutputField(desc="The core task the cloud model should perform, without raw PII.")
    safe_context = dspy.OutputField(desc="Useful context rewritten to avoid raw PII and expressed with placeholders.")
    style_constraints = dspy.OutputField(desc="Tone, formatting, and stylistic requirements that preserve utility.")


class InfoAggregator(dspy.Signature):
    """
    You are a helpful assistant. Respond to queries from the user.
    """

    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    modelExampleResponses = dspy.InputField(
        desc="Information from a more powerful language model responding to related queries. Complete the user query by referencing this information. Only you have access to this information."
    )
    finalOutput = dspy.OutputField()


class StructuredPAPILLON(dspy.Module):
    def __init__(
        self,
        untrusted_model,
        privacy_filter,
        allow_direct_bypass=True,
        pii_score_threshold=0.5,
    ):
        super().__init__()
        self.structured_prompt_creator = dspy.ChainOfThought(CreateStructuredDelegation)
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.untrusted_model = untrusted_model
        self.privacy_filter = privacy_filter
        self.allow_direct_bypass = allow_direct_bypass
        self.pii_score_threshold = pii_score_threshold

    def analyze_query(self, user_query):
        filter_result = self.privacy_filter.analyze(user_query)
        route_decision = decide_route(filter_result, allow_direct_bypass=self.allow_direct_bypass)
        return filter_result, route_decision

    def preview(self, user_query):
        filter_result, route_decision = self.analyze_query(user_query)
        structured_fields = {}
        cloud_prompt = user_query

        if route_decision.route == "protected":
            structured_fields = self._build_structured_fields(user_query, filter_result)
            cloud_prompt = self.render_cloud_prompt(structured_fields)

        return {
            "route": route_decision.route,
            "route_reason": route_decision.reason,
            "cloud_prompt": cloud_prompt,
            "structured_fields": structured_fields,
            "detected_pii": [entity.to_dict() for entity in filter_result.entities],
            "redacted_query": filter_result.redacted_query,
            "placeholder_map": dict(filter_result.placeholder_map),
            "detector_available": filter_result.detector_available,
            "detector_uncertain": filter_result.uncertain,
            "detector_error": filter_result.error,
            "can_bypass": filter_result.can_bypass,
        }

    def run_with_prompt(self, user_query, cloud_prompt=None):
        preview = self.preview(user_query)
        effective_prompt = cloud_prompt or preview["cloud_prompt"]
        return self._execute(user_query, preview, effective_prompt)

    def forward(self, user_query):
        try:
            preview = self.preview(user_query)
            return self._execute(user_query, preview, preview["cloud_prompt"])
        except Exception:
            return dspy.Prediction(
                prompt="",
                output="",
                gptResponse="",
                route="protected",
                detected_pii=[],
                cloud_prompt="",
                structured_fields={},
                redacted_query="",
                placeholder_map={},
                detector_available=False,
                detector_uncertain=True,
            )

    def _build_structured_fields(self, user_query, filter_result):
        placeholder_hints = ", ".join(filter_result.placeholder_map.keys()) or "NONE"
        structured_plan = self.structured_prompt_creator(
            userQuery=user_query,
            redactedQuery=filter_result.redacted_query,
            placeholderHints=placeholder_hints,
        )
        return {
            "task": self._clean_text(getattr(structured_plan, "task", "")),
            "safe_context": self._clean_text(getattr(structured_plan, "safe_context", "")),
            "style_constraints": self._clean_text(getattr(structured_plan, "style_constraints", "")),
        }

    def _execute(self, user_query, preview, cloud_prompt):
        route = preview["route"]
        remote_prompt = user_query if route == "direct" else cloud_prompt
        response = self.untrusted_model(remote_prompt)[0]
        if route == "direct":
            final_output = response
        else:
            final_output = self.info_aggregator(
                userQuery=user_query,
                modelExampleResponses=response,
            ).finalOutput

        return dspy.Prediction(
            prompt=remote_prompt,
            output=final_output,
            gptResponse=response,
            route=route,
            detected_pii=preview["detected_pii"],
            cloud_prompt=remote_prompt,
            structured_fields=preview["structured_fields"],
            redacted_query=preview["redacted_query"],
            placeholder_map=preview["placeholder_map"],
            detector_available=preview["detector_available"],
            detector_uncertain=preview["detector_uncertain"],
            route_reason=preview["route_reason"],
        )

    @staticmethod
    def render_cloud_prompt(structured_fields):
        return (
            f"Task:\n{structured_fields.get('task', '').strip()}\n\n"
            f"Context:\n{structured_fields.get('safe_context', '').strip()}\n\n"
            f"Style:\n{structured_fields.get('style_constraints', '').strip()}"
        ).strip()

    @staticmethod
    def _clean_text(value):
        return " ".join((value or "").strip().split())
