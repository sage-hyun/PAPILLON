import os
import hashlib
import threading

import dspy
import time

try:
    from .pipeline_types import decide_route
except ImportError:
    from pipeline_types import decide_route


os.environ["LITELLM_LOG"] = "ERROR"


def _debug_enabled():
    return os.getenv("PAPILLON_DEBUG_THREADS", "").lower() in {"1", "true", "yes", "y"}


def _debug_raise_enabled():
    return os.getenv("PAPILLON_DEBUG_RAISE", "").lower() in {"1", "true", "yes", "y"}


def _debug_max_preview():
    try:
        return max(int(os.getenv("PAPILLON_DEBUG_QUERY_PREVIEW", "80")), 0)
    except ValueError:
        return 80


def _debug_log(event, **kwargs):
    if not _debug_enabled():
        return
    thread_id = threading.get_ident()
    parts = [f"[THREAD-DEBUG] event={event}", f"thread={thread_id}"]
    for key, value in kwargs.items():
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def _query_fingerprint(user_query):
    query = user_query or ""
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    preview = query[:_debug_max_preview()].replace("\n", "\\n")
    return digest, preview


def _retry_call(fn, *, max_retries=3, **kwargs):
    last_exc = None
    for _ in range(max_retries):
        try:
            return fn(**kwargs), None
        except Exception as exc:
            last_exc = exc
    return None, last_exc


class CreateStructuredDelegation(dspy.Signature):
    """
    You are a trusted local privacy planner. Given the original user query plus a locally redacted version,
    create a structured cloud prompt that preserves utility without revealing raw personally identifiable
    information. Never copy raw names, organizations, locations, dates, phone numbers, email addresses, URLs,
    IDs, or other identifying details into any output field. Use abstract placeholder-based descriptions instead.
    Preserve the user's original task and intent as literally as possible. Do not introduce a new domain,
    scenario, diagnosis, product, platform, workflow, or background story that is not already supported by the
    userQuery or redactedQuery. Do not convert a general request into a narrower special case. If unsure, stay
    close to the wording and task structure of the original request and only abstract the sensitive details.
    """

    userQuery = dspy.InputField(desc="The full user request. This stays local.")
    redactedQuery = dspy.InputField(desc="A locally redacted version of the user request.")
    placeholderHints = dspy.InputField(desc="The detected placeholders available in the redacted query.")
    task = dspy.OutputField(desc="The same core task as the original request, rewritten without raw PII and without inventing a new domain or scenario.")
    safe_context = dspy.OutputField(desc="Only the context already present in the request, rewritten to avoid raw PII and expressed with placeholders.")
    style_constraints = dspy.OutputField(desc="Tone, formatting, and stylistic requirements that preserve utility without adding unsupported assumptions.")


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
        planner_mode="cot",
    ):
        super().__init__()
        if planner_mode not in {"cot", "predict"}:
            raise ValueError(f"Unsupported planner_mode: {planner_mode}")
        self.planner_mode = planner_mode
        self.structured_prompt_creator = (
            dspy.ChainOfThought(CreateStructuredDelegation)
            if planner_mode == "cot"
            else dspy.Predict(CreateStructuredDelegation)
        )
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.untrusted_model = untrusted_model
        self.privacy_filter = privacy_filter
        self.allow_direct_bypass = allow_direct_bypass
        self.pii_score_threshold = pii_score_threshold

    def analyze_query(self, user_query):
        fp, preview = _query_fingerprint(user_query)
        start = time.perf_counter()
        _debug_log("analyze_query.start", query_fp=fp, query_preview=repr(preview))
        filter_result = self.privacy_filter.analyze(user_query)
        route_decision = decide_route(filter_result, allow_direct_bypass=self.allow_direct_bypass)
        _debug_log(
            "analyze_query.done",
            query_fp=fp,
            route=route_decision.route,
            reason=route_decision.reason,
            pii_count=len(filter_result.entities),
            placeholders=len(filter_result.placeholder_map),
            detector_available=filter_result.detector_available,
            detector_uncertain=filter_result.uncertain,
            elapsed_ms=f"{(time.perf_counter() - start) * 1000:.1f}",
        )
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
        start_time = time.perf_counter()
        fp, preview = _query_fingerprint(user_query)
        _debug_log("forward.start", query_fp=fp, query_preview=repr(preview))
        try:
            preview = self.preview(user_query)
            response = self._execute(user_query, preview, preview["cloud_prompt"], start_time)
            _debug_log(
                "forward.done",
                query_fp=fp,
                route=preview["route"],
                latency_ms=f"{response.latency * 1000:.1f}",
            )
            return response
        except Exception as exc:
            _debug_log(
                "forward.exception",
                query_fp=fp,
                exc_type=type(exc).__name__,
                exc_msg=repr(str(exc)[:300]),
            )
            if _debug_raise_enabled():
                raise
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
                latency=0.0
            )

    def _build_structured_fields(self, user_query, filter_result):
        fp, _ = _query_fingerprint(user_query)
        placeholder_hints = ", ".join(filter_result.placeholder_map.keys()) or "NONE"
        start = time.perf_counter()
        _debug_log(
            "structured_prompt_creator.start",
            query_fp=fp,
            planner_mode=self.planner_mode,
            redacted_len=len(filter_result.redacted_query),
            placeholder_count=len(filter_result.placeholder_map),
        )
        structured_plan, exc = _retry_call(
            self.structured_prompt_creator,
            max_retries=3,
            userQuery=user_query,
            redactedQuery=filter_result.redacted_query,
            placeholderHints=placeholder_hints,
        )
        if structured_plan is None:
            raise exc
        _debug_log(
            "structured_prompt_creator.done",
            query_fp=fp,
            planner_mode=self.planner_mode,
            task_len=len(getattr(structured_plan, "task", "") or ""),
            safe_context_len=len(getattr(structured_plan, "safe_context", "") or ""),
            style_len=len(getattr(structured_plan, "style_constraints", "") or ""),
            rationale_len=len(getattr(structured_plan, "rationale", "") or ""),
            elapsed_ms=f"{(time.perf_counter() - start) * 1000:.1f}",
        )
        return {
            "task": self._clean_text(getattr(structured_plan, "task", "") or ""),
            "safe_context": self._clean_text(getattr(structured_plan, "safe_context", "") or ""),
            "style_constraints": self._clean_text(getattr(structured_plan, "style_constraints", "") or ""),
            "rationale": self._clean_text(getattr(structured_plan, "rationale", "")),
        }

    def _execute(self, user_query, preview, cloud_prompt, start_time):
        fp, _ = _query_fingerprint(user_query)
        route = preview["route"]
        remote_prompt = user_query if route == "direct" else cloud_prompt
        remote_start = time.perf_counter()
        _debug_log(
            "untrusted_model.start",
            query_fp=fp,
            route=route,
            prompt_len=len(remote_prompt or ""),
        )
        response = self.untrusted_model(remote_prompt)[0]
        _debug_log(
            "untrusted_model.done",
            query_fp=fp,
            route=route,
            response_len=len(response or ""),
            elapsed_ms=f"{(time.perf_counter() - remote_start) * 1000:.1f}",
        )
        aggregator_output = response
        if route == "direct":
            final_output = response
        else:
            agg_start = time.perf_counter()
            _debug_log(
                "info_aggregator.start",
                query_fp=fp,
                response_len=len(response or ""),
            )
            aggregator_prediction = self.info_aggregator(
                userQuery=user_query,
                modelExampleResponses=response,
            )
            aggregator_output = getattr(aggregator_prediction, "finalOutput", "") or ""
            final_output = aggregator_output
            _debug_log(
                "info_aggregator.done",
                query_fp=fp,
                output_len=len(final_output or ""),
                elapsed_ms=f"{(time.perf_counter() - agg_start) * 1000:.1f}",
            )
        latency = time.perf_counter() - start_time

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
            structured_delegation_output=preview["structured_fields"],
            info_aggregator_output=aggregator_output,
            latency=latency
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
