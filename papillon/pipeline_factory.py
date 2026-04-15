try:
    from .privacy_filter import PrivacyFilter
    from .run_llama_dspy import PAPILLON
    from .structured_pipeline import StructuredPAPILLON
except ImportError:
    from privacy_filter import PrivacyFilter
    from run_llama_dspy import PAPILLON
    from structured_pipeline import StructuredPAPILLON


def build_pipeline(
    pipeline_name,
    untrusted_model,
    allow_direct_bypass=True,
    privacy_filter_name="regex_presidio",
    pii_score_threshold=0.5,
):
    if pipeline_name == "legacy":
        return PAPILLON(untrusted_model)
    if pipeline_name == "structured_v1":
        if privacy_filter_name != "regex_presidio":
            raise NotImplementedError(f"Unsupported privacy filter: {privacy_filter_name}")
        return StructuredPAPILLON(
            untrusted_model=untrusted_model,
            privacy_filter=PrivacyFilter(score_threshold=pii_score_threshold),
            allow_direct_bypass=allow_direct_bypass,
            pii_score_threshold=pii_score_threshold,
        )
    raise ValueError(f"Unsupported pipeline: {pipeline_name}")
