from .pipeline_types import DetectedEntity, PrivacyFilterResult, RouteDecision, decide_route
from .privacy_filter import PrivacyFilter

__all__ = [
    "DetectedEntity",
    "PrivacyFilter",
    "PrivacyFilterResult",
    "RouteDecision",
    "decide_route",
]

try:
    from .run_llama_dspy import PAPILLON

    __all__.append("PAPILLON")
except Exception:
    PAPILLON = None

try:
    from .structured_pipeline import StructuredPAPILLON

    __all__.append("StructuredPAPILLON")
except Exception:
    StructuredPAPILLON = None
