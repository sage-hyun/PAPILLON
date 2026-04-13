from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DetectedEntity:
    entity_type: str
    text: str
    start: int
    end: int
    score: float = 1.0
    source: str = "regex"

    def to_dict(self) -> Dict[str, object]:
        return {
            "entity_type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "source": self.source,
        }


@dataclass(frozen=True)
class PrivacyFilterResult:
    entities: List[DetectedEntity]
    redacted_query: str
    placeholder_map: Dict[str, str]
    has_pii: bool
    detector_available: bool
    uncertain: bool
    error: Optional[str] = None
    low_confidence_entities: List[DetectedEntity] = field(default_factory=list)

    @property
    def can_bypass(self) -> bool:
        return self.detector_available and not self.uncertain and not self.has_pii

    def to_dict(self) -> Dict[str, object]:
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "redacted_query": self.redacted_query,
            "placeholder_map": dict(self.placeholder_map),
            "has_pii": self.has_pii,
            "detector_available": self.detector_available,
            "uncertain": self.uncertain,
            "error": self.error,
            "low_confidence_entities": [entity.to_dict() for entity in self.low_confidence_entities],
            "can_bypass": self.can_bypass,
        }


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    allow_direct_bypass: bool
    detector_available: bool
    detector_uncertain: bool
    detected_pii: List[DetectedEntity] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "route": self.route,
            "reason": self.reason,
            "allow_direct_bypass": self.allow_direct_bypass,
            "detector_available": self.detector_available,
            "detector_uncertain": self.detector_uncertain,
            "detected_pii": [entity.to_dict() for entity in self.detected_pii],
        }


def decide_route(filter_result: PrivacyFilterResult, allow_direct_bypass: bool = True) -> RouteDecision:
    if not allow_direct_bypass:
        reason = "direct_bypass_disabled"
        route = "protected"
    elif not filter_result.detector_available:
        reason = "detector_unavailable"
        route = "protected"
    elif filter_result.uncertain:
        reason = "detector_uncertain"
        route = "protected"
    elif filter_result.has_pii:
        reason = "pii_detected"
        route = "protected"
    else:
        reason = "no_pii_detected"
        route = "direct"

    return RouteDecision(
        route=route,
        reason=reason,
        allow_direct_bypass=allow_direct_bypass,
        detector_available=filter_result.detector_available,
        detector_uncertain=filter_result.uncertain,
        detected_pii=list(filter_result.entities),
    )
