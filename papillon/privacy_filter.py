import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

try:
    from .pipeline_types import DetectedEntity, PrivacyFilterResult
except ImportError:
    from pipeline_types import DetectedEntity, PrivacyFilterResult


REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "URL": re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE),
    "PHONE_NUMBER": re.compile(
        r"(?:(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{2,4}\)?[\s.\-]?)?\d{3,4}[\s.\-]?\d{4})"
    ),
    "DATE_TIME": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:19|20)\d{2}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
        re.IGNORECASE,
    ),
    "BENCHMARK_TAG": re.compile(
        r"(?:<)?presidio_anonymized_[a-z_]+(?:>)?",
        re.IGNORECASE
    ),
    "USER_FILE_PATH": re.compile(
        r"(?:[a-zA-Z]:\\Users\\[^\s\"'\\]+|/home/[^\s\"'\\]+|/Users/[^\s\"'\\]+)", 
        re.IGNORECASE
    ),
    # "ID_LIKE": re.compile(r"\b(?=[A-Z0-9-]{6,}\b)(?=.*\d)(?=.*[A-Z])[A-Z0-9-]+\b", re.IGNORECASE),
}

PRESIDIO_ENTITY_TYPES = (
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "CRYPTO",
    # "EMAIL_ADDRESS",  # Covered by custom REGEX_PATTERNS
    # "PHONE_NUMBER",   # Covered by custom REGEX_PATTERNS
    # "URL",            # Covered by custom REGEX_PATTERNS
    # "DATE_TIME",      # Covered by custom REGEX_PATTERNS
)


class PrivacyFilter:
    def __init__(
        self,
        score_threshold: float = 0.5,
        analyzer=None,
        analyzer_factory: Optional[Callable[[], object]] = None,
        model_name: str = "en_core_web_lg",
    ):
        self.score_threshold = score_threshold
        self.model_name = model_name
        self._analyzer = analyzer
        self._analyzer_factory = analyzer_factory or self._default_analyzer_factory
        self._analyzer_attempted = analyzer is not None
        self._analyzer_error: Optional[str] = None

    def analyze(self, text: str) -> PrivacyFilterResult:
        query = text or ""
        regex_entities = self._detect_with_regex(query)
        analyzer = self._ensure_analyzer()
        detector_available = analyzer is not None
        uncertain = not detector_available
        high_confidence_entities: List[DetectedEntity] = list(regex_entities)
        low_confidence_entities: List[DetectedEntity] = []
        error_message = self._analyzer_error

        if analyzer is not None:
            try:
                analyzer_entities, low_confidence_entities = self._detect_with_presidio(analyzer, query)
                high_confidence_entities.extend(analyzer_entities)
                uncertain = bool(low_confidence_entities)
            except Exception as exc:
                # We do NOT set self._analyzer = None to avoid disabling it for subsequent queries.
                # Instead, we mark detector_available = False for this query, record the error,
                # and fall back to regex entities for this specific query.
                detector_available = False
                uncertain = True
                error_message = f"RuntimeError during Presidio detection: {type(exc).__name__}: {exc}"
                
                # Fetch regex entities since Presidio failed
                regex_entities = self._detect_with_regex(query)
                high_confidence_entities = list(regex_entities)
                low_confidence_entities = []

        if not detector_available:
            high_confidence_entities = self._detect_with_regex(query)

        merged_entities = self._merge_entities(query, high_confidence_entities)
        redacted_query, placeholder_map = self._redact_text(query, merged_entities)

        return PrivacyFilterResult(
            entities=merged_entities,
            redacted_query=redacted_query,
            placeholder_map=placeholder_map,
            has_pii=bool(merged_entities),
            detector_available=detector_available,
            uncertain=uncertain,
            error=error_message,
            low_confidence_entities=low_confidence_entities,
        )

    def _ensure_analyzer(self):
        if self._analyzer_attempted:
            return self._analyzer

        self._analyzer_attempted = True
        try:
            self._analyzer = self._analyzer_factory()
            self._analyzer_error = None
        except ImportError as imp_err:
            # Let ImportError propagate and terminate immediately at startup if packages are missing
            raise ImportError(
                f"Presidio analyzer packages are not installed: {imp_err}. "
                "Please run 'pip install presidio-analyzer spacy' before running."
            ) from imp_err
        except Exception as exc:
            exc_str = str(exc)
            # If the SpaCy model is missing, try self-healing download
            if "Can't find model" in exc_str or "en_core_web" in exc_str:
                print(f"\n[INFO] SpaCy model '{self.model_name}' not found. Attempting to download it automatically...")
                try:
                    import spacy
                    spacy.cli.download(self.model_name)
                    self._analyzer = self._analyzer_factory()
                    self._analyzer_error = None
                    print(f"[INFO] Successfully downloaded and loaded '{self.model_name}'!\n")
                    return self._analyzer
                except Exception as dl_exc:
                    # Propagate OSError if automatic download fails
                    raise OSError(
                        f"SpaCy model '{self.model_name}' is missing and automatic download failed: {dl_exc}. "
                        f"Please run 'python -m spacy download {self.model_name}' manually."
                    ) from exc
            else:
                # Propagate any other startup exceptions
                raise exc

        return self._analyzer

    def _default_analyzer_factory(self):
        import spacy
        spacy.prefer_gpu()

        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": self.model_name}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    def _detect_with_regex(self, text: str) -> List[DetectedEntity]:
        entities: List[DetectedEntity] = []
        for entity_type, pattern in REGEX_PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                if entity_type == "PHONE_NUMBER" and not self._looks_like_phone_number(value):
                    continue
                entities.append(
                    DetectedEntity(
                        entity_type=entity_type,
                        text=value,
                        start=match.start(),
                        end=match.end(),
                        score=1.0,
                        source="regex",
                    )
                )
        return entities

    def _detect_with_presidio(self, analyzer, text: str) -> Tuple[List[DetectedEntity], List[DetectedEntity]]:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=list(PRESIDIO_ENTITY_TYPES),
            score_threshold=0.0,
        )
        high_confidence = []
        low_confidence = []

        for result in results:
            entity = DetectedEntity(
                entity_type=self._normalize_entity_type(result.entity_type),
                text=text[result.start:result.end],
                start=result.start,
                end=result.end,
                score=float(result.score),
                source="presidio",
            )
            if entity.score >= self.score_threshold:
                high_confidence.append(entity)
            else:
                low_confidence.append(entity)

        return high_confidence, low_confidence

    def _merge_entities(self, text: str, entities: Iterable[DetectedEntity]) -> List[DetectedEntity]:
        sorted_entities = sorted(entities, key=lambda item: (item.start, item.end, -item.score))
        if not sorted_entities:
            return []

        merged: List[DetectedEntity] = [sorted_entities[0]]
        for entity in sorted_entities[1:]:
            previous = merged[-1]
            if entity.start <= previous.end:
                start = min(previous.start, entity.start)
                end = max(previous.end, entity.end)
                top_entity = previous if previous.score >= entity.score else entity
                source = "+".join(sorted(set((previous.source + "+" + entity.source).split("+"))))
                merged[-1] = DetectedEntity(
                    entity_type=top_entity.entity_type,
                    text=text[start:end],
                    start=start,
                    end=end,
                    score=max(previous.score, entity.score),
                    source=source,
                )
                continue
            merged.append(entity)

        return merged

    def _redact_text(self, text: str, entities: Iterable[DetectedEntity]) -> Tuple[str, Dict[str, str]]:
        placeholder_counts: Dict[str, int] = {}
        placeholder_map: Dict[str, str] = {}
        parts: List[str] = []
        cursor = 0

        for entity in entities:
            label = self._placeholder_label(entity.entity_type)
            placeholder_counts[label] = placeholder_counts.get(label, 0) + 1
            placeholder = f"[{label}_{placeholder_counts[label]}]"
            parts.append(text[cursor:entity.start])
            parts.append(placeholder)
            placeholder_map[placeholder] = text[entity.start:entity.end]
            cursor = entity.end

        parts.append(text[cursor:])
        return "".join(parts), placeholder_map

    @staticmethod
    def _normalize_entity_type(entity_type: str) -> str:
        if entity_type == "ORG":
            return "ORGANIZATION"
        if entity_type == "GPE":
            return "LOCATION"
        return entity_type.upper()

    @staticmethod
    def _placeholder_label(entity_type: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", entity_type.upper()).strip("_") or "PII"

    @staticmethod
    def _looks_like_phone_number(value: str) -> bool:
        digits_only = re.sub(r"\D", "", value)
        return len(digits_only) >= 8 and any(char in value for char in "+-(). ")
