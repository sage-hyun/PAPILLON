import re
from typing import Dict, Iterable, List, Optional


def parse_pii_units(pii_str: Optional[str]) -> List[str]:
    if not isinstance(pii_str, str):
        return []

    seen = set()
    pii_units = []
    for piece in pii_str.split("||"):
        cleaned = piece.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        pii_units.append(cleaned)
    return pii_units


def normalized_leakage(leakage_score: int, pii_str: Optional[str]) -> float:
    pii_units = parse_pii_units(pii_str)
    if not pii_units:
        return 0.0
    return leakage_score / len(pii_units)


def exposed_token_count(pii_str: Optional[str], cloud_prompt: Optional[str]) -> int:
    prompt = (cloud_prompt or "").lower()
    pii_units = parse_pii_units(pii_str)
    
    total_count = 0
    for pii in pii_units:
        pii_lower = pii.lower()
        pattern = rf"\b{re.escape(pii_lower)}\b"
        total_count += len(re.findall(pattern, prompt))
        
    return total_count


def entity_retention_rate(
    pii_str: Optional[str],
    target_response: Optional[str],
    final_output: Optional[str],
) -> float:
    target = (target_response or "").lower()
    output = (final_output or "").lower()
    retained_candidates = [pii for pii in parse_pii_units(pii_str) if pii.lower() in target]
    if not retained_candidates:
        return 0.0
    retained = sum(1 for pii in retained_candidates if pii.lower() in output)
    return retained / len(retained_candidates)


def parse_structured_prompt_sections(prompt: Optional[str]) -> Dict[str, str]:
    rendered_prompt = (prompt or "").strip()
    pattern = re.compile(
        r"(?is)^\s*Task:\s*(.*?)\n+\s*Context:\s*(.*?)\n+\s*Style:\s*(.*?)\s*$"
    )
    match = pattern.match(rendered_prompt)
    if not match:
        return {}
    task, context, style = match.groups()
    return {
        "task": task.strip(),
        "safe_context": context.strip(),
        "style_constraints": style.strip(),
    }


def schema_valid(
    route: Optional[str],
    structured_fields: Optional[Dict[str, str]],
    cloud_prompt: Optional[str],
) -> bool:
    if route != "protected":
        return True

    parsed_sections = parse_structured_prompt_sections(cloud_prompt)
    if not parsed_sections:
        return False

    if not structured_fields:
        return True

    required_fields = ("task", "safe_context", "style_constraints")
    return all(isinstance(structured_fields.get(field), str) and structured_fields.get(field).strip() for field in required_fields)


def collect_deterministic_metrics(
    pii_str: Optional[str],
    target_response: Optional[str],
    final_output: Optional[str],
    cloud_prompt: Optional[str],
    route: Optional[str],
    structured_fields: Optional[Dict[str, str]],
    latency: Optional[float]
) -> Dict[str, object]:
    return {
        "exposed_token_count": exposed_token_count(pii_str, cloud_prompt),
        "entity_retention_rate": entity_retention_rate(pii_str, target_response, final_output),
        "schema_valid": schema_valid(route, structured_fields, cloud_prompt),
        "latency": latency,
        "route": route or "legacy",
    }
