import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm
import openai


L12_SYSTEM_PROMPT = """You are a privacy labeling assistant.
Task: Categorize the provided PII units into two distinct buckets: L1 or L2.

Definitions & Boundaries:
- L1 (Direct Identifiers - Highest Sensitivity): Data that unambiguously identifies a specific individual on its own. 
  * Examples: Full names, email addresses, phone numbers, exact physical addresses, national/government IDs, financial account/card numbers, passwords/credentials, biometric data.
- L2 (Quasi-Identifiers - High Sensitivity): Data that does not uniquely identify someone on its own, but defines their affiliations, locations, or broad profile. 
  * Examples: Partial names/aliases, school/university names, company/organization names, job titles, cities/regions, dates of birth.

Rules:
1) Mutual Exclusivity: Every provided unit must be placed in exactly one bucket (L1 or L2).
2) Strict Preservation: Output the exact strings provided. Do not invent, alter, or translate the units.
3) Tie-breaker: If a unit's classification is ambiguous between L1 and L2, default to L2.

Output format (Strict JSON only):
{"l1_units": ["..."], "l2_units": ["..."]}"""


L3_SYSTEM_PROMPT = """You are a privacy labeling assistant.
Task: Extract L3 terms from the provided redacted user query.

Definition:
- L3 (Entity Context & Specific Traces): Specific details describing the situations, actions, history, or operations of the subjects in the text.
  * INCLUDE: Specific achievements/performance metrics, distinct activities/projects, operational schedules, or specific circumstantial details.
  * EXCLUDE: General skills/software, standardized test names, common industry terms, and broad abstract jargon.

Rules:
1) Extract exact substrings from `redacted_query` only. Do not summarize or alter.
2) Track Coreferences: The context does not need to be immediately adjacent to a "[REDACTED]" tag. Track pronouns (e.g., she, he, it, they) and implicit references to capture the entity's context throughout the text.
3) Focus on Actions & States: Extract meaningful phrases that show *what* the subject specifically did, experienced, or operates, rather than universal facts.
4) Do not extract "[REDACTED]" tags.
5) If no L3 terms exist, return an empty list.

Output JSON only:
{"l3_terms": [...]}"""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for item in items:
        val = normalize_whitespace(str(item))
        if not val:
            continue
        key = val.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(val)
    return output


def split_pii_units(raw_value: str, unit_delimiter: str) -> List[str]:
    if not isinstance(raw_value, str):
        return []
    parts = [p.strip() for p in raw_value.split(unit_delimiter)]
    cleaned = []
    for part in parts:
        low = part.lower()
        if not part or low in {"none", "null", "nan"}:
            continue
        cleaned.append(part)
    return dedupe_keep_order(cleaned)


def extract_json_object(raw_content: str) -> Dict:
    text = (raw_content or "").strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return {}
    return {}


def safe_list_strings(data: Dict, key: str) -> List[str]:
    value = data.get(key, [])
    if not isinstance(value, list):
        return []
    output = []
    for item in value:
        if isinstance(item, str):
            output.append(item)
    return dedupe_keep_order(output)


class LLMLevelingClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_retries: int,
        sleep_seconds: float,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.sleep_seconds = sleep_seconds
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.l12_cache: Dict[Tuple[str, str], Tuple[List[str], List[str]]] = {}
        self.l3_cache: Dict[Tuple[str, str, str, str], List[str]] = {}

    def _chat_json(self, system_prompt: str, user_prompt: str) -> Dict:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                parsed = extract_json_object(content)
                if parsed:
                    return parsed
            except Exception as exc:
                last_exception = exc

            if attempt < self.max_retries - 1:
                wait_s = max(0.0, self.sleep_seconds)
                if wait_s > 0:
                    time.sleep(wait_s)
        if last_exception:
            raise last_exception
        return {}

    def classify_l1_l2(self, user_query: str, pii_units: List[str]) -> Tuple[List[str], List[str]]:
        key = (normalize_whitespace(user_query), "||".join(pii_units))
        if key in self.l12_cache:
            return self.l12_cache[key]

        if not pii_units:
            self.l12_cache[key] = ([], [])
            return [], []

        units_json = json.dumps(pii_units, ensure_ascii=False)
        user_prompt = (
            f"user_query:\n{user_query}\n\n"
            f"pii_units:\n{units_json}\n\n"
            "Return only JSON."
        )
        parsed = self._chat_json(L12_SYSTEM_PROMPT, user_prompt)
        l1_raw = safe_list_strings(parsed, "l1_units")
        l2_raw = safe_list_strings(parsed, "l2_units")

        canonical = {u.lower(): u for u in pii_units}
        l1_set = {item.lower() for item in l1_raw}
        l2_set = {item.lower() for item in l2_raw}

        l1: List[str] = []
        l2: List[str] = []
        for unit in pii_units:
            low = unit.lower()
            if low in l1_set and low in canonical:
                l1.append(canonical[low])
            elif low in l2_set and low in canonical:
                l2.append(canonical[low])
            else:
                # Fallback only for malformed model output: keep unresolved units in L2.
                l2.append(unit)

        l1 = dedupe_keep_order(l1)
        l2 = dedupe_keep_order(l2)
        self.l12_cache[key] = (l1, l2)
        return l1, l2

    def classify_l3(
        self,
        user_query: str,
        redacted_query: str,
        l1_units: List[str],
        l2_units: List[str],
    ) -> List[str]:
        key = (
            normalize_whitespace(user_query),
            normalize_whitespace(redacted_query),
            "||".join(l1_units),
            "||".join(l2_units),
        )
        if key in self.l3_cache:
            return self.l3_cache[key]

        if not isinstance(redacted_query, str) or not redacted_query.strip():
            self.l3_cache[key] = []
            return []

        user_prompt = (
            f"user_query:\n{user_query}\n\n"
            f"redacted_query:\n{redacted_query}\n\n"
            f"l1_units:\n{json.dumps(l1_units, ensure_ascii=False)}\n\n"
            f"l2_units:\n{json.dumps(l2_units, ensure_ascii=False)}\n\n"
            "Return only JSON."
        )
        parsed = self._chat_json(L3_SYSTEM_PROMPT, user_prompt)
        l3_terms = safe_list_strings(parsed, "l3_terms")
        self.l3_cache[key] = l3_terms
        return l3_terms


def process_rows(
    rows: List[dict],
    llm_client: LLMLevelingClient,
    unit_delimiter: str,
) -> List[dict]:
    processed = []
    for row in tqdm(rows, desc="Processing rows"):
        units = split_pii_units(row.get("pii_units", ""), unit_delimiter=unit_delimiter)
        user_query = row.get("user_query", "") if isinstance(row.get("user_query", ""), str) else ""
        redacted_query = row.get("redacted_query", "") if isinstance(row.get("redacted_query", ""), str) else ""

        l1_units, l2_units = llm_client.classify_l1_l2(user_query=user_query, pii_units=units)
        l3_terms = llm_client.classify_l3(
            user_query=user_query,
            redacted_query=redacted_query,
            l1_units=l1_units,
            l2_units=l2_units,
        )

        out = dict(row)
        out["l1_units"] = unit_delimiter.join(l1_units)
        out["l2_units"] = unit_delimiter.join(l2_units)
        out["l3_terms"] = unit_delimiter.join(l3_terms)
        processed.append(out)
    return processed


def create_leveling_dataset(
    input_csv: Path,
    output_csv: Path,
    unit_delimiter: str,
    encoding: str,
    llm_client: LLMLevelingClient,
) -> None:
    with input_csv.open("r", newline="", encoding=encoding) as infile:
        reader = csv.DictReader(infile)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {input_csv}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    for col in ("l1_units", "l2_units", "l3_terms"):
        if col not in fieldnames:
            fieldnames.append(col)

    processed_rows = process_rows(rows, llm_client=llm_client, unit_delimiter=unit_delimiter)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding=encoding) as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    l1_non_empty = sum(1 for row in processed_rows if row.get("l1_units"))
    l2_non_empty = sum(1 for row in processed_rows if row.get("l2_units"))
    l3_non_empty = sum(1 for row in processed_rows if row.get("l3_terms"))

    print(f"Input rows: {len(processed_rows)}")
    print(f"Rows with l1_units: {l1_non_empty}")
    print(f"Rows with l2_units: {l2_non_empty}")
    print(f"Rows with l3_terms: {l3_non_empty}")
    print(f"Wrote: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new CSV with l1_units/l2_units/l3_terms using LLM calls "
            "(2-stage: L1/L2 from pii_units, then L3 from redacted_query)."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "PUPA_TNB.csv",
        help="Input PUPA-style CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "PUPA_TNB_leveling.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--unit_delimiter",
        type=str,
        default="||",
        help="Delimiter for pii_units and output unit columns",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("LEVELING_MODEL", "gpt-4-turbo"),
        help="Model name for chat completion",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="API base URL (default: OPENAI_BASE_URL or OpenAI API URL)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retries for each LLM call",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=1.0,
        help="Sleep seconds between retries",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass --api_key.")

    llm_client = LLMLevelingClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
        sleep_seconds=args.sleep_seconds,
    )
    create_leveling_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        unit_delimiter=args.unit_delimiter,
        encoding=args.encoding,
        llm_client=llm_client,
    )
