"""
Re-validate and augment pii_units using LLM.

For each row:
- All existing pii_units (including presidio_anonymized_* ones) are passed to LLM for validation
- LLM also discovers missed PII in the original query
- Result is written to a new CSV with updated pii_units column
"""

import json
import os
import re
import time
from pathlib import Path

import openai
import pandas as pd
from tqdm import tqdm


client = openai.OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """\
You are a PII (Personally Identifiable Information) expert.

You will be given a user query and a list of candidate PII units extracted from that query.

PII includes: person names, company names, places of origin, current living locations, \
addresses, and social media links.

Do two things:
A) VALIDATE: From the candidate list, keep only units that are genuinely PII by the definition above.
B) DISCOVER: Find any PII present in the query that is not already in the candidate list.

Respond ONLY with JSON (no markdown):
{
  "validated": ["...", ...],
  "discovered": ["...", ...]
}
"""


def split_units(pii_str: str) -> list[str]:
    if not isinstance(pii_str, str) or not pii_str.strip():
        return []
    seen = set()
    result = []
    for piece in pii_str.split("||"):
        cleaned = piece.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def call_llm(user_query: str, candidate_units: list[str], max_retries: int = 5) -> dict:
    user_msg = f"USER QUERY:\n{user_query}\n\nCANDIDATE PII UNITS:\n{json.dumps(candidate_units, ensure_ascii=False)}"
    base_delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
            )
            return extract_json(response.choices[0].message.content or "")
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = min(30.0, base_delay * (2 ** attempt))
            print(f"\n[WARN] attempt {attempt+1}/{max_retries} failed: {exc}. Retry in {delay:.1f}s")
            time.sleep(delay)
    return {}


def revalidate_row(user_query: str, pii_str: str) -> str:
    candidate_units = split_units(pii_str)

    result = call_llm(user_query, candidate_units)

    validated = [u for u in (result.get("validated") or []) if isinstance(u, str) and u.strip()]
    discovered = [u for u in (result.get("discovered") or []) if isinstance(u, str) and u.strip()]

    seen = set()
    final = []
    for u in validated + discovered:
        key = u.strip().lower()
        if key and key not in seen:
            seen.add(key)
            final.append(u.strip())

    return "||".join(final) if final else ""


def process_file(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    assert "user_query" in df.columns and "pii_units" in df.columns, \
        f"Expected columns missing in {input_path.name}"

    new_pii_units = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=input_path.name):
        new_pii_units.append(revalidate_row(str(row["user_query"]), row["pii_units"]))

    df["pii_units"] = new_pii_units
    df.to_csv(output_path, index=False)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    pupa_dir = Path(__file__).resolve().parent

    for name in ("PUPA_New.csv", "PUPA_TNB.csv"):
        input_path = pupa_dir / name
        output_path = pupa_dir / name.replace(".csv", "_repii.csv")
        process_file(input_path, output_path)
