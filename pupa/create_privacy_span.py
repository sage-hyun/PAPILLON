import openai
import re
import time
from pathlib import Path


client = openai.OpenAI(api_key="<YOUR_OPENAI_API_KEY>")


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "extract_privacy_span.txt"
prompt_text = PROMPT_PATH.read_text(encoding="utf-8")


def generate_extract(og, redacted):
    msgs = [{"role": "system", "content": f"Given the original string and the redacted string, what are the contents of the [REDACTED] segments? Give your answers one line per segment.\n\nORIGINAL: {og}\n\nREDACTED: {redacted}"}]
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=msgs
    )
    return list(set(response.choices[0].message.content.lower().split("\n")))


def redact_text(user_prompt):
    msgs = [{"role": "system", "content": prompt_text + user_prompt}]
    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=msgs
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response content from redact_text")
            return content
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            sleep_seconds = min(30.0, base_delay * (2 ** attempt))
            print(
                f"[WARN] redact_text failed ({attempt + 1}/{max_retries}): "
                f"{type(exc).__name__}: {exc}. Retry in {sleep_seconds:.1f}s"
            )
            time.sleep(sleep_seconds)

def unredact_information(original_query, redacted):
    redact_segment = redacted.split("[REDACTED]")
    unredact_segments = []
    for i in range(len(redact_segment) - 1):
        try:
            redact_info = re.search(fr"{redact_segment[i]}[\s\S]*?{redact_segment[i + 1]}", original_query).group(0)
        except AttributeError:
            return generate_extract(original_query, redacted)
        except re.error:
            return generate_extract(original_query, redacted)
        unredaction = redact_info[len(redact_segment[i]):]
        unredaction = unredaction[:-len(redact_segment[i + 1])]
        if len(unredaction.split(" ")) >= 5:
            return generate_extract(original_query, redacted)
        unredact_segments.append(unredaction.lower().strip())
    unredact_segments = list(set(unredact_segments))
    return unredact_segments

def process_user_query(query):
    user_query_redacted = redact_text(query)
    pii_units = None
    if user_query_redacted and "[REDACTED]" in user_query_redacted:
        all_redacted_spans = unredact_information(query, user_query_redacted)
        if len(all_redacted_spans):
            pii_units = "||".join(all_redacted_spans)
        else:
            pii_units = None
    return pii_units, user_query_redacted

def process_user_query_for_no_pii(query):
    user_query_redacted = redact_text(query)
    if user_query_redacted and "[REDACTED]" in user_query_redacted:
        pii_units = "something"
    else:
        pii_units = ""
    return pii_units, user_query_redacted
