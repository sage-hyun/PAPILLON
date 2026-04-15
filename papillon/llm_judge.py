import dspy
import json
import re


class JudgeQuality(dspy.Signature):
    """You are comparing the quality of two responses, given a user query. Is response A at least as good as response B? Respond with yes or no."""

    user_query = dspy.InputField(desc="The user's request to be fulfilled.")
    response_A = dspy.InputField()
    response_B = dspy.InputField()
    output = dspy.OutputField()

class JudgeLeakage(dspy.Signature):
    """Determine whether a piece of information is present in the prompt given. Respond with yes or no."""

    information_piece = dspy.InputField(desc="The piece of information that we wish to check. If this piece of information is a celebrity name, respond with no.")
    prompt = dspy.InputField()
    output = dspy.OutputField()

class JudgePromptQual(dspy.Signature):
    """Determine whether a piece of text is a valid prompt for a language model. Respond with yes or no."""
    prompt_input = dspy.InputField()
    output = dspy.OutputField()

class JudgeL3LeakageBatch(dspy.Signature):
    """Given L3 context terms and a prompt, return JSON {"hits":[0 or 1,...]} in the same order.
    Mark 1 if the same contextual meaning is preserved or paraphrased in the prompt."""

    l3_terms = dspy.InputField(desc="JSON array string of L3 terms in order.")
    prompt = dspy.InputField()
    output = dspy.OutputField()


class LLMJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.fact_checker = dspy.Predict(JudgeLeakage)
        self.prompt_qual = dspy.ChainOfThought(JudgePromptQual)
        self.l3_batch_checker = dspy.Predict(JudgeL3LeakageBatch)

    @staticmethod
    def _split_units(unit_str):
        if not isinstance(unit_str, str):
            return []
        seen = set()
        output = []
        for piece in unit_str.split("||"):
            cleaned = piece.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            output.append(cleaned)
        return output

    @staticmethod
    def _extract_json(raw_text):
        text = (raw_text or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
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

    @staticmethod
    def _normalize_hits(raw_hits, expected_len):
        if not isinstance(raw_hits, list):
            return [0] * expected_len
        hits = []
        for item in raw_hits:
            if isinstance(item, bool):
                hits.append(1 if item else 0)
            elif isinstance(item, (int, float)):
                hits.append(1 if item >= 0.5 else 0)
            elif isinstance(item, str):
                lowered = item.strip().lower()
                hits.append(1 if lowered in {"1", "yes", "y", "true"} else 0)
            else:
                hits.append(0)
        if len(hits) < expected_len:
            hits.extend([0] * (expected_len - len(hits)))
        elif len(hits) > expected_len:
            hits = hits[:expected_len]
        return hits

    def forward(
        self,
        user_query,
        og_resp,
        new_resp=None,
        updated_query=None,
        pii_str=None,
        l1_str=None,
        l2_str=None,
        l3_str=None,
    ):
        if new_resp:
            answer = self.quality_judge(user_query=user_query, response_A=new_resp, response_B=og_resp)
            judgment_1 = int((answer.output.lower()).startswith("yes"))
            answer = self.quality_judge(user_query=user_query, response_A=og_resp, response_B=new_resp)
            judgment_2 = int((answer.output.lower()).startswith("yes"))
            judgment = process_scores(judgment_1, judgment_2)
        else:
            judgment = -1
        leaked_l1 = -1
        leaked_l2 = -1
        leaked_l3 = -1
        total_l1 = 0
        total_l2 = 0
        total_l3 = 0
        if updated_query:
            pii_hits_by_piece = {}
            if isinstance(pii_str, str):
                all_pii_pieces = self._split_units(pii_str)
                pii_score = 0
                for p in all_pii_pieces:
                    answer = self.fact_checker(information_piece=p, prompt=updated_query)
                    hit = int((answer.output.lower()).startswith("yes"))
                    pii_score += hit
                    pii_hits_by_piece[p.lower()] = hit
            else:
                pii_score = -1

            # L1/L2: reuse existing fact_checker results from pii_score computation.
            l1_pieces = self._split_units(l1_str)
            l2_pieces = self._split_units(l2_str)
            total_l1 = len(l1_pieces)
            total_l2 = len(l2_pieces)
            if pii_hits_by_piece:
                leaked_l1 = sum(pii_hits_by_piece.get(p.lower(), 0) for p in l1_pieces)
                leaked_l2 = sum(pii_hits_by_piece.get(p.lower(), 0) for p in l2_pieces)
            else:
                # Fallback for cases where pii_str is absent but level columns are present.
                leaked_l1 = 0
                leaked_l2 = 0
                for p in l1_pieces:
                    answer = self.fact_checker(information_piece=p, prompt=updated_query)
                    leaked_l1 += int((answer.output.lower()).startswith("yes"))
                for p in l2_pieces:
                    answer = self.fact_checker(information_piece=p, prompt=updated_query)
                    leaked_l2 += int((answer.output.lower()).startswith("yes"))

            # L3: one batch call returning hit list.
            l3_pieces = self._split_units(l3_str)
            total_l3 = len(l3_pieces)
            leaked_l3 = 0
            if total_l3 > 0:
                payload = json.dumps(l3_pieces, ensure_ascii=False)
                answer = self.l3_batch_checker(l3_terms=payload, prompt=updated_query)
                parsed = self._extract_json(getattr(answer, "output", ""))
                hits = self._normalize_hits(parsed.get("hits", []), expected_len=total_l3)
                leaked_l3 = sum(hits)

            ans = self.prompt_qual(prompt_input=updated_query)
            prompt_score = int((ans.output.lower()).startswith("yes"))
        else:
            pii_score = -1
            prompt_score = -1
        
        return dspy.Prediction(
            quality=judgment,
            leakage=pii_score,
            prompt=prompt_score,
            leaked_l1=leaked_l1,
            leaked_l2=leaked_l2,
            leaked_l3=leaked_l3,
            total_l1=total_l1,
            total_l2=total_l2,
            total_l3=total_l3,
        )


def process_scores(s1, s2):
    if s1 == s2:
        return 1
    return s1
