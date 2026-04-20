import dspy


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


def _robust_call(primary, fallback, *, max_retries=3, **kwargs):
    """Call a dspy module with retries; if all retries fail, call the fallback module.
    Returns the Prediction-like object (has .output), or None if everything fails.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return primary(**kwargs)
        except (ValueError, Exception) as e:
            last_exc = e
            # Continue to next retry
    # Primary failed max_retries times — try fallback (Predict without rationale)
    if fallback is not None:
        try:
            return fallback(**kwargs)
        except Exception as e:
            last_exc = e
    return None


class LLMJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.quality_judge_fallback = dspy.Predict(JudgeQuality)
        self.fact_checker = dspy.Predict(JudgeLeakage)
        self.prompt_qual = dspy.ChainOfThought(JudgePromptQual)
        self.prompt_qual_fallback = dspy.Predict(JudgePromptQual)

    def _safe_binary(self, answer, default=0):
        """Extract binary yes/no from answer.output, defaulting on failure."""
        if answer is None or not hasattr(answer, "output") or answer.output is None:
            return default
        try:
            return int(str(answer.output).lower().strip().startswith("yes"))
        except Exception:
            return default

    def forward(self, user_query, og_resp, new_resp=None, updated_query=None, pii_str=None):
        if new_resp:
            ans1 = _robust_call(
                self.quality_judge, self.quality_judge_fallback,
                user_query=user_query, response_A=new_resp, response_B=og_resp,
            )
            judgment_1 = self._safe_binary(ans1, default=0)
            ans2 = _robust_call(
                self.quality_judge, self.quality_judge_fallback,
                user_query=user_query, response_A=og_resp, response_B=new_resp,
            )
            judgment_2 = self._safe_binary(ans2, default=0)
            judgment = process_scores(judgment_1, judgment_2)
        else:
            judgment = -1
        if updated_query:
            if isinstance(pii_str, str):
                # all_pii_pieces = set(pii_str.split("||"))
                all_pii_pieces = [piece.strip() for piece in pii_str.split("||") if piece and piece.strip()] # main 브랜치에서 수정된 코드
                pii_score = 0
                for p in all_pii_pieces:
                    # fact_checker is already Predict; retry only
                    ans = _robust_call(
                        self.fact_checker, None,
                        information_piece=p, prompt=updated_query,
                    )
                    pii_score += self._safe_binary(ans, default=0)
            else:
                pii_score = -1
            ans = _robust_call(
                self.prompt_qual, self.prompt_qual_fallback,
                prompt_input=updated_query,
            )
            prompt_score = self._safe_binary(ans, default=1)  # assume valid prompt if judge fails
        else:
            pii_score = -1
            prompt_score = -1
        
        return dspy.Prediction(
            quality=judgment,
            leakage=pii_score,
            prompt=prompt_score
        )


def process_scores(s1, s2):
    if s1 == s2:
        return 1
    return s1
