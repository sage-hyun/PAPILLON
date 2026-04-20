import dspy
import time

import os; os.environ['LITELLM_LOG'] = 'ERROR'


class CreateOnePrompt(dspy.Signature):
    """
    You are a helpful assistant that is very mindful of user privacy. You have access to a powerful large language model that you can query. Given a user request, create a prompt for your large language model that preserves user privacy, so that this model can help you complete the user request. Provide the prompt directly without any preamble. DO NOT COMPLETE THE USER QUERY, ONLY GENERATE A PROMPT.
    """
    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    createdPrompt = dspy.OutputField()

class InfoAggregator(dspy.Signature):
    """
    You are a helpful assistant. Respond to queries from the user.
    """

    userQuery = dspy.InputField(desc="The user's request to be fulfilled.")
    modelExampleResponses = dspy.InputField(desc="You have the following information from a better language model responding to related query or queries. Complete the user query by referencing this information. Only you have access to this information.", format=lambda s: f'======\n\n{s.strip()}\n\n======')
    finalOutput = dspy.OutputField()


def _retry_call(fn, *, max_retries=3, **kwargs):
    """Call dspy module with retries on ValueError (DSPy parsing errors).
    Returns (result, last_exception). result is None if all retries failed."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(**kwargs), None
        except (ValueError, Exception) as e:
            last_exc = e
    return None, last_exc


class PrivacyOnePrompter(dspy.Module):
    def __init__(self, trusted_model, untrusted_model):
        super().__init__()
        self.prompt_creater = dspy.ChainOfThought(CreateOnePrompt)
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.trusted_model = trusted_model
        dspy.configure(lm=self.trusted_model)
        self.untrusted_model = untrusted_model
        # Lazily-built fallback (not a tracked dspy parameter so `load()` of old states still works)
        self._prompt_creater_fallback = None

    def _get_prompt_creater_fallback(self):
        # Store via __dict__ to avoid dspy.Module registering this as a tracked parameter.
        if self._prompt_creater_fallback is None:
            self.__dict__["_prompt_creater_fallback"] = dspy.Predict(CreateOnePrompt)
        return self._prompt_creater_fallback

    def forward(self, user_query):
        timing = {}

        # Stage 1: Local trusted model creates privacy-preserving prompt (with retries + fallback)
        t0 = time.time()
        prompt, exc = _retry_call(self.prompt_creater, max_retries=3, userQuery=user_query)
        if prompt is None:
            # Fall back to Predict (no rationale requirement)
            prompt, exc = _retry_call(self._get_prompt_creater_fallback(), max_retries=2, userQuery=user_query)
        timing["stage1_prompt_creation"] = time.time() - t0
        if prompt is None or not getattr(prompt, "createdPrompt", None):
            return dspy.Prediction(
                prompt="", output="", gptResponse="", timing=timing
            )

        # Stage 2a: Untrusted cloud model generates response (with retries)
        t1 = time.time()
        response = ""
        last_exc = None
        for attempt in range(3):
            try:
                if isinstance(self.untrusted_model, dspy.LM):
                    response = self.untrusted_model(messages=[{"role": "user", "content": prompt.createdPrompt}])[0]
                else:
                    response = self.untrusted_model(prompt.createdPrompt)[0]
                break
            except (ValueError, Exception) as e:
                last_exc = e
                response = ""
        timing["stage2a_cloud_response"] = time.time() - t1
        if not response:
            return dspy.Prediction(
                prompt=prompt.createdPrompt, output="", gptResponse="", timing=timing
            )

        # Stage 2b: Local trusted model aggregates final response (with retries)
        t2 = time.time()
        final_output, exc = _retry_call(
            self.info_aggregator, max_retries=3,
            userQuery=user_query, modelExampleResponses=response,
        )
        timing["stage2b_local_aggregation"] = time.time() - t2
        timing["total_pipeline"] = time.time() - t0
        if final_output is None or not getattr(final_output, "finalOutput", None):
            # Even if aggregator fails, we still return cloud response as output fallback
            return dspy.Prediction(
                prompt=prompt.createdPrompt, output=response, gptResponse=response, timing=timing
            )

        return dspy.Prediction(
            prompt=prompt.createdPrompt,
            output=final_output.finalOutput,
            gptResponse=response,
            timing=timing
        )
