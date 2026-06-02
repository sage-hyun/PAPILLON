import dspy
import time

import os; os.environ['LITELLM_LOG'] = 'ERROR'

try:
    from .structured_pipeline import empty_latency_breakdown
except ImportError:
    from structured_pipeline import empty_latency_breakdown


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
    modelExampleResponses = dspy.InputField(desc="Information from a more powerful language model responding to related queries. Complete the user query by referencing this information. Only you have access to this information.")
    finalOutput = dspy.OutputField()


class PAPILLON(dspy.Module):
    def __init__(self, untrusted_model):
        self.prompt_creater = dspy.ChainOfThought(CreateOnePrompt)
        self.info_aggregator = dspy.Predict(InfoAggregator)
        self.untrusted_model = untrusted_model

    def forward(self, user_query):
        start_time = time.perf_counter()
        breakdown = empty_latency_breakdown()
        try:
            prompt_start = time.perf_counter()
            prompt_prediction = self.prompt_creater(userQuery=user_query)
            prompt = getattr(prompt_prediction, "createdPrompt", "") or ""
            breakdown["prompt_creator_ms"] = (time.perf_counter() - prompt_start) * 1000

            cloud_start = time.perf_counter()
            response = self.untrusted_model(prompt)[0]
            breakdown["cloud_ms"] = (time.perf_counter() - cloud_start) * 1000

            agg_start = time.perf_counter()
            output_prediction = self.info_aggregator(userQuery=user_query, modelExampleResponses=response)
            final_output = getattr(output_prediction, "finalOutput", "") or ""
            breakdown["aggregator_ms"] = (time.perf_counter() - agg_start) * 1000

            breakdown["total_ms"] = (time.perf_counter() - start_time) * 1000
        except Exception as e:
            print(f"{e}")
            return dspy.Prediction(
                prompt="",
                output="",
                gptResponse="",
                total_ms=0.0,
                latency_breakdown=empty_latency_breakdown(),
            )

        return dspy.Prediction(
            prompt=prompt,
            output=final_output,
            gptResponse=response,
            total_ms=breakdown["total_ms"],
            latency_breakdown=breakdown,
        )
