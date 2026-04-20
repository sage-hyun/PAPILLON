"""
Custom dspy.LM wrapper for Ollama models that need thinking disabled (e.g., Qwen3.5).
Calls Ollama's native /api/chat endpoint with think=false, then wraps the response
so that dspy can use it like a normal LM.
"""
import dspy
import requests
import re


class OllamaNoThinkLM(dspy.LM):
    """A dspy.LM that calls Ollama's native API with think=false to disable reasoning mode."""

    def __init__(self, model_name, ollama_base="http://0.0.0.0:11434", **kwargs):
        self.ollama_model = model_name
        self.ollama_base = ollama_base.rstrip("/")
        # Initialize parent with a dummy openai-compatible config
        super().__init__(
            f"openai/{model_name}",
            api_base=f"{self.ollama_base}/v1",
            api_key="nokey",
            **kwargs
        )

    def __call__(self, prompt=None, messages=None, **kwargs):
        if messages is None and prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if "max_tokens" in kwargs:
            payload["options"] = {"num_predict": kwargs["max_tokens"]}
        elif hasattr(self, 'kwargs') and self.kwargs.get("max_tokens"):
            payload["options"] = {"num_predict": self.kwargs["max_tokens"]}

        resp = requests.post(f"{self.ollama_base}/api/chat", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        # Strip any residual <think> tags just in case
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return [content]
