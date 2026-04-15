import dspy

### we use openrouter
def normalize_openai_model_name(model_name: str) -> str:
    return model_name if model_name.startswith("openrouter/openai/") else f"openrouter/openai/{model_name}"


def normalize_openai_compatible_model_name(model_name: str) -> str:
    if model_name.startswith("openai/openai/"):
        return model_name
    if model_name.startswith("openai/"):
        return f"openai/{model_name}"
    if "/" in model_name:
        return f"openai/{model_name}"
    return f"openai/openai/{model_name}"


def build_openai_lm(model_name: str, **kwargs):
    return dspy.LM(normalize_openai_model_name(model_name), **kwargs)


def build_openai_compatible_lm(model_name: str, **kwargs):
    return dspy.LM(normalize_openai_compatible_model_name(model_name), **kwargs)


def build_local_lm(model_name: str, host: str, port: int, api_key: str, **kwargs):
    return dspy.LM(
        f"openai/{model_name}",
        api_base=f"http://{host}:{port}/v1",
        api_key=api_key,
        **kwargs,
    )
