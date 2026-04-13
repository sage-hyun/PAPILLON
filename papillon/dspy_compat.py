import dspy


def normalize_openai_model_name(model_name: str) -> str:
    return model_name if "/" in model_name else f"openai/{model_name}"


def build_openai_lm(model_name: str, **kwargs):
    return dspy.LM(normalize_openai_model_name(model_name), **kwargs)
