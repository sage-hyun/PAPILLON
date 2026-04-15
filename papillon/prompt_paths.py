import json
import os
import tempfile


def parse_model_prompt(model_name):
    lowered_model_name = model_name.lower()
    if "llama" in lowered_model_name:
        if "1b-instruct" in lowered_model_name:
            return "optimized_prompts/llama_32_1b_instruct_prompt.json"
        if "3b-instruct" in lowered_model_name:
            return "optimized_prompts/llama_32_3b_instruct_prompt.json"
        if "8b-instruct" in lowered_model_name:
            if "3.1" in lowered_model_name:
                return "optimized_prompts/llama_31_8b_instruct_prompt.json"
            return "optimized_prompts/llama_3_8b_instruct_prompt.json"
    elif "mistral" in lowered_model_name:
        if "small" in lowered_model_name:
            return "optimized_prompts/mistral_small_prompt.json"
        if "7b" in lowered_model_name:
            return "optimized_prompts/mistral_7b_instruct_prompt.json"
    elif "gemma" in lowered_model_name:
        if "e4b" in lowered_model_name:
            return "optimized_prompts/gemma_4_e4b_prompt.json"
    elif "qwen" in lowered_model_name:
        if "9b" in lowered_model_name:
            return "optimized_prompts/qwen_3.5_9b_prompt.json"
    raise NotImplementedError("Model currently not supported! You will have to optimize it yourself!")


def load_prompt_with_pipeline_compat(pipeline, prompt_file):
    """Load only prompt blocks that exist on the current pipeline object."""
    if not prompt_file:
        return

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        pipeline.load(prompt_file)
        return

    if not isinstance(payload, dict):
        pipeline.load(prompt_file)
        return

    filtered_payload = {k: v for k, v in payload.items() if hasattr(pipeline, k)}
    if not filtered_payload:
        return

    if len(filtered_payload) == len(payload):
        pipeline.load(prompt_file)
        return

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(filtered_payload, tf, ensure_ascii=False, indent=2)
            temp_path = tf.name
        pipeline.load(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
