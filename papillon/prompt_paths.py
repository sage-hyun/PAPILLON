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
    if not prompt_file or not os.path.exists(prompt_file):
        print(f"Prompt file not found: {prompt_file}")
        return

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return

    pipeline_params = dict(pipeline.named_parameters())
    
    loaded_keys = []
    skipped_keys = []

    for name, param in pipeline_params.items():
        if name in payload:
            param.load_state(payload[name], use_legacy_loading=True)
            loaded_keys.append(name)
        else:
            skipped_keys.append(name)

    if loaded_keys:
        print(f"Partially loaded from {os.path.basename(prompt_file)}: {loaded_keys}")
    if skipped_keys:
        print(f"Skipped (not in JSON): {skipped_keys}")
