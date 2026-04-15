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
    raise NotImplementedError("Model currently not supported! You will have to optimize it yourself!")
