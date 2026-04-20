import dspy
import os
from argparse import ArgumentParser

from evaluate_papillon import parse_model_prompt
from run_llama_dspy import PrivacyOnePrompter


def _load_prompt_if_needed(priv_prompt, prompt_file):
    if prompt_file == "NONE" or not prompt_file or not os.path.exists(prompt_file):
        print("[INFO] No pre-optimized prompt. Zero-shot baseline.")
        return
    try:
        priv_prompt.load(prompt_file)
    except ValueError as e:
        if "prior to v2.5.3" in str(e) or "use_legacy_loading" in str(e):
            print("[INFO] Legacy DSPy prompt format detected; loading with use_legacy_loading=True")
            priv_prompt.load(prompt_file, use_legacy_loading=True)
        else:
            raise

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openrouter_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--prompt_file", type=str, default="ORIGINAL", help="The DSPy-optimized prompt, stored as a json file")
    parser.add_argument("--model_name", type=str, help="The Huggingface identifier / name for your local LM")
    
    args = parser.parse_args()

    if args.prompt_file == "ORIGINAL":
        args.prompt_file = parse_model_prompt(args.model_name)
    
    local_lm = dspy.LM(
        f"openai/{args.model_name}",
        api_base=f"http://127.0.0.1:{args.port}/v1",
        api_key="",
        max_tokens=8000,
        cache=False,
    )
    dspy.configure(lm=local_lm)

    openai_lm = dspy.LM(
        f"openai/{args.openai_model}",
        api_base=args.openrouter_base,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        max_tokens=8000,
        cache=False,
    )

    priv_prompt = PrivacyOnePrompter(local_lm, openai_lm)
    _load_prompt_if_needed(priv_prompt, args.prompt_file)

    while True:
        user_query = input("Your Query > ")
        pred = priv_prompt(user_query)
        print("PAPILLON PROMPT > ", pred.prompt)
        print("PAPILLON OUTPUT > ", pred.output)
