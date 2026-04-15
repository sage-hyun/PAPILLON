from argparse import ArgumentParser
import os
import dspy
from dspy_compat import build_local_lm, build_openai_lm
from pipeline_factory import build_pipeline
from prompt_paths import parse_model_prompt
from run_dspy_optimization_llama import str_to_bool

LOCAL_LM_API_KEY = "local-openai-compatible-key"
LOCAL_LM_API_HOST = os.getenv("PAPILLON_LOCAL_LM_HOST", "127.0.0.1")

### ignore warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--prompt_file", type=str, default="ORIGINAL", help="The DSPy-optimized prompt, stored as a json file")
    parser.add_argument("--model_name", type=str, help="The Huggingface identifier / name for your local LM")
    parser.add_argument("--pipeline", type=str, choices=["legacy", "structured_v1"], default="legacy")
    parser.add_argument("--allow_direct_bypass", type=str_to_bool, default=True)
    parser.add_argument("--privacy_filter", type=str, default="regex_presidio")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    
    args = parser.parse_args()

    if args.prompt_file == "ORIGINAL":
        args.prompt_file = parse_model_prompt(args.model_name) if args.pipeline == "legacy" else None
    
    local_lm = build_local_lm(
        args.model_name,
        host=LOCAL_LM_API_HOST,
        port=args.port,
        api_key=LOCAL_LM_API_KEY,
        max_tokens=4000,
    )
    dspy.configure(lm=local_lm)

    openai_lm = build_openai_lm(args.openai_model, max_tokens=4000)

    priv_prompt = build_pipeline(
        pipeline_name=args.pipeline,
        untrusted_model=openai_lm,
        allow_direct_bypass=args.allow_direct_bypass,
        privacy_filter_name=args.privacy_filter,
        pii_score_threshold=args.pii_score_threshold,
    )

    if args.prompt_file:
        priv_prompt.load(args.prompt_file)

    while True:
        user_query = input("Your Query > ")
        pred = priv_prompt(user_query)
        print("ROUTE > ", getattr(pred, "route", "legacy"))
        print("PAPILLON PROMPT > ", getattr(pred, "cloud_prompt", pred.prompt))
        print("PAPILLON OUTPUT > ", pred.output)
