import argparse
import json
import os
import time

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

import run_dspy_optimization_llama as opt
from ollama_nothink_lm import OllamaNoThinkLM
from run_llama_dspy import PrivacyOnePrompter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PAPILLON DSPy prompt optimization with Qwen through OllamaNoThinkLM."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Ollama model tag, e.g. qwen3:8b or qwen3:4b.",
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="CSV containing PUPA-format data.",
    )
    parser.add_argument(
        "--prompt_output",
        required=True,
        help="Path to save the optimized DSPy prompt JSON.",
    )
    parser.add_argument(
        "--ollama_base",
        default="http://127.0.0.1:11434",
        help="Ollama native API base URL. Use /api/chat capable endpoint, not /v1.",
    )
    parser.add_argument(
        "--openai_model",
        default="gpt-4o-mini",
        help="Cloud model used as untrusted model and judge.",
    )
    parser.add_argument(
        "--openrouter_base",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible API base URL.",
    )
    parser.add_argument("--max_tokens", type=int, default=4000)
    parser.add_argument("--num_batches", type=int, default=2)
    parser.add_argument("--num_candidates", type=int, default=2)
    parser.add_argument("--train_limit", type=int, default=10)
    parser.add_argument("--val_limit", type=int, default=5)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument(
        "--skip_before_eval",
        action="store_true",
        help="Skip the pre-optimization evaluation pass.",
    )
    return parser.parse_args()


def main():
    run_start = time.perf_counter()
    args = parse_args()

    if not args.prompt_output.endswith(".json"):
        raise ValueError("--prompt_output must end with .json")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[WARN] OPENAI_API_KEY is empty. OpenRouter calls will likely fail.")

    local_lm = OllamaNoThinkLM(
        args.model_name,
        ollama_base=args.ollama_base,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=local_lm)

    openai_lm = dspy.LM(
        f"openai/{args.openai_model}",
        api_base=args.openrouter_base,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )
    opt.openai_lm_gpt4o = openai_lm

    train, val, _ = opt.synthesize_tvt(args.data_file)
    if args.train_limit > 0:
        train = train[: args.train_limit]
    if args.val_limit > 0:
        val = val[: args.val_limit]

    print(f"[INFO] Local model: {args.model_name} via {args.ollama_base}/api/chat")
    print(f"[INFO] Cloud/judge model: {args.openai_model} via {args.openrouter_base}")
    print(f"[INFO] Train examples: {len(train)}, validation examples: {len(val)}")
    print(f"[INFO] num_batches={args.num_batches}, num_candidates={args.num_candidates}")

    zeroshot = PrivacyOnePrompter(local_lm, openai_lm)
    eval_scores = {}
    timing = {
        "model_name": args.model_name,
        "openai_model": args.openai_model,
        "train_examples": len(train),
        "validation_examples": len(val),
        "num_batches": args.num_batches,
        "num_candidates": args.num_candidates,
        "num_threads": args.num_threads,
        "max_tokens": args.max_tokens,
    }

    evaluate = Evaluate(
        metric=opt.metric,
        devset=val,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=2,
        max_errors=100,
    )

    if not args.skip_before_eval:
        before_start = time.perf_counter()
        try:
            before_score = evaluate(zeroshot)
            eval_scores["before_optimization"] = before_score
            print(f"[INFO] Before optimization score: {before_score}")
        except Exception as exc:
            print(f"[WARN] Before optimization failed: {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            timing["before_optimization_seconds"] = time.perf_counter() - before_start

    teleprompter = MIPROv2(
        prompt_model=openai_lm,
        task_model=local_lm,
        metric=opt.metric,
        num_candidates=args.num_candidates,
        init_temperature=1.0,
    )

    compile_start = time.perf_counter()
    compiled_prompt = teleprompter.compile(
        zeroshot,
        trainset=train,
        num_batches=args.num_batches,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        eval_kwargs={
            "num_threads": args.num_threads,
            "display_progress": True,
            "display_table": 0,
        },
        requires_permission_to_run=False,
    )

    timing["compile_seconds"] = time.perf_counter() - compile_start

    save_start = time.perf_counter()
    compiled_prompt.save(args.prompt_output)
    timing["save_prompt_seconds"] = time.perf_counter() - save_start
    print(f"[INFO] Saved optimized prompt to {args.prompt_output}")

    after_start = time.perf_counter()
    try:
        after_score = evaluate(compiled_prompt)
        eval_scores["after_optimization"] = after_score
        print(f"[INFO] After optimization score: {after_score}")
    except Exception as exc:
        print(f"[WARN] After optimization evaluation failed: {type(exc).__name__}: {str(exc)[:300]}")
    finally:
        timing["after_optimization_seconds"] = time.perf_counter() - after_start

    timing["total_seconds"] = time.perf_counter() - run_start

    score_output = args.prompt_output.replace(".json", "_eval_scores.json")
    with open(score_output, "w") as f:
        json.dump(eval_scores, f, indent=2)
    print(f"[INFO] Saved eval scores to {score_output}")

    timing_output = args.prompt_output.replace(".json", "_timing.json")
    with open(timing_output, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"[INFO] Saved timing to {timing_output}")


if __name__ == "__main__":
    main()
