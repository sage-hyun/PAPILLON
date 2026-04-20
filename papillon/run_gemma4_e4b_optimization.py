import argparse
import json
import os
import time
from pathlib import Path

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

import run_dspy_optimization_llama as opt
from run_llama_dspy import PrivacyOnePrompter


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PAPILLON DSPy prompt optimization for gemma4:e4b via Ollama."
    )
    parser.add_argument(
        "--ollama_base",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--model_name",
        default="gemma4:e4b",
        help="Ollama model tag for the local trusted model.",
    )
    parser.add_argument(
        "--data_file",
        default=str(ROOT.parent / "pupa" / "PUPA_TNB.csv"),
        help="CSV containing PUPA-format data.",
    )
    parser.add_argument(
        "--prompt_output",
        default=str(ROOT / "optimized_prompts" / "gemma4_e4b_prompt.json"),
        help="Path to save the optimized DSPy prompt JSON.",
    )
    parser.add_argument(
        "--openai_model",
        default="gpt-4o-mini",
        help="Cloud model used as the untrusted model and judge.",
    )
    parser.add_argument(
        "--openrouter_base",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible API base URL.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8000,
        help="Max tokens for both local and cloud model calls.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Number of MIPROv2 optimization batches/trials.",
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=10,
        help="Number of MIPROv2 instruction candidates.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=10,
        help="Number of threads used during DSPy evaluation.",
    )
    parser.add_argument(
        "--train_limit",
        type=int,
        default=0,
        help="Optional cap on train examples. Use 0 to keep all available train examples.",
    )
    parser.add_argument(
        "--val_limit",
        type=int,
        default=0,
        help="Optional cap on validation examples. Use 0 to keep all available validation examples.",
    )
    parser.add_argument(
        "--skip_before_eval",
        action="store_true",
        help="Skip the pre-optimization evaluation pass.",
    )
    return parser.parse_args()


def build_local_lm(args):
    api_base = f"{args.ollama_base.rstrip('/')}/v1"
    return dspy.LM(
        f"openai/{args.model_name}",
        api_base=api_base,
        api_key="nokey",
        max_tokens=args.max_tokens,
    )


def compile_prompt(teleprompter, program, trainset, num_batches, eval_kwargs):
    common_kwargs = dict(
        trainset=trainset,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        eval_kwargs=eval_kwargs,
        requires_permission_to_run=False,
    )
    try:
        return teleprompter.compile(
            program,
            num_batches=num_batches,
            **common_kwargs,
        )
    except TypeError as exc:
        print(
            "[WARN] teleprompter.compile() rejected num_batches; "
            f"falling back to num_trials. ({type(exc).__name__}: {str(exc)[:200]})"
        )
        return teleprompter.compile(
            program,
            num_trials=num_batches,
            **common_kwargs,
        )


def main():
    run_start = time.perf_counter()
    args = parse_args()

    prompt_output = Path(args.prompt_output)
    if prompt_output.suffix != ".json":
        raise ValueError("--prompt_output must end with .json")
    prompt_output.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("[WARN] OPENAI_API_KEY is empty. OpenRouter calls will likely fail.")

    local_lm = build_local_lm(args)
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

    print(f"[INFO] Local model: {args.model_name} via {args.ollama_base.rstrip('/')}/v1")
    print(f"[INFO] Cloud/judge model: {args.openai_model} via {args.openrouter_base}")
    print(f"[INFO] Train examples: {len(train)}, validation examples: {len(val)}")
    print(
        "[INFO] "
        f"num_batches={args.num_batches}, num_candidates={args.num_candidates}, "
        f"num_threads={args.num_threads}"
    )

    zeroshot = PrivacyOnePrompter(local_lm, openai_lm)
    eval_scores = {}
    timing = {
        "model_name": args.model_name,
        "local_api_base": f"{args.ollama_base.rstrip('/')}/v1",
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
    teleprompter._checkpoint_path = str(prompt_output)

    compile_start = time.perf_counter()
    compiled_prompt = None
    try:
        compiled_prompt = compile_prompt(
            teleprompter,
            zeroshot,
            train,
            args.num_batches,
            eval_kwargs={
                "num_threads": args.num_threads,
                "display_progress": True,
                "display_table": 0,
            },
        )
    except Exception as exc:
        print(f"[WARN] Prompt optimization failed: {type(exc).__name__}: {str(exc)[:300]}")
    finally:
        timing["compile_seconds"] = time.perf_counter() - compile_start

    if compiled_prompt is not None:
        save_start = time.perf_counter()
        try:
            compiled_prompt.save(str(prompt_output))
            print(f"[INFO] Saved optimized prompt to {prompt_output}")
        except Exception as exc:
            print(f"[WARN] Failed to save optimized prompt: {type(exc).__name__}: {str(exc)[:300]}")
        finally:
            timing["save_prompt_seconds"] = time.perf_counter() - save_start

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

    score_output = prompt_output.with_name(f"{prompt_output.stem}_eval_scores.json")
    with open(score_output, "w") as f:
        json.dump(eval_scores, f, indent=2)
    print(f"[INFO] Saved eval scores to {score_output}")

    timing_output = prompt_output.with_name(f"{prompt_output.stem}_timing.json")
    with open(timing_output, "w") as f:
        json.dump(timing, f, indent=2)
    print(f"[INFO] Saved timing to {timing_output}")


if __name__ == "__main__":
    main()
