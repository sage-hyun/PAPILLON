import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMPAT_DIR = ROOT / "optimized_prompts" / "_compat"

### DO NOT ERASE COMMENTED JOBS ###
EVAL_JOBS = [
    # {
    #     "label": "llama31_8b_before",
    #     "model_name": "llama3.1:8b-instruct-fp16",
    #     "prompt_file": "NONE",
    #     "output_file_name": "eval_llama31_8b_PUPA_TNB_before.csv",
    # },
    # {
    #     "label": "llama31_8b_after",
    #     "model_name": "llama3.1:8b-instruct-fp16",
    #     "prompt_file": "optimized_prompts/llama_31_8b_instruct_prompt.json",
    #     "output_file_name": "eval_llama31_8b_PUPA_TNB_after.csv",
    # },
    # {
    #     "label": "gemma3_4b_before",
    #     "model_name": "gemma3:4b",
    #     "prompt_file": "NONE",
    #     "output_file_name": "eval_gemma3_4b_PUPA_TNB_before.csv",
    # },
    {
        "label": "gemma3_4b_after",
        "model_name": "gemma3:4b",
        "prompt_file": "optimized_prompts/gemma3_4b_prompt.json",
        "output_file_name": "eval_gemma3_4b_PUPA_TNB_after.csv",
    },
    # {
    #     "label": "gemma4_e4b_before",
    #     "model_name": "gemma4:e4b",
    #     "prompt_file": "NONE",
    #     "output_file_name": "eval_gemma4_e4b_PUPA_TNB_before.csv",
    # },
    # {
    #     "label": "gemma4_e4b_after",
    #     "model_name": "gemma4:e4b",
    #     "prompt_file": "optimized_prompts/gemma4_e4b_prompt.json",
    #     "output_file_name": "eval_gemma4_e4b_PUPA_TNB_after.csv",
    # },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run before/after PAPILLON evaluations for local Ollama models on PUPA-TNB."
    )
    parser.add_argument("--port", type=int, default=11434, help="Port for the local model server.")
    parser.add_argument(
        "--data_file",
        default="../pupa/PUPA_TNB.csv",
        help="PUPA-format CSV to evaluate on.",
    )
    parser.add_argument(
        "--openai_model",
        default="gpt-4o-mini",
        help="Cloud model used as proprietary/judge model.",
    )
    parser.add_argument(
        "--openrouter_base",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible API base URL.",
    )
    parser.add_argument(
        "--python_bin",
        default=sys.executable,
        help="Python executable to use for the child evaluations.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if one evaluation fails.",
    )
    return parser.parse_args()


def _extract_legacy_signature_fields(signature_state):
    if not isinstance(signature_state, dict):
        return None, None
    instructions = signature_state.get("instructions")
    fields = signature_state.get("fields") or []
    prefix = None
    if fields and isinstance(fields, list):
        last_field = fields[-1]
        if isinstance(last_field, dict):
            prefix = last_field.get("prefix")
    return instructions, prefix


def ensure_legacy_compatible_prompt(prompt_file):
    if prompt_file == "NONE":
        return prompt_file

    prompt_path = ROOT / prompt_file
    if not prompt_path.exists():
        return prompt_file

    with open(prompt_path) as f:
        state = json.load(f)

    if not isinstance(state, dict):
        return prompt_file

    changed = False
    compat_state = json.loads(json.dumps(state))
    for predictor_name, predictor_state in compat_state.items():
        if not isinstance(predictor_state, dict):
            continue

        signature = predictor_state.pop("signature", None)
        if signature is not None:
            instructions, prefix = _extract_legacy_signature_fields(signature)
            if instructions is not None:
                predictor_state["signature_instructions"] = instructions
            if prefix is not None:
                predictor_state["signature_prefix"] = prefix
            changed = True

        extended_signature = predictor_state.pop("extended_signature", None)
        if extended_signature is not None:
            instructions, prefix = _extract_legacy_signature_fields(extended_signature)
            if instructions is not None:
                predictor_state["extended_signature_instructions"] = instructions
            if prefix is not None:
                predictor_state["extended_signature_prefix"] = prefix
            changed = True

    if not changed:
        return prompt_file

    COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    compat_path = COMPAT_DIR / f"{prompt_path.stem}.legacy_compat.json"
    with open(compat_path, "w") as f:
        json.dump(compat_state, f, indent=2)

    rel_compat_path = compat_path.relative_to(ROOT)
    print(f"[INFO] Using legacy-compatible prompt shim: {rel_compat_path}")
    return str(rel_compat_path)


def run_one_job(job, args):
    prompt_file = ensure_legacy_compatible_prompt(job["prompt_file"])
    cmd = [
        args.python_bin,
        "evaluate_papillon.py",
        "--port",
        str(args.port),
        "--model_name",
        job["model_name"],
        "--data_file",
        args.data_file,
        "--prompt_file",
        prompt_file,
        "--output_file_name",
        job["output_file_name"],
        "--openai_model",
        args.openai_model,
        "--openrouter_base",
        args.openrouter_base,
    ]

    print("\n" + "=" * 80)
    print(f"[RUN] {job['label']}")
    print(" ".join(cmd))
    print("=" * 80)

    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - start

    print(f"[DONE] {job['label']} exit_code={proc.returncode} elapsed={elapsed/60:.1f} min")
    return {
        "label": job["label"],
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "output_file_name": job["output_file_name"],
    }


def main():
    args = parse_args()
    results = []
    overall_start = time.time()

    print(f"[INFO] Running {len(EVAL_JOBS)} evaluation jobs sequentially:")
    for job in EVAL_JOBS:
        print(f"  - {job['label']} -> {job['output_file_name']}")

    for job in EVAL_JOBS:
        result = run_one_job(job, args)
        results.append(result)
        if result["returncode"] != 0 and args.stop_on_error:
            break

    total_elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    for result in results:
        status = "OK" if result["returncode"] == 0 else f"FAIL({result['returncode']})"
        print(
            f"  {result['label']:20s} {status:10s} "
            f"{result['elapsed_sec']/60:7.1f} min  {result['output_file_name']}"
        )
    print(f"  Total wall time: {total_elapsed/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
