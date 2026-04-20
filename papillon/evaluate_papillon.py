import json
import os
import time
from argparse import ArgumentParser

import dspy
import litellm
import pandas
import tqdm
from dspy import Example

import run_dspy_optimization_llama
from run_dspy_optimization_llama import metric_finegrained
from run_llama_dspy import PrivacyOnePrompter


MAX_PIPELINE_RETRIES = 3
MAX_METRIC_RETRIES = 3


def parse_model_prompt(model_name):
    model_name = model_name.lower()
    if "llama" in model_name:
        if "1b-instruct" in model_name:
            return "optimized_prompts/llama_32_1b_instruct_prompt.json"
        if "3b-instruct" in model_name:
            return "optimized_prompts/llama_32_3b_instruct_prompt.json"
        if "8b-instruct" in model_name:
            if "3.1" in model_name:
                return "optimized_prompts/llama_31_8b_instruct_prompt.json"
            return "optimized_prompts/llama_3_8b_instruct_prompt.json"
    if "mistral" in model_name:
        if "small" in model_name:
            return "optimized_prompts/mistral_small_prompt.json"
        if "7b" in model_name:
            return "optimized_prompts/mistral_7b_instruct_prompt.json"
    if "gemma3" in model_name and "4b" in model_name:
        return "optimized_prompts/gemma3_4b_prompt.json"
    if "qwen" in model_name:
        return "optimized_prompts/qwen3.5_9b_nothink.json"
    raise NotImplementedError("Model currently not supported! You will have to optimize it yourself!")


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


def _write_outputs(
    output_file_name,
    qual_scores,
    leak_scores,
    all_user_queries,
    all_redacted_queries,
    all_final_outputs,
    all_cloud_responses,
    timing_records,
):
    result_df = pandas.DataFrame()
    result_df["quals"] = qual_scores
    result_df["leaks"] = leak_scores
    result_df["queries"] = all_user_queries
    result_df["redacted_query"] = all_redacted_queries
    result_df["final_output"] = all_final_outputs
    result_df["cloud_response"] = all_cloud_responses
    result_df.to_csv(output_file_name, index=False)

    timing_df = pandas.DataFrame(timing_records)
    timing_output = output_file_name.replace(".csv", "_timing.csv")
    timing_df.to_csv(timing_output, index=False)


def _build_summary(
    args,
    qual_scores,
    leak_scores,
    timing_records,
    priv_prompt_fail_count,
    metric_fail_count,
):
    summary = {
        "model_name": args.model_name,
        "openai_model": args.openai_model,
        "data_file": args.data_file,
        "prompt_file": args.prompt_file,
        "total_rows": len(pandas.read_csv(args.data_file)),
        "evaluated_rows": len(qual_scores),
        "priv_prompt_failures": priv_prompt_fail_count,
        "metric_finegrained_failures": metric_fail_count,
        "avg_quality": float(sum(qual_scores) / len(qual_scores)) if qual_scores else None,
        "avg_leakage": float(sum(leak_scores) / len(leak_scores)) if leak_scores else None,
    }

    timing_df = pandas.DataFrame(timing_records)
    if not timing_df.empty:
        total_series = timing_df["time_total_row_sec"].dropna()
        stage1_series = timing_df["stage1_prompt_creation_sec"].dropna()
        stage2a_series = timing_df["stage2a_cloud_response_sec"].dropna()
        stage2b_series = timing_df["stage2b_local_aggregation_sec"].dropna()
        judge_series = timing_df["judge_evaluation_sec"].dropna()

        mean_total = float(total_series.mean()) if not total_series.empty else None
        mean_stage1 = float(stage1_series.mean()) if not stage1_series.empty else None
        mean_stage2a = float(stage2a_series.mean()) if not stage2a_series.empty else None
        mean_stage2b = float(stage2b_series.mean()) if not stage2b_series.empty else None
        mean_judge = float(judge_series.mean()) if not judge_series.empty else None

        summary["timing"] = {
            "mean_per_row_sec": round(mean_total, 2) if mean_total is not None else None,
            "median_per_row_sec": round(float(total_series.median()), 2) if not total_series.empty else None,
            "std_per_row_sec": round(float(total_series.std()), 2) if len(total_series) > 1 else 0.0,
            "min_per_row_sec": round(float(total_series.min()), 2) if not total_series.empty else None,
            "max_per_row_sec": round(float(total_series.max()), 2) if not total_series.empty else None,
            "stage1_prompt_creation_mean_sec": round(mean_stage1, 3) if mean_stage1 is not None else None,
            "stage2a_cloud_response_mean_sec": round(mean_stage2a, 3) if mean_stage2a is not None else None,
            "stage2b_local_aggregation_mean_sec": round(mean_stage2b, 3) if mean_stage2b is not None else None,
            "judge_evaluation_mean_sec": round(mean_judge, 3) if mean_judge is not None else None,
            "stage1_pct": round(100.0 * mean_stage1 / mean_total, 2) if mean_total and mean_stage1 is not None else None,
            "stage2a_pct": round(100.0 * mean_stage2a / mean_total, 2) if mean_total and mean_stage2a is not None else None,
            "stage2b_pct": round(100.0 * mean_stage2b / mean_total, 2) if mean_total and mean_stage2b is not None else None,
            "judge_pct": round(100.0 * mean_judge / mean_total, 2) if mean_total and mean_judge is not None else None,
        }

    summary_output = args.output_file_name.replace(".csv", "_summary.json")
    with open(summary_output, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model")
    parser.add_argument("--data_file", type=str, help="The data file containing PUPA-style queries and target responses")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openrouter_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--prompt_file", type=str, default="ORIGINAL", help="The DSPy-optimized prompt, stored as a json file")
    parser.add_argument("--model_name", type=str, help="The Huggingface identifier / name for your local LM")
    parser.add_argument("--output_file_name", type=str, default="output.csv")
    args = parser.parse_args()

    data_file = pandas.read_csv(args.data_file)
    qual_scores = []
    leak_scores = []
    all_user_queries = []
    all_redacted_queries = []
    all_final_outputs = []
    all_cloud_responses = []
    timing_records = []
    priv_prompt_fail_count = 0
    metric_fail_count = 0

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
    run_dspy_optimization_llama.openai_lm_gpt4o = openai_lm

    priv_prompt = PrivacyOnePrompter(local_lm, openai_lm)

    if args.prompt_file == "ORIGINAL":
        args.prompt_file = parse_model_prompt(args.model_name)

    _load_prompt_if_needed(priv_prompt, args.prompt_file)

    for i, row in tqdm.tqdm(data_file.iterrows(), total=len(data_file)):
        if row["target_response"] is None or not isinstance(row["target_response"], str) or not isinstance(row["pii_units"], str):
            continue

        row_start = time.time()
        gold = Example({
            "target_response": row["target_response"],
            "user_query": row["user_query"],
            "pii_str": row["pii_units"],
        }).with_inputs("user_query")

        pred = None
        last_pipeline_exc = None
        for _ in range(MAX_PIPELINE_RETRIES):
            try:
                pred = priv_prompt(row["user_query"])
                break
            except (litellm.exceptions.BadRequestError, ValueError, Exception) as exc:
                last_pipeline_exc = exc
                pred = None

        if pred is None:
            priv_prompt_fail_count += 1
            timing_records.append({
                "row_idx": i,
                "stage1_prompt_creation_sec": None,
                "stage2a_cloud_response_sec": None,
                "stage2b_local_aggregation_sec": None,
                "judge_evaluation_sec": None,
                "time_total_row_sec": time.time() - row_start,
                "quality": None,
                "leakage": None,
                "status": "pipeline_failed",
                "error": f"{type(last_pipeline_exc).__name__}: {str(last_pipeline_exc)[:300]}" if last_pipeline_exc else "",
            })
            continue

        judge_start = time.time()
        qual, leak = -1, -1
        last_metric_exc = None
        for _ in range(MAX_METRIC_RETRIES):
            try:
                qual, leak = metric_finegrained(gold, pred)
                break
            except (ValueError, Exception) as exc:
                last_metric_exc = exc
                qual, leak = -1, -1
        judge_elapsed = time.time() - judge_start
        row_elapsed = time.time() - row_start

        pipeline_timing = getattr(pred, "timing", {}) or {}
        timing_record = {
            "row_idx": i,
            "stage1_prompt_creation_sec": pipeline_timing.get("stage1_prompt_creation"),
            "stage2a_cloud_response_sec": pipeline_timing.get("stage2a_cloud_response"),
            "stage2b_local_aggregation_sec": pipeline_timing.get("stage2b_local_aggregation"),
            "judge_evaluation_sec": judge_elapsed,
            "time_total_row_sec": row_elapsed,
            "quality": qual if qual != -1 else None,
            "leakage": leak if leak != -1 else None,
            "status": "ok",
            "error": "",
        }

        if qual != -1 and leak != -1:
            qual_scores.append(qual)
            leak_scores.append(leak)
            all_user_queries.append(row["user_query"])
            all_redacted_queries.append(getattr(pred, "prompt", ""))
            all_final_outputs.append(getattr(pred, "output", ""))
            all_cloud_responses.append(getattr(pred, "gptResponse", ""))
            timing_records.append(timing_record)
            _write_outputs(
                args.output_file_name,
                qual_scores,
                leak_scores,
                all_user_queries,
                all_redacted_queries,
                all_final_outputs,
                all_cloud_responses,
                timing_records,
            )
        else:
            metric_fail_count += 1
            timing_record["status"] = "metric_failed"
            timing_record["error"] = f"{type(last_metric_exc).__name__}: {str(last_metric_exc)[:300]}" if last_metric_exc else ""
            timing_records.append(timing_record)

    _write_outputs(
        args.output_file_name,
        qual_scores,
        leak_scores,
        all_user_queries,
        all_redacted_queries,
        all_final_outputs,
        all_cloud_responses,
        timing_records,
    )

    _build_summary(
        args,
        qual_scores,
        leak_scores,
        timing_records,
        priv_prompt_fail_count,
        metric_fail_count,
    )

    print("AVERAGE QUALITY SCORE", sum(qual_scores) / len(qual_scores) if qual_scores else None)
    print("AVERAGE LEAKAGE SCORE", sum(leak_scores) / len(leak_scores) if leak_scores else None)
    print("==============")
