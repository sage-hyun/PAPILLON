import json
from argparse import ArgumentParser
import os

import dspy
import litellm
import pandas
import tqdm
from dspy import Example

from dspy_compat import build_local_lm, build_openai_lm
from pipeline_factory import build_pipeline
from prompt_paths import parse_model_prompt, load_prompt_with_pipeline_compat
from run_dspy_optimization_llama import metric_finegrained, str_to_bool


LOCAL_LM_API_KEY = "local-openai-compatible-key"
LOCAL_LM_API_HOST = os.getenv("PAPILLON_LOCAL_LM_HOST", "127.0.0.1")


def safe_average(values):
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model")
    parser.add_argument("--data_file", type=str, help="The data file containing PUPA-style queries and target responses")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--prompt_file", type=str, default="ORIGINAL", help="The DSPy-optimized prompt, stored as a json file")
    parser.add_argument("--model_name", type=str, help="The Huggingface identifier / name for your local LM")
    parser.add_argument("--output_file_name", type=str, default="output.csv")
    parser.add_argument("--pipeline", type=str, choices=["legacy", "structured_v1"], default="legacy")
    parser.add_argument("--allow_direct_bypass", type=str_to_bool, default=True)
    parser.add_argument("--privacy_filter", type=str, default="regex_presidio")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_frame = pandas.read_csv(args.data_file)
    local_lm = build_local_lm(
        args.model_name,
        host=LOCAL_LM_API_HOST,
        port=args.port,
        api_key=LOCAL_LM_API_KEY,
        max_tokens=4000,
    )
    dspy.configure(lm=local_lm)
    openai_lm = build_openai_lm(args.openai_model, max_tokens=4000)

    pipeline = build_pipeline(
        pipeline_name=args.pipeline,
        untrusted_model=openai_lm,
        allow_direct_bypass=args.allow_direct_bypass,
        privacy_filter_name=args.privacy_filter,
        pii_score_threshold=args.pii_score_threshold,
    )

    resolved_prompt_file = args.prompt_file
    if resolved_prompt_file == "ORIGINAL":
        resolved_prompt_file = parse_model_prompt(args.model_name)

    if resolved_prompt_file:
        load_prompt_with_pipeline_compat(pipeline, resolved_prompt_file)

    rows = []
    qual_scores = []
    leak_scores = []
    wpl_scores = []   
    etc_scores = []   
    err_scores = []     
    latency_scores = []

    for _, row in tqdm.tqdm(data_frame.iterrows(), total=len(data_frame)):
        gold = Example(
            {
                "target_response": row["target_response"],
                "user_query": row["user_query"],
                "pii_str": row["pii_units"],
            }
        ).with_inputs("user_query")
        try:
            pred = pipeline(row["user_query"])
        except litellm.exceptions.BadRequestError:
            continue

        if not isinstance(row["target_response"], str):
            continue

        l1_str = row["l1_units"] if "l1_units" in row and isinstance(row["l1_units"], str) else ""
        l2_str = row["l2_units"] if "l2_units" in row and isinstance(row["l2_units"], str) else ""
        l3_str = row["l3_terms"] if "l3_terms" in row and isinstance(row["l3_terms"], str) else ""
        metrics = metric_finegrained(gold, pred, openai_lm, l1_str=l1_str, l2_str=l2_str, l3_str=l3_str)
        if metrics["quality"] != -1 and metrics["leakage"] != -1:
            qual_scores.append(metrics["quality"])
            leak_scores.append(metrics["leakage"])
            wpl_scores.append(metrics.get("weighted_leakage", metrics["leakage"]))
            etc_scores.append(metrics["exposed_token_count"])
            err_scores.append(metrics["entity_retention_rate"])
            latency_scores.append(metrics["latency"])

        rows.append(
            {
                "quals": metrics["quality"],
                "leaks": metrics["leakage"],
                "weighted_leakage": metrics.get("weighted_leakage", metrics["leakage"]),
                #"leakage_l1_ratio": metrics.get("leakage_l1_ratio", 0.0),
                #"leakage_l2_ratio": metrics.get("leakage_l2_ratio", 0.0),
                #"leakage_l3_ratio": metrics.get("leakage_l3_ratio", 0.0),
                #"leaked_l1": metrics.get("leaked_l1", 0),
                #"total_l1": metrics.get("total_l1", 0),
                #"leaked_l2": metrics.get("leaked_l2", 0),
                #"total_l2": metrics.get("total_l2", 0),
                #"leaked_l3": metrics.get("leaked_l3", 0),
                #"total_l3": metrics.get("total_l3", 0),
                "exposed_token_count": metrics["exposed_token_count"],
                "entity_retention_rate": metrics["entity_retention_rate"],
                "schema_valid": metrics["schema_valid"],
                "latency": metrics["latency"],
                "route": metrics["route"],
                "queries": row["user_query"],
                "targets": row["target_response"],
                "papillon_completion": getattr(pred, "output", ""),
                "papillon_prompt": getattr(pred, "cloud_prompt", getattr(pred, "prompt", "")),
                "pii_str": row["pii_units"],
                "l1_units": l1_str,
                "l2_units": l2_str,
                "l3_terms": l3_str,
                #"structured_fields_json": json.dumps(getattr(pred, "structured_fields", {})),
                #"detected_pii_json": json.dumps(getattr(pred, "detected_pii", [])),
            }
        )

        pandas.DataFrame(rows).to_csv(args.output_file_name, index=False)

    print("AVERAGE QUALITY SCORE:", safe_average(qual_scores))
    print("AVERAGE LEAKAGE SCORE:", safe_average(leak_scores))
    print("AVERAGE WPL SCORE    :", safe_average(wpl_scores))
    print("AVERAGE ETC SCORE    :", safe_average(etc_scores))
    print("AVERAGE ERR SCORE    :", safe_average(err_scores))
    print("AVERAGE LATENCY      :", safe_average(latency_scores))
    print("==============")
