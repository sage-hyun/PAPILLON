import os
import dspy
from dspy import Example

from dspy_compat import build_openai_lm
from llm_judge import LLMJudge
from evaluation_metrics import collect_deterministic_metrics, normalized_leakage
from pipeline_factory import build_pipeline
import pandas
from dspy.evaluate.evaluate import Evaluate

from dspy.teleprompt import MIPROv2

from argparse import ArgumentParser
import json

os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')


llm_judge = LLMJudge()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")

def metric(gold, pred, trace=None):
    og_model_output, og_user_query, og_pii = gold.target_response, gold.user_query, gold.pii_str
    pred_prompt, pred_out = pred.prompt or "", pred.output
    if len(pred_prompt) == 0:
        return 0
    with dspy.context(lm=openai_lm_gpt4o):
        score_dict = llm_judge(user_query=og_user_query, new_resp=pred_out, og_resp=og_model_output,
                                            updated_query=pred_prompt, pii_str=og_pii)       
        final_quality_score = score_dict.quality
        leakage_sc = score_dict.leakage
        prompt_sc = score_dict.prompt
        try:
            assert leakage_sc != -1
        except AssertionError:
            return 0
    # Want to maximize quality and minimize leakage
    final_total_score = (final_quality_score - normalized_leakage(leakage_sc, og_pii) + prompt_sc) / 2
    if trace is not None: return final_total_score >= 1
    return final_total_score

def metric_finegrained(gold, pred, openai_lm):
    og_model_output, og_user_query, og_pii = gold.target_response, gold.user_query, gold.pii_str
    pred_prompt, pred_out = pred.prompt or "", pred.output
    if pred_prompt is not None and len(pred_prompt) == 0:
        return {
            "quality": -1,
            "leakage": -1,
            "exposed_token_count": -1,
            "entity_retention_rate": -1,
            "schema_valid": False,
            "route": getattr(pred, "route", "legacy"),
        }
    with dspy.context(lm=openai_lm):
        score_dict = llm_judge(user_query=og_user_query, new_resp=pred_out, og_resp=og_model_output,
                                            updated_query=pred_prompt, pii_str=og_pii)
    deterministic_metrics = collect_deterministic_metrics(
        pii_str=og_pii,
        target_response=og_model_output,
        final_output=pred_out,
        cloud_prompt=getattr(pred, "cloud_prompt", pred_prompt),
        route=getattr(pred, "route", "legacy"),
        structured_fields=getattr(pred, "structured_fields", {}),
    )
    return {
        "quality": score_dict.quality,
        "leakage": normalized_leakage(score_dict.leakage, og_pii),
        **deterministic_metrics,
    }



def synthesize_tvt(data_file):
    df = pandas.read_csv(data_file, index_col=False)
    train, val, test = [], [], []
    for i, row in df.iterrows():
        if pandas.isna(row["pii_units"]) or not isinstance(row["pii_units"], str) or len(row["pii_units"]) == 0:
            continue
        new_dp = Example({"target_response": row["target_response"],
                          "user_query": row["user_query"],
                          "pii_str": row["pii_units"]}).with_inputs("user_query")
        if i < 150:
            train.append(new_dp)
        elif 150 <= i < 300:
            val.append(new_dp)
        else:
            test.append(new_dp)
    return train, val, test



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--prompt_output", type=str, help="The json file path where we will store the optimized prompts")
    parser.add_argument("--data_file", type=str, help="The csv containing PUPA-format data for optimization")
    parser.add_argument("--pipeline", type=str, choices=["legacy", "structured_v1"], default="legacy")
    parser.add_argument("--allow_direct_bypass", type=str_to_bool, default=True)
    parser.add_argument("--privacy_filter", type=str, default="regex_presidio")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    local_lm = dspy.LM('openai/default', api_base=f"http://127.0.0.1:{args.port}/v1", api_key="", max_tokens=4000)
    dspy.configure(lm=local_lm)

    openai_lm = build_openai_lm(args.openai_model, max_tokens=4000)
    openai_lm_gpt4o = build_openai_lm("gpt-4o-mini", max_tokens=4000)

    assert isinstance(args.prompt_output, str) and args.prompt_output.endswith(".json")
    prompt_dir = os.path.dirname(args.prompt_output)
    if prompt_dir:
        os.makedirs(prompt_dir, exist_ok=True)


    train, val, test = synthesize_tvt(args.data_file)
    zeroshot = build_pipeline(
        pipeline_name=args.pipeline,
        untrusted_model=openai_lm,
        allow_direct_bypass=args.allow_direct_bypass,
        privacy_filter_name=args.privacy_filter,
        pii_score_threshold=args.pii_score_threshold,
    )
    INCOMPLIANCE = 0
    evaluate = Evaluate(metric=metric, devset=val, num_threads=8, display_progress=True, display_table=5, max_errors=100)
    try:
        eval_score = evaluate(zeroshot)
    except Exception as e:
        INCOMPLIANCE += 1
    eval_scores = {}
    eval_scores.update({"before_optimization": eval_score})
    print(eval_score)
    try:
        teleprompter = MIPROv2(prompt_model=openai_lm, task_model=local_lm, metric=metric, num_candidates=10, init_temperature=1.0)
        kwargs = dict(num_threads=8, display_progress=True, display_table=0)
        compiled_prompt_opt = teleprompter.compile(zeroshot, trainset=train, num_batches=200, max_bootstrapped_demos=0, max_labeled_demos=0, eval_kwargs=kwargs)
        eval_score = evaluate(compiled_prompt_opt, devset=val, **kwargs)
        print(eval_score)
        eval_scores.update({"after_optimization": eval_score})

        compiled_prompt_opt.save(args.prompt_output)
    except ValueError as e:
        print(e)
        local_lm.inspect_history()
    EVAL_FILE = args.prompt_output.replace(".json", "_eval_socres.json")
    json.dump(eval_scores, open(EVAL_FILE, "w+"))
