import json
import os
from argparse import ArgumentParser

import dspy
import pandas
from dspy import Example
from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt import MIPROv2

from llm_judge import LLMJudge
from run_llama_dspy import PrivacyOnePrompter


os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), "cache")
openai_lm_gpt4o = None
llm_judge = LLMJudge()


def _normalized_pii_pieces(pii_str):
    if not isinstance(pii_str, str):
        return set()
    return {piece.strip() for piece in pii_str.split("||") if piece and piece.strip()}


def metric(gold, pred, trace=None):
    og_model_output, og_user_query, og_pii = gold.target_response, gold.user_query, gold.pii_str
    pred_prompt, pred_out = pred.prompt, pred.output
    if len(pred_prompt) == 0:
        return 0
    with dspy.context(lm=openai_lm_gpt4o):
        score_dict = llm_judge(
            user_query=og_user_query,
            new_resp=pred_out,
            og_resp=og_model_output,
            updated_query=pred_prompt,
            pii_str=og_pii,
        )
        final_quality_score = score_dict.quality
        leakage_sc = score_dict.leakage
        prompt_sc = score_dict.prompt
        if leakage_sc == -1:
            return 0
    denom = max(len(_normalized_pii_pieces(og_pii)), 1)
    final_total_score = (final_quality_score - leakage_sc / denom + prompt_sc) / 2
    if trace is not None:
        return final_total_score >= 1
    return final_total_score


def metric_finegrained(gold, pred):
    og_model_output, og_user_query, og_pii = gold.target_response, gold.user_query, gold.pii_str
    pred_prompt, pred_out = pred.prompt, pred.output
    if pred_prompt is not None and len(pred_prompt) == 0:
        return -1, -1
    with dspy.context(lm=openai_lm_gpt4o):
        score_dict = llm_judge(
            user_query=og_user_query,
            new_resp=pred_out,
            og_resp=og_model_output,
            updated_query=pred_prompt,
            pii_str=og_pii,
        )
    denom = max(len(_normalized_pii_pieces(og_pii)), 1)
    return score_dict.quality, score_dict.leakage / denom


def synthesize_tvt(data_file):
    df = pandas.read_csv(data_file, index_col=False)
    train, val, test = [], [], []
    for i, row in df.iterrows():
        if pandas.isna(row["pii_units"]) or not isinstance(row["pii_units"], str) or len(row["pii_units"]) == 0:
            continue
        new_dp = Example({
            "target_response": row["target_response"],
            "user_query": row["user_query"],
            "pii_str": row["pii_units"],
        }).with_inputs("user_query")
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
    parser.add_argument("--model_name", type=str, required=True, help="The local Ollama/OpenAI-compatible model name")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openrouter_base", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--prompt_output", type=str, help="The json file path where we will store the optimized prompts")
    parser.add_argument("--data_file", type=str, help="The csv containing PUPA-format data for optimization")
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--num_candidates", type=int, default=10)
    args = parser.parse_args()

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
    openai_lm_gpt4o = openai_lm

    assert isinstance(args.prompt_output, str) and args.prompt_output.endswith(".json")

    train, val, _ = synthesize_tvt(args.data_file)
    zeroshot = PrivacyOnePrompter(local_lm, openai_lm)
    evaluate = Evaluate(metric=metric, devset=val, num_threads=8, display_progress=True, display_table=5, max_errors=100)

    eval_scores = {}
    try:
        before_score = evaluate(zeroshot)
        eval_scores["before_optimization"] = before_score
        print(before_score)
    except Exception as e:
        print(f"[WARN] before-opt eval failed: {type(e).__name__}: {str(e)[:300]}")

    compiled_prompt_opt = None
    try:
        teleprompter = MIPROv2(
            prompt_model=openai_lm,
            task_model=local_lm,
            metric=metric,
            num_candidates=args.num_candidates,
            init_temperature=1.0,
        )
        kwargs = dict(num_threads=8, display_progress=True, display_table=0)
        compiled_prompt_opt = teleprompter.compile(
            zeroshot,
            trainset=train,
            num_batches=args.num_batches,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
            eval_kwargs=kwargs,
            requires_permission_to_run=False,
        )
    except Exception as e:
        print(f"[WARN] compile failed: {type(e).__name__}: {str(e)[:300]}")
        try:
            local_lm.inspect_history()
        except Exception:
            pass

    if compiled_prompt_opt is not None:
        try:
            compiled_prompt_opt.save(args.prompt_output)
            print(f"[INFO] Saved optimized prompt to {args.prompt_output}")
        except Exception as e:
            print(f"[WARN] save failed: {type(e).__name__}: {str(e)[:300]}")

        try:
            after_score = evaluate(compiled_prompt_opt)
            eval_scores["after_optimization"] = after_score
            print(after_score)
        except Exception as e:
            print(f"[WARN] after-opt eval failed: {type(e).__name__}: {str(e)[:300]}")

    eval_file = args.prompt_output.replace(".json", "_eval_scores.json")
    with open(eval_file, "w") as f:
        json.dump(eval_scores, f, indent=2)
