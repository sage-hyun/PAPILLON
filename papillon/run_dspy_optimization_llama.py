import os
import dspy
from dspy import Example
import time
from pathlib import Path
import inspect
import csv
from contextlib import suppress
import threading
import shutil
import sys
import traceback

from dspy_compat import build_local_lm, build_openai_lm
from llm_judge import LLMJudge
from evaluation_metrics import (
    collect_deterministic_metrics,
    normalized_leakage,
    weighted_leakage_from_level_counts,
)
from pipeline_factory import build_pipeline
from prompt_paths import load_prompt_with_pipeline_compat
import pandas
from dspy.evaluate.evaluate import Evaluate

from dspy.teleprompt import MIPROv2


### ignore warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


LOCAL_LM_API_KEY = "local-openai-compatible-key"
LOCAL_LM_API_HOST = os.getenv("PAPILLON_LOCAL_LM_HOST", "127.0.0.1")

from argparse import ArgumentParser
import json
import re

os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')


llm_judge = LLMJudge()
L1_PENALTY_ALPHA = 0.5
HISTORY_FLUSHER = None
TASK_FIDELITY_ALPHA = 0.35
DOMAIN_INVENTION_ALPHA = 0.12
MAX_DOMAIN_INVENTION_PENALTY = 0.36
MIN_TASK_FIDELITY_OVERLAP = 0.35
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "given", "help", "how", "i", "if",
    "in", "into", "is", "it", "me", "my", "of", "on", "or", "please", "provide", "rewrite", "show", "so",
    "summarize", "tell", "that", "the", "this", "to", "use", "using", "want", "with", "without", "you",
    "your", "user", "request", "query", "response", "task", "context", "style", "output", "final", "model",
    "assistant", "placeholder", "placeholders", "safe", "sensitive", "private", "privacy",
}
PLACEHOLDER_TOKEN_PREFIXES = ("person", "location", "organization", "email", "phone", "date", "time", "url", "id")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


def format_minutes(seconds):
    return f"{seconds / 60:.1f} min"


def _tokenize_content(text):
    tokens = re.findall(r"[a-z0-9_:+.-]+", (text or "").lower())
    return [token for token in tokens if len(token) >= 3]


def _is_placeholder_like(token):
    normalized = token.strip("_[]")
    return normalized.startswith(PLACEHOLDER_TOKEN_PREFIXES)


def _content_token_set(text):
    result = set()
    for token in _tokenize_content(text):
        if token in CONTENT_STOPWORDS:
            continue
        if _is_placeholder_like(token):
            continue
        if token.isdigit():
            continue
        result.add(token)
    return result


def _planner_text(pred):
    structured_fields = getattr(pred, "structured_fields", {}) or {}
    pieces = [
        structured_fields.get("task", ""),
        structured_fields.get("safe_context", ""),
        structured_fields.get("style_constraints", ""),
    ]
    return " ".join(piece for piece in pieces if piece)


def compute_task_fidelity_metrics(gold, pred):
    if getattr(pred, "route", "") != "protected":
        return {
            "task_fidelity_overlap": 1.0,
            "task_fidelity_penalty": 0.0,
            "domain_invention_penalty": 0.0,
            "invented_domain_terms": [],
        }

    source_text = " ".join(
        part for part in [
            getattr(gold, "user_query", ""),
            getattr(pred, "redacted_query", ""),
        ] if part
    )
    generated_text = _planner_text(pred)

    source_tokens = _content_token_set(source_text)
    generated_tokens = _content_token_set(generated_text)
    if not generated_tokens:
        return {
            "task_fidelity_overlap": 0.0,
            "task_fidelity_penalty": TASK_FIDELITY_ALPHA,
            "domain_invention_penalty": 0.0,
            "invented_domain_terms": [],
        }

    overlap = len(source_tokens & generated_tokens) / max(len(generated_tokens), 1)
    if overlap >= MIN_TASK_FIDELITY_OVERLAP:
        task_fidelity_penalty = 0.0
    else:
        task_fidelity_penalty = TASK_FIDELITY_ALPHA * (MIN_TASK_FIDELITY_OVERLAP - overlap) / MIN_TASK_FIDELITY_OVERLAP

    invented_domain_terms = sorted(
        token for token in (generated_tokens - source_tokens)
        if len(token) >= 5 and not token.isdigit()
    )
    domain_invention_penalty = min(
        MAX_DOMAIN_INVENTION_PENALTY,
        DOMAIN_INVENTION_ALPHA * len(invented_domain_terms),
    )
    return {
        "task_fidelity_overlap": overlap,
        "task_fidelity_penalty": task_fidelity_penalty,
        "domain_invention_penalty": domain_invention_penalty,
        "invented_domain_terms": invented_domain_terms[:12],
    }


class BestPromptTracker:
    def __init__(self, checkpoint_path, prompt_output_path, archive_dir):
        self.checkpoint_path = checkpoint_path
        self.prompt_output_path = prompt_output_path
        self.archive_dir = archive_dir
        self.history_path = os.path.join(self.archive_dir, "best_prompt_history.jsonl")
        self.last_checkpoint_signature = None
        os.makedirs(self.archive_dir, exist_ok=True)

    @staticmethod
    def _file_signature(path):
        if not os.path.exists(path):
            return None
        stat = os.stat(path)
        return f"{stat.st_size}:{int(stat.st_mtime_ns)}"

    def sync_from_checkpoint(self, *, reason, event_index=None, eval_kind=None, score=None, force=False):
        if not os.path.exists(self.checkpoint_path):
            return False

        checkpoint_signature = self._file_signature(self.checkpoint_path)
        prompt_signature = self._file_signature(self.prompt_output_path)
        changed = checkpoint_signature != self.last_checkpoint_signature
        needs_copy = force or changed or prompt_signature != checkpoint_signature

        if needs_copy:
            prompt_dir = os.path.dirname(self.prompt_output_path)
            if prompt_dir:
                os.makedirs(prompt_dir, exist_ok=True)
            shutil.copy2(self.checkpoint_path, self.prompt_output_path)

        if changed:
            event_suffix = f"{event_index:03d}" if isinstance(event_index, int) else "latest"
            score_suffix = f"{score:.2f}" if isinstance(score, (int, float)) else "na"
            snapshot_name = f"best_prompt_event_{event_suffix}_{eval_kind or 'unknown'}_score_{score_suffix}.json"
            snapshot_path = os.path.join(self.archive_dir, snapshot_name)
            shutil.copy2(self.checkpoint_path, snapshot_path)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "reason": reason,
                            "event_index": event_index,
                            "eval_kind": eval_kind,
                            "score": score,
                            "checkpoint_path": self.checkpoint_path,
                            "prompt_output_path": self.prompt_output_path,
                            "snapshot_path": snapshot_path,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"[INFO] Synced new best prompt from checkpoint to {self.prompt_output_path} "
                f"and archived {snapshot_path}",
                flush=True,
            )

        self.last_checkpoint_signature = checkpoint_signature
        return needs_copy or changed


class LMHistoryFlusher:
    def __init__(self, output_path, flush_interval=25):
        self.output_path = output_path
        self.flush_interval = max(int(flush_interval), 1)
        self.registered_lms = []
        self.event_count = 0
        if self.output_path:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

    def register(self, name, lm):
        self.registered_lms.append((name, lm))

    def tick(self):
        self.event_count += 1
        if self.output_path and self.event_count % self.flush_interval == 0:
            self.flush()

    def flush(self):
        if not self.output_path:
            return

        rows = []
        for name, lm in self.registered_lms:
            history = list(getattr(lm, "history", []))
            if not history:
                continue
            for entry in history:
                rows.append(
                    {
                        "lm_name": name,
                        "entry": entry,
                    }
                )
            lm.history.clear()

        from dspy.clients import base_lm as dspy_base_lm

        global_history = list(getattr(dspy_base_lm, "GLOBAL_HISTORY", []))
        if global_history:
            for entry in global_history:
                rows.append(
                    {
                        "lm_name": "__global__",
                        "entry": entry,
                    }
                )
            dspy_base_lm.GLOBAL_HISTORY.clear()

        if not rows:
            return

        with open(self.output_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def maybe_flush_lm_history(force=False):
    global HISTORY_FLUSHER
    if HISTORY_FLUSHER is None:
        return
    if force:
        HISTORY_FLUSHER.flush()
    else:
        HISTORY_FLUSHER.tick()


def write_resume_state(
    state_path,
    *,
    status,
    args,
    checkpoint_path,
    prompt_output,
    optimization_log_dir,
    optimization_sample_csv,
    extra=None,
):
    payload = {
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": args.model_name,
        "pipeline": args.pipeline,
        "num_trials": args.num_trials,
        "num_candidates": args.num_candidates,
        "num_threads": args.num_threads,
        "data_file": args.data_file,
        "prompt_output": prompt_output,
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": os.path.exists(checkpoint_path),
        "optimization_log_dir": optimization_log_dir,
        "optimization_sample_csv": optimization_sample_csv,
        "optimization_sample_csv_exists": os.path.exists(optimization_sample_csv),
        "resume_hint": {
            "resume_from_checkpoint": os.path.exists(checkpoint_path),
            "resume_prompt_file": checkpoint_path if os.path.exists(checkpoint_path) else "",
        },
    }
    if extra:
        payload.update(extra)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def metric(gold, pred, trace=None):
    maybe_flush_lm_history()
    if os.getenv("PAPILLON_DEBUG_THREADS", "").lower() in {"1", "true", "yes", "y"}:
        print(
            f"[THREAD-DEBUG] event=metric.start thread={threading.get_ident()} "
            f"query_len={len(getattr(gold, 'user_query', '') or '')} route={getattr(pred, 'route', 'unknown')}",
            flush=True,
        )
    og_model_output, og_user_query, og_pii = gold.target_response, gold.user_query, gold.pii_str
    l1_str = getattr(gold, "l1_str", "")
    l2_str = getattr(gold, "l2_str", "")
    l3_str = getattr(gold, "l3_str", "")
    pred_prompt, pred_out = pred.prompt or "", pred.output
    if len(pred_prompt) == 0:
        return 0
    try:
        with dspy.context(lm=openai_lm_gpt4o):
            score_dict = llm_judge(user_query=og_user_query, new_resp=pred_out, og_resp=og_model_output,
                                                updated_query=pred_prompt, pii_str=og_pii, l1_str=l1_str, l2_str=l2_str, l3_str=l3_str)
            final_quality_score = score_dict.quality
            leakage_sc = score_dict.leakage
            prompt_sc = score_dict.prompt
            try:
                assert leakage_sc != -1
            except AssertionError:
                return 0
    except Exception as exc:
        print(f"[WARN] metric judge failure: {type(exc).__name__}: {str(exc)[:200]}")
        return 0

    level_counts_available = any(
        isinstance(value, str) and value.strip()
        for value in (l1_str, l2_str, l3_str)
    )
    if level_counts_available:
        weighted_parts = weighted_leakage_from_level_counts(
            leaked_l1=getattr(score_dict, "leaked_l1", 0),
            total_l1=getattr(score_dict, "total_l1", 0),
            leaked_l2=getattr(score_dict, "leaked_l2", 0),
            total_l2=getattr(score_dict, "total_l2", 0),
            leaked_l3=getattr(score_dict, "leaked_l3", 0),
            total_l3=getattr(score_dict, "total_l3", 0),
        )
        leakage_value = weighted_parts["weighted_leakage"]
        l1_leak_ratio = weighted_parts["leakage_l1_ratio"]
    else:
        leakage_value = normalized_leakage(leakage_sc, og_pii)
        l1_leak_ratio = 0.0

    fidelity_parts = compute_task_fidelity_metrics(gold, pred)

    # Want to maximize quality and minimize leakage
    final_total_score = (
        (final_quality_score - leakage_value + prompt_sc) / 2
        - (L1_PENALTY_ALPHA * l1_leak_ratio)
        - fidelity_parts["task_fidelity_penalty"]
        - fidelity_parts["domain_invention_penalty"]
    )
    if os.getenv("PAPILLON_DEBUG_THREADS", "").lower() in {"1", "true", "yes", "y"}:
        print(
            f"[THREAD-DEBUG] event=metric.done thread={threading.get_ident()} "
            f"quality={final_quality_score} leakage={leakage_value} prompt={prompt_sc} "
            f"task_fidelity_penalty={fidelity_parts['task_fidelity_penalty']} "
            f"domain_invention_penalty={fidelity_parts['domain_invention_penalty']} total={final_total_score}",
            flush=True,
        )
    if trace is not None: return final_total_score >= 1
    return final_total_score

def metric_finegrained(
    gold,
    pred,
    openai_lm,
    l1_str=None,
    l2_str=None,
    l3_str=None,
):
    maybe_flush_lm_history()
    og_model_output, og_user_query, og_pii = gold.target_response, gold.user_query, gold.pii_str
    pred_prompt, pred_out = pred.prompt or "", pred.output
    if pred_prompt is not None and len(pred_prompt) == 0:
        return {
            "quality": -1,
            "leakage": -1,
            "weighted_leakage": -1,
            "leakage_l1_ratio": 0.0,
            "leakage_l2_ratio": 0.0,
            "leakage_l3_ratio": 0.0,
            "leaked_l1": 0,
            "total_l1": 0,
            "leaked_l2": 0,
            "total_l2": 0,
            "leaked_l3": 0,
            "total_l3": 0,
            "task_fidelity_overlap": 0.0,
            "task_fidelity_penalty": 0.0,
            "domain_invention_penalty": 0.0,
            "invented_domain_terms": "",
            "exposed_token_count": -1,
            "entity_retention_rate": -1,
            "schema_valid": False,
            "route": getattr(pred, "route", "legacy"),
            "latency": getattr(pred, "latency", 0.0),
        }
    try:
        with dspy.context(lm=openai_lm):
            score_dict = llm_judge(user_query=og_user_query, new_resp=pred_out, og_resp=og_model_output,
                                                updated_query=pred_prompt, pii_str=og_pii, l1_str=l1_str, l2_str=l2_str, l3_str=l3_str)
    except Exception as exc:
        print(f"[WARN] finegrained judge failure: {type(exc).__name__}: {str(exc)[:200]}")
        return {
            "quality": -1,
            "leakage": -1,
            "weighted_leakage": -1,
            "leakage_l1_ratio": 0.0,
            "leakage_l2_ratio": 0.0,
            "leakage_l3_ratio": 0.0,
            "leaked_l1": 0,
            "total_l1": 0,
            "leaked_l2": 0,
            "total_l2": 0,
            "leaked_l3": 0,
            "total_l3": 0,
            "task_fidelity_overlap": 0.0,
            "task_fidelity_penalty": 0.0,
            "domain_invention_penalty": 0.0,
            "invented_domain_terms": "",
            "exposed_token_count": -1,
            "entity_retention_rate": -1,
            "schema_valid": False,
            "route": getattr(pred, "route", "legacy"),
            "latency": getattr(pred, "latency", 0.0),
        }
    deterministic_metrics = collect_deterministic_metrics(
        pii_str=og_pii,
        target_response=og_model_output,
        final_output=pred_out,
        cloud_prompt=getattr(pred, "cloud_prompt", pred_prompt),
        route=getattr(pred, "route", "legacy"),
        structured_fields=getattr(pred, "structured_fields", {}),
        latency=getattr(pred, "latency", 0.0)
    )
    level_counts_available = any(
        isinstance(value, str) and value.strip()
        for value in (l1_str, l2_str, l3_str)
    )
    if level_counts_available:
        weighted_parts = weighted_leakage_from_level_counts(
            leaked_l1=getattr(score_dict, "leaked_l1", 0),
            total_l1=getattr(score_dict, "total_l1", 0),
            leaked_l2=getattr(score_dict, "leaked_l2", 0),
            total_l2=getattr(score_dict, "total_l2", 0),
            leaked_l3=getattr(score_dict, "leaked_l3", 0),
            total_l3=getattr(score_dict, "total_l3", 0),
        )
        leakage_value = weighted_parts["weighted_leakage"]
    else:
        weighted_parts = {
            "weighted_leakage": normalized_leakage(score_dict.leakage, og_pii),
            "leakage_l1_ratio": 0.0,
            "leakage_l2_ratio": 0.0,
            "leakage_l3_ratio": 0.0,
        }
        leakage_value = weighted_parts["weighted_leakage"]
    fidelity_parts = compute_task_fidelity_metrics(gold, pred)
    return {
        "quality": score_dict.quality,
        "leakage": leakage_value,
        "weighted_leakage": weighted_parts["weighted_leakage"],
        "leakage_l1_ratio": weighted_parts["leakage_l1_ratio"],
        "leakage_l2_ratio": weighted_parts["leakage_l2_ratio"],
        "leakage_l3_ratio": weighted_parts["leakage_l3_ratio"],
        "leaked_l1": getattr(score_dict, "leaked_l1", 0),
        "total_l1": getattr(score_dict, "total_l1", 0),
        "leaked_l2": getattr(score_dict, "leaked_l2", 0),
        "total_l2": getattr(score_dict, "total_l2", 0),
        "leaked_l3": getattr(score_dict, "leaked_l3", 0),
        "total_l3": getattr(score_dict, "total_l3", 0),
        "task_fidelity_overlap": fidelity_parts["task_fidelity_overlap"],
        "task_fidelity_penalty": fidelity_parts["task_fidelity_penalty"],
        "domain_invention_penalty": fidelity_parts["domain_invention_penalty"],
        "invented_domain_terms": ", ".join(fidelity_parts["invented_domain_terms"]),
        "optimization_score": (
            (score_dict.quality - leakage_value + score_dict.prompt) / 2
            - (L1_PENALTY_ALPHA * weighted_parts["leakage_l1_ratio"])
            - fidelity_parts["task_fidelity_penalty"]
            - fidelity_parts["domain_invention_penalty"]
        ),
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
                          "pii_str": row["pii_units"],
                          "l1_str": row["l1_units"] if "l1_units" in row and isinstance(row["l1_units"], str) else "",
                          "l2_str": row["l2_units"] if "l2_units" in row and isinstance(row["l2_units"], str) else "",
                          "l3_str": row["l3_terms"] if "l3_terms" in row and isinstance(row["l3_terms"], str) else "",
                          }).with_inputs("user_query")
        if i < 150:
            train.append(new_dp)
        elif 150 <= i < 300:
            val.append(new_dp)
        else:
            test.append(new_dp)
    return train, val, test


def build_mipro_teleprompter(prompt_model, task_model, metric_fn, num_candidates, num_threads, log_dir):
    init_kwargs = {
        "prompt_model": prompt_model,
        "task_model": task_model,
        "metric": metric_fn,
        "num_candidates": num_candidates,
        "init_temperature": 1.0,
        "log_dir": log_dir,
    }
    init_signature = inspect.signature(MIPROv2.__init__)
    if "num_threads" in init_signature.parameters:
        init_kwargs["num_threads"] = num_threads
    return MIPROv2(**init_kwargs)


def compile_with_mipro_compat(teleprompter, program, trainset, valset, num_trials, num_threads):
    compile_signature = inspect.signature(teleprompter.compile)
    common_kwargs = {
        "trainset": trainset,
        "valset": valset,
        "max_bootstrapped_demos": 0,
        "max_labeled_demos": 0,
        "requires_permission_to_run": False,
    }
    if "eval_kwargs" in compile_signature.parameters:
        common_kwargs["eval_kwargs"] = {
            "num_threads": num_threads,
            "display_progress": True,
            "display_table": 0,
            "max_errors": 100,
        }
    if "num_trials" in compile_signature.parameters:
        return teleprompter.compile(program, num_trials=num_trials, **common_kwargs)
    if "num_batches" in compile_signature.parameters:
        return teleprompter.compile(program, num_batches=num_trials, **common_kwargs)
    return teleprompter.compile(program, **common_kwargs)


def build_optimization_sample_row(trial_index, eval_kind, gold, pred, metrics):
    return {
        "trial_index": trial_index,
        "eval_kind": eval_kind,
        "quality": metrics["quality"],
        "leakage": metrics["leakage"],
        "weighted_leakage": metrics.get("weighted_leakage", metrics["leakage"]),
        "leakage_l1_ratio": metrics.get("leakage_l1_ratio", 0.0),
        "leakage_l2_ratio": metrics.get("leakage_l2_ratio", 0.0),
        "leakage_l3_ratio": metrics.get("leakage_l3_ratio", 0.0),
        "leaked_l1": metrics.get("leaked_l1", 0),
        "total_l1": metrics.get("total_l1", 0),
        "leaked_l2": metrics.get("leaked_l2", 0),
        "total_l2": metrics.get("total_l2", 0),
        "leaked_l3": metrics.get("leaked_l3", 0),
        "total_l3": metrics.get("total_l3", 0),
        "task_fidelity_overlap": metrics.get("task_fidelity_overlap", 0.0),
        "task_fidelity_penalty": metrics.get("task_fidelity_penalty", 0.0),
        "domain_invention_penalty": metrics.get("domain_invention_penalty", 0.0),
        "invented_domain_terms": metrics.get("invented_domain_terms", ""),
        "optimization_score": metrics.get("optimization_score", 0.0),
        "exposed_token_count": metrics.get("exposed_token_count", -1),
        "entity_retention_rate": metrics.get("entity_retention_rate", -1),
        "schema_valid": metrics.get("schema_valid", False),
        "latency": metrics.get("latency", getattr(pred, "latency", 0.0)),
        "route": metrics.get("route", getattr(pred, "route", "legacy")),
        "queries": getattr(gold, "user_query", ""),
        "targets": getattr(gold, "target_response", ""),
        "papillon_completion": getattr(pred, "output", ""),
        "papillon_prompt": getattr(pred, "cloud_prompt", getattr(pred, "prompt", "")),
        "cloud_model_raw_response": getattr(pred, "gptResponse", ""),
        "structured_delegation_output_json": json.dumps(
            getattr(pred, "structured_delegation_output", getattr(pred, "structured_fields", {})),
            ensure_ascii=False,
        ),
        "structured_task": getattr(pred, "structured_fields", {}).get("task", ""),
        "structured_safe_context": getattr(pred, "structured_fields", {}).get("safe_context", ""),
        "structured_style_constraints": getattr(pred, "structured_fields", {}).get("style_constraints", ""),
        "structured_rationale": getattr(pred, "structured_fields", {}).get("rationale", ""),
        "info_aggregator_output": getattr(pred, "info_aggregator_output", getattr(pred, "output", "")),
        "pii_str": getattr(gold, "pii_str", ""),
        "l1_units": getattr(gold, "l1_str", ""),
        "l2_units": getattr(gold, "l2_str", ""),
        "l3_terms": getattr(gold, "l3_str", ""),
    }


def install_optimization_sample_logger(sample_csv_path, sample_count, openai_lm):
    if sample_count <= 0:
        return None

    from dspy.teleprompt import mipro_optimizer_v2 as mipro_module
    from dspy.teleprompt import utils as teleprompt_utils

    original_eval_candidate_program = mipro_module.eval_candidate_program
    state = {"trial_index": 0}

    def append_rows(rows):
        if not rows:
            return
        file_exists = os.path.exists(sample_csv_path)
        with open(sample_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def wrapped_eval_candidate_program(*args, **kwargs):
        score = original_eval_candidate_program(*args, **kwargs)
        state["trial_index"] += 1

        dataset = args[1] if len(args) > 1 else kwargs.get("trainset") or kwargs.get("valset") or []
        candidate_program = args[2] if len(args) > 2 else kwargs.get("candidate_program")
        batch_size = args[0] if len(args) > 0 else kwargs.get("batch_size", len(dataset))
        eval_kind = "full_eval" if batch_size >= len(dataset) else "minibatch_eval"

        sample_rows = []
        for gold in list(dataset)[:sample_count]:
            try:
                pred = candidate_program(gold.user_query)
                metrics = metric_finegrained(
                    gold,
                    pred,
                    openai_lm,
                    l1_str=getattr(gold, "l1_str", ""),
                    l2_str=getattr(gold, "l2_str", ""),
                    l3_str=getattr(gold, "l3_str", ""),
                )
            except Exception as exc:
                print(f"[WARN] optimization sample logging failure: {type(exc).__name__}: {str(exc)[:200]}")
                pred = dspy.Prediction(output="", prompt="", cloud_prompt="", gptResponse="", latency=0.0)
                metrics = {
                    "quality": -1,
                    "leakage": -1,
                    "weighted_leakage": -1,
                    "leakage_l1_ratio": 0.0,
                    "leakage_l2_ratio": 0.0,
                    "leakage_l3_ratio": 0.0,
                    "leaked_l1": 0,
                    "total_l1": 0,
                    "leaked_l2": 0,
                    "total_l2": 0,
                    "leaked_l3": 0,
                    "total_l3": 0,
                    "task_fidelity_overlap": 0.0,
                    "task_fidelity_penalty": 0.0,
                    "domain_invention_penalty": 0.0,
                    "invented_domain_terms": "",
                    "exposed_token_count": -1,
                    "entity_retention_rate": -1,
                    "schema_valid": False,
                    "latency": 0.0,
                    "route": "legacy",
                }
            sample_rows.append(
                build_optimization_sample_row(
                    trial_index=state["trial_index"],
                    eval_kind=eval_kind,
                    gold=gold,
                    pred=pred,
                    metrics=metrics,
                )
            )

        append_rows(sample_rows)
        maybe_flush_lm_history()
        return score

    mipro_module.eval_candidate_program = wrapped_eval_candidate_program
    teleprompt_utils.eval_candidate_program = wrapped_eval_candidate_program

    def restore():
        mipro_module.eval_candidate_program = original_eval_candidate_program
        teleprompt_utils.eval_candidate_program = original_eval_candidate_program

    return restore


def install_optimization_progress_logger(
    num_trials,
    valset_size,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    best_prompt_tracker=None,
):
    from dspy.teleprompt import mipro_optimizer_v2 as mipro_module
    from dspy.teleprompt import utils as teleprompt_utils

    original_eval_candidate_program = mipro_module.eval_candidate_program
    state = {
        "event_index": 0,
        "optimization_start": time.time(),
        "expected_events": 1 + num_trials + (num_trials // minibatch_full_eval_steps),
    }
    if num_trials % minibatch_full_eval_steps != 0:
        state["expected_events"] += 1

    def wrapped_eval_candidate_program(*args, **kwargs):
        score = original_eval_candidate_program(*args, **kwargs)
        state["event_index"] += 1

        dataset = args[1] if len(args) > 1 else kwargs.get("trainset") or kwargs.get("valset") or []
        batch_size = args[0] if len(args) > 0 else kwargs.get("batch_size", len(dataset))
        eval_kind = "full_eval" if batch_size >= len(dataset) else "minibatch_eval"
        elapsed = time.time() - state["optimization_start"]
        avg_event_sec = elapsed / max(state["event_index"], 1)
        remaining_events = max(state["expected_events"] - state["event_index"], 0)
        eta_sec = avg_event_sec * remaining_events

        trial_progress = "default_eval"
        if state["event_index"] > 1:
            completed_trial_events = state["event_index"] - 1
            minibatch_trials_done = min(completed_trial_events, num_trials)
            trial_progress = f"trial~{minibatch_trials_done}/{num_trials}"

        print(
            f"[ETA] optimization {eval_kind} event={state['event_index']}/{state['expected_events']} "
            f"{trial_progress} elapsed={format_minutes(elapsed)} eta~={format_minutes(eta_sec)} "
            f"batch_size={batch_size if batch_size else valset_size} score={score:.2f}"
        )
        if best_prompt_tracker is not None:
            best_prompt_tracker.sync_from_checkpoint(
                reason="optimization_progress",
                event_index=state["event_index"],
                eval_kind=eval_kind,
                score=score,
            )
        return score

    mipro_module.eval_candidate_program = wrapped_eval_candidate_program
    teleprompt_utils.eval_candidate_program = wrapped_eval_candidate_program

    def restore():
        mipro_module.eval_candidate_program = original_eval_candidate_program
        teleprompt_utils.eval_candidate_program = original_eval_candidate_program

    return restore



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, help="The port where you are hosting your local model")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--prompt_output", type=str, help="The json file path where we will store the optimized prompts")
    parser.add_argument("--data_file", type=str, help="The csv containing PUPA-format data for optimization")
    parser.add_argument("--model_name", type=str, help="The Huggingface identifier / name for your local LM")
    parser.add_argument("--pipeline", type=str, choices=["legacy", "structured_v1"], default="legacy")
    parser.add_argument("--allow_direct_bypass", type=str_to_bool, default=True)
    parser.add_argument("--privacy_filter", type=str, default="regex_presidio")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    parser.add_argument("--structured_planner_mode", choices=["cot", "predict"], default="cot")
    parser.add_argument("--l1_penalty_alpha", type=float, default=0.5)
    parser.add_argument("--num_threads", type=int, default=8, help="Thread count for DSPy evaluation/optimization.")
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of prompt candidates for MIPROv2.")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of MIPROv2 optimization trials.")
    parser.add_argument("--optimization_log_dir", type=str, default=None, help="Directory for MIPRO trial logs and candidate program snapshots.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to save the best prompt checkpoint during optimization.")
    parser.add_argument("--optimization_sample_csv", type=str, default=None, help="CSV file to append sampled per-trial optimization evaluations.")
    parser.add_argument("--optimization_sample_count", type=int, default=5, help="How many examples to log for each optimization trial evaluation.")
    parser.add_argument("--disable_lm_cache", type=str_to_bool, default=True, help="Disable DSPy/LiteLLM in-memory caching to reduce RAM usage.")
    parser.add_argument("--save_lm_history", type=str_to_bool, default=False, help="If true, stream LM history to JSONL for debugging.")
    parser.add_argument("--lm_history_file", type=str, default=None, help="Optional JSONL file to stream LM history to disk and clear from memory.")
    parser.add_argument("--history_flush_interval", type=int, default=25, help="Flush LM history to disk every N metric/sample events.")
    parser.add_argument("--resume_prompt_file", type=str, default=None, help="Load this prompt/checkpoint JSON before optimization starts.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="If checkpoint_path exists, load it before starting a new optimization run.")
    parser.add_argument("--debug_threads", type=str_to_bool, default=False, help="Enable verbose thread/stage debug logs during optimization.")
    parser.add_argument("--debug_query_preview", type=int, default=80, help="Max query preview length in thread debug logs.")
    args = parser.parse_args()

    L1_PENALTY_ALPHA = args.l1_penalty_alpha

    local_lm = build_local_lm(
        args.model_name,
        host=LOCAL_LM_API_HOST,
        port=args.port,
        api_key=LOCAL_LM_API_KEY,
        max_tokens=4000,
        cache=not args.disable_lm_cache,
    )
    dspy.configure(lm=local_lm)

    openai_lm = build_openai_lm(args.openai_model, max_tokens=4000, cache=not args.disable_lm_cache)
    openai_lm_gpt4o = build_openai_lm("gpt-4o-mini", max_tokens=4000, cache=not args.disable_lm_cache)

    assert isinstance(args.prompt_output, str) and args.prompt_output.endswith(".json")
    os.environ["PAPILLON_DEBUG_THREADS"] = "1" if args.debug_threads else "0"
    os.environ["PAPILLON_DEBUG_QUERY_PREVIEW"] = str(args.debug_query_preview)
    os.environ["PAPILLON_DEBUG_RAISE"] = "1" if args.debug_threads else "0"
    prompt_dir = os.path.dirname(args.prompt_output)
    if prompt_dir:
        os.makedirs(prompt_dir, exist_ok=True)
    prompt_output_path = Path(args.prompt_output)
    optimization_log_dir = args.optimization_log_dir
    if not optimization_log_dir:
        optimization_log_dir = str(prompt_output_path.with_suffix("")) + "_mipro_logs"
    os.makedirs(optimization_log_dir, exist_ok=True)
    checkpoint_path = args.checkpoint_path or str(prompt_output_path.with_suffix("")) + "_checkpoint.json"
    optimization_sample_csv = args.optimization_sample_csv or str(prompt_output_path.with_suffix("")) + "_optimization_samples.csv"
    resume_state_path = str(Path(optimization_log_dir) / "resume_state.json")
    best_prompt_tracker = BestPromptTracker(
        checkpoint_path=checkpoint_path,
        prompt_output_path=args.prompt_output,
        archive_dir=str(Path(optimization_log_dir) / "best_prompt_snapshots"),
    )

    lm_history_file = None
    if args.save_lm_history:
        lm_history_file = args.lm_history_file or str(Path(optimization_log_dir) / "lm_history.jsonl")
        HISTORY_FLUSHER = LMHistoryFlusher(lm_history_file, flush_interval=args.history_flush_interval)
        HISTORY_FLUSHER.register("local_lm", local_lm)
        HISTORY_FLUSHER.register("openai_lm", openai_lm)
        HISTORY_FLUSHER.register("openai_lm_gpt4o", openai_lm_gpt4o)
    else:
        HISTORY_FLUSHER = None


    train, val, test = synthesize_tvt(args.data_file)
    print(
        f"[INFO] Optimization config: train={len(train)} val={len(val)} test={len(test)} "
        f"threads={args.num_threads} trials={args.num_trials} candidates={args.num_candidates} "
        f"l1_penalty_alpha={args.l1_penalty_alpha} planner_mode={args.structured_planner_mode}"
    )
    print(f"[INFO] Optimization logs: {optimization_log_dir}")
    print(f"[INFO] Best-checkpoint path: {checkpoint_path}")
    print(f"[INFO] Optimization sample CSV: {optimization_sample_csv} (rows_per_eval={args.optimization_sample_count})")
    print(f"[INFO] LM history file: {lm_history_file or 'DISABLED'}")
    print(f"[INFO] Resume state file: {resume_state_path}")
    write_resume_state(
        resume_state_path,
        status="initialized",
        args=args,
        checkpoint_path=checkpoint_path,
        prompt_output=args.prompt_output,
        optimization_log_dir=optimization_log_dir,
        optimization_sample_csv=optimization_sample_csv,
    )
    zeroshot = build_pipeline(
        pipeline_name=args.pipeline,
        untrusted_model=openai_lm,
        allow_direct_bypass=args.allow_direct_bypass,
        privacy_filter_name=args.privacy_filter,
        pii_score_threshold=args.pii_score_threshold,
        structured_planner_mode=args.structured_planner_mode,
    )
    resume_prompt_file = args.resume_prompt_file
    if args.resume_from_checkpoint and os.path.exists(checkpoint_path):
        resume_prompt_file = checkpoint_path
    if resume_prompt_file:
        load_prompt_with_pipeline_compat(zeroshot, resume_prompt_file)
        print(f"[INFO] Loaded starting prompt from: {resume_prompt_file}")
    INCOMPLIANCE = 0
    evaluate = Evaluate(
        metric=metric,
        devset=val,
        num_threads=args.num_threads,
        display_progress=True,
        display_table=5,
        max_errors=100,
    )
    eval_score = 0
    try:
        before_eval_start = time.time()
        eval_score = evaluate(zeroshot)
        maybe_flush_lm_history(force=True)
        print(f"[INFO] Before-optimization eval finished in {(time.time() - before_eval_start) / 60:.1f} min")
    except Exception as e:
        INCOMPLIANCE += 1
        print(f"[WARN] before-optimization eval failed: {type(e).__name__}: {str(e)[:300]}")
    eval_scores = {}
    eval_scores.update({"before_optimization": eval_score})
    print(eval_score)
    try:
        teleprompter = build_mipro_teleprompter(
            prompt_model=openai_lm,
            task_model=local_lm,
            metric_fn=metric,
            num_candidates=args.num_candidates,
            num_threads=args.num_threads,
            log_dir=optimization_log_dir,
        )
        teleprompter._checkpoint_path = checkpoint_path
        restore_progress_logger = install_optimization_progress_logger(
            num_trials=args.num_trials,
            valset_size=len(val),
            best_prompt_tracker=best_prompt_tracker,
        )
        restore_sample_logger = install_optimization_sample_logger(
            sample_csv_path=optimization_sample_csv,
            sample_count=args.optimization_sample_count,
            openai_lm=openai_lm,
        )
        compile_start = time.time()
        compiled_prompt_opt = compile_with_mipro_compat(
            teleprompter=teleprompter,
            program=zeroshot,
            trainset=train,
            valset=val,
            num_trials=args.num_trials,
            num_threads=args.num_threads,
        )
        maybe_flush_lm_history(force=True)
        best_prompt_tracker.sync_from_checkpoint(reason="post_compile", force=True)
        if restore_progress_logger is not None:
            restore_progress_logger()
        if restore_sample_logger is not None:
            restore_sample_logger()
        print(f"[INFO] Optimization compile finished in {(time.time() - compile_start) / 60:.1f} min")
        write_resume_state(
            resume_state_path,
            status="compiled",
            args=args,
            checkpoint_path=checkpoint_path,
            prompt_output=args.prompt_output,
            optimization_log_dir=optimization_log_dir,
            optimization_sample_csv=optimization_sample_csv,
            extra={"compile_minutes": round((time.time() - compile_start) / 60, 2)},
        )
        after_eval_start = time.time()
        eval_score = evaluate(compiled_prompt_opt, devset=val)
        maybe_flush_lm_history(force=True)
        best_prompt_tracker.sync_from_checkpoint(reason="post_after_eval", force=True)
        print(f"[INFO] After-optimization eval finished in {(time.time() - after_eval_start) / 60:.1f} min")
        print(eval_score)
        eval_scores.update({"after_optimization": eval_score})

        compiled_prompt_opt.save(args.prompt_output)
        write_resume_state(
            resume_state_path,
            status="completed",
            args=args,
            checkpoint_path=checkpoint_path,
            prompt_output=args.prompt_output,
            optimization_log_dir=optimization_log_dir,
            optimization_sample_csv=optimization_sample_csv,
            extra={
                "before_optimization_score": eval_scores.get("before_optimization"),
                "after_optimization_score": eval_score,
            },
        )
    except ValueError as e:
        print(e)
        local_lm.inspect_history()
        with suppress(Exception):
            best_prompt_tracker.sync_from_checkpoint(reason="value_error_recovery", force=True)
        write_resume_state(
            resume_state_path,
            status="failed_value_error",
            args=args,
            checkpoint_path=checkpoint_path,
            prompt_output=args.prompt_output,
            optimization_log_dir=optimization_log_dir,
            optimization_sample_csv=optimization_sample_csv,
            extra={"error": str(e)[:500]},
        )
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"[WARN] optimization failed: {type(e).__name__}: {str(e)[:300]}")
        if os.path.exists(checkpoint_path):
            print(f"[INFO] Latest best checkpoint is available at: {checkpoint_path}")
        with suppress(Exception):
            best_prompt_tracker.sync_from_checkpoint(reason="exception_recovery", force=True)
        write_resume_state(
            resume_state_path,
            status="failed_exception",
            args=args,
            checkpoint_path=checkpoint_path,
            prompt_output=args.prompt_output,
            optimization_log_dir=optimization_log_dir,
            optimization_sample_csv=optimization_sample_csv,
            extra={"error": f"{type(e).__name__}: {str(e)[:500]}"},
        )
        traceback.print_exc()
        sys.exit(1)
    finally:
        with suppress(Exception):
            maybe_flush_lm_history(force=True)
    EVAL_FILE = args.prompt_output.replace(".json", "_eval_socres.json")
    json.dump(eval_scores, open(EVAL_FILE, "w+"))
