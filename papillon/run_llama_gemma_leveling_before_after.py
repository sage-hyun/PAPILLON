import argparse
import os
import select
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODEL_JOBS = [
    {
        "label": "llama31_8b",
        "model_name": "llama3.1:8b-instruct-fp16",
        "prompt_output_template": "optimized_prompts/llama_31_8b_instruct_{pipeline}_prompt.json",
        "before_output_template": "eval_results/eval_llama31_8b_{pipeline}_PUPA_TNB_leveling_before.csv",
        "after_output_template": "eval_results/eval_llama31_8b_{pipeline}_PUPA_TNB_leveling_after.csv",
        "run_optimization": False,
        "run_before_eval": False,
        "run_after_eval": True,
    },
    # {
    #     "label": "gemma4_e4b",
    #     "model_name": "gemma4:e4b",
    #     "prompt_output_template": "optimized_prompts/gemma_4_e4b_{pipeline}_prompt.json",
    #     "before_output_template": "eval_results/eval_gemma4_e4b_{pipeline}_PUPA_TNB_leveling_before.csv",
    #     "after_output_template": "eval_results/eval_gemma4_e4b_{pipeline}_PUPA_TNB_leveling_after.csv",
    #     "run_optimization": True,
    #     "run_before_eval": True,
    #     "run_after_eval": True,
    # },
]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize prompts on PUPA_New_leveling.csv, then run before/after evals on PUPA_TNB_leveling.csv."
    )
    parser.add_argument("--port", type=int, default=11434, help="Port for the local model server.")
    parser.add_argument(
        "--optimization_data_file",
        default="../pupa/PUPA_New_leveling.csv",
        help="Leveling CSV used for prompt optimization.",
    )
    parser.add_argument(
        "--eval_data_file",
        default="../pupa/PUPA_TNB_leveling.csv",
        help="Leveling CSV used for before/after evaluation.",
    )
    parser.add_argument("--openai_model", default="gpt-4o-mini", help="Judge/prompt model.")
    parser.add_argument("--python_bin", default=sys.executable, help="Python executable to use.")
    parser.add_argument("--pipeline", choices=["legacy", "structured_v1"], default="structured_v1")
    parser.add_argument("--allow_direct_bypass", type=str_to_bool, default=True)
    parser.add_argument("--privacy_filter", default="regex_presidio")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    parser.add_argument("--structured_planner_mode", choices=["cot", "predict"], default="predict")
    parser.add_argument("--l1_penalty_alpha", type=float, default=0.5)
    parser.add_argument("--num_threads", type=int, default=8, help="Thread count passed into DSPy evaluation/optimization.")
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of prompt candidates for MIPROv2.")
    parser.add_argument("--num_trials", type=int, default=100, help="Number of MIPROv2 optimization trials.")
    parser.add_argument("--optimization_log_root", type=str, default=None, help="Root directory where per-model MIPRO logs/checkpoints will be written.")
    parser.add_argument("--optimization_sample_count", type=int, default=5, help="How many per-trial optimization samples to log.")
    parser.add_argument("--disable_lm_cache", type=str_to_bool, default=True, help="Disable DSPy/LiteLLM in-memory caching to reduce RAM usage.")
    parser.add_argument("--save_lm_history", type=str_to_bool, default=False, help="If true, store LM history JSONL during optimization.")
    parser.add_argument("--history_flush_interval", type=int, default=25, help="Flush LM history to disk every N metric/sample events.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Start optimization from an existing best checkpoint if present.")
    parser.add_argument("--debug_threads", type=str_to_bool, default=False, help="Enable verbose thread/stage debug logs during optimization.")
    parser.add_argument("--debug_query_preview", type=int, default=80, help="Max query preview length in thread debug logs.")
    parser.add_argument(
        "--skip_optimization_models",
        type=str,
        default="",
        help="Comma-separated model job labels to skip optimization for, e.g. 'llama31_8b'.",
    )
    parser.add_argument("--progress_interval_sec", type=int, default=60, help="How often to print elapsed/ETA while a step runs.")
    parser.add_argument("--step_log_root", type=str, default="run_logs", help="Directory where per-step .log files will be written.")
    parser.add_argument("--skip_optimization", action="store_true", help="Reuse existing optimized prompts.")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop immediately on failure.")
    return parser.parse_args()


def format_minutes(seconds):
    return f"{seconds / 60:.1f} min"


def estimate_remaining_seconds(step_durations, pending_step_count):
    if not step_durations or pending_step_count <= 0:
        return None
    avg_step_sec = sum(step_durations) / len(step_durations)
    return avg_step_sec * pending_step_count


def emit_line(text, log_file=None):
    print(text)
    if log_file is not None:
        log_file.write(text + "\n")
        log_file.flush()


def stream_process_output(proc, log_file):
    stdout = proc.stdout
    if stdout is None:
        return
    while True:
        ready, _, _ = select.select([stdout], [], [], 0)
        if not ready:
            break
        chunk = os.read(stdout.fileno(), 4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()
        if log_file is not None:
            log_file.buffer.write(chunk)
            log_file.flush()


def run_cmd(label, cmd, progress_interval_sec, expected_step_sec=None, log_path=None):
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    emit_line("\n" + "=" * 80, log_file)
    emit_line(f"[RUN] {label}", log_file)
    emit_line(" ".join(cmd), log_file)
    if log_path:
        emit_line(f"[LOG] {log_path}", log_file)
    emit_line("=" * 80, log_file)
    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        next_report = start + max(progress_interval_sec, 1)
        while True:
            returncode = proc.poll()
            stream_process_output(proc, log_file)
            now = time.time()
            if returncode is not None:
                elapsed = now - start
                stream_process_output(proc, log_file)
                break
            if now >= next_report:
                elapsed = now - start
                progress_line = f"[PROGRESS] {label} elapsed={format_minutes(elapsed)}"
                if expected_step_sec is not None and expected_step_sec > elapsed:
                    progress_line += f" eta_step={format_minutes(expected_step_sec - elapsed)}"
                emit_line(progress_line, log_file)
                next_report = now + max(progress_interval_sec, 1)
            time.sleep(1)

        emit_line(f"[DONE] {label} exit_code={proc.returncode} elapsed={elapsed/60:.1f} min", log_file)
        return {"label": label, "returncode": proc.returncode, "elapsed_sec": elapsed, "log_path": log_path}
    finally:
        if log_file is not None:
            log_file.close()


def optimization_command(job, args):
    log_root = Path(args.optimization_log_root) if args.optimization_log_root else ROOT / "optimization_runs"
    model_log_dir = log_root / f"{job['label']}_{args.pipeline}"
    checkpoint_path = model_log_dir / "best_checkpoint.json"
    sample_csv_path = model_log_dir / "optimization_samples.csv"
    lm_history_path = model_log_dir / "lm_history.jsonl"
    prompt_output = resolve_output_path(job["prompt_output_template"], args.pipeline)
    return [
        args.python_bin,
        "run_dspy_optimization_llama.py",
        "--port",
        str(args.port),
        "--openai_model",
        args.openai_model,
        "--prompt_output",
        prompt_output,
        "--data_file",
        args.optimization_data_file,
        "--model_name",
        job["model_name"],
        "--pipeline",
        args.pipeline,
        "--allow_direct_bypass",
        str(args.allow_direct_bypass),
        "--privacy_filter",
        args.privacy_filter,
        "--pii_score_threshold",
        str(args.pii_score_threshold),
        "--structured_planner_mode",
        args.structured_planner_mode,
        "--l1_penalty_alpha",
        str(args.l1_penalty_alpha),
        "--num_threads",
        str(args.num_threads),
        "--num_candidates",
        str(args.num_candidates),
        "--num_trials",
        str(args.num_trials),
        "--optimization_log_dir",
        str(model_log_dir),
        "--checkpoint_path",
        str(checkpoint_path),
        "--optimization_sample_csv",
        str(sample_csv_path),
        "--optimization_sample_count",
        str(args.optimization_sample_count),
        "--disable_lm_cache",
        str(args.disable_lm_cache),
        "--save_lm_history",
        str(args.save_lm_history),
        "--history_flush_interval",
        str(args.history_flush_interval),
        "--debug_threads",
        str(args.debug_threads),
        "--debug_query_preview",
        str(args.debug_query_preview),
        *(
            ["--lm_history_file", str(lm_history_path)]
            if args.save_lm_history
            else []
        ),
        *(
            ["--resume_from_checkpoint"]
            if args.resume_from_checkpoint
            else []
        ),
    ]


def resolve_output_path(template, pipeline_name):
    return template.format(pipeline=pipeline_name)


def parse_label_set(raw_value):
    return {item.strip() for item in (raw_value or "").split(",") if item.strip()}


def should_skip_optimization(job, args, skip_labels):
    return (
        args.skip_optimization
        or job["label"] in skip_labels
        or not job.get("run_optimization", True)
    )


def eval_command(job, args, prompt_file, output_file_name):
    output_path = ROOT / output_file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        args.python_bin,
        "evaluate_papillon.py",
        "--port",
        str(args.port),
        "--data_file",
        args.eval_data_file,
        "--openai_model",
        args.openai_model,
        "--prompt_file",
        prompt_file,
        "--model_name",
        job["model_name"],
        "--output_file_name",
        output_file_name,
        "--pipeline",
        args.pipeline,
        "--allow_direct_bypass",
        str(args.allow_direct_bypass),
        "--privacy_filter",
        args.privacy_filter,
        "--pii_score_threshold",
        str(args.pii_score_threshold),
        "--structured_planner_mode",
        args.structured_planner_mode,
    ]


def main():
    args = parse_args()
    results = []
    overall_start = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    completed_step_durations = []
    skip_optimization_labels = parse_label_set(args.skip_optimization_models)
    log_root = Path(args.step_log_root)
    if not log_root.is_absolute():
        log_root = ROOT / log_root
    log_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Running {len(MODEL_JOBS)} model pipelines:")
    for job in MODEL_JOBS:
        print(f"  - {job['label']} ({job['model_name']})")
    print(f"[INFO] Step logs: {log_root}")

    all_steps = []
    for job in MODEL_JOBS:
        prompt_output = resolve_output_path(job["prompt_output_template"], args.pipeline)
        if not should_skip_optimization(job, args, skip_optimization_labels):
            all_steps.append((f"{job['label']}_optimize", optimization_command(job, args)))
        if job.get("run_before_eval", True):
            all_steps.append((
                f"{job['label']}_before",
                eval_command(job, args, "NONE", resolve_output_path(job["before_output_template"], args.pipeline)),
            ))
        if job.get("run_after_eval", True):
            all_steps.append((
                f"{job['label']}_after",
                eval_command(job, args, prompt_output, resolve_output_path(job["after_output_template"], args.pipeline)),
            ))

    total_steps = len(all_steps)
    completed_steps = 0

    for job in MODEL_JOBS:
        steps = []
        prompt_output = resolve_output_path(job["prompt_output_template"], args.pipeline)
        if not should_skip_optimization(job, args, skip_optimization_labels):
            steps.append(
                (
                    f"{job['label']}_optimize",
                    optimization_command(job, args),
                )
            )
        if job.get("run_before_eval", True):
            steps.append(
                (
                    f"{job['label']}_before",
                    eval_command(job, args, "NONE", resolve_output_path(job["before_output_template"], args.pipeline)),
                )
            )
        if job.get("run_after_eval", True):
            steps.append(
                (
                    f"{job['label']}_after",
                    eval_command(job, args, prompt_output, resolve_output_path(job["after_output_template"], args.pipeline)),
                )
            )

        for label, cmd in steps:
            log_path = log_root / f"{run_id}_{label}.log"
            pending_steps_after_this = total_steps - completed_steps - 1
            expected_step_sec = None
            if completed_step_durations:
                expected_step_sec = sum(completed_step_durations) / len(completed_step_durations)
                overall_eta = estimate_remaining_seconds(completed_step_durations, pending_steps_after_this + 1)
                if overall_eta is not None:
                    print(
                        f"[ETA] Starting step {completed_steps + 1}/{total_steps}: {label} | "
                        f"expected_step={format_minutes(expected_step_sec)} total_remaining~={format_minutes(overall_eta)}"
                    )
            else:
                print(f"[ETA] Starting step {completed_steps + 1}/{total_steps}: {label} | collecting timing baseline")

            result = run_cmd(
                label,
                cmd,
                progress_interval_sec=args.progress_interval_sec,
                expected_step_sec=expected_step_sec,
                log_path=str(log_path),
            )
            results.append(result)
            completed_steps += 1
            completed_step_durations.append(result["elapsed_sec"])
            if result["returncode"] != 0 and args.stop_on_error:
                break
        if results and results[-1]["returncode"] != 0 and args.stop_on_error:
            break

    total_elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    for result in results:
        status = "OK" if result["returncode"] == 0 else f"FAIL({result['returncode']})"
        log_suffix = f" log={result['log_path']}" if result.get("log_path") else ""
        print(f"  {result['label']:28s} {status:10s} {result['elapsed_sec']/60:7.1f} min{log_suffix}")
    print(f"  Total wall time: {total_elapsed/60:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
