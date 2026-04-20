import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import dspy
import litellm
import pandas
import tqdm
from dspy import Example

import run_dspy_optimization_llama as opt_module
from run_dspy_optimization_llama import metric_finegrained
from ollama_nothink_lm import OllamaNoThinkLM
from run_llama_dspy import PrivacyOnePrompter


_THREAD_LOCAL = threading.local()
_WORKER_CONFIG = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate PAPILLON on PUPA-TNB with Qwen via OllamaNoThinkLM.'
    )
    parser.add_argument('--model_name', required=True, help='Ollama model tag, e.g. qwen3.5:9b-nothink')
    parser.add_argument('--data_file', default='../pupa/PUPA_TNB.csv', help='CSV containing PUPA-format data')
    parser.add_argument('--output_file_name', required=True, help='Per-row CSV to write evaluation results to')
    parser.add_argument('--prompt_file', default='NONE', help='DSPy prompt JSON to load. Use NONE for zero-shot baseline.')
    parser.add_argument('--ollama_base', default='http://127.0.0.1:11434', help='Ollama native API base URL')
    parser.add_argument('--openai_model', default='gpt-4o-mini', help='Cloud proprietary/judge model')
    parser.add_argument('--openrouter_base', default='https://openrouter.ai/api/v1', help='OpenRouter-compatible API base URL')
    parser.add_argument('--max_tokens', type=int, default=8000)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0, help='Optional row limit for quick tests; 0 uses all rows')
    parser.add_argument('--prompt_creator_instruction', type=str, default=None, help='Override prompt creator instruction text')
    parser.add_argument('--prompt_creator_instruction_file', type=str, default=None, help='Read prompt creator instruction text from file')
    parser.add_argument('--info_aggregator_instruction', type=str, default=None, help='Override info aggregator instruction text')
    parser.add_argument('--info_aggregator_instruction_file', type=str, default=None, help='Read info aggregator instruction text from file')
    return parser.parse_args()


def _load_text_arg(text_value, file_path):
    if text_value and file_path:
        raise ValueError('Provide either the inline instruction text or the *_file variant, not both.')
    if file_path:
        with open(file_path, 'r') as f:
            return f.read().strip()
    if text_value:
        return text_value.strip()
    return None


def _apply_instruction_overrides(priv_prompt, prompt_creator_instruction=None, info_aggregator_instruction=None):
    if prompt_creator_instruction:
        priv_prompt.prompt_creater.signature = priv_prompt.prompt_creater.signature.with_instructions(prompt_creator_instruction)
        fallback = priv_prompt._get_prompt_creater_fallback()
        fallback.signature = fallback.signature.with_instructions(prompt_creator_instruction)
    if info_aggregator_instruction:
        priv_prompt.info_aggregator.signature = priv_prompt.info_aggregator.signature.with_instructions(info_aggregator_instruction)


def _load_prompt_file_if_needed(priv_prompt, prompt_file):
    if prompt_file == 'NONE' or not prompt_file:
        return
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f'Prompt file not found: {prompt_file}')
    try:
        priv_prompt.load(prompt_file)
    except ValueError as e:
        if 'prior to v2.5.3' in str(e) or 'use_legacy_loading' in str(e):
            print('[INFO] Legacy DSPy prompt format detected; loading with use_legacy_loading=True')
            priv_prompt.load(prompt_file, use_legacy_loading=True)
        else:
            raise


def _build_pipeline(cfg):
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print('[WARN] OPENAI_API_KEY is empty. OpenRouter calls will likely fail.')

    local_lm = OllamaNoThinkLM(
        cfg['model_name'],
        ollama_base=cfg['ollama_base'],
        max_tokens=cfg['max_tokens'],
        cache=False,
    )
    dspy.configure(lm=local_lm)

    openai_lm = dspy.LM(
        f"openai/{cfg['openai_model']}" ,
        api_base=cfg['openrouter_base'],
        api_key=api_key,
        max_tokens=cfg['max_tokens'],
        cache=False,
    )
    opt_module.openai_lm_gpt4o = openai_lm

    priv_prompt = PrivacyOnePrompter(local_lm, openai_lm)
    _load_prompt_file_if_needed(priv_prompt, cfg['prompt_file'])
    _apply_instruction_overrides(
        priv_prompt,
        prompt_creator_instruction=cfg['prompt_creator_instruction'],
        info_aggregator_instruction=cfg['info_aggregator_instruction'],
    )
    return priv_prompt


def _get_thread_pipeline():
    pipeline = getattr(_THREAD_LOCAL, 'pipeline', None)
    if pipeline is None:
        pipeline = _build_pipeline(_WORKER_CONFIG)
        _THREAD_LOCAL.pipeline = pipeline
    return pipeline


def _evaluate_row(row_idx, row):
    row_start = time.time()
    result = {
        'row_idx': row_idx,
        'user_prompt': row.get('user_query', ''),
        'privacy_preserved_prompt': '',
        'proprietary_llm_response': '',
        'final_aggregated_response': '',
        'qual': None,
        'leak': None,
        'stage1_prompt_creation_sec': None,
        'stage2a_cloud_response_sec': None,
        'stage2b_local_aggregation_sec': None,
        'judge_evaluation_sec': None,
        'time_total_row_sec': None,
        'status': 'ok',
        'error': '',
    }

    if not isinstance(row.get('target_response'), str) or not isinstance(row.get('pii_units'), str):
        result['status'] = 'skipped_invalid_gold'
        result['error'] = 'Missing target_response or pii_units'
        result['time_total_row_sec'] = time.time() - row_start
        return result

    gold = Example({
        'target_response': row['target_response'],
        'user_query': row['user_query'],
        'pii_str': row['pii_units'],
    }).with_inputs('user_query')

    priv_prompt = _get_thread_pipeline()

    pred = None
    last_pipeline_exc = None
    for _ in range(3):
        try:
            pred = priv_prompt(row['user_query'])
            break
        except (litellm.exceptions.BadRequestError, ValueError, Exception) as exc:
            last_pipeline_exc = exc
            pred = None

    if pred is None:
        result['status'] = 'pipeline_failed'
        result['error'] = f'{type(last_pipeline_exc).__name__}: {str(last_pipeline_exc)[:300]}' if last_pipeline_exc else 'Unknown pipeline failure'
        result['time_total_row_sec'] = time.time() - row_start
        return result

    result['privacy_preserved_prompt'] = getattr(pred, 'prompt', '')
    result['proprietary_llm_response'] = getattr(pred, 'gptResponse', '')
    result['final_aggregated_response'] = getattr(pred, 'output', '')

    pipeline_timing = getattr(pred, 'timing', {}) or {}
    result['stage1_prompt_creation_sec'] = pipeline_timing.get('stage1_prompt_creation')
    result['stage2a_cloud_response_sec'] = pipeline_timing.get('stage2a_cloud_response')
    result['stage2b_local_aggregation_sec'] = pipeline_timing.get('stage2b_local_aggregation')

    judge_start = time.time()
    qual, leak = -1, -1
    last_metric_exc = None
    for _ in range(3):
        try:
            qual, leak = metric_finegrained(gold, pred)
            break
        except (ValueError, Exception) as exc:
            last_metric_exc = exc
            qual, leak = -1, -1
    result['judge_evaluation_sec'] = time.time() - judge_start
    result['time_total_row_sec'] = time.time() - row_start

    if qual == -1 and leak == -1:
        result['status'] = 'judge_failed'
        result['error'] = f'{type(last_metric_exc).__name__}: {str(last_metric_exc)[:300]}' if last_metric_exc else 'Unknown judge failure'
        return result

    result['qual'] = qual
    result['leak'] = leak
    return result


def _write_outputs(results, args, prompt_config, total_wall):
    ordered = sorted(results, key=lambda x: x['row_idx'])
    df = pandas.DataFrame(ordered)
    df.to_csv(args.output_file_name, index=False)

    timing_path = args.output_file_name.replace('.csv', '_timing.csv')
    timing_cols = [
        'row_idx',
        'stage1_prompt_creation_sec',
        'stage2a_cloud_response_sec',
        'stage2b_local_aggregation_sec',
        'judge_evaluation_sec',
        'time_total_row_sec',
        'status',
        'error',
    ]
    df[timing_cols].to_csv(timing_path, index=False)

    successes = df[df['qual'].notna() & df['leak'].notna()].copy()
    summary = {
        'model_name': args.model_name,
        'openai_model': args.openai_model,
        'data_file': args.data_file,
        'prompt_file': args.prompt_file,
        'total_rows': len(df),
        'evaluated_rows': int(len(successes)),
        'pipeline_failures': int((df['status'] == 'pipeline_failed').sum()),
        'judge_failures': int((df['status'] == 'judge_failed').sum()),
        'skipped_invalid_gold': int((df['status'] == 'skipped_invalid_gold').sum()),
        'avg_quality': float(successes['qual'].mean()) if len(successes) else None,
        'avg_leakage': float(successes['leak'].mean()) if len(successes) else None,
        'total_wall_time_sec': round(total_wall, 2),
        'prompt_creator_instruction': prompt_config['prompt_creator_instruction'],
        'info_aggregator_instruction': prompt_config['info_aggregator_instruction'],
    }
    if len(successes):
        summary['timing'] = {
            'mean_per_row_sec': round(float(successes['time_total_row_sec'].mean()), 2),
            'median_per_row_sec': round(float(successes['time_total_row_sec'].median()), 2),
            'std_per_row_sec': round(float(successes['time_total_row_sec'].std()), 2) if len(successes) > 1 else 0.0,
            'stage1_local_prompt_mean_sec': round(float(successes['stage1_prompt_creation_sec'].dropna().mean()), 3) if successes['stage1_prompt_creation_sec'].notna().any() else None,
            'stage2a_cloud_mean_sec': round(float(successes['stage2a_cloud_response_sec'].dropna().mean()), 3) if successes['stage2a_cloud_response_sec'].notna().any() else None,
            'stage2b_local_agg_mean_sec': round(float(successes['stage2b_local_aggregation_sec'].dropna().mean()), 3) if successes['stage2b_local_aggregation_sec'].notna().any() else None,
            'judge_mean_sec': round(float(successes['judge_evaluation_sec'].dropna().mean()), 3) if successes['judge_evaluation_sec'].notna().any() else None,
        }

    summary_path = args.output_file_name.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    prompt_config_path = args.output_file_name.replace('.csv', '_prompt_config.json')
    with open(prompt_config_path, 'w') as f:
        json.dump(prompt_config, f, indent=2)

    return summary_path, timing_path, prompt_config_path


def main():
    args = parse_args()
    prompt_creator_instruction = _load_text_arg(args.prompt_creator_instruction, args.prompt_creator_instruction_file)
    info_aggregator_instruction = _load_text_arg(args.info_aggregator_instruction, args.info_aggregator_instruction_file)

    df = pandas.read_csv(args.data_file)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    _WORKER_CONFIG.update({
        'model_name': args.model_name,
        'ollama_base': args.ollama_base,
        'max_tokens': args.max_tokens,
        'openai_model': args.openai_model,
        'openrouter_base': args.openrouter_base,
        'prompt_file': args.prompt_file,
        'prompt_creator_instruction': prompt_creator_instruction,
        'info_aggregator_instruction': info_aggregator_instruction,
    })

    reference_pipeline = _build_pipeline(_WORKER_CONFIG)
    prompt_config = {
        'model_name': args.model_name,
        'prompt_file': args.prompt_file,
        'prompt_creator_instruction': reference_pipeline.prompt_creater.signature.instructions,
        'info_aggregator_instruction': reference_pipeline.info_aggregator.signature.instructions,
    }

    print(f'[INFO] Evaluating model: {args.model_name}')
    print(f'[INFO] Data file: {args.data_file} ({len(df)} rows)')
    print(f'[INFO] Prompt file: {args.prompt_file}')
    print(f'[INFO] num_threads: {args.num_threads}')
    print(f'[INFO] Output CSV: {args.output_file_name}')

    results = []
    wall_start = time.time()
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {
            executor.submit(_evaluate_row, row_idx, row.to_dict()): row_idx
            for row_idx, row in df.iterrows()
        }
        progress = tqdm.tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True)
        for idx, future in enumerate(progress, start=1):
            results.append(future.result())
            elapsed = time.time() - wall_start
            avg_per_row = elapsed / idx if idx else 0.0
            remaining = len(futures) - idx
            eta_seconds = avg_per_row * remaining
            progress.set_postfix({
                'elapsed_min': f'{elapsed / 60:.1f}',
                'avg_s_row': f'{avg_per_row:.2f}',
                'eta_min': f'{eta_seconds / 60:.1f}',
            })

    total_wall = time.time() - wall_start
    summary_path, timing_path, prompt_config_path = _write_outputs(results, args, prompt_config, total_wall)

    ordered = sorted(results, key=lambda x: x['row_idx'])
    success_rows = [r for r in ordered if r['qual'] is not None and r['leak'] is not None]

    print('\n' + '=' * 60)
    print(f'EVALUATION COMPLETE — {args.model_name}')
    print(f'Dataset: {args.data_file} ({len(df)} rows)')
    print('=' * 60)
    print(f'  Evaluated rows:      {len(success_rows)}/{len(df)}')
    print(f'  Pipeline failures:   {sum(r["status"] == "pipeline_failed" for r in ordered)}')
    print(f'  Judge failures:      {sum(r["status"] == "judge_failed" for r in ordered)}')
    if success_rows:
        avg_q = sum(r['qual'] for r in success_rows) / len(success_rows)
        avg_l = sum(r['leak'] for r in success_rows) / len(success_rows)
        mean_row = sum(r['time_total_row_sec'] or 0 for r in success_rows) / len(success_rows)
        print(f'  AVERAGE QUALITY:     {avg_q:.4f}')
        print(f'  AVERAGE LEAKAGE:     {avg_l:.4f}')
        print(f'  Mean time per row:   {mean_row:.2f}s')
    else:
        print('  WARNING: No rows were successfully evaluated!')
    print(f'  Total wall time:     {total_wall:.1f}s ({total_wall/60:.1f}min)')
    print(f'\nSummary saved to: {summary_path}')
    print(f'Timing CSV saved to: {timing_path}')
    print(f'Prompt config saved to: {prompt_config_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
