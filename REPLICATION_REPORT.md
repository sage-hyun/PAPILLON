# Replication Report: PAPILLON

**Paper**: PAPILLON: PrivAcy Preservation from Internet-based and Local Language MOdel ENsembles  
**Authors**: Vithursan Thangarasa et al.  
**Link**: [arXiv:2410.17127](https://arxiv.org/abs/2410.17127)

---

## 1. How Did We Replicate the Original Paper?

### 1.1 Overview of the PAPILLON System

PAPILLON is a two-stage privacy-preserving pipeline that enables users to leverage powerful cloud-based LLMs (e.g., GPT-4o-mini) without exposing their personally identifiable information (PII). The system consists of:

1. **Stage 1 — Privacy-Preserving Prompt Creation**: A local *trusted* model (e.g., Llama-3.1-8B) reads the user's original query containing PII and generates a sanitized prompt that removes sensitive information while preserving the intent.
2. **Stage 2 — Response Aggregation**: The sanitized prompt is sent to an *untrusted* cloud model (GPT-4o-mini), which generates a response. The trusted local model then aggregates this response with the original user query to produce a final, complete answer.

The pipeline is optimized using DSPy's MIPROv2 teleprompter, which automatically tunes the system prompts for both stages. Evaluation uses an LLM-based judge that measures two metrics:
- **Quality Score** (0 or 1): Whether PAPILLON's response is at least as good as a direct response.
- **Leakage Score** (0 to 1): The fraction of PII units that appear in the sanitized prompt sent to the untrusted model.

### 1.2 Replication Strategy

We used the authors' official codebase from the `v1.0` branch, which contains:
- The core PAPILLON pipeline (`run_llama_dspy.py`)
- Pre-optimized prompts for 6 model configurations
- The LLM judge evaluation framework (`llm_judge.py`, `evaluate_papillon.py`)
- The PUPA benchmark datasets (`PUPA_New.csv`, `PUPA_TNB.csv`)

Our replication focused on **evaluating the pre-optimized PAPILLON pipeline** using the provided prompts, rather than re-running the full MIPROv2 optimization (which is extremely time-consuming and API-costly). This is equivalent to evaluating the authors' trained model on the same test data.

### 1.3 Replication Configuration

| Component | Original Paper | Our Replication |
|-----------|---------------|-----------------|
| **Trusted Model** | Llama-3.1-8B-Instruct via SGLang | Llama-3.1-8B-Instruct via Ollama (FP16) |
| **Untrusted Model** | GPT-4o-mini via OpenAI API | GPT-4o-mini via OpenRouter |
| **LLM Judge** | GPT-4o-mini via OpenAI API | GPT-4o-mini via OpenRouter |
| **Datasets** | PUPA-New (664 rows), PUPA-TNB (237 rows) | Same |
| **DSPy Version** | 2.5.35 | 2.4.16 |
| **Optimized Prompts** | Authors' MIPROv2-optimized prompts | Same (loaded from provided JSON files) |
| **Hardware** | Not specified | 3x NVIDIA RTX 4090 (24GB each) |

Key differences from the original:
- **Model serving**: We used Ollama instead of SGLang for local model hosting. Both provide OpenAI-compatible APIs, so the model behavior should be identical for the same weights.
- **API routing**: We accessed GPT-4o-mini through OpenRouter rather than the OpenAI API directly. OpenRouter proxies to the same OpenAI models, so responses should be equivalent.
- **DSPy version**: We used 2.4.16 instead of 2.5.35, which required minor code adaptations (see Section 2).

---

## 2. Problems Faced During Replication and Solutions

### Problem 1: OpenAI API Key Incompatibility

**Issue**: The provided API key (`sk-or-v1-...`) was an OpenRouter key, but the original code was hardcoded to use the OpenAI API directly via `dspy.OpenAI(model="gpt-4o-mini")`.

**Attempted solutions**:
1. Direct OpenAI connection → `401 AuthenticationError` (key format mismatch)
2. OpenRouter routing → Initially `401 User not found` (expired key)

**Final solution**: Obtained a valid OpenRouter key and modified all code to route GPT-4o-mini calls through OpenRouter's API (`https://openrouter.ai/api/v1`).

### Problem 2: `dspy.OpenAI` vs `dspy.LM` Interface Mismatch

**Issue**: The original code used `dspy.OpenAI` for the untrusted model, but this class failed with Ollama's OpenAI-compatible endpoint (`404 Not Found`). Investigation revealed that `dspy.OpenAI` internally modifies the `openai` Python module's global state, which conflicted with Ollama's endpoint format.

**Solution**: Replaced all `dspy.OpenAI(...)` calls with `dspy.LM('openai/...', api_base=..., api_key=...)`. This required an additional change in `run_llama_dspy.py` because `dspy.LM` uses a messages-based calling convention (`lm(messages=[...])`) while `dspy.OpenAI` uses a prompt-based one (`lm(prompt_string)`). We added a type check:

```python
if isinstance(self.untrusted_model, dspy.LM):
    response = self.untrusted_model(messages=[{"role": "user", "content": prompt.createdPrompt}])[0]
else:
    response = self.untrusted_model(prompt.createdPrompt)[0]
```

### Problem 3: Package Version Conflicts

**Issue**: The conda environment was missing `litellm`, and installing it triggered a cascade of version conflicts:
- `litellm` latest required `openai>=2.0`, but `dspy-ai 2.4.16` required `openai<2.0`
- `httpx 0.28.1` removed the `proxies` parameter that `openai 1.x` relied on

**Solution**: Pinned specific compatible versions:
```bash
pip install litellm==1.51.0 openai==1.55.0 httpx==0.27.0
```

### Problem 4: DSPy API Changes Between Versions

**Issue**: The code called `priv_prompt.load(path, use_legacy_loading=True)`, but DSPy 2.4.16's `Module.load()` only accepts `path` — the `use_legacy_loading` parameter does not exist.

**Solution**: Removed the `use_legacy_loading=True` argument. The JSON format of the optimized prompts was compatible with the standard `load()` method.

### Problem 5: Global Variable Scoping Bug

**Issue**: `evaluate_papillon.py` imports `metric_finegrained` from `run_dspy_optimization_llama.py`. This function references the global variable `openai_lm_gpt4o`, but that variable was only defined inside the `if __name__ == "__main__"` block. When imported as a module, `openai_lm_gpt4o` was undefined, causing `NameError`.

**Solution**: 
1. Added `openai_lm_gpt4o = None` at module level in `run_dspy_optimization_llama.py`
2. Set it from `evaluate_papillon.py` before calling `metric_finegrained`:
```python
run_dspy_optimization_llama.openai_lm_gpt4o = judge_lm
```

### Problem 6: Original Code Bug — Typo in Parameter Name

**Issue**: In `run_dspy_optimization_llama.py`, the `metric_finegrained()` function passed `ppi_str=og_pii` instead of `pii_str=og_pii` (note the typo: `ppi` vs `pii`). Since `LLMJudge.forward()` expects `pii_str`, the misspelled keyword was silently ignored, causing the leakage score to always return -1.

**Solution**: Fixed the typo from `ppi_str` to `pii_str`. This is a bug in the original codebase that may have affected the authors' own fine-grained evaluation results.

### Problem 7: DataFrame Length Mismatch in Evaluation

**Issue**: The evaluation script maintained separate lists for `qual_scores` and `leak_scores`. When quality was valid but leakage was -1 (or vice versa), the lists grew to different lengths, causing a `ValueError` when constructing the output DataFrame.

**Solution**: Changed the logic to only append scores when both quality and leakage are valid (`qual != -1 and leak != -1`).

### Problem 8: LLM Output Parsing Failures

**Issue**: For some data rows, the Llama-3.1-8B model produced outputs that DSPy's `ChatAdapter` could not parse (e.g., `ValueError: Expected dict_keys(['output']) but got dict_keys([])`). This occurred in approximately 3-5% of rows.

**Solution**: Added try/except error handling around both the pipeline execution and the metric computation in the evaluation loop, allowing the script to skip problematic rows and continue.

---

## 3. Experiment Setting

### 3.1 Datasets

| Dataset | Rows | Categories | Description |
|---------|------|------------|-------------|
| **PUPA-New** | 664 | 4 (job applications, financial info, healthcare, emails) | Primary evaluation dataset |
| **PUPA-TNB** | 237 | 4 (same categories) | Secondary evaluation dataset |

Each row contains:
- `user_query`: Original user query containing PII
- `target_response`: Ideal response (ground truth)
- `pii_units`: PII elements separated by `||` (avg. 2.7 per row)
- `redacted_query`: LLM-generated redacted version of the query

For optimization, PUPA-New is split: Train (rows 0–149), Validation (150–299), Test (300+).

### 3.2 Models Under Evaluation

The authors provide pre-optimized prompts for 6 local model configurations, all paired with GPT-4o-mini as the untrusted cloud model:

| Local (Trusted) Model | Parameters | Optimized Prompt File |
|-----------------------|------------|----------------------|
| Llama-3.1-8B-Instruct | 8B | `llama_31_8b_instruct_prompt.json` |
| Llama-3.2-3B-Instruct | 3B | `llama_32_3b_instruct_prompt.json` |
| Llama-3.2-1B-Instruct | 1B | `llama_32_1b_instruct_prompt.json` |
| Llama-3-8B-Instruct | 8B | `llama_3_8b_instruct_prompt.json` |
| Mistral-7B-Instruct | 7B | `mistral_7b_instruct_prompt.json` |
| Mistral-Small | 22B | `mistral_small_prompt.json` |

### 3.3 Evaluation Metrics

Two metrics are computed by the LLM judge (GPT-4o-mini):

1. **Quality Score**: Binary (0 or 1). Two symmetric comparisons ask whether PAPILLON's response is "at least as good as" the target response. If both agree, score = 1; otherwise, the first judgment is used.

2. **Leakage Score**: Continuous (0 to 1). For each PII unit in the original query, the judge checks whether it appears in the sanitized prompt. The score is the fraction of leaked PII units: `leaked_count / total_pii_units`.

**Ideal outcome**: Quality = 1.0 (responses are as good as direct responses) and Leakage = 0.0 (no PII leaked to the cloud model).

### 3.4 Evaluation Command

```bash
export OPENAI_API_KEY="<OPENROUTER_KEY>"
cd papillon/

# Evaluate Llama-3.1-8B-Instruct on PUPA-New
python evaluate_papillon.py \
  --port 11434 \
  --data_file ../pupa/PUPA_New.csv \
  --model_name "llama3.1:8b-instruct-fp16" \
  --openai_model "gpt-4o-mini" \
  --output_file_name eval_llama31_8b_PUPA_New.csv

# Evaluate on PUPA-TNB
python evaluate_papillon.py \
  --port 11434 \
  --data_file ../pupa/PUPA_TNB.csv \
  --model_name "llama3.1:8b-instruct-fp16" \
  --openai_model "gpt-4o-mini" \
  --output_file_name eval_llama31_8b_PUPA_TNB.csv
```

---

## 4. Results

### 4.1 Baseline Implementation Verification

Before running full-scale evaluation, we verified the PAPILLON pipeline works correctly with a single qualitative test:

**Input query**: *"Write a cover letter for John Smith applying for a software engineer position at Google"*

| Stage | Output |
|-------|--------|
| **Privacy-preserving prompt** (sent to GPT-4o-mini) | *"Write a cover letter for a software engineer applying for a position at Google."* |
| **GPT-4o-mini response** | Generic cover letter without personal details |
| **Final PAPILLON output** | Complete cover letter with "John Smith" restored, incorporating GPT-4o-mini's writing quality |

This confirms the pipeline correctly: (1) strips PII ("John Smith") from the prompt, (2) obtains a high-quality response from the cloud model, and (3) re-personalizes the response locally.

### 4.2 Partial Replication Results — Llama-3.1-8B-Instruct

Evaluation is currently in progress. Results so far:

| Dataset | Rows Evaluated | Total Rows | Quality | Leakage |
|---------|---------------|------------|---------|---------|
| **PUPA-New** | 192 | 664 | **0.932** | **0.194** |
| **PUPA-TNB** | 86 | 237 | **0.919** | **0.155** |

**Quality Score Breakdown (PUPA-New, 192 rows)**:
- Score = 1 (response as good as target): **179 rows (93.2%)**
- Score = 0 (response worse than target): **13 rows (6.8%)**

**Interpretation**:
- The high quality score (0.93) indicates that PAPILLON successfully preserves response quality even after privacy sanitization. Over 93% of responses match or exceed the quality of direct (non-private) responses.
- The leakage score of ~0.19 means approximately 19% of PII units still appear in the sanitized prompt. While not zero, this represents a significant reduction from the baseline of 100% leakage (sending the original query directly).
- Rows that failed evaluation (~3-5% skipped due to parsing errors) are excluded from these statistics.

### 4.3 Files Produced

| Output File | Description | Status |
|-------------|-------------|--------|
| `papillon/eval_llama31_8b_PUPA_New.csv` | Llama-3.1-8B evaluation on PUPA-New | In progress (192/664) |
| `papillon/eval_llama31_8b_PUPA_TNB.csv` | Llama-3.1-8B evaluation on PUPA-TNB | In progress (86/237) |
| `papillon/eval_gpt4o_test10.csv` | Small-scale GPT-4o-mini test (10 rows) | Complete |
| `papillon/eval_test10_output.csv` | Small-scale local-only test (10 rows) | Complete |

---

## 5. Plan for Completing the Remaining Experiments

### 5.1 Current Evaluation (In Progress)

The Llama-3.1-8B-Instruct evaluation on both datasets is running concurrently. At the current rate of ~2 rows/minute per dataset:
- **PUPA-New**: ~3.5 hours remaining (472 rows left)
- **PUPA-TNB**: ~1.3 hours remaining (151 rows left)

### 5.2 Additional Model Evaluations (Planned)

After the 8B evaluation completes, we will sequentially evaluate the remaining models using GPUs 0, 1, 2:

| Priority | Model | Dataset | Est. Time | Status |
|----------|-------|---------|-----------|--------|
| 1 | Llama-3.2-1B-Instruct | PUPA-New + PUPA-TNB | ~3h | Model downloaded, waiting |
| 2 | Llama-3.2-3B-Instruct | PUPA-New + PUPA-TNB | ~3h | Model downloaded, waiting |
| 3 | Llama-3-8B-Instruct | PUPA-New + PUPA-TNB | ~4h | Needs download |
| 4 | Mistral-7B-Instruct | PUPA-New + PUPA-TNB | ~4h | Needs download |
| 5 | Mistral-Small (22B) | PUPA-New + PUPA-TNB | ~5h | Needs download, may not fit in GPU memory |

**Total estimated time for all remaining evaluations**: ~19 hours (sequential on shared GPUs)

### 5.3 Prompt Optimization (Optional)

The full MIPROv2 prompt optimization (`run_dspy_optimization_llama.py`) can be run to verify that the optimization process reproduces similar prompts:

```bash
python run_dspy_optimization_llama.py \
  --port 11434 \
  --openai_model "gpt-4o-mini" \
  --prompt_output "new_optimized_prompt.json" \
  --data_file "../pupa/PUPA_New.csv"
```

This is significantly more time-consuming (200 optimization batches x validation set) and API-costly, but would provide a more complete replication of the paper's methodology.

---

## 6. Code Modifications Summary

All modifications were necessary to adapt the codebase to our environment (Ollama + OpenRouter) while preserving the original semantics:

| File | Changes Made | Reason |
|------|-------------|--------|
| `evaluate_papillon.py` | Replaced `dspy.OpenAI` with `dspy.LM`, added OpenRouter routing, added error handling, removed `use_legacy_loading`, set global judge LM | Environment compatibility |
| `run_dspy_optimization_llama.py` | Replaced `dspy.OpenAI` with `dspy.LM`, added OpenRouter routing, fixed `ppi_str` → `pii_str` typo, added module-level `openai_lm_gpt4o` init | Environment compatibility + bug fix |
| `run_papillon_interactive.py` | Replaced `dspy.OpenAI` with `dspy.LM`, added OpenRouter routing, removed `use_legacy_loading` | Environment compatibility |
| `run_llama_dspy.py` | Added `dspy.LM` messages-based calling support in `forward()` | Interface compatibility |

---

## 7. Key Takeaways

1. **The PAPILLON pipeline works as described**: Our qualitative and quantitative tests confirm that the two-stage privacy-preserving architecture successfully strips PII from prompts while maintaining response quality.

2. **Quality is high, leakage is non-trivial**: With Llama-3.1-8B as the trusted model, we observe ~93% quality preservation but ~19% PII leakage. This suggests that smaller local models may struggle to completely sanitize all PII, particularly for complex queries with many PII elements.

3. **The codebase required non-trivial adaptation**: Despite being a v1.0 release, 7 distinct issues needed resolution before the evaluation pipeline could run. The most critical was a typo bug (`ppi_str` vs `pii_str`) in the original code that would have caused incorrect leakage measurements.

4. **Evaluation is computationally bottlenecked by API calls**: Each row requires ~6 GPT-4o-mini API calls (1 untrusted + 2 quality + N leakage + 1 prompt quality), making the evaluation time approximately 25-30 seconds per row regardless of local model size.
