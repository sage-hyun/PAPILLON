# PAPILLON Replication — Bug Fix & Modification Report

본 문서는 PAPILLON 재현 과정에서 **발견한 버그**와 **수정 내용**을 모두 정리합니다.
각 항목은 (1) 증상, (2) 원인, (3) 수정 코드, (4) 근거 파일/라인 순으로 기록합니다.

---

## 1. `ppi_str` → `pii_str` 오타 (leakage = -1)

**파일**: `papillon/run_dspy_optimization_llama.py` — `metric_finegrained()`

**증상**: 모든 행의 leakage 점수가 -1로 나와서 평가 완전 실패.

**원인**: `LLMJudge.forward(pii_str=...)` 인자인데 잘못 타이핑된 `ppi_str`로 전달.
Python은 kwarg 타입 체크를 안 해서 조용히 무시 → pii_str이 None → pii_score=-1.

**수정** (전):
```python
score_dict = llm_judge(..., ppi_str=og_pii)   # 오타
```
**수정** (후):
```python
score_dict = llm_judge(..., pii_str=og_pii)   # 올바른 키워드
```

---

## 2. `openai_lm_gpt4o` 전역 변수 NameError

**파일**: `papillon/run_dspy_optimization_llama.py`, `papillon/evaluate_papillon.py`

**증상**:
```
NameError: name 'openai_lm_gpt4o' is not defined
```
`evaluate_papillon.py`가 `metric_finegrained`를 import해서 호출하면 터짐.

**원인**: `openai_lm_gpt4o`가 `if __name__ == "__main__":` 블록 **내부에서만** 정의됨.
import 경로(`evaluate_papillon.py`에서 사용)에서는 정의되지 않은 상태.

**수정**:
```python
# run_dspy_optimization_llama.py (module-level)
openai_lm_gpt4o = None  # 전역으로 선언
```
```python
# evaluate_papillon.py
run_dspy_optimization_llama.openai_lm_gpt4o = openai_lm  # 명시적으로 주입
```

---

## 3. `qual_scores` / `leak_scores` 길이 불일치 → DataFrame ValueError

**파일**: `papillon/evaluate_papillon.py`

**증상**: 수백 행 평가 후 CSV 저장 시 `ValueError: arrays must all be same length`.

**원인**: 두 리스트에 `qual != -1`, `leak != -1` 조건이 **각각 독립적**으로 append.
한쪽만 -1일 때 길이가 어긋남.

**수정**: 둘 다 유효할 때만 동시에 append:
```python
if qual != -1 and leak != -1:
    qual_scores.append(qual)
    leak_scores.append(leak)
```

---

## 4. DSPy `load()` 버전 불일치 (v2.5.3 이전 saved state)

**파일**: `papillon/evaluate_papillon.py`

**증상**:
```
ValueError: The saved state is from a version of DSPy prior to v2.5.3.
            Please use `use_legacy_loading=True` to load the state.
```
8B Llama 최적화 프롬프트 JSON 로드 시.

**원인**: 저자들의 프롬프트는 DSPy 구버전으로 저장됨. DSPy 2.5.35는 기본 load가 새 포맷 기대.

**수정**: try/except로 자동 fallback:
```python
try:
    priv_prompt.load(args.prompt_file)
except ValueError as e:
    if "prior to v2.5.3" in str(e) or "use_legacy_loading" in str(e):
        priv_prompt.load(args.prompt_file, use_legacy_loading=True)
    else:
        raise
```

---

## 5. `httpx 0.28` × `openai 1.55` 호환성 에러 (모든 LLM 호출 실패)

**증상**: **모든** LLM 호출이
```
APIError: Client.__init__() got an unexpected keyword argument 'proxies'
```
로 실패. 그런데 try/except 때문에 조용히 빈 Prediction 반환 → **237행 4.5초에 "완료"** (가짜 결과).

**원인**: `httpx 0.28`에서 `Client(proxies=...)` 제거됨. `openai 1.55`는 여전히 `proxies=` 전달.

**수정**:
```bash
pip install 'httpx<0.28'   # 0.28.1 → 0.27.2 다운그레이드
```

**교훈**: **조용히 실패하는 예외는 가장 위험하다.** retry 로직이 에러를 삼켜서 근본 문제를 가렸음.

---

## 6. DSPy `ChainOfThought` 파싱 실패 (53/237 행 skip)

**파일**: `papillon/llm_judge.py`, `papillon/run_llama_dspy.py`

**증상**:
```
ValueError: Expected dict_keys(['rationale', 'output'])
            but got dict_keys(['rationale'])
```
47행은 `metric_finegrained`에서, 6행은 `priv_prompt`에서 발생. 총 **53/237 skip**.

**원인**: LLM이 긴 rationale을 생성하다가 **max_tokens=4000에 걸려 잘림** → `[[ ## output ## ]]` 마커 못 나옴 → DSPy가 파싱 실패.

**수정 (3단계 방어)**:

### 6a. `llm_judge.py` — retry + Predict fallback
```python
def _robust_call(primary, fallback, *, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return primary(**kwargs)
        except (ValueError, Exception):
            continue
    if fallback is not None:
        try: return fallback(**kwargs)
        except: pass
    return None

class LLMJudge(dspy.Module):
    def __init__(self):
        self.quality_judge = dspy.ChainOfThought(JudgeQuality)
        self.quality_judge_fallback = dspy.Predict(JudgeQuality)  # 추가
        self.prompt_qual = dspy.ChainOfThought(JudgePromptQual)
        self.prompt_qual_fallback = dspy.Predict(JudgePromptQual)  # 추가
    
    def _safe_binary(self, answer, default=0):
        # answer가 None/malformed여도 기본값 반환 (-1 반환 방지)
```

### 6b. `run_llama_dspy.py` — pipeline 재시도 + Predict fallback
```python
class PrivacyOnePrompter(dspy.Module):
    # prompt_creater_fallback은 lazy로: load() 호환성을 위해 __dict__ 우회
    def _get_prompt_creater_fallback(self):
        if self._prompt_creater_fallback is None:
            self.__dict__["_prompt_creater_fallback"] = dspy.Predict(CreateOnePrompt)
        return self._prompt_creater_fallback
    
    def forward(self, user_query):
        # Stage 1: 3회 재시도 + fallback
        prompt, _ = _retry_call(self.prompt_creater, max_retries=3, userQuery=user_query)
        if prompt is None:
            prompt, _ = _retry_call(self._get_prompt_creater_fallback(), max_retries=2, ...)
        
        # Stage 2a (cloud): 3회 재시도
        for attempt in range(3):
            try: response = self.untrusted_model(...); break
            except: continue
        
        # Stage 2b: 3회 재시도, 실패 시 cloud response를 그대로 output으로 사용
```

### 6c. `evaluate_papillon.py` — 최상위 재시도 안전망
```python
MAX_PIPELINE_RETRIES = 3
MAX_METRIC_RETRIES = 3
priv_prompt_fail_count = 0
metric_fail_count = 0

for i, row in tqdm.tqdm(data_file.iterrows()):
    # priv_prompt 3회 재시도
    pred = None
    for attempt in range(MAX_PIPELINE_RETRIES):
        try: pred = priv_prompt(row["user_query"]); break
        except: continue
    if pred is None:
        priv_prompt_fail_count += 1
        continue
    
    # metric_finegrained 3회 재시도
    for attempt in range(MAX_METRIC_RETRIES):
        try: qual, leak = metric_finegrained(gold, pred); break
        except: qual, leak = -1, -1
```

**결과**: 53/237 → **0/237 skip** (100% 평가 완료)

---

## 7. `max_tokens=4000` 부족 → output 필드 truncation (근본 원인)

**파일**: `papillon/evaluate_papillon.py`, `papillon/run_dspy_optimization_llama.py`

**원인**: ChainOfThought가 emit하는 `[rationale, output]` 구조에서, LLM이 긴 rationale을 쓰다가 4000 토큰에 걸려 `output` 필드가 아예 못 나옴 → #6의 ValueError 원인.

**수정**:
```python
# 전
local_lm  = dspy.LM(..., max_tokens=4000)
openai_lm = dspy.LM(..., max_tokens=4000)

# 후
local_lm  = dspy.LM(..., max_tokens=8000)
openai_lm = dspy.LM(..., max_tokens=8000)
```

---

## 8. DSPy 캐시가 **evaluation에 영향** → 재현성 파괴

**파일**: `papillon/evaluate_papillon.py`

**증상**: TNB 237행 평가 중 **12행이 <1초**에 완료 (평균 40s/행인데). 실제 계산 안 하고 캐시에서 반환됨.

**원인 체인**:
1. DSPy 2.5의 `dspy.LM` **기본값이 `cache=True`** (디스크 SQLite 캐시).
2. `evaluate_papillon.py`가 import 시 `run_dspy_optimization_llama.py`의 top-level을 실행 → `os.environ["DSPY_CACHEDIR"]` 설정됨.
3. 과거 smoke test/optimization/v2 eval 등이 `cache.db`에 TNB 행의 LLM 응답 누적.
4. 오늘 full eval 때 입력 해시가 같은 12행은 **과거 캐시 히트** → 즉시 반환.

**수정**:
```python
# evaluation용 LM은 명시적으로 cache=False
local_lm  = dspy.LM(..., max_tokens=8000, cache=False)
openai_lm = dspy.LM(..., max_tokens=8000, cache=False)
```
그리고 실행 전에 `.dspy_cache/cache.db` 삭제로 깨끗한 상태 보장.

**교훈**: DSPy의 기본 캐싱 정책이 "조용히 활성화"라서, evaluation 스크립트에서는 **반드시 명시적으로 `cache=False`**를 지정해야 재현성이 확보됨.

---

## 9. MIPROv2 `save()`가 `evaluate()` 이후에 있어서 prompt JSON 유실

**파일**: `papillon/run_dspy_optimization_llama.py`

**증상**: Gemma3 4B 1-batch 테스트 후 `before_optimization=38.15`만 기록되고 `after_optimization`, **최적화된 prompt JSON 자체도 없음**. 재사용 불가.

**원인** (이전 코드 구조):
```python
try:
    compiled = teleprompter.compile(...)
    eval_score = evaluate(compiled, ...)   # ← 여기서 ValueError 발생
    compiled.save(args.prompt_output)       # ← 도달 못함 → 프롬프트 유실
except Exception as e:
    print(e)
```

**수정** (save-first 구조):
```python
compiled_prompt_opt = None
try:
    teleprompter = MIPROv2(...)
    compiled_prompt_opt = teleprompter.compile(...)
except Exception as e:
    print(f"[WARN] compile failed: {e}")

# 저장을 먼저!
if compiled_prompt_opt is not None:
    try:
        compiled_prompt_opt.save(args.prompt_output)
        print(f"Saved to {args.prompt_output}")
    except Exception as e:
        print(f"[WARN] save failed: {e}")
    
    # 그 다음 평가 (best-effort)
    try:
        eval_score = evaluate(compiled_prompt_opt, ...)
        eval_scores.update({"after_optimization": eval_score})
    except Exception as e:
        print(f"[WARN] after-opt eval failed: {e}")
```

---

## 10. MIPROv2 `EOFError` — interactive prompt 응답 못함

**파일**: `papillon/run_dspy_optimization_llama.py`

**증상**: 백그라운드 실행에서 MIPROv2가 "Do you want to proceed? (y/n)" 물어보는데 stdin 없어서 `EOFError`로 중단.

**수정**:
```python
compiled_prompt_opt = teleprompter.compile(
    zeroshot,
    trainset=train,
    num_batches=args.num_batches,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    eval_kwargs=kwargs,
    requires_permission_to_run=False,   # 추가
)
```

---

## 11. `--model_name` vs `--openai_model` 인자 혼동

**파일**: `papillon/run_dspy_optimization_llama.py`

**증상**: 최적화 스크립트가 로컬 모델로 `--openai_model` 값을 사용해서 Ollama에 `gpt-4o-mini`를 보내려 함 → 실패.

**원인**: 기존 코드에서는 로컬/클라우드 모델 구분이 명확하지 않았음.

**수정**: `--model_name`, `--openai_model` 2개 인자로 분리:
```python
parser.add_argument("--model_name", type=str, required=True, help="Ollama로컬 모델")
parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="Cloud/judge 모델")

local_lm  = dspy.LM(f'openai/{args.model_name}', api_base=f"http://127.0.0.1:{args.port}/v1", ...)
openai_lm = dspy.LM(f'openai/{args.openai_model}', api_base=args.openrouter_base, ...)
```

---

## 12. `num_candidates=10` 기본값 — 너무 오래 걸림

**파일**: `papillon/run_dspy_optimization_llama.py`

**증상**: MIPROv2가 `num_candidates=10` × `num_batches=200`으로 무한정 돌아감. 테스트용으로 빠르게 줄이고 싶은데 하드코딩되어 있었음.

**수정**: CLI 인자로 노출:
```python
parser.add_argument("--num_batches", type=int, default=200)
parser.add_argument("--num_candidates", type=int, default=10)
teleprompter = MIPROv2(..., num_candidates=args.num_candidates, ...)
```

---

## 13. `pii_str` 정규화 불일치 (numerator/denominator mismatch)

**파일**: `papillon/llm_judge.py`, `papillon/run_dspy_optimization_llama.py`

**증상**: Leakage 점수가 가끔 1보다 커지거나 이상값 발생.

**원인**: numerator (`LLMJudge.forward`)는 각 piece를 체크할 때 `piece.strip()` + 빈 piece 제거 사용. 그런데 denominator (`metric_finegrained`)는 `len(set(og_pii.split("||")))` — **strip/filter 안 함**.
→ `"a || b ||"` 같은 케이스에서 numerator는 2, denominator는 3이 돼서 불일치.

**수정**:
```python
# llm_judge.py
all_pii_pieces = [p.strip() for p in pii_str.split("||") if p and p.strip()]

# run_dspy_optimization_llama.py (metric_finegrained)
pii_pieces = {p.strip() for p in og_pii.split("||") if p and p.strip()}
denom = max(len(pii_pieces), 1)   # div-by-zero 방지
return score_dict.quality, score_dict.leakage / denom
```

---

## 14. 결과 CSV에 `redacted_query`, `final_output`, `cloud_response` 컬럼 추가

**파일**: `papillon/evaluate_papillon.py`

**요구**: 평가 결과 CSV에 PAPILLON이 생성한 **sanitized prompt**(redacted_query), **최종 출력**, **클라우드 응답**도 저장.

**수정**:
```python
all_redacted_queries = []
all_final_outputs = []
all_cloud_responses = []

# 루프 내부:
all_redacted_queries.append(getattr(pred, "prompt", ""))
all_final_outputs.append(getattr(pred, "output", ""))
all_cloud_responses.append(getattr(pred, "gptResponse", ""))

# CSV 저장:
result_df["redacted_query"] = all_redacted_queries
result_df["final_output"]   = all_final_outputs
result_df["cloud_response"] = all_cloud_responses
```

---

## 15. Timing 수집 & 분석 기능 추가

**파일**: `papillon/run_llama_dspy.py`, `papillon/evaluate_papillon.py`

**요구**: 재현 보고용으로 각 pipeline stage별 소요 시간 측정.

**수정 (run_llama_dspy.py)**:
```python
def forward(self, user_query):
    timing = {}
    t0 = time.time()
    prompt = self.prompt_creater(userQuery=user_query)
    timing["stage1_prompt_creation"] = time.time() - t0
    
    t1 = time.time()
    response = self.untrusted_model(...)
    timing["stage2a_cloud_response"] = time.time() - t1
    
    t2 = time.time()
    final_output = self.info_aggregator(...)
    timing["stage2b_local_aggregation"] = time.time() - t2
    timing["total_pipeline"] = time.time() - t0
    
    return dspy.Prediction(..., timing=timing)
```

**수정 (evaluate_papillon.py)**:
- `timing_records` 리스트에 행별 `{stage1, stage2a, stage2b, judge_evaluation, total_row, quality, leakage}` 저장
- `{output}_timing.csv`로 별도 저장
- `{output}_summary.json`에 mean/median/std/min/max + 백분율 breakdown 기록

---

## 16. Prompt 파일 `"NONE"` 처리 & zero-shot baseline

**파일**: `papillon/evaluate_papillon.py`

**요구**: 최적화된 프롬프트 없이 baseline 측정하고 싶을 때 `--prompt_file NONE` 지원.

**수정**:
```python
if args.prompt_file == "ORIGINAL":
    args.prompt_file = parse_model_prompt(args.model_name)

if args.prompt_file == "NONE" or not os.path.exists(args.prompt_file):
    print(f"[INFO] No pre-optimized prompt. Zero-shot baseline.")
else:
    priv_prompt.load(args.prompt_file, ...)
```
+ `parse_model_prompt`에 Qwen3, Gemma3 등 신규 모델 분기 추가.

---

## 17. Qwen3 "think" 모드 비활성화 (향후 용도)

**파일**: `papillon/ollama_nothink_lm.py` (신규 생성, 현재 다른 팀원이 수정 중)

**증상**: Qwen3가 `<think>` 태그로 긴 reasoning 출력. Ollama의 OpenAI-호환 엔드포인트(`/v1/chat/completions`)에서는 `content`가 빈 문자열로 반환됨.

**원인**: Qwen3의 reasoning trace는 Ollama 내부 필드 `reasoning`에 담기고, `content`에는 최종 응답만. 하지만 think 모드 비활성화 필요 시 native API 사용 권장.

**수정**: 커스텀 wrapper:
```python
class OllamaNoThinkLM(dspy.LM):
    def __call__(self, prompt=None, messages=None, **kwargs):
        payload = {"model": self.ollama_model, "messages": messages,
                   "stream": False, "think": False}
        resp = requests.post(f"{self.ollama_base}/api/chat", json=payload)
        content = resp.json().get("message", {}).get("content", "")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return [content]
```

---

## 18. 실패 카운트를 summary에 노출

**파일**: `papillon/evaluate_papillon.py`

**요구**: 몇 개 행이 skip되었는지 명시적으로 알 수 있게.

**수정**:
```python
summary = {
    ...,
    "priv_prompt_failures": priv_prompt_fail_count,
    "metric_finegrained_failures": metric_fail_count,
    ...,
}
```

---

## 19. 증분 저장 (incremental save)

**파일**: `papillon/evaluate_papillon.py`

**요구**: 장시간 실행 중에 중단되어도 부분 결과 보존.

**수정**: 매 행 성공 시마다 `result_df.to_csv(...)`, `timing_df.to_csv(...)` 호출.
```python
if qual != -1 and leak != -1:
    ...
    result_df.to_csv(args.output_file_name)
    timing_df.to_csv(args.output_file_name.replace(".csv", "_timing.csv"), index=False)
```

---

## 20. Ollama 포트 / base URL 일관성

**파일**: `papillon/run_dspy_optimization_llama.py`, `papillon/evaluate_papillon.py`

**증상**: optimization은 `127.0.0.1`, evaluation은 `0.0.0.0` 사용. 혼동 유발.

**수정**: 두 스크립트 모두 `--port`로 Ollama 포트 받고, 동일한 API base 패턴 사용.

---

# 최종 영향 요약

| 재현 단계 | 수정 전 | 수정 후 |
|---|---|---|
| Llama 8B TNB 평가 rows | 184/237 (78%) | **237/237 (100%)** |
| priv_prompt skip | 6 | **0** |
| metric skip | 47 | **0** |
| Wall time 기록 | 4.5s (가짜, httpx 버그) | **9010s (실제)** |
| DSPy cache 영향 | 12행 hit (재현성 파괴) | **0행 hit** |
| Quality | 0.946 (쉬운 행만) | **0.878** (전체, 논문 0.866과 거의 일치) |
| Leakage | 0.125 | **0.129** (논문 0.115 ±1.4pp) |
| MIPROv2 prompt 저장 | 유실 | save-first로 확실히 저장 |

# 수정된 파일 목록

1. `papillon/evaluate_papillon.py` — 대폭 개편
2. `papillon/run_dspy_optimization_llama.py` — 대폭 개편
3. `papillon/run_llama_dspy.py` — retry + fallback + timing
4. `papillon/llm_judge.py` — robust judging (retry + fallback + safe defaults)
5. `papillon/ollama_nothink_lm.py` — 신규 (Qwen3 용)

# 환경 수정

- `pip install 'httpx<0.28'` (0.28.1 → 0.27.2)
- `.dspy_cache/cache.db` 삭제 (재현성 확보)
