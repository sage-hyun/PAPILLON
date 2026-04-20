# PAPILLON 논문 재현 실행 로그

## 1. 프로젝트 개요

**논문**: PAPILLON: PrivAcy Preservation from Internet-based and Local Language MOdel ENsembles (arXiv:2410.17127)

**목표**: 논문에서 제시한 PAPILLON 프라이버시 보존 파이프라인의 결과 재현

**핵심 아이디어**: 사용자의 개인정보가 포함된 쿼리를 로컬(trusted) 모델이 프라이버시 보호 프롬프트로 변환하고, 클라우드(untrusted) 모델에 전송하여 응답을 받은 뒤, 다시 로컬 모델이 최종 응답을 합성하는 2단계 파이프라인

## 2. 환경 구성

### 하드웨어
- GPU: NVIDIA RTX 4090 x 3 (24GB VRAM each)
- CUDA: 13.0 (Driver 580.105.08)

### 소프트웨어 환경
- Conda 환경: `papillon_v1.0` (Python 3.10)
- 주요 패키지: `dspy-ai 2.4.16`, `litellm 1.51.0`, `openai 1.55.0`
- 모델 서빙: Ollama (llama3.1:8b-instruct-fp16)
- 클라우드 모델: GPT-4o-mini (OpenRouter API 경유)

### 사용한 conda 환경
```bash
conda activate papillon_v1.0
```

## 3. 트러블슈팅 과정

### 3.1 OpenAI API 키 문제
- **문제**: `.bashrc`에 설정된 키(`sk-or-v1-...`)가 OpenRouter 키였으나, 원래 코드는 OpenAI 직접 API를 사용
- **시도 1**: OpenAI 직접 연결 → `401 AuthenticationError` (키 형식 불일치)
- **시도 2**: OpenRouter 경유 시도 → 처음에는 `401 User not found` (키가 만료됨)
- **해결**: 새 OpenRouter 키를 발급하여 `.bashrc` 업데이트
- **코드 수정**: `dspy.OpenAI(model="gpt-4o-mini")` → `dspy.LM('openai/gpt-4o-mini', api_base='https://openrouter.ai/api/v1', api_key=...)` 로 변경

### 3.2 dspy.OpenAI vs dspy.LM 호환성
- **문제**: `dspy.OpenAI`는 Ollama의 OpenAI-호환 API에서 `404 Not Found` 에러 발생
- **원인**: `dspy.OpenAI`가 내부적으로 `openai` 모듈의 전역 설정을 변경하는데, Ollama 엔드포인트와 호환되지 않음
- **해결**: 모든 코드에서 `dspy.OpenAI` → `dspy.LM` 타입으로 통일
- **추가 수정**: `run_llama_dspy.py`의 `forward()` 메서드에서 untrusted_model 호출 시 `dspy.LM`의 messages 기반 인터페이스 지원 추가

### 3.3 패키지 버전 충돌
- **문제 1**: `litellm`이 설치되어 있지 않음 → `ImportError: The LiteLLM package is not installed`
- **해결**: `pip install litellm==1.51.0`
- **문제 2**: litellm 최신 버전이 `openai 2.x`를 설치하면서 `dspy-ai 2.4.16`과 충돌
- **해결**: `pip install litellm==1.51.0 openai==1.55.0` (environment.yml 사양에 맞춤)
- **문제 3**: `httpx 0.28.1`에서 `proxies` 파라미터 미지원 에러
  ```
  TypeError: Client.__init__() got an unexpected keyword argument 'proxies'
  ```
- **해결**: `pip install httpx==0.27.0`

### 3.4 DSPy 모듈 로딩 호환성
- **문제**: `priv_prompt.load(args.prompt_file, use_legacy_loading=True)` → `load()` 메서드가 `use_legacy_loading` 파라미터를 지원하지 않음 (DSPy 2.4.16)
- **해결**: `use_legacy_loading=True` 파라미터 제거

### 3.5 전역 변수 스코프 문제
- **문제**: `evaluate_papillon.py`가 `metric_finegrained`를 `run_dspy_optimization_llama`에서 import할 때, `openai_lm_gpt4o` 전역변수가 `__main__` 블록 안에서만 정의되어 `NameError` 발생
- **해결**:
  1. `run_dspy_optimization_llama.py` 모듈 레벨에 `openai_lm_gpt4o = None` 초기화
  2. `evaluate_papillon.py`에서 `run_dspy_optimization_llama.openai_lm_gpt4o = judge_lm` 으로 설정

### 3.6 코드 버그 (원본 코드)
- **버그**: `run_dspy_optimization_llama.py`의 `metric_finegrained()` 함수에서 `ppi_str=og_pii` (오타) → `pii_str=og_pii`로 수정
- **영향**: 이 버그로 인해 leakage 점수가 항상 -1로 반환되었을 가능성

### 3.7 LLM 출력 파싱 에러
- **문제**: 일부 행에서 모델 출력이 DSPy 형식에 맞지 않아 `ValueError: Expected dict_keys(['output']) but got dict_keys([])` 발생
- **해결**: `evaluate_papillon.py`의 평가 루프에 try/except 에러 핸들링 추가

## 4. 실행한 명령어 및 결과물

### 4.1 모델 다운로드 (Ollama)
```bash
ollama pull llama3.1:8b-instruct-fp16
# → 16GB FP16 모델 다운로드 (~25분)
```

### 4.2 단일 쿼리 파이프라인 테스트
```bash
cd /home/cs575/PAPILLON_replication/papillon
conda run -n papillon_v1.0 python -c "
import dspy
from run_llama_dspy import PrivacyOnePrompter
# ... (파이프라인 초기화 후)
pred = priv_prompt('Write a cover letter for John Smith applying for a software engineer position at Google')
"
```
**결과**:
- 원본 쿼리: "Write a cover letter for **John Smith** applying for ... **Google**"
- 프라이버시 보호 프롬프트: "Write a cover letter for a software engineer applying for a position at Google" (개인정보 "John Smith" 제거)
- 최종 출력: 원래 이름을 복원하여 완전한 커버 레터 생성

### 4.3 소규모 평가 테스트 (10 샘플, 로컬 모델만)
```bash
conda run -n papillon_v1.0 python evaluate_papillon.py \
  --port 11434 \
  --data_file test_10.csv \
  --model_name "llama3.1:8b-instruct-fp16" \
  --openai_model "llama3.1:8b-instruct-fp16" \
  --output_file_name eval_test10_output.csv
```
**결과**: Quality 0.778, Leakage 0.167

### 4.4 소규모 평가 (10 샘플, GPT-4o-mini)
```bash
export OPENAI_API_KEY="<OPENROUTER_KEY>"
conda run -n papillon_v1.0 python evaluate_papillon.py \
  --port 11434 \
  --data_file test_10.csv \
  --model_name "llama3.1:8b-instruct-fp16" \
  --openai_model "gpt-4o-mini" \
  --output_file_name eval_gpt4o_test10.csv
```
**결과**: Quality 1.0, Leakage 0.370

### 4.5 전체 PUPA_New 데이터셋 평가 (진행 중)
```bash
export OPENAI_API_KEY="<OPENROUTER_KEY>"
conda run -n papillon_v1.0 python evaluate_papillon.py \
  --port 11434 \
  --data_file ../pupa/PUPA_New.csv \
  --model_name "llama3.1:8b-instruct-fp16" \
  --openai_model "gpt-4o-mini" \
  --output_file_name eval_llama31_8b_PUPA_New.csv
```
**결과 파일**: `papillon/eval_llama31_8b_PUPA_New.csv`
**중간 결과** (24행 기준): Quality 1.0, Leakage 0.194

## 5. 파일 구조 및 역할

### 입력 데이터
| 파일 | 설명 |
|------|------|
| `pupa/PUPA_New.csv` | 주 데이터셋 (664행, 4개 카테고리) |
| `pupa/PUPA_TNB.csv` | 보조 데이터셋 (237행) |

### PAPILLON 파이프라인 코드
| 파일 | 역할 |
|------|------|
| `papillon/run_llama_dspy.py` | 핵심 모듈 - `PrivacyOnePrompter` (2단계 프라이버시 파이프라인) |
| `papillon/llm_judge.py` | LLM 기반 품질/누출 판정기 (`LLMJudge`) |
| `papillon/evaluate_papillon.py` | 최적화된 파이프라인 평가 스크립트 |
| `papillon/run_dspy_optimization_llama.py` | DSPy MIPROv2 프롬프트 최적화 |
| `papillon/run_papillon_interactive.py` | 대화형 모드 |

### 사전 최적화된 프롬프트
| 파일 | 모델 |
|------|------|
| `papillon/optimized_prompts/llama_31_8b_instruct_prompt.json` | Llama-3.1-8B-Instruct |
| `papillon/optimized_prompts/llama_32_1b_instruct_prompt.json` | Llama-3.2-1B-Instruct |
| `papillon/optimized_prompts/llama_32_3b_instruct_prompt.json` | Llama-3.2-3B-Instruct |
| `papillon/optimized_prompts/llama_3_8b_instruct_prompt.json` | Llama-3-8B-Instruct |
| `papillon/optimized_prompts/mistral_7b_instruct_prompt.json` | Mistral-7B-Instruct |
| `papillon/optimized_prompts/mistral_small_prompt.json` | Mistral-Small |

### 출력 결과물
| 파일 | 설명 |
|------|------|
| `papillon/eval_test10_output.csv` | 10 샘플 평가 (로컬 모델만) |
| `papillon/eval_gpt4o_test10.csv` | 10 샘플 평가 (GPT-4o-mini) |
| `papillon/eval_llama31_8b_PUPA_New.csv` | 전체 PUPA_New 평가 (진행 중) |

### PUPA 데이터 파이프라인 코드
| 파일 | 역할 |
|------|------|
| `pupa/turn_processor.py` | 대화 턴 → PUPA 데이터 변환 |
| `pupa/create_privacy_span.py` | PII 추출 및 redaction |
| `pupa/filter_context_dependence.py` | 문맥 독립성 필터링 |
| `pupa/prompts/extract_privacy_span.txt` | PII redaction 시스템 프롬프트 |

## 6. 수정한 코드 목록

| 파일 | 수정 내용 |
|------|-----------|
| `evaluate_papillon.py` | OpenRouter 지원, dspy.LM 통일, 에러 핸들링 추가, use_legacy_loading 제거, 전역 변수 설정 |
| `run_dspy_optimization_llama.py` | OpenRouter 지원, dspy.LM 통일, ppi_str→pii_str 오타 수정, openai_lm_gpt4o 초기화 |
| `run_papillon_interactive.py` | OpenRouter 지원, dspy.LM 통일, use_legacy_loading 제거 |
| `run_llama_dspy.py` | dspy.LM의 messages 기반 호출 지원 추가 |

## 7. 재현 구성 비교

| 항목 | 논문 원본 | 재현 |
|------|-----------|------|
| Trusted Model | Llama-3.1-8B-Instruct (SGLang) | Llama-3.1-8B-Instruct (Ollama FP16) |
| Untrusted Model | GPT-4o-mini (OpenAI API) | GPT-4o-mini (OpenRouter) |
| LLM Judge | GPT-4o-mini (OpenAI API) | GPT-4o-mini (OpenRouter) |
| 데이터셋 | PUPA-New (664행) | PUPA-New (664행) |
| DSPy 버전 | 2.5.35 | 2.4.16 |
