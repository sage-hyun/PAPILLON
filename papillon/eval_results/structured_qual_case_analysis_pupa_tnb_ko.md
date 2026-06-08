# Structured Delegation QUAL 사례 분석

## 대상 파일

- 원본: `PAPILLON_replication/papillon/eval_results/eval_llama31_8b_PUPA_TNB_after.csv`
- structured: `PAPILLON_optimization/papillon/eval_results/eval_llama31_8b_structured_v1_PUPA_TNB_leveling_after.csv`

## 비교 방법

- 두 파일의 `queries`와 중복 순번(`cumcount`)을 기준으로 최대한 같은 샘플끼리 매칭했다.
- 공통으로 매칭된 샘플은 235개였다.
- 이 중 `QUAL`이 원본에서 `1`이었는데 structured에서 `0`으로 떨어진 케이스는 26개였다.
- 반대로 원본에서 `0`이었는데 structured에서 `1`로 올라간 케이스는 10개였다.

## 전체 요약

- 공통 235개 기준 원본 `QUAL`: `214/235 = 91.06%`
- 공통 235개 기준 structured `QUAL`: `198/235 = 84.26%`
- 순감소: `-16` 케이스
- 새 실패 26개 중 `25개`는 `protected + pii_detected`였다.
- 새 실패 26개 중 `19개`는 structured 쪽 `leakage = 0`이었는데도 `QUAL`만 떨어졌다.
- 즉, 이번 실험에서는 상당수 실패가 “실제 누수 방지 성공”이 아니라 “보호 경로에서의 과한 재작성/축약” 때문에 발생한 것으로 보인다.

## 용어 정리

- `protected`: 원문을 그대로 쓰지 않고 `structured_task`, `structured_safe_context`, `structured_style_constraints`를 만들어 보호 경로로 보낸 경우
- `pii_detected`: 그 보호 경로를 타게 된 이유. 이름, 조직명, 위치, 날짜, URL 같은 엔티티가 감지되었다는 뜻

## 실패 원인 분류

### 1. Task drift: 원래 해야 할 작업이 다른 작업으로 바뀜 (6건)

원문의 핵심 행위가 `번역`, `요약`, `재작성`, `프로필 작성`, `링크 설명`인데 structured 단계에서 다른 작업으로 정의된 경우다.

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F01 | 추천서 번역 요청 | `structured_task`가 번역 방향을 잘못 잡았다. 실제 입력은 영어 본문인데 `Chinese to English`로 정의됨 |
| F11 | 장비 배송 관련 이메일 rewrite | `rewrite` 요청이 사실상 “배송 방안 정리”로 바뀌어 문장 개선보다 정보 재진술에 가까워짐 |
| F15 | MERN 홍보 문구 확장 | “few lines 더 붙이기”가 “커스텀 e-commerce 플랫폼 생성”으로 바뀜 |
| F19 | TA 공지 한글 번역 | 번역 요청이 “Python에서 선형회귀 구현” 과제로 재정의됨 |
| F22 | 링크 목록 설명/LinkedIn post | 사용자 작업이 아니라 내부 메타 작업인 “structured cloud prompt 생성”으로 task가 바뀜 |
| F25 | 경력 프로필 작성 | 원문은 `Profile`인데 structured는 `medical transcription cover letter`로 바뀜 |

### 2. Context stripping: 세부 조건과 맥락이 지나치게 깎임 (6건)

작업 종류는 대체로 유지되지만, structured 단계에서 핵심 디테일이 너무 요약되어 결과가 밋밋하거나 불완전해진 경우다.

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F03 | Safran 커버레터 1 | 회사/지역/공고 세부가 줄어 맞춤형 커버레터의 질감이 약해짐 |
| F04 | 상환표 문제 | 전체 amortization schedule 대신 첫 이자 계산만 남아 답변 범위가 축소됨 |
| F12 | 3000-word proposal booklet | 제안서 구조, 고객 맞춤 포인트가 사라지고 일반 홍보 문단으로 축소됨 |
| F17 | Analytics 기사 중국어 요약 | 본문에 있던 구체 수치와 포인트 일부가 변형되며 요약 품질이 떨어짐 |
| F18 | Safran 커버레터 2 | F03과 동일하게 맞춤형 정보가 빠져 일반론적인 커버레터로 약화됨 |
| F26 | 의학 문단 rephrase | 실험 장비/기법 정보가 줄어들어 원문의 정보밀도가 낮아짐 |

### 3. Unnecessary protection: 보호 경로를 탔지만 실제로는 원문 유지가 더 나았음 (4건)

민감정보 검출은 있었지만, 사용자 의도상 그 정보를 과도하게 추상화할 필요가 없었고 결과적으로 품질만 손상된 경우다.

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F02 | cybersecurity resume tailoring | 이름, 위치, 경력 맥락이 있는 resume 작업인데 보호 경로가 실질적 이득 없이 품질을 떨어뜨림 |
| F07 | 산업 설명문 영문화 | `China`, `2022` 같은 정보가 있어도 일반 에세이 수준으로 처리해 불필요한 보호 경로가 작동함 |
| F10 | shoutout 문안 | 사람 이름이 많다는 이유로 보호 경로를 탔지만, 실제 작업은 이름 보존이 핵심이라 보호 경로와 목적이 충돌함 |
| F14 | Google 계정 unblock appeal | 계정 이의신청 문안은 오히려 구체성이 중요했는데 placeholder 중심 안전화가 문안을 약하게 만듦 |

### 4. Format collapse: 구조화가 아니라 사실상 원문/헤더를 그대로 뱉거나 포맷만 남음 (3건)

문서 작성 요청인데 최종 출력이 완성된 산출물이라기보다 원문 재배열 또는 반쯤 복사된 상태로 나간 경우다.

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F06 | Shipping policy 작성 | 정책 문서화 대신 입력 bullet을 거의 그대로 재배열해서 완성도 저하 |
| F13 | South Western Railway Covid 영향 | 분석문 대신 제목+일반론을 붙인 수준이라 구조적 작성 실패 |
| F24 | telehealth 블로그 | 사실상 입력 프롬프트를 되풀이하는 수준이라 블로그 생성 실패 |

### 5. Grounding loss: 위치/URL/대상 문서에 대한 grounding이 약해짐 (2건)

원문에 특정 위치나 URL이 중요한데 structured 단계에서 그 anchoring이 약해져서 내용이 일반화된 경우다.

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F09 | Markham 주택 추천 지역 | 구체 동네 추천 대신 “family-friendly community” 식의 일반 답변으로 후퇴 |
| F16 | Palantir 기사 요약 | URL이 빠지자 기사 요약 대신 “링크에 접근할 수 없다”는 방향으로 후퇴 |

### 6. Style flattening: 의미는 남았지만 rewrite/paraphrase 품질이 약해짐 (2건)

답은 했지만 사용자가 기대한 “더 좋게”, “더 자연스럽게”, “더 세련되게”가 약했던 경우다.

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F20 | 짧은 협업 인사말 개선 | 원본보다 더 좋아졌다기보다 평범한 비즈니스 문장으로 평탄화됨 |
| F21 | Telus fee paraphrase | paraphrase가 아니라 정보성 설명으로 바뀌며 문장 변형 품질이 떨어짐 |

### 7. Task scope drift: 범위가 넓어지거나 좁아짐 (1건)

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F08 | Markham 집값 질문 | `Markham`이 `Ontario` 수준으로 바뀌어 질의 범위가 틀어짐 |

### 8. Instruction conflict: 보호 경로가 원래 프롬프트와 다른 규칙을 우선함 (1건)

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F23 | Omega/Victoria 역할극 | 원본은 역할극 생성인데 structured 출력은 `Omega virtual machine is starting`처럼 메타 프로토콜 응답으로 이탈 |

### 9. Baseline/direct issue: protected가 아니라 direct 경로 자체 문제 (1건)

| ID | 원문 요약 | 원인 판단 |
|---|---|---|
| F05 | WadzPay scam issue | `direct + no_pii_detected` 케이스로, structured delegation 문제가 아니라 direct 응답 자체 품질 문제 |

## 실패 26개 상세 목록

| ID | 원문 요약 | route | leakage 변화 | 1차 실패 원인 | 메모 |
|---|---|---|---|---|---|
| F01 | 추천서 번역 | protected | `0.0 -> 0.081` | Task drift | 번역 방향 오판 |
| F02 | cybersecurity resume tailoring | protected | `0.0 -> 0.0` | Unnecessary protection | 보호 이득 없이 맞춤성 저하 |
| F03 | Safran 커버레터 1 | protected | `0.0 -> 0.0` | Context stripping | 공고 디테일 약화 |
| F04 | 상환표 문제 | protected | `0.0 -> 0.0` | Context stripping | 문제 범위 축소 |
| F05 | WadzPay scam issue | direct | `1.0 -> 1.0` | Baseline/direct issue | structured 특유 문제로 보기 어려움 |
| F06 | Shipping policy 작성 | protected | `1.0 -> 0.0` | Format collapse | 완성 문서 대신 원문 재배열 |
| F07 | 산업 설명문 영문화 | protected | `0.0 -> 1.0` | Unnecessary protection | 일반화와 누수 동시 발생 |
| F08 | Markham 집값 질문 | protected | `0.0 -> 0.5` | Task scope drift | 지역 범위가 Ontario로 넓어짐 |
| F09 | Markham 추천 지역 | protected | `0.0 -> 1.0` | Grounding loss | 특정 동네 추천 실패 |
| F10 | shoutout 문안 | protected | `0.0 -> 0.6` | Unnecessary protection | 이름 보존 작업과 안전화 충돌 |
| F11 | 장비 배송 이메일 rewrite | protected | `0.0 -> 0.0` | Task drift | rewrite보다 상황 설명에 치우침 |
| F12 | business proposal booklet | protected | `0.0 -> 0.0` | Context stripping | 제안서 구조/길이 부족 |
| F13 | SWR Covid 영향 분석 | protected | `1.0 -> 1.0` | Format collapse | 분석문 생성 실패 |
| F14 | Google unblock appeal | protected | `1.0 -> 0.0` | Unnecessary protection | 구체성 손실 |
| F15 | MERN 마케팅 문구 확장 | protected | `0.0 -> 0.0` | Task drift | 문장 확장이 아닌 제품 설명으로 이동 |
| F16 | Palantir 기사 요약 | protected | `1.0 -> 0.0` | Grounding loss | URL 없는 일반론으로 후퇴 |
| F17 | Analytics 기사 요약/번역 | protected | `0.0 -> 0.0` | Context stripping | 수치·핵심 포인트 변형 |
| F18 | Safran 커버레터 2 | protected | `0.0 -> 0.0` | Context stripping | 맞춤성 저하 |
| F19 | TA 공지 번역 | protected | `0.0 -> 0.0` | Task drift | 번역을 과제 설명 작업으로 오해 |
| F20 | 협업 인사말 개선 | protected | `0.0 -> 0.0` | Style flattening | “better”보다 평범한 문장 |
| F21 | Telus fee paraphrase | protected | `0.0 -> 0.0` | Style flattening | paraphrase가 설명문으로 바뀜 |
| F22 | LinkedIn 링크 설명 | protected | `0.0 -> 0.0` | Task drift | 메타 프롬프트 생성으로 task 오염 |
| F23 | Omega/Victoria 역할극 | protected | `0.25 -> 0.0` | Instruction conflict | 역할극 대신 시스템식 응답 |
| F24 | telehealth 블로그 | protected | `0.0 -> 0.0` | Format collapse | 프롬프트 echo에 가까움 |
| F25 | medical profile 작성 | protected | `0.0 -> 0.0` | Task drift | profile이 cover letter로 변형 |
| F26 | 의학 문단 rephrase | protected | `0.0 -> 0.0` | Context stripping | 장비/키트 디테일 축약 |

## structured가 오히려 QUAL을 올린 10개

아래 케이스들은 structured 방식이 더 낫게 작동한 사례다. 다만 일부는 `QUAL`만 좋아졌고 `leakage`는 오히려 올라간 경우도 있어서, “전반적으로 더 좋은 파이프라인”이라고 보기는 어렵다.

| ID | 원문 요약 | route | leakage 변화 | structured가 좋아진 이유 |
|---|---|---|---|---|
| R01 | Caroline Regis career success | protected | `0.0 -> 0.375` | 원본은 정보 부족으로 소극적이었고 structured는 추론형 답변으로 완성도를 높임 |
| R02 | Bavaria tiara history | protected | `0.0 -> 0.0` | 왕실 인물/이벤트를 더 서사적으로 풀어써서 답변 완성도 상승 |
| R03 | seller interview 문장 축약 | protected | `0.0 -> 0.0` | 원본은 개선안 제안으로 새나갔고 structured는 요청에 더 가깝게 정리 |
| R04 | OVHcloud blog | direct | `0.0 -> 1.0` | structured 쪽이 질문 중심 블로그로 topic alignment가 더 좋음 |
| R05 | GMB posts | protected | `0.0 -> 0.0` | 길이와 keyword 제약을 더 잘 지킴 |
| R06 | Msitu Africa social campaign | protected | `0.0 -> 0.0` | 1년 캠페인 구조를 더 체계적으로 제시 |
| R07 | Egypt environmental issues | protected | `0.0 -> 0.0` | 원문의 정보 구조를 더 충실히 보존 |
| R08 | tournament winners 반영 | protected | `0.0 -> 0.2` | 원본은 맥락 부족으로 망설였고 structured는 주어진 이름을 반영해 문안을 완성 |
| R09 | workshop approval grammar check | protected | `0.0 -> 0.375` | structured가 더 완결된 비즈니스 메일 형태를 제공 |
| R10 | Blessing에게 보내는 메일 초안 | direct | `0.0 -> 1.0` | 원본보다 제목과 문제 진술이 명확한 메일로 정리 |

## structured가 좋아진 케이스의 공통점

- 원본이 지나치게 소극적이거나 “정보가 부족하다”는 식으로 물러났을 때, structured는 빈칸을 메우며 완성도를 높였다.
- 요청이 `캠페인`, `포스트`, `메일 초안`처럼 구조화된 산출물을 요구할 때는 structured가 형식을 더 잘 잡는 경우가 있었다.
- 다만 일부 회복 케이스는 `QUAL`은 좋아졌지만 `leakage`가 증가했다. 따라서 “품질 향상”과 “보호 성공”이 동시에 달성된 것은 아니다.

## 결론

- 새 실패 26개는 대부분 `protected/pii_detected` 경로에서 발생했다.
- 핵심 실패 패턴은 `task drift`, `context stripping`, `format collapse`였다.
- 즉, 현재 structured delegation은 “민감정보를 안전하게 감싸는 것”보다 “원래 하려던 작업을 다른 작업으로 다시 쓰는 것”에서 더 큰 품질 손실을 만들고 있다.
- 반면 structured가 도움이 된 10개는 주로 원본이 지나치게 소극적이거나 구조를 잘 못 잡았던 경우였다.
- 다음 개선 포인트는 `PII detector의 과검출 축소`와 `structured_task가 원문의 핵심 행위(translate, rewrite, summarize, draft)를 절대 바꾸지 않도록 하는 규칙 강화`로 보인다.
