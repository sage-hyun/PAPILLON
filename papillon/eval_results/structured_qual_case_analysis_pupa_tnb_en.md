# Structured Delegation QUAL Case Analysis

## Target Files

- Baseline: `eval_llama31_8b_PUPA_TNB_after.csv`
- Structured: `eval_llama31_8b_structured_v1_PUPA_TNB_leveling_after.csv`

## Comparison Method

- Samples were matched as closely as possible using `queries` plus duplicate order (`cumcount`).
- 235 samples were matched across both files.
- Among them, 26 cases had `QUAL = 1` in the baseline but `QUAL = 0` in the structured run.
- Conversely, 10 cases had `QUAL = 0` in the baseline but improved to `QUAL = 1` in the structured run.

## Overall Summary

- Baseline `QUAL` on the 235 matched samples: `214/235 = 91.06%`
- Structured `QUAL` on the 235 matched samples: `198/235 = 84.26%`
- Net decrease: `-16` cases
- Of the 26 new failures, `25` were `protected + pii_detected`.
- Of the 26 new failures, `19` had `leakage = 0` on the structured side, yet still lost `QUAL`.
- In other words, many failures in this experiment appear to come not from successful privacy protection, but from over-rewriting or over-compression inside the protected path.

## Terminology

- `protected`: the request was not sent in its raw form; instead, the pipeline created `structured_task`, `structured_safe_context`, and `structured_style_constraints` and routed through the protected path.
- `pii_detected`: the reason the protected path was used. It means entities such as names, organizations, locations, dates, or URLs were detected.

## Failure Cause Taxonomy

### 1. Task drift: the original requested action changed into a different task (6 cases)

These are cases where the core action in the user request, such as `translate`, `summarize`, `rewrite`, `write a profile`, or `describe links`, was changed during the structured step.

| ID | Query summary | Reason |
|---|---|---|
| F01 | Recommendation letter translation | `structured_task` got the translation direction wrong. The input was English, but the task became `Chinese to English` |
| F11 | Rewrite of an equipment shipping email | A `rewrite` request became closer to “arrange shipping” than to improving the wording |
| F15 | Expand a MERN marketing blurb | “Add a few more lines” shifted into “create a custom e-commerce platform” |
| F19 | Korean translation of a TA announcement | The translation request was redefined as a linear regression assignment task |
| F22 | Describe a list of links / LinkedIn post task | The user task was replaced by an internal meta-task: “create a structured cloud prompt” |
| F25 | Professional profile writing | The original request was a `Profile`, but structured turned it into a `cover letter` |

### 2. Context stripping: too much detail and constraint information was removed (6 cases)

The task type mostly stayed the same, but the structured representation removed too many critical details, making the answer flatter or incomplete.

| ID | Query summary | Reason |
|---|---|---|
| F03 | Safran cover letter 1 | Company, location, and posting details were reduced, weakening personalization |
| F04 | Amortization schedule problem | The full schedule task collapsed into only the first interest calculation |
| F12 | 3000-word proposal booklet | Proposal structure and client-specific positioning were lost and reduced to generic business copy |
| F17 | Chinese summary/translation of analytics article | Some concrete numbers and key points in the source were altered or weakened |
| F18 | Safran cover letter 2 | Same pattern as F03: the response became more generic and less tailored |
| F26 | Medical paragraph rephrase | Experimental device and assay details were reduced, lowering informational density |

### 3. Unnecessary protection: the protected path was triggered, but preserving more of the original would likely have been better (4 cases)

PII was detected, but the abstraction was not actually helpful for the user’s goal and instead hurt utility.

| ID | Query summary | Reason |
|---|---|---|
| F02 | Cybersecurity resume tailoring | Resume tasks naturally contain names, locations, and work history, but the protected path reduced quality without clear privacy gain |
| F07 | English polishing of industrial description | Data like `China` and `2022` pushed the sample into the protected path even though a normal rewrite would have been more useful |
| F10 | Team shoutout text | The task required preserving names, so the privacy path conflicted with the task’s purpose |
| F14 | Google account unblock appeal | This appeal needed specificity; placeholder-heavy safety handling weakened the letter |

### 4. Format collapse: output became raw rearrangement or partial prompt echo instead of a finished deliverable (3 cases)

These are generation failures where the output looked more like reorganized input or header-only output than a completed response.

| ID | Query summary | Reason |
|---|---|---|
| F06 | Shipping policy drafting | Instead of composing a polished policy, the output mostly rearranged the input bullets |
| F13 | South Western Railway Covid impact analysis | Instead of a real analysis, the output became title-plus-generalities |
| F24 | Telehealth blog | The output was close to repeating the prompt instead of generating a real blog post |

### 5. Grounding loss: the answer lost anchoring to the specific place, URL, or source document (2 cases)

These are cases where the request depended on a concrete location or article reference, and the structured path weakened that grounding.

| ID | Query summary | Reason |
|---|---|---|
| F09 | Best area to buy a home in Markham | The answer backed away from specific neighborhoods and became generic |
| F16 | Palantir article summary | Once the URL was abstracted away, the model backed off into generic commentary |

### 6. Style flattening: the meaning remained, but rewrite/paraphrase quality weakened (2 cases)

The system still answered the request, but it failed to deliver the requested improvement in tone or writing quality.

| ID | Query summary | Reason |
|---|---|---|
| F20 | Improve a short collaboration greeting | The rewrite became a flat business sentence rather than a stronger version |
| F21 | Telus fee paraphrase | The result became explanatory prose rather than a proper paraphrase |

### 7. Task scope drift: the request scope became broader or narrower than intended (1 case)

| ID | Query summary | Reason |
|---|---|---|
| F08 | Markham home price query | `Markham` was effectively broadened into `Ontario`, changing the query scope |

### 8. Instruction conflict: the protected path prioritized a different instruction frame than the original prompt (1 case)

| ID | Query summary | Reason |
|---|---|---|
| F23 | Omega/Victoria roleplay prompt | The original request was roleplay generation, but the structured output shifted into a meta-protocol response |

### 9. Baseline/direct issue: this was a direct-path problem rather than a structured-protection problem (1 case)

| ID | Query summary | Reason |
|---|---|---|
| F05 | WadzPay scam issue | This was `direct + no_pii_detected`, so it is better treated as a direct answer quality issue rather than a structured delegation failure |

## Detailed List of the 26 New Failures

| ID | Query summary | route | leakage change | Primary cause | Notes |
|---|---|---|---|---|---|
| F01 | Recommendation letter translation | protected | `0.0 -> 0.081` | Task drift | Translation direction misread |
| F02 | Cybersecurity resume tailoring | protected | `0.0 -> 0.0` | Unnecessary protection | Privacy path reduced tailoring quality |
| F03 | Safran cover letter 1 | protected | `0.0 -> 0.0` | Context stripping | Job-post details weakened |
| F04 | Amortization schedule problem | protected | `0.0 -> 0.0` | Context stripping | Full problem scope reduced |
| F05 | WadzPay scam issue | direct | `1.0 -> 1.0` | Baseline/direct issue | Not specific to structured protection |
| F06 | Shipping policy drafting | protected | `1.0 -> 0.0` | Format collapse | Rearranged input instead of polished output |
| F07 | Industrial description polishing | protected | `0.0 -> 1.0` | Unnecessary protection | Over-generalization plus leakage |
| F08 | Markham home price query | protected | `0.0 -> 0.5` | Task scope drift | Region broadened to Ontario |
| F09 | Best area to buy in Markham | protected | `0.0 -> 1.0` | Grounding loss | Failed to stay neighborhood-specific |
| F10 | Team shoutout writing | protected | `0.0 -> 0.6` | Unnecessary protection | Name preservation conflicted with abstraction |
| F11 | Equipment shipping email rewrite | protected | `0.0 -> 0.0` | Task drift | Rewriting shifted into scenario explanation |
| F12 | Business proposal booklet | protected | `0.0 -> 0.0` | Context stripping | Proposal structure and length were weakened |
| F13 | SWR Covid impact analysis | protected | `1.0 -> 1.0` | Format collapse | Failed to generate a real analysis |
| F14 | Google unblock appeal | protected | `1.0 -> 0.0` | Unnecessary protection | Lost useful specificity |
| F15 | MERN marketing copy expansion | protected | `0.0 -> 0.0` | Task drift | Expansion request turned into product framing |
| F16 | Palantir article summary | protected | `1.0 -> 0.0` | Grounding loss | Became generic once the URL context weakened |
| F17 | Analytics article summary/translation | protected | `0.0 -> 0.0` | Context stripping | Key figures and details weakened |
| F18 | Safran cover letter 2 | protected | `0.0 -> 0.0` | Context stripping | Less tailored than baseline |
| F19 | TA announcement translation | protected | `0.0 -> 0.0` | Task drift | Translation misread as assignment content |
| F20 | Collaboration greeting rewrite | protected | `0.0 -> 0.0` | Style flattening | “Better” became merely neutral |
| F21 | Telus fee paraphrase | protected | `0.0 -> 0.0` | Style flattening | Became explanation instead of paraphrase |
| F22 | LinkedIn link description task | protected | `0.0 -> 0.0` | Task drift | User task contaminated by meta prompt generation |
| F23 | Omega/Victoria roleplay | protected | `0.25 -> 0.0` | Instruction conflict | Roleplay replaced with system-style response |
| F24 | Telehealth blog | protected | `0.0 -> 0.0` | Format collapse | Prompt echo rather than generated article |
| F25 | Medical profile writing | protected | `0.0 -> 0.0` | Task drift | Profile turned into cover letter |
| F26 | Medical paragraph rephrase | protected | `0.0 -> 0.0` | Context stripping | Reduced procedural details |

## 10 Cases Where Structured Actually Improved QUAL

The following cases are examples where structured delegation improved `QUAL`. However, some of them also increased `leakage`, so they should not automatically be treated as globally better outcomes.

| ID | Query summary | route | leakage change | Why structured helped |
|---|---|---|---|---|
| R01 | Caroline Regis career success | protected | `0.0 -> 0.375` | The baseline was too hesitant; structured gave a more complete inferential answer |
| R02 | Bavaria tiara history | protected | `0.0 -> 0.0` | The structured answer was more coherent and narrative-rich |
| R03 | Seller interview text shortening | protected | `0.0 -> 0.0` | The baseline drifted into recommendations; structured stayed closer to rewriting |
| R04 | OVHcloud blog | direct | `0.0 -> 1.0` | The structured-side output aligned better with the actual blog topic |
| R05 | GMB posts | protected | `0.0 -> 0.0` | Better adherence to length and keyword constraints |
| R06 | Msitu Africa social campaign | protected | `0.0 -> 0.0` | Better campaign structure over a one-year horizon |
| R07 | Egypt environmental issues | protected | `0.0 -> 0.0` | The structured answer preserved the original content more faithfully |
| R08 | Tournament winners insertion | protected | `0.0 -> 0.2` | The baseline hesitated due to context shortage; structured completed the requested wording |
| R09 | Workshop approval grammar check | protected | `0.0 -> 0.375` | The structured answer became a more complete formal business email |
| R10 | Draft email to Blessing | direct | `0.0 -> 1.0` | Clearer subject line and issue description than the baseline |

## Common Patterns in the Improved Cases

- When the baseline became too hesitant or claimed insufficient information, the structured path sometimes produced a more complete answer.
- For requests such as `campaign`, `post`, or `email draft`, structured sometimes imposed a clearer output structure.
- However, some recovered cases improved `QUAL` while increasing `leakage`, so they should not be interpreted as unambiguous wins for the privacy pipeline.

## Conclusion

- Most of the 26 new failures came from the `protected / pii_detected` path.
- The main failure patterns were `task drift`, `context stripping`, and `format collapse`.
- In short, the current structured delegation pipeline appears to lose more quality by rewriting the task itself than it gains from privacy-safe reframing.
- The 10 improved cases mostly happened where the baseline was too hesitant or poorly structured.
- The most promising next steps are to reduce over-triggering in PII detection and to enforce stronger rules so that `structured_task` never changes the user’s core action such as `translate`, `rewrite`, `summarize`, or `draft`.
