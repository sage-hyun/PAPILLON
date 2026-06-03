"""Analyze CSV outputs produced by evaluate_papillon.py.

Input pool: any number of eval CSVs (e.g. one from PUPA_TNB_leveling that mixes
PII and no-PII rows, plus one from PUPA_No_PII which is pure no-PII). All input
CSVs are concatenated into a single pool and each row is classified by
`pii_str` (non-empty -> with_pii, empty/NaN -> no_pii).

Subsampling is deterministic + monotonic by default: rows in each group are
ordered by `conversation_hash` (then original index), and a target row count
takes the *prefix*. This guarantees the strict-superset property the user wants
-- e.g. pii_ratio=1.0 selects all 206 with_pii rows only, pii_ratio=0.5 selects
the same 206 with_pii rows plus the first 206 no_pii rows of the sorted pool,
and a smaller pii_ratio just appends more no_pii rows on top of the same prefix.

Ratio convention: pii_ratio = n_with_pii / (n_with_pii + n_no_pii). So pii_ratio
== 1.0 means "PII only", pii_ratio == 0.0 means "no_pii only", pii_ratio == 0.5
means 1:1.

Usage:
  # combine TNB_leveling eval (mixed) + PUPA_No_PII eval (pure no_pii)
  python analyze_eval_csv.py \\
    --eval_csv eval_results/<method_TNB>.csv eval_results/<method_NoPII>.csv \\
    --baseline_csv eval_results/<base_TNB>.csv eval_results/<base_NoPII>.csv \\
    --pii_ratio 0.5 --sweep \\
    --out_dir papillon/eval_results/analysis/<run_name>

Tip: invoke with --print_counts_only first to see how many with_pii / no_pii
rows are in your pool, then choose a ratio sweep that fits.
"""

import argparse
import json
import os
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd


METRIC_COLS = [
    "quality",
    "leakage",
    "weighted_leakage",
    "leakage_l1_ratio",
    "leakage_l2_ratio",
    "leakage_l3_ratio",
    "exposed_token_count",
    "entity_retention_rate",
]

LATENCY_COLS = [
    "latency_privacy_filter_ms",
    "latency_prompt_creator_ms",
    "latency_cloud_ms",
    "latency_aggregator_ms",
    "latency_total_ms",
]


def _is_no_pii(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "nan"


def load_eval_csvs(paths: Iterable[str], head: Optional[int] = None) -> pd.DataFrame:
    """Load and concatenate one or more eval CSVs. `head` applies per file."""
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if head is not None and head > 0:
            df = df.iloc[:head].copy()
        df["_source_csv"] = os.path.basename(path)
        df["_row_in_source"] = np.arange(len(df))
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool["_is_no_pii"] = pool["pii_str"].apply(_is_no_pii)
    pool["_valid"] = (pool["quality"] != -1) & (pool["leakage"] != -1)
    # Stable sort key for deterministic monotonic subsampling.
    sort_key = pool.get("conversation_hash", pd.Series([""] * len(pool))).fillna("").astype(str)
    pool["_sort_key"] = sort_key + "|" + pool["_source_csv"].astype(str) + "|" + pool["_row_in_source"].astype(str).str.zfill(8)
    return pool


def _monotonic_prefix(sub: pd.DataFrame, k: int) -> pd.DataFrame:
    if k >= len(sub):
        return sub
    return sub.sort_values("_sort_key").iloc[:k]


def _random_sample(sub: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if k >= len(sub):
        return sub
    idx = rng.choice(len(sub), size=k, replace=False)
    return sub.iloc[np.sort(idx)]


def select_rows(
    df: pd.DataFrame,
    no_pii_n: Optional[int] = None,
    pii_n: Optional[int] = None,
    pii_ratio: Optional[float] = None,
    mode: str = "monotonic",
    seed: int = 0,
) -> pd.DataFrame:
    """Pick a subset of rows with a target with_pii / no_pii split.

    - mode='monotonic' (default): deterministic prefix of sort_key. Growing
      either target only *adds* rows -- prior selections stay.
    - mode='random': deterministic seeded random sample.

    Priority of target spec: explicit --no_pii_n/--pii_n > --pii_ratio > no-op.

    For --pii_ratio, the rule keeps all with_pii rows (pii_n = total with_pii)
    and chooses no_pii count so that pii_n / (pii_n + no_pii_n) == pii_ratio,
    capped by the available no_pii pool. Edge cases:
        pii_ratio == 1.0 -> only with_pii rows (no_pii_n = 0)
        pii_ratio == 0.0 -> only no_pii rows  (pii_n = 0)
    """
    rng = np.random.default_rng(seed)
    with_pii_df = df[~df["_is_no_pii"]]
    no_pii_df = df[df["_is_no_pii"]]

    if no_pii_n is None and pii_n is None and pii_ratio is None:
        return df

    if pii_ratio is not None and no_pii_n is None and pii_n is None:
        if pii_ratio >= 1.0:
            pii_n = len(with_pii_df)
            no_pii_n = 0
        elif pii_ratio <= 0.0:
            pii_n = 0
            no_pii_n = len(no_pii_df)
        else:
            pii_n = len(with_pii_df)
            target_no_pii = int(round(pii_n * (1.0 - pii_ratio) / pii_ratio))
            no_pii_n = min(target_no_pii, len(no_pii_df))

    pii_n = len(with_pii_df) if pii_n is None else min(pii_n, len(with_pii_df))
    no_pii_n = len(no_pii_df) if no_pii_n is None else min(no_pii_n, len(no_pii_df))

    if mode == "monotonic":
        pii_sel = _monotonic_prefix(with_pii_df, pii_n)
        no_pii_sel = _monotonic_prefix(no_pii_df, no_pii_n)
    elif mode == "random":
        pii_sel = _random_sample(with_pii_df, pii_n, rng)
        no_pii_sel = _random_sample(no_pii_df, no_pii_n, rng)
    else:
        raise ValueError(f"unknown mode: {mode}")

    return pd.concat([pii_sel, no_pii_sel]).sort_index()


def _avg(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else float("nan")


def metric_summary(df: pd.DataFrame, label: str) -> dict:
    valid = df[df["_valid"]]
    out = {"group": label, "n_rows": int(len(df)), "n_valid": int(len(valid))}
    for col in METRIC_COLS:
        if col in valid.columns:
            out[col] = _avg(valid[col])
    if "schema_valid" in valid.columns:
        out["schema_valid_rate"] = _avg(valid["schema_valid"].astype(float))
    return out


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        metric_summary(df, "overall"),
        metric_summary(df[~df["_is_no_pii"]], "with_pii"),
        metric_summary(df[df["_is_no_pii"]], "no_pii"),
    ]
    return pd.DataFrame(rows)


def build_latency_table(df: pd.DataFrame, baseline_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    valid = df[df["_valid"]]
    rec = {"source": "method"}
    for col in LATENCY_COLS:
        if col in valid.columns:
            rec[col] = _avg(valid[col])
    out = [rec]
    if baseline_df is not None:
        b_valid = baseline_df[baseline_df["_valid"]]
        b = {"source": "baseline"}
        for col in LATENCY_COLS:
            if col in b_valid.columns:
                b[col] = _avg(b_valid[col])
        out.append(b)
        delta = {"source": "delta (method - baseline)"}
        for col in LATENCY_COLS:
            if col in rec and col in b:
                delta[col] = rec[col] - b[col]
        out.append(delta)
    return pd.DataFrame(out)


# ---------- NER FP / FN ----------

_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _split_gold(pii_str) -> list:
    if _is_no_pii(pii_str):
        return []
    parts = [p for p in str(pii_str).split("||") if p.strip()]
    return [_norm(p) for p in parts if _norm(p)]


def _parse_detected(detected_json) -> list:
    if not isinstance(detected_json, str) or not detected_json.strip():
        return []
    try:
        items = json.loads(detected_json)
    except Exception:
        return []
    out = []
    for it in items or []:
        text = (it or {}).get("text", "")
        n = _norm(text)
        if n:
            out.append(n)
    return out


def _ner_match(gold: list, detected: list):
    """Soft match: gold[i] hits detected[j] if either fully contains the other."""
    matched_gold, matched_det = set(), set()
    for i, g in enumerate(gold):
        for j, d in enumerate(detected):
            if j in matched_det:
                continue
            if g == d or g in d or d in g:
                matched_gold.add(i)
                matched_det.add(j)
                break
    tp = len(matched_gold)
    fn = len(gold) - tp
    fp = len(detected) - len(matched_det)
    return tp, fp, fn


def ner_per_row(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        gold = _split_gold(r.get("pii_str", ""))
        detected = _parse_detected(r.get("detected_pii_json", ""))
        tp, fp, fn = _ner_match(gold, detected)
        rows.append({
            "conversation_hash": r.get("conversation_hash", ""),
            "source_csv": r.get("_source_csv", ""),
            "is_no_pii": bool(r.get("_is_no_pii", False)),
            "n_gold": len(gold),
            "n_detected": len(detected),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    return pd.DataFrame(rows)


def ner_aggregate(per_row: pd.DataFrame) -> pd.DataFrame:
    def agg(sub, label):
        tp, fp, fn = int(sub["tp"].sum()), int(sub["fp"].sum()), int(sub["fn"].sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        return {
            "group": label,
            "n_rows": int(len(sub)),
            "fp_per_row": float(sub["fp"].mean()) if len(sub) else float("nan"),
            "fn_per_row": float(sub["fn"].mean()) if len(sub) else float("nan"),
            "tp_total": tp,
            "fp_total": fp,
            "fn_total": fn,
            "precision": precision,
            "recall": recall,
        }
    return pd.DataFrame([
        agg(per_row, "overall"),
        agg(per_row[~per_row["is_no_pii"]], "with_pii"),
        agg(per_row[per_row["is_no_pii"]], "no_pii"),
    ])


# ---------- ratio sweep (monotonic) ----------

def ratio_sweep(
    df: pd.DataFrame,
    ratios=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    seed: int = 0,
    mode: str = "monotonic",
    baseline_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for r in ratios:
        sub = select_rows(df, pii_ratio=r, mode=mode, seed=seed)
        valid = sub[sub["_valid"]]
        rec = {
            "pii_ratio": r,
            "n_rows": len(sub),
            "n_with_pii": int((~sub["_is_no_pii"]).sum()),
            "n_no_pii": int(sub["_is_no_pii"].sum()),
            "n_valid": len(valid),
            "quality": _avg(valid["quality"]) if "quality" in valid else float("nan"),
            "weighted_leakage": _avg(valid["weighted_leakage"]) if "weighted_leakage" in valid else float("nan"),
            "latency_total_ms": _avg(valid["latency_total_ms"]) if "latency_total_ms" in valid else float("nan"),
            "source": "method",
        }
        rows.append(rec)
        if baseline_df is not None:
            sub_b = select_rows(baseline_df, pii_ratio=r, mode=mode, seed=seed)
            valid_b = sub_b[sub_b["_valid"]]
            rows.append({
                "pii_ratio": r,
                "n_rows": len(sub_b),
                "n_with_pii": int((~sub_b["_is_no_pii"]).sum()),
                "n_no_pii": int(sub_b["_is_no_pii"].sum()),
                "n_valid": len(valid_b),
                "quality": _avg(valid_b["quality"]) if "quality" in valid_b else float("nan"),
                "weighted_leakage": _avg(valid_b["weighted_leakage"]) if "weighted_leakage" in valid_b else float("nan"),
                "latency_total_ms": _avg(valid_b["latency_total_ms"]) if "latency_total_ms" in valid_b else float("nan"),
                "source": "baseline",
            })
    return pd.DataFrame(rows)


def plot_sweep(sweep_df: pd.DataFrame, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for src, sub in sweep_df.groupby("source"):
        sub = sub.sort_values("pii_ratio")
        axes[0].plot(sub["pii_ratio"], sub["quality"], marker="o", label=src)
        axes[1].plot(sub["pii_ratio"], sub["weighted_leakage"], marker="o", label=src)
        axes[2].plot(sub["pii_ratio"], sub["latency_total_ms"], marker="o", label=src)
    axes[0].set_title("Quality vs pii ratio"); axes[0].set_xlabel("pii ratio"); axes[0].set_ylabel("quality"); axes[0].legend()
    axes[1].set_title("Weighted leakage vs pii ratio"); axes[1].set_xlabel("pii ratio"); axes[1].set_ylabel("weighted_leakage"); axes[1].legend()
    axes[2].set_title("Total latency (ms) vs pii ratio"); axes[2].set_xlabel("pii ratio"); axes[2].set_ylabel("latency_total_ms"); axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_counts(df: pd.DataFrame, label: str) -> None:
    n = len(df)
    n_no = int(df["_is_no_pii"].sum())
    n_with = n - n_no
    n_valid = int(df["_valid"].sum())
    sources = ", ".join(sorted(df["_source_csv"].unique()))
    print(f"[{label}] total={n}  with_pii={n_with}  no_pii={n_no}  valid_for_avg={n_valid}  sources=[{sources}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_csv", nargs="+", required=True,
                   help="One or more eval CSVs from evaluate_papillon.py. They are concatenated into one pool.")
    p.add_argument("--baseline_csv", nargs="+", default=None,
                   help="One or more baseline eval CSVs for comparison.")
    p.add_argument("--out_dir", default="papillon/eval_results/analysis")
    p.add_argument("--head", type=int, default=None, help="Use only the first N rows of each input CSV")
    p.add_argument("--no_pii_n", type=int, default=None)
    p.add_argument("--pii_n", type=int, default=None)
    p.add_argument("--pii_ratio", type=float, default=None,
                   help="Fix the pii share of analyzed rows: pii / (pii + no_pii), 0..1. "
                        "1.0 = pii only, 0.5 = 1:1, 0.0 = no_pii only. "
                        "Ignored if --no_pii_n/--pii_n set.")
    p.add_argument("--sample_mode", choices=["monotonic", "random"], default="monotonic",
                   help="monotonic = sorted-prefix (strict superset as the no_pii target grows); random = seeded random sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true", help="Also produce ratio sweep CSV + PNG (pii_ratio on x-axis)")
    p.add_argument("--sweep_ratios", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    p.add_argument("--print_counts_only", action="store_true",
                   help="Just load the pool, print counts, and exit -- helps you pick a ratio.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_raw = load_eval_csvs(args.eval_csv, head=args.head)
    print_counts(df_raw, "eval/raw")

    baseline_raw = None
    if args.baseline_csv:
        baseline_raw = load_eval_csvs(args.baseline_csv, head=args.head)
        print_counts(baseline_raw, "baseline/raw")

    if args.print_counts_only:
        return

    df = select_rows(df_raw, no_pii_n=args.no_pii_n, pii_n=args.pii_n,
                     pii_ratio=args.pii_ratio, mode=args.sample_mode, seed=args.seed)
    print_counts(df, "eval/selected")

    baseline = None
    if baseline_raw is not None:
        baseline = select_rows(baseline_raw, no_pii_n=args.no_pii_n, pii_n=args.pii_n,
                               pii_ratio=args.pii_ratio, mode=args.sample_mode, seed=args.seed)
        print_counts(baseline, "baseline/selected")

    # ---- summary table (utility + privacy per group) ----
    summary = build_summary_table(df)
    print("\n=== metric summary (method) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    if baseline is not None:
        b_summary = build_summary_table(baseline)
        print("\n=== metric summary (baseline) ===")
        print(b_summary.to_string(index=False))
        b_summary.to_csv(os.path.join(args.out_dir, "summary_baseline.csv"), index=False)

    # ---- latency table ----
    latency = build_latency_table(df, baseline)
    print("\n=== latency table (ms, fixed pii ratio) ===")
    print(latency.to_string(index=False))
    latency.to_csv(os.path.join(args.out_dir, "latency_table.csv"), index=False)

    # ---- NER FP / FN ----
    if "detected_pii_json" in df.columns:
        per_row = ner_per_row(df)
        per_row.to_csv(os.path.join(args.out_dir, "ner_per_row.csv"), index=False)
        agg = ner_aggregate(per_row)
        print("\n=== NER FP / FN (method, fixed pii ratio) ===")
        print(agg.to_string(index=False))
        agg.to_csv(os.path.join(args.out_dir, "ner_aggregate.csv"), index=False)
    else:
        print("[WARN] eval CSV has no detected_pii_json column; skipping NER stats")

    # ---- ratio sweep ----
    if args.sweep:
        ratios = tuple(float(x) for x in args.sweep_ratios.split(","))
        sweep = ratio_sweep(df_raw, ratios=ratios, seed=args.seed,
                            mode=args.sample_mode, baseline_df=baseline_raw)
        sweep.to_csv(os.path.join(args.out_dir, "ratio_sweep.csv"), index=False)
        plot_sweep(sweep, os.path.join(args.out_dir, "ratio_sweep.png"))
        print(f"\nratio sweep written -> {args.out_dir}/ratio_sweep.csv (+ .png)")

    print(f"\nAll outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
