"""NER (privacy_filter) FP/FN by pii_ratio, run directly on the source datasets.

Why a separate script: evaluate_papillon.py's CSV stores `detected_pii_json`
that was produced at eval time, but for a clean NER-only evaluation we
re-run the privacy filter on raw queries here. The unit-level TP/FP/FN
matching logic is the same as evaluate_privacy_filter_by_level.py
(`evaluate_row_pii`).

Per-row NER is the expensive step, so we run it ONCE per unique row in the
combined pool, cache (tp, fp, fn) per row, then reuse the cache to aggregate
across pii_ratio values via the monotonic-prefix subsampling rule.

Ratio convention matches analyze_eval_csv.py:
    pii_ratio = n_with_pii / (n_with_pii + n_no_pii)
    1.0 = with_pii only, 0.5 = 1:1, 0.0 = no_pii only.

Usage:
    python papillon/analyze_ner_by_ratio.py \\
        --dataset_file pupa/PUPA_TNB_leveling.csv pupa/PUPA_No_PII.csv \\
        --pii_ratio 0.5 --sweep \\
        --out_dir papillon/eval_results/analysis/ner_run1
"""

import argparse
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import tqdm

try:
    from .privacy_filter import PrivacyFilter
    from .evaluate_privacy_filter_by_level import (
        parse_pii_units,
        unique_predicted_entities,
        evaluate_row_pii,
        safe_div,
    )
except ImportError:
    from privacy_filter import PrivacyFilter
    from evaluate_privacy_filter_by_level import (
        parse_pii_units,
        unique_predicted_entities,
        evaluate_row_pii,
        safe_div,
    )


def _is_no_pii(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "nan"


def load_datasets(paths: Iterable[str], head: Optional[int] = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if head is not None and head > 0:
            df = df.iloc[:head].copy()
        df["_source_csv"] = os.path.basename(path)
        df["_row_in_source"] = np.arange(len(df))
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool["_is_no_pii"] = pool["pii_units"].apply(_is_no_pii)
    sort_key = pool.get("conversation_hash", pd.Series([""] * len(pool))).fillna("").astype(str)
    pool["_sort_key"] = (
        sort_key + "|" + pool["_source_csv"].astype(str)
        + "|" + pool["_row_in_source"].astype(str).str.zfill(8)
    )
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
    pii_ratio: Optional[float],
    mode: str = "monotonic",
    seed: int = 0,
    no_pii_n: Optional[int] = None,
    pii_n: Optional[int] = None,
) -> pd.DataFrame:
    """Same rule as analyze_eval_csv.py.select_rows, but for the dataset pool."""
    rng = np.random.default_rng(seed)
    with_pii_df = df[~df["_is_no_pii"]]
    no_pii_df = df[df["_is_no_pii"]]

    if no_pii_n is None and pii_n is None and pii_ratio is None:
        return df

    if pii_ratio is not None and no_pii_n is None and pii_n is None:
        if pii_ratio >= 1.0:
            pii_n = len(with_pii_df); no_pii_n = 0
        elif pii_ratio <= 0.0:
            pii_n = 0; no_pii_n = len(no_pii_df)
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


def run_ner_per_row(
    pool: pd.DataFrame,
    privacy_filter: PrivacyFilter,
    text_column: str = "user_query",
) -> pd.DataFrame:
    """Run the privacy filter once per row in `pool` and return per-row TP/FP/FN."""
    records = []
    for _, row in tqdm.tqdm(pool.iterrows(), total=len(pool), desc="NER per row"):
        query = row.get(text_column, "")
        text = query if isinstance(query, str) else ""

        # PUPA_No_PII has no l1_units/l2_units columns -> empty gt
        l1_gt = parse_pii_units(row.get("l1_units")) if "l1_units" in pool.columns else []
        l2_gt = parse_pii_units(row.get("l2_units")) if "l2_units" in pool.columns else []

        result = privacy_filter.analyze(text)
        pred_entities = unique_predicted_entities(result.entities)
        res = evaluate_row_pii(l1_gt, l2_gt, pred_entities)

        records.append({
            "conversation_hash": row.get("conversation_hash", ""),
            "source_csv": row.get("_source_csv", ""),
            "row_in_source": int(row.get("_row_in_source", -1)),
            "is_no_pii": bool(row.get("_is_no_pii", False)),
            "_sort_key": row["_sort_key"],
            "n_gt_l1": len(l1_gt),
            "n_gt_l2": len(l2_gt),
            "n_pred": len(pred_entities),
            "tp": res["total_tp"],
            "fp": res["total_fp"],
            "fn": res["total_fn"],
            "l1_tp": res["l1_tp"], "l1_fn": res["l1_fn"],
            "l2_tp": res["l2_tp"], "l2_fn": res["l2_fn"],
            "pred_entities": "; ".join(e["text"] for e in pred_entities),
        })
    return pd.DataFrame(records)


def _agg_block(sub: pd.DataFrame, label: str) -> dict:
    tp = int(sub["tp"].sum()); fp = int(sub["fp"].sum()); fn = int(sub["fn"].sum())
    l1_tp = int(sub["l1_tp"].sum()); l1_fn = int(sub["l1_fn"].sum())
    l2_tp = int(sub["l2_tp"].sum()); l2_fn = int(sub["l2_fn"].sum())
    n = len(sub)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "group": label,
        "n_rows": n,
        "fp_per_row": float(sub["fp"].mean()) if n else float("nan"),
        "fn_per_row": float(sub["fn"].mean()) if n else float("nan"),
        "tp_total": tp, "fp_total": fp, "fn_total": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "l1_recall": safe_div(l1_tp, l1_tp + l1_fn),
        "l2_recall": safe_div(l2_tp, l2_tp + l2_fn),
    }


def aggregate_per_row(per_row: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        _agg_block(per_row, "overall"),
        _agg_block(per_row[~per_row["is_no_pii"]], "with_pii"),
        _agg_block(per_row[per_row["is_no_pii"]], "no_pii"),
    ])


def _select_per_row(per_row: pd.DataFrame, pii_ratio: float, mode: str, seed: int) -> pd.DataFrame:
    """Apply the same select_rows rule to the cached per-row table."""
    sub = per_row.rename(columns={"is_no_pii": "_is_no_pii"})
    sub = select_rows(sub, pii_ratio=pii_ratio, mode=mode, seed=seed)
    sub = sub.rename(columns={"_is_no_pii": "is_no_pii"})
    return sub


def sweep_ratios(
    per_row: pd.DataFrame,
    ratios=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    mode: str = "monotonic",
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for r in ratios:
        sub = _select_per_row(per_row, pii_ratio=r, mode=mode, seed=seed)
        agg = _agg_block(sub, "overall")
        agg.update({
            "pii_ratio": r,
            "n_with_pii": int((~sub["is_no_pii"]).sum()),
            "n_no_pii": int(sub["is_no_pii"].sum()),
        })
        rows.append(agg)
    out = pd.DataFrame(rows)
    cols = ["pii_ratio", "n_rows", "n_with_pii", "n_no_pii",
            "fp_per_row", "fn_per_row", "precision", "recall", "f1",
            "l1_recall", "l2_recall", "tp_total", "fp_total", "fn_total"]
    return out[[c for c in cols if c in out.columns]]


def plot_sweep(sweep_df: pd.DataFrame, out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    s = sweep_df.sort_values("pii_ratio")
    axes[0].plot(s["pii_ratio"], s["fp_per_row"], marker="o", label="FP / row")
    axes[0].plot(s["pii_ratio"], s["fn_per_row"], marker="s", label="FN / row")
    axes[0].set_xlabel("pii ratio"); axes[0].set_ylabel("count per row")
    axes[0].set_title("NER FP / FN per row vs pii_ratio"); axes[0].legend()
    axes[1].plot(s["pii_ratio"], s["precision"], marker="o", label="precision")
    axes[1].plot(s["pii_ratio"], s["recall"], marker="s", label="recall")
    axes[1].plot(s["pii_ratio"], s["f1"], marker="^", label="f1")
    axes[1].set_xlabel("pii ratio"); axes[1].set_ylabel("score")
    axes[1].set_title("NER precision / recall / F1 vs pii_ratio"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_pool_counts(pool: pd.DataFrame) -> None:
    n = len(pool)
    n_no = int(pool["_is_no_pii"].sum())
    n_with = n - n_no
    sources = ", ".join(sorted(pool["_source_csv"].unique()))
    print(f"[pool] total={n}  with_pii={n_with}  no_pii={n_no}  sources=[{sources}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_file", nargs="+", required=True,
                   help="One or more raw dataset CSVs (e.g. pupa/PUPA_TNB_leveling.csv pupa/PUPA_No_PII.csv).")
    p.add_argument("--text_column", default="user_query")
    p.add_argument("--pii_score_threshold", type=float, default=0.5)
    p.add_argument("--model_name", default="en_core_web_lg")
    p.add_argument("--out_dir", default="papillon/eval_results/analysis/ner")
    p.add_argument("--head", type=int, default=None, help="Only the first N rows per dataset file (debug).")
    p.add_argument("--pii_ratio", type=float, default=None,
                   help="Fix pii share = pii / (pii + no_pii), 0..1. 1.0 = pii only, 0.5 = 1:1.")
    p.add_argument("--sample_mode", choices=["monotonic", "random"], default="monotonic")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", action="store_true", help="Also produce a pii_ratio sweep CSV + PNG")
    p.add_argument("--sweep_ratios", default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    p.add_argument("--per_row_cache", default=None,
                   help="Optional path to load/save the per-row NER cache CSV. "
                        "If the file exists, NER is skipped and cache is used.")
    p.add_argument("--print_counts_only", action="store_true",
                   help="Load datasets, print pool counts, and exit -- helps pick ratios.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pool = load_datasets(args.dataset_file, head=args.head)
    print_pool_counts(pool)
    if args.print_counts_only:
        return

    cache_path = args.per_row_cache or os.path.join(args.out_dir, "ner_per_row.csv")
    if args.per_row_cache and os.path.exists(args.per_row_cache):
        per_row = pd.read_csv(args.per_row_cache)
        print(f"[cache] loaded per-row NER results from {args.per_row_cache} ({len(per_row)} rows)")
    else:
        privacy_filter = PrivacyFilter(score_threshold=args.pii_score_threshold,
                                       model_name=args.model_name)
        per_row = run_ner_per_row(pool, privacy_filter, text_column=args.text_column)
        per_row.to_csv(cache_path, index=False)
        print(f"[cache] wrote per-row NER results to {cache_path}")

    # ---- fixed-ratio aggregate ----
    if args.pii_ratio is not None:
        sub = _select_per_row(per_row, pii_ratio=args.pii_ratio,
                              mode=args.sample_mode, seed=args.seed)
        agg = aggregate_per_row(sub)
        print(f"\n=== NER FP/FN at pii_ratio={args.pii_ratio} "
              f"(with_pii={int((~sub['is_no_pii']).sum())}, "
              f"no_pii={int(sub['is_no_pii'].sum())}) ===")
        print(agg.to_string(index=False))
        agg.to_csv(os.path.join(args.out_dir, "ner_aggregate_fixed.csv"), index=False)
    else:
        agg = aggregate_per_row(per_row)
        print("\n=== NER FP/FN (full pool) ===")
        print(agg.to_string(index=False))
        agg.to_csv(os.path.join(args.out_dir, "ner_aggregate_fullpool.csv"), index=False)

    # ---- ratio sweep ----
    if args.sweep:
        ratios = tuple(float(x) for x in args.sweep_ratios.split(","))
        sweep = sweep_ratios(per_row, ratios=ratios,
                             mode=args.sample_mode, seed=args.seed)
        print("\n=== NER metrics vs pii_ratio ===")
        print(sweep.to_string(index=False))
        sweep.to_csv(os.path.join(args.out_dir, "ner_ratio_sweep.csv"), index=False)
        plot_sweep(sweep, os.path.join(args.out_dir, "ner_ratio_sweep.png"))
        print(f"\nsweep -> {args.out_dir}/ner_ratio_sweep.csv (+ .png)")

    print(f"\nAll outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()
