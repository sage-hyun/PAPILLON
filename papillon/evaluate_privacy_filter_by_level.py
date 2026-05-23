import argparse
import os
import re
import pandas as pd
import tqdm

try:
    from .privacy_filter import PrivacyFilter
except ImportError:
    from privacy_filter import PrivacyFilter


def parse_pii_units(pii_str):
    if not isinstance(pii_str, str):
        return []
    seen = set()
    units = []
    for piece in pii_str.split("||"):
        cleaned = piece.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        units.append(cleaned)
    return units


def normalize_text(text):
    return " ".join((text or "").strip().lower().split())


def alnum_text(text):
    return re.sub(r"[^a-z0-9]+", "", normalize_text(text))


def unit_match(gt_unit, pred_unit):
    gt_norm = normalize_text(gt_unit)
    pred_norm = normalize_text(pred_unit)
    if not gt_norm or not pred_norm:
        return False
    if gt_norm == pred_norm:
        return True

    gt_alnum = alnum_text(gt_norm)
    pred_alnum = alnum_text(pred_norm)
    if gt_alnum and pred_alnum and gt_alnum == pred_alnum:
        return True

    shorter_len = min(len(gt_norm), len(pred_norm))
    if shorter_len >= 4 and (gt_norm in pred_norm or pred_norm in gt_norm):
        return True
    return False


def unique_predicted_entities(entities):
    seen = set()
    output = []
    for entity in entities:
        raw_text = (getattr(entity, "text", "") or "").strip()
        text = normalize_text(raw_text)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(
            {
                "text": text,
                "raw_text": raw_text,
                "entity_type": getattr(entity, "entity_type", ""),
                "source": getattr(entity, "source", ""),
                "score": getattr(entity, "score", ""),
            }
        )
    return output


def safe_div(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator



def evaluate_row_pii(l1_gt_raw, l2_gt_raw, pred_entities_raw):
    """
    Calculate TP, FP, FN on Total PII, and track matched/missed counts for L1 and L2.
    """
    l1_gt = [normalize_text(u) for u in l1_gt_raw]
    l2_gt = [normalize_text(u) for u in l2_gt_raw]
    total_gt = l1_gt + l2_gt
    l1_len = len(l1_gt)

    pred_units = [normalize_text(entity["text"]) for entity in pred_entities_raw]

    matched_gt = set()
    matched_pred = set()

    for gt_idx, gt_unit in enumerate(total_gt):
        for pred_idx, pred_unit in enumerate(pred_units):
            if pred_idx in matched_pred:
                continue
            if unit_match(gt_unit, pred_unit):
                matched_gt.add(gt_idx)
                matched_pred.add(pred_idx)
                break

    # Calculate L1 and L2 matched counts
    l1_tp = sum(1 for idx in matched_gt if idx < l1_len)
    l1_fn = l1_len - l1_tp

    l2_tp = sum(1 for idx in matched_gt if idx >= l1_len)
    l2_fn = len(l2_gt) - l2_tp

    # Calculate total TP, FP, FN
    total_tp = len(matched_gt)
    total_fn = len(total_gt) - total_tp
    total_fp = len(pred_units) - total_tp

    return {
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "l1_tp": l1_tp,
        "l1_fn": l1_fn,
        "l2_tp": l2_tp,
        "l2_fn": l2_fn,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Privacy Filter (Total PII metrics + L1/L2 Coverage)")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to leveling CSV (e.g. pupa/PUPA_TNB_leveling.csv)")
    parser.add_argument("--text_column", type=str, default="user_query")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    parser.add_argument("--model_name", type=str, default="en_core_web_lg", help="SpaCy model name (en_core_web_sm, en_core_web_lg)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed row-by-row metrics to terminal")
    parser.add_argument("--save_details", type=str, default=None, help="Path to save row-by-row metrics as a CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_file)
    privacy_filter = PrivacyFilter(score_threshold=args.pii_score_threshold, model_name=args.model_name)

    # Accumulators for metrics
    metrics = {
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "total_gt_count": 0,
        "l1_tp": 0,
        "l1_fn": 0,
        "l1_gt_count": 0,
        "l2_tp": 0,
        "l2_fn": 0,
        "l2_gt_count": 0,
    }

    # Row-level presence metrics
    row_tp = 0
    row_fp = 0
    row_fn = 0
    row_tn = 0

    detailed_rows = []

    print(f"Loaded {len(df)} rows from {args.dataset_file}.")
    print(f"Evaluating Privacy Filter (Presidio NER using {args.model_name} + Regex Hybrid) performance...")

    for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        query = row.get(args.text_column, "")
        query_text = query if isinstance(query, str) else ""

        # Parse ground truths
        l1_gt = parse_pii_units(row.get("l1_units"))
        l2_gt = parse_pii_units(row.get("l2_units"))

        # Analyze query using Presidio + Regex
        result = privacy_filter.analyze(query_text)
        pred_entities = unique_predicted_entities(result.entities)

        # Evaluate unit-level row metrics
        row_res = evaluate_row_pii(l1_gt, l2_gt, pred_entities)

        # Calculate unit-level metrics for this specific row
        r_tp = row_res["total_tp"]
        r_fp = row_res["total_fp"]
        r_fn = row_res["total_fn"]
        r_precision = safe_div(r_tp, r_tp + r_fp)
        r_recall = safe_div(r_tp, r_tp + r_fn)
        r_f1 = safe_div(2 * r_precision * r_recall, r_precision + r_recall)

        if args.verbose:
            print(f"\n[Row {idx}] Text: {query_text}")
            print(f"  GT L1: {l1_gt}")
            print(f"  GT L2: {l2_gt}")
            print(f"  Pred : {[e['text'] for e in pred_entities]}")
            print(f"  Row Metrics -> Precision: {r_precision:.4f}, Recall: {r_recall:.4f}, F1: {r_f1:.4f}")

        if args.save_details:
            detailed_rows.append({
                "row_index": idx,
                "text": query_text,
                "l1_gt": str(l1_gt),
                "l2_gt": str(l2_gt),
                "predictions": str([e['text'] for e in pred_entities]),
                "tp": r_tp,
                "fp": r_fp,
                "fn": r_fn,
                "precision": r_precision,
                "recall": r_recall,
                "f1": r_f1
            })

        # Accumulate unit-level metrics
        metrics["total_tp"] += r_tp
        metrics["total_fp"] += r_fp
        metrics["total_fn"] += r_fn
        metrics["total_gt_count"] += (len(l1_gt) + len(l2_gt))

        metrics["l1_tp"] += row_res["l1_tp"]
        metrics["l1_fn"] += row_res["l1_fn"]
        metrics["l1_gt_count"] += len(l1_gt)

        metrics["l2_tp"] += row_res["l2_tp"]
        metrics["l2_fn"] += row_res["l2_fn"]
        metrics["l2_gt_count"] += len(l2_gt)

        # Row-level presence (Binary classification: Does row have any PII?)
        gt_has_pii = (len(l1_gt) + len(l2_gt)) > 0
        pred_has_pii = len(pred_entities) > 0

        if gt_has_pii and pred_has_pii:
            row_tp += 1
        elif (not gt_has_pii) and pred_has_pii:
            row_fp += 1
        elif gt_has_pii and (not pred_has_pii):
            row_fn += 1
        else:
            row_tn += 1

    # Compute Total Unit Metrics
    tp = metrics["total_tp"]
    fp = metrics["total_fp"]
    fn = metrics["total_fn"]
    total_gt = metrics["total_gt_count"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    # Compute Coverage Breakdown
    l1_recall = safe_div(metrics["l1_tp"], metrics["l1_gt_count"])
    l2_recall = safe_div(metrics["l2_tp"], metrics["l2_gt_count"])

    # Compute Row-Level Metrics
    row_precision = safe_div(row_tp, row_tp + row_fp)
    row_recall = safe_div(row_tp, row_tp + row_fn)
    row_f1 = safe_div(2 * row_precision * row_recall, row_precision + row_recall)
    row_accuracy = safe_div(row_tp + row_tn, len(df))

    # Print Report
    print("\n" + "="*50)
    print(f"      PRIVACY FILTER PERFORMANCE ({args.model_name.upper()})")
    print("="*50)

    print("\n[UNIT-LEVEL PII METRICS (L1 + L2 combined)]")
    print(f"  Ground Truth Count : {total_gt}")
    print(f"  TP (True Positives) : {tp}")
    print(f"  FP (False Positives): {fp}")
    print(f"  FN (False Negatives): {fn}")
    print(f"  Precision           : {precision:.6f}")
    print(f"  Recall (Overall)    : {recall:.6f}")
    print(f"  F1-Score            : {f1:.6f}")

    print("\n[LEVEL BREAKDOWN - COVERAGE / RECALL]")
    print(f"  L1 (Direct Identifiers)")
    print(f"    Ground Truth Count: {metrics['l1_gt_count']}")
    print(f"    Matched (TP)      : {metrics['l1_tp']}")
    print(f"    Missed (FN)       : {metrics['l1_fn']}")
    print(f"    L1 Recall         : {l1_recall:.6f}")
    print(f"  L2 (Quasi-Identifiers)")
    print(f"    Ground Truth Count: {metrics['l2_gt_count']}")
    print(f"    Matched (TP)      : {metrics['l2_tp']}")
    print(f"    Missed (FN)       : {metrics['l2_fn']}")
    print(f"    L2 Recall         : {l2_recall:.6f}")

    print("\n[ROW-LEVEL BINARY PII PRESENCE METRICS]")
    print(f"  Total Rows          : {len(df)}")
    print(f"  Row TP (Correct PII): {row_tp}")
    print(f"  Row FP (False Alarm): {row_fp}")
    print(f"  Row FN (Missed PII) : {row_fn}")
    print(f"  Row TN (Correct non): {row_tn}")
    print(f"  Row Precision       : {row_precision:.6f}")
    print(f"  Row Recall          : {row_recall:.6f}")
    print(f"  Row F1-Score        : {row_f1:.6f}")
    print(f"  Row Accuracy        : {row_accuracy:.6f}")
    print("="*50 + "\n")

    if args.save_details and detailed_rows:
        details_df = pd.DataFrame(detailed_rows)
        details_df.to_csv(args.save_details, index=False)
        print(f"Saved detailed row-by-row metrics to: {args.save_details}\n")
