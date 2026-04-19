from argparse import ArgumentParser
import re

import pandas
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

    # Allow containment for partially detected spans (e.g., "johnny bay" vs "johnny").
    # Guard very short strings to reduce accidental matches.
    shorter_len = min(len(gt_norm), len(pred_norm))
    if shorter_len >= 4 and (gt_norm in pred_norm or pred_norm in gt_norm):
        return True
    return False


def unique_predicted_units(entities):
    seen = set()
    units = []
    for entity in entities:
        text = normalize_text(getattr(entity, "text", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        units.append(text)
    return units


def safe_div(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="user_query")
    parser.add_argument("--pii_column", type=str, default="pii_units")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    data_frame = pandas.read_csv(args.dataset_file)
    privacy_filter = PrivacyFilter(score_threshold=args.pii_score_threshold)

    tp = 0
    fp = 0
    fn = 0
    total_gt_units = 0
    total_pred_units = 0
    rows_with_fp = 0
    rows_with_fn = 0
    rows_with_gt_pii = 0
    rows_without_gt_pii = 0

    for _, row in tqdm.tqdm(data_frame.iterrows(), total=len(data_frame)):
        query = row.get(args.text_column, "")
        gt_units = [normalize_text(u) for u in parse_pii_units(row.get(args.pii_column))]
        result = privacy_filter.analyze(query if isinstance(query, str) else "")
        pred_units = unique_predicted_units(result.entities)

        if gt_units:
            rows_with_gt_pii += 1
        else:
            rows_without_gt_pii += 1

        total_gt_units += len(gt_units)
        total_pred_units += len(pred_units)

        matched_gt = set()
        matched_pred = set()

        for gt_idx, gt_unit in enumerate(gt_units):
            for pred_idx, pred_unit in enumerate(pred_units):
                if pred_idx in matched_pred:
                    continue
                if unit_match(gt_unit, pred_unit):
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    break

        row_tp = len(matched_gt)
        row_fn = len(gt_units) - row_tp
        row_fp = len(pred_units) - row_tp

        tp += row_tp
        fn += row_fn
        fp += row_fp

        if row_fp > 0:
            rows_with_fp += 1
        if row_fn > 0:
            rows_with_fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fp_per_row = safe_div(fp, len(data_frame))
    fn_per_gt_row = safe_div(rows_with_fn, rows_with_gt_pii)

    print("===== Privacy Filter Unit-Level Detection =====")
    print(f"dataset_file            : {args.dataset_file}")
    print(f"rows_total              : {len(data_frame)}")
    print(f"rows_with_gt_pii        : {rows_with_gt_pii}")
    print(f"rows_without_gt_pii     : {rows_without_gt_pii}")
    print(f"total_gt_units          : {total_gt_units}")
    print(f"total_pred_units        : {total_pred_units}")
    print("-----")
    print(f"tp_units                : {tp}")
    print(f"fp_units                : {fp}")
    print(f"fn_units                : {fn}")
    print("-----")
    print(f"precision               : {precision:.6f}")
    print(f"recall                  : {recall:.6f}")
    print(f"f1                      : {f1:.6f}")
    print(f"avg_fp_units_per_row    : {fp_per_row:.6f}")
    print(f"rows_with_fp            : {rows_with_fp}")
    print(f"rows_with_fn            : {rows_with_fn}")
    print(f"fn_row_rate_on_gt_pii   : {fn_per_gt_row:.6f}")
