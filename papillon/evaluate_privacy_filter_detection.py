from argparse import ArgumentParser
import os
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


def join_units(units):
    return "||".join(units)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="user_query")
    parser.add_argument("--pii_column", type=str, default="pii_units")
    parser.add_argument("--pii_score_threshold", type=float, default=0.5)
    parser.add_argument("--output_csv", type=str, default=None)
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

    # Row-level binary confusion (PII exists in row or not)
    row_level_tp = 0
    row_level_fp = 0
    row_level_fn = 0
    row_level_tn = 0

    row_records = []

    for row_idx, row in tqdm.tqdm(data_frame.iterrows(), total=len(data_frame)):
        query = row.get(args.text_column, "")
        query_text = query if isinstance(query, str) else ""

        gt_units_raw = parse_pii_units(row.get(args.pii_column))
        gt_units = [normalize_text(u) for u in gt_units_raw]

        result = privacy_filter.analyze(query_text)
        pred_entities = unique_predicted_entities(result.entities)
        pred_units = [entity["text"] for entity in pred_entities]
        pred_units_raw = [entity["raw_text"] for entity in pred_entities]

        gt_has_pii = len(gt_units) > 0
        pred_has_pii = len(pred_units) > 0
        if gt_has_pii and pred_has_pii:
            row_level_case = "TP"
            row_level_tp += 1
        elif (not gt_has_pii) and pred_has_pii:
            row_level_case = "FP"
            row_level_fp += 1
        elif gt_has_pii and (not pred_has_pii):
            row_level_case = "FN"
            row_level_fn += 1
        else:
            row_level_case = "TN"
            row_level_tn += 1

        if gt_units:
            rows_with_gt_pii += 1
        else:
            rows_without_gt_pii += 1

        total_gt_units += len(gt_units)
        total_pred_units += len(pred_units)

        matched_gt = set()
        matched_pred = set()
        matched_pairs = {}

        for gt_idx, gt_unit in enumerate(gt_units):
            for pred_idx, pred_unit in enumerate(pred_units):
                if pred_idx in matched_pred:
                    continue
                if unit_match(gt_unit, pred_unit):
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    matched_pairs[gt_idx] = pred_idx
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

        row_tp_units = [gt_units_raw[i] for i in sorted(matched_gt)]
        row_fn_units = [gt_units_raw[i] for i in range(len(gt_units_raw)) if i not in matched_gt]
        row_fp_units = [pred_units_raw[i] for i in range(len(pred_units_raw)) if i not in matched_pred]
        matched_pred_units = [pred_units_raw[matched_pairs[i]] for i in sorted(matched_pairs.keys())]

        row_records.append(
            {
                "row_index": row_idx,
                "conversation_hash": row.get("conversation_hash", ""),
                "user_query": query_text,
                "gt_has_pii": gt_has_pii,
                "pred_has_pii": pred_has_pii,
                "row_level_case": row_level_case,
                "gt_units": join_units(gt_units_raw),
                "pred_units": join_units(pred_units_raw),
                "tp_units": join_units(row_tp_units),
                "matched_pred_units": join_units(matched_pred_units),
                "fn_units": join_units(row_fn_units),
                "fp_units": join_units(row_fp_units),
                "tp_count": row_tp,
                "fp_count": row_fp,
                "fn_count": row_fn,
                "detector_available": result.detector_available,
                "detector_uncertain": result.uncertain,
                "detector_error": result.error or "",
            }
        )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fp_per_row = safe_div(fp, len(data_frame))
    fn_per_gt_row = safe_div(rows_with_fn, rows_with_gt_pii)

    row_precision = safe_div(row_level_tp, row_level_tp + row_level_fp)
    row_recall = safe_div(row_level_tp, row_level_tp + row_level_fn)
    row_f1 = safe_div(2 * row_precision * row_recall, row_precision + row_recall)
    row_accuracy = safe_div(row_level_tp + row_level_tn, len(data_frame))

    base_path = os.path.splitext(args.dataset_file)[0]
    output_csv = args.output_csv or f"{base_path}_privacy_filter_eval.csv"
    row_columns = [
        "row_index",
        "conversation_hash",
        "user_query",
        "gt_has_pii",
        "pred_has_pii",
        "row_level_case",
        "gt_units",
        "pred_units",
        "tp_units",
        "matched_pred_units",
        "fn_units",
        "fp_units",
        "tp_count",
        "fp_count",
        "fn_count",
        "detector_available",
        "detector_uncertain",
        "detector_error",
    ]
    pandas.DataFrame(row_records, columns=row_columns).to_csv(output_csv, index=False)

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
    print("-----")
    print("===== Row-Level PII Presence Confusion =====")
    print(f"row_tp                  : {row_level_tp}")
    print(f"row_fp                  : {row_level_fp}")
    print(f"row_fn                  : {row_level_fn}")
    print(f"row_tn                  : {row_level_tn}")
    print(f"row_precision           : {row_precision:.6f}")
    print(f"row_recall              : {row_recall:.6f}")
    print(f"row_f1                  : {row_f1:.6f}")
    print(f"row_accuracy            : {row_accuracy:.6f}")
    print("-----")
    print(f"output_csv              : {output_csv}")
