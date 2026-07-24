import math

import numpy as np
from scipy import ndimage


CLASS_NAMES = {
    1: "ncr_net",
    2: "edema",
    3: "enhancing",
}

REGIONS = {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3],
}


def binary_confusion(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    tn = int(np.logical_and(~pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    return tp, fp, tn, fn


def overlap_metrics_from_confusion(tp, fp, tn, fn):
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) else 1.0
    f1 = dice
    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "Sensitivity": float(recall),
        "Specificity": float(specificity),
        "F1": float(f1),
    }


def surface_distances(pred, target, spacing=(1.0, 1.0, 1.0)):
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return {"HD95": 0.0, "ASD": 0.0, "ASSD": 0.0, "surface_status": "both_empty"}
    if pred.any() != target.any():
        return {"HD95": math.inf, "ASD": math.inf, "ASSD": math.inf, "surface_status": "one_empty"}

    pred_border = np.logical_xor(pred, ndimage.binary_erosion(pred))
    target_border = np.logical_xor(target, ndimage.binary_erosion(target))
    dt_pred = ndimage.distance_transform_edt(~pred_border, sampling=spacing)
    dt_target = ndimage.distance_transform_edt(~target_border, sampling=spacing)
    d_pred_to_target = dt_target[pred_border]
    d_target_to_pred = dt_pred[target_border]
    all_dist = np.concatenate([d_pred_to_target, d_target_to_pred])
    return {
        "HD95": float(np.percentile(all_dist, 95)) if all_dist.size else 0.0,
        "ASD": float(np.mean(d_pred_to_target)) if d_pred_to_target.size else 0.0,
        "ASSD": float(np.mean(all_dist)) if all_dist.size else 0.0,
        "surface_status": "ok",
    }


def rvd(pred, target):
    pred_v = float(pred.astype(bool).sum())
    target_v = float(target.astype(bool).sum())
    if target_v == 0:
        return 0.0 if pred_v == 0 else math.inf
    return float((pred_v - target_v) / target_v)


def compute_case_metrics(pred_label, target_label, case_id, include_surface=True):
    row = {"case_id": case_id, "empty_label": bool(np.max(target_label) == 0), "exception": ""}
    confusion = {}
    targets = {**{name: [idx] for idx, name in CLASS_NAMES.items()}, **REGIONS}
    for name, labels in targets.items():
        pred_bin = np.isin(pred_label, labels)
        target_bin = np.isin(target_label, labels)
        tp, fp, tn, fn = binary_confusion(pred_bin, target_bin)
        confusion[name] = (tp, fp, tn, fn)
        metrics = overlap_metrics_from_confusion(tp, fp, tn, fn)
        metrics["RVD"] = rvd(pred_bin, target_bin)
        if include_surface:
            metrics.update(surface_distances(pred_bin, target_bin))
        for key, value in metrics.items():
            row[f"{name}_{key}"] = value
        row[f"{name}_TP"] = tp
        row[f"{name}_FP"] = fp
        row[f"{name}_TN"] = tn
        row[f"{name}_FN"] = fn
    return row, confusion


def summarize_case_metrics(case_rows, global_confusion, metadata):
    summary = dict(metadata)
    metric_suffixes = ["Dice", "IoU", "Precision", "Recall", "Sensitivity", "Specificity", "F1", "HD95", "ASD", "ASSD", "RVD"]
    metric_columns = [c for c in case_rows[0].keys() if any(c.endswith("_" + s) for s in metric_suffixes)] if case_rows else []
    for col in metric_columns:
        values = np.array([r[col] for r in case_rows if np.isfinite(r[col])], dtype=np.float64)
        excluded = len(case_rows) - len(values)
        if len(values) == 0:
            summary[f"{col}_case_mean"] = math.nan
            summary[f"{col}_case_std"] = math.nan
            summary[f"{col}_case_median"] = math.nan
        else:
            summary[f"{col}_case_mean"] = float(np.mean(values))
            summary[f"{col}_case_std"] = float(np.std(values))
            summary[f"{col}_case_median"] = float(np.median(values))
            summary[f"{col}_case_q25"] = float(np.percentile(values, 25))
            summary[f"{col}_case_q75"] = float(np.percentile(values, 75))
        summary[f"{col}_nonfinite_excluded"] = int(excluded)

    for name, conf in global_confusion.items():
        tp, fp, tn, fn = conf
        for key, value in overlap_metrics_from_confusion(tp, fp, tn, fn).items():
            summary[f"{name}_{key}_global"] = value
        summary[f"{name}_TP_global"] = int(tp)
        summary[f"{name}_FP_global"] = int(fp)
        summary[f"{name}_TN_global"] = int(tn)
        summary[f"{name}_FN_global"] = int(fn)
    return summary
