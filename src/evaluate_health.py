"""Health metric evaluation (post-hoc).

Reads `test_predictions_*.json` and `usda_mapping.json`, prints / saves
Δ-nutrient stats and satisfaction rates (PDF: "건강 달성 — 영양 개선율,
목표 달성률"). Iterates `HEALTH_NUTRIENT_KEYS` so the metric set stays in
sync with the goal-vector / L_health definition.

The `ranks` field added to recent test_predictions JSON is intentionally
ignored here — health metrics are computed only on the predicted top-1.

Usage:
    python evaluate_health.py \\
        --predictions ./out/v3/test_predictions_v3_auto.json \\
        --usda ./data/usda_mapping.json
"""

import argparse
import json
import os

import numpy as np

from dataset import HEALTH_NUTRIENT_KEYS, _safe_float


def _safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def _rate(xs, threshold=0.0):
    if not xs:
        return float("nan")
    return float((np.asarray(xs) > threshold).mean() * 100)


def compute_health_metrics(predictions_path, usda_path, save_to=None):
    with open(predictions_path) as f:
        pred = json.load(f)
    with open(usda_path) as f:
        usda = {int(k): v for k, v in json.load(f).items()}

    sources = pred["sources"]
    top1 = pred["top1"]
    targets = pred["targets"]

    # Δ relative to source: positive = predicted ingredient has LESS of the nutrient
    deltas_pred = {k: [] for k in HEALTH_NUTRIENT_KEYS}
    deltas_gt = {k: [] for k in HEALTH_NUTRIENT_KEYS}
    n_unmapped = 0

    for s, p, y in zip(sources, top1, targets):
        if s in usda and p in usda:
            for k in HEALTH_NUTRIENT_KEYS:
                deltas_pred[k].append(_safe_float(usda[s].get(k))
                                       - _safe_float(usda[p].get(k)))
        else:
            n_unmapped += 1

        if s in usda and y in usda:
            for k in HEALTH_NUTRIENT_KEYS:
                deltas_gt[k].append(_safe_float(usda[s].get(k))
                                     - _safe_float(usda[y].get(k)))

    out = {
        "mode": pred.get("mode", "?"),
        "g_label": pred.get("g_label"),
        "n_total": len(sources),
        "n_evaluated_pred": len(deltas_pred[HEALTH_NUTRIENT_KEYS[0]]),
        "n_unmapped": n_unmapped,
        "nutrient_keys": list(HEALTH_NUTRIENT_KEYS),
        "pred": {
            k: {"avg_delta": _safe_mean(deltas_pred[k]),
                "satisfaction_rate": _rate(deltas_pred[k], threshold=0.0)}
            for k in HEALTH_NUTRIENT_KEYS
        },
        "gt": {
            k: {"avg_delta": _safe_mean(deltas_gt[k]),
                "satisfaction_rate": _rate(deltas_gt[k], threshold=0.0)}
            for k in HEALTH_NUTRIENT_KEYS
        },
    }

    print(f"\n=== Health Metrics ({out['mode']}, g={out['g_label']}) ===")
    print(f"  n_total      : {out['n_total']}")
    print(f"  n_unmapped   : {out['n_unmapped']}")
    print(f"  n_evaluated  : {out['n_evaluated_pred']}")
    print(f"  --- Predicted top-1 vs Source (Δ = source − pred, positive = healthier) ---")
    for k in HEALTH_NUTRIENT_KEYS:
        unit = "g" if k.endswith("_g") else ("mg" if k.endswith("_mg") else "")
        print(f"  {k:<14} avg Δ {out['pred'][k]['avg_delta']:+.3f} {unit:<2} "
              f"sat. {out['pred'][k]['satisfaction_rate']:5.1f} %")
    print(f"  --- Ground truth y vs Source (reference) ---")
    for k in HEALTH_NUTRIENT_KEYS:
        unit = "g" if k.endswith("_g") else ("mg" if k.endswith("_mg") else "")
        print(f"  {k:<14} avg Δ {out['gt'][k]['avg_delta']:+.3f} {unit:<2} "
              f"sat. {out['gt'][k]['satisfaction_rate']:5.1f} %")

    if save_to:
        with open(save_to, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {save_to}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to test_predictions_*.json from train_*.py")
    parser.add_argument("--usda", type=str, required=True,
                        help="Path to usda_mapping.json")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional output JSON path")
    args = parser.parse_args()

    save_to = args.save
    if save_to is None:
        base, _ = os.path.splitext(args.predictions)
        save_to = f"{base}_health.json"

    compute_health_metrics(args.predictions, args.usda, save_to=save_to)
