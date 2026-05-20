"""ID vs OOD split evaluation (post-hoc).

Reads a test_predictions_*.json (which now includes per-pair `ranks`) and
splits the test pairs into:

  ID  : (source, target) was seen as a substitution pair in pairs_train.csv
  OOD : (source, target) never appeared in training — model is generalizing
        to an unseen substitution direction.

Reports MRR / Hit@k separately on each subset and as an overall sanity check.

This is the GISMo-paper-style robustness check: "does the model still
substitute well when it has never seen this swap before?".

Usage:
    python evaluate_id_ood.py \\
        --predictions ./out/v3/test_predictions_v3_auto.json \\
        --train_pairs ./data/pairs_train.csv \\
        [--save ./out/v3/test_predictions_v3_auto_idood.json]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def split_id_ood(sources, targets, train_pair_set):
    """Return boolean mask: True = ID (seen in train), False = OOD."""
    is_id = np.zeros(len(sources), dtype=bool)
    for i, (s, t) in enumerate(zip(sources, targets)):
        if (int(s), int(t)) in train_pair_set:
            is_id[i] = True
    return is_id


def metrics_from_ranks(ranks):
    if len(ranks) == 0:
        return {k: float("nan") for k in ("MRR", "Hit@1", "Hit@3", "Hit@10")}
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "MRR":   float((1.0 / ranks).mean() * 100),
        "Hit@1": float((ranks <= 1).mean() * 100),
        "Hit@3": float((ranks <= 3).mean() * 100),
        "Hit@10": float((ranks <= 10).mean() * 100),
        "n":     int(len(ranks)),
    }


def evaluate_id_ood(predictions_path, train_pairs_csv, save_to=None):
    with open(predictions_path) as f:
        pred = json.load(f)

    if "ranks" not in pred:
        raise ValueError(
            f"{predictions_path} has no 'ranks' field — re-run training "
            f"with the latest train_*.py (now stores per-pair ranks)."
        )

    sources = pred["sources"]
    targets = pred["targets"]
    ranks = np.asarray(pred["ranks"], dtype=np.float64)

    # Build training pair set
    train_df = pd.read_csv(train_pairs_csv)
    train_pair_set = set(zip(train_df["source_id"].astype(int).tolist(),
                              train_df["target_id"].astype(int).tolist()))

    is_id = split_id_ood(sources, targets, train_pair_set)
    n_id = int(is_id.sum())
    n_ood = int((~is_id).sum())

    out = {
        "mode": pred.get("mode", "?"),
        "g_label": pred.get("g_label"),
        "n_total": len(ranks),
        "n_id": n_id,
        "n_ood": n_ood,
        "id_fraction": float(n_id / len(ranks)) if len(ranks) else 0.0,
        "overall": metrics_from_ranks(ranks),
        "id":      metrics_from_ranks(ranks[is_id]),
        "ood":     metrics_from_ranks(ranks[~is_id]),
    }

    print(f"\n=== ID vs OOD ({out['mode']}, g={out['g_label']}) ===")
    print(f"  n_total={out['n_total']}  n_id={n_id} ({out['id_fraction']*100:.1f}%)  n_ood={n_ood}")
    print(f"  {'subset':<10} {'MRR':>7} {'Hit@1':>7} {'Hit@3':>7} {'Hit@10':>7}  n")
    print(f"  {'-'*60}")
    for tag, key in [("overall", "overall"), ("ID",  "id"), ("OOD", "ood")]:
        m = out[key]
        print(f"  {tag:<10} {m['MRR']:>7.2f} {m['Hit@1']:>7.2f} "
              f"{m['Hit@3']:>7.2f} {m['Hit@10']:>7.2f}  {m['n']}")

    if save_to:
        with open(save_to, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {save_to}")

    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, required=True,
                   help="Path to test_predictions_*.json (must contain 'ranks')")
    p.add_argument("--train_pairs", type=str, required=True,
                   help="Path to pairs_train.csv")
    p.add_argument("--save", type=str, default=None,
                   help="Optional output JSON. Default: <predictions>_idood.json")
    args = p.parse_args()

    save_to = args.save
    if save_to is None:
        base, _ = os.path.splitext(args.predictions)
        save_to = f"{base}_idood.json"

    evaluate_id_ood(args.predictions, args.train_pairs, save_to=save_to)
