"""Flavor preservation metric (post-hoc).

PDF spec: "대체 전후 flavor compound cosine similarity → 이게 높으면 맛이 유지된 것"

For each test pair (source, top1_prediction), we build a flavor profile
vector indexed over flavor-compound (F) nodes and compute cosine similarity
between source and prediction. Higher = the predicted substitute shares
more flavor compounds (= taste is preserved).

Flavor profile definition:
  Each ingredient i has a vector x_i ∈ R^|F| where
    x_i[j] = weight of the I-F edge from ingredient i to F-compound j,
             or 0 if no edge exists.
  We use the raw FlavorGraph edge weights (no learning involved).

Caveat: only ~416 of 6,313 ingredients are FlavorGraph "Chemical-hub"
ingredients with I-F edges (i.e. non-zero flavor profile). The rest are
"Non-hub" — their flavor profile is the zero vector, which makes cosine
undefined. We report metrics on the subset where BOTH source and prediction
are hub ingredients (cosine well-defined), and also the fraction of
test pairs that are fully hub-covered.

Usage:
    python evaluate_flavor.py \\
        --predictions ./out/v3/test_predictions_v3_auto.json \\
        --edges ./data/flavorgraph_edges.csv \\
        [--save ./out/v3/test_predictions_v3_auto_flavor.json]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def build_profile_matrix(edges_csv):
    """Build a [max_id+1, n_f] flavor profile matrix from I-F edges.

    Each row i is the sparse flavor profile for node id i (zero for nodes
    that have no I-F edges, or for non-hub ingredients). Columns index
    distinct I-F edge endpoints — for an ingredient row, the non-zero
    columns correspond to flavor compounds connected via I-F edges.

    Returns:
        M:       [max_id + 1, n_f] float32 numpy array
        n_f:     int, number of distinct I-F edge endpoints
    """
    df = pd.read_csv(edges_csv)
    if "edge_type" not in df.columns:
        raise ValueError("edges CSV must have 'edge_type' column")
    if_edges = df[df["edge_type"] == "I-F"].reset_index(drop=True)
    if len(if_edges) == 0:
        raise ValueError("No I-F edges found in CSV. Cannot build flavor profiles.")

    src = if_edges["src_id"].to_numpy(dtype=np.int64)
    dst = if_edges["dst_id"].to_numpy(dtype=np.int64)
    if "weight" in if_edges.columns:
        w = if_edges["weight"].to_numpy(dtype=np.float32)
        bad = np.isnan(w) | np.isinf(w)
        w[bad] = 1.0
    else:
        w = np.ones(len(if_edges), dtype=np.float32)

    # Treat I-F as undirected: each edge contributes to both endpoint rows.
    # In practice source/target ingredients are always queried as row
    # indices and their flavor compounds end up as the active columns.
    all_endpoints = np.concatenate([src, dst])
    unique_endpoints = np.unique(all_endpoints)
    col_id_to_pos = {int(v): i for i, v in enumerate(unique_endpoints)}
    n_f = len(unique_endpoints)

    max_id = int(unique_endpoints.max())
    M = np.zeros((max_id + 1, n_f), dtype=np.float32)
    # src→dst column
    dst_pos = np.array([col_id_to_pos[int(v)] for v in dst], dtype=np.int64)
    src_pos = np.array([col_id_to_pos[int(v)] for v in src], dtype=np.int64)
    M[src, dst_pos] = w
    M[dst, src_pos] = w
    return M, n_f


def stats(xs):
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p25": float("nan"), "p75": float("nan")}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
    }


def evaluate_flavor(predictions_path, edges_csv, save_to=None):
    with open(predictions_path) as f:
        pred = json.load(f)
    sources = np.asarray(pred["sources"], dtype=np.int64)
    top1 = np.asarray(pred["top1"], dtype=np.int64)
    targets = np.asarray(pred["targets"], dtype=np.int64)

    print(f"[flavor] building profiles from {edges_csv} ...")
    M, n_f = build_profile_matrix(edges_csv)
    n_rows = M.shape[0]
    print(f"[flavor] profile matrix: {M.shape} "
          f"({int((M.sum(axis=1) > 0).sum())} non-zero rows)")

    # Pad ids that exceed matrix rows to the (zero) row at index n_rows-1
    # by clipping; their norms will be 0 and they'll be excluded from the
    # hub-only subset anyway.
    def safe_rows(ids):
        bad = ids >= n_rows
        if bad.any():
            ids = ids.copy()
            ids[bad] = 0  # row 0 is also zero unless it's an ingredient w/ I-F
        return ids

    s_idx = safe_rows(sources)
    p_idx = safe_rows(top1)
    y_idx = safe_rows(targets)

    src_vecs = M[s_idx]      # [N, n_f]
    pred_vecs = M[p_idx]
    gt_vecs = M[y_idx]

    src_n = np.linalg.norm(src_vecs, axis=1)
    pred_n = np.linalg.norm(pred_vecs, axis=1)
    gt_n = np.linalg.norm(gt_vecs, axis=1)

    eps = 1e-12
    dot_pred = (src_vecs * pred_vecs).sum(axis=1)
    dot_gt = (src_vecs * gt_vecs).sum(axis=1)
    cos_pred = dot_pred / np.maximum(src_n * pred_n, eps)
    cos_gt = dot_gt / np.maximum(src_n * gt_n, eps)

    both_pred_hub = (src_n > 0) & (pred_n > 0)
    both_gt_hub = (src_n > 0) & (gt_n > 0)

    out = {
        "mode": pred.get("mode", "?"),
        "g_label": pred.get("g_label"),
        "n_total": int(len(sources)),
        "n_src_hub": int((src_n > 0).sum()),
        "n_pred_hub": int((pred_n > 0).sum()),
        "n_gt_hub": int((gt_n > 0).sum()),
        "pred_cosine": stats(cos_pred[both_pred_hub]),    # predicted top-1 vs source
        "gt_cosine":   stats(cos_gt[both_gt_hub]),        # ground-truth y vs source (reference)
    }

    print(f"\n=== Flavor preservation ({out['mode']}, g={out['g_label']}) ===")
    print(f"  n_total      : {out['n_total']}")
    print(f"  n_src_hub    : {out['n_src_hub']}    "
          f"({out['n_src_hub'] / max(out['n_total'],1) * 100:.1f}%)")
    print(f"  n_pred_hub   : {out['n_pred_hub']}")
    print(f"  n_gt_hub     : {out['n_gt_hub']}")
    print(f"  --- Cosine(source profile, predicted top-1 profile) ---")
    print(f"  evaluated on : {out['pred_cosine']['n']} pairs "
          f"(both endpoints hub)")
    print(f"  mean / median: {out['pred_cosine']['mean']:.4f} / "
          f"{out['pred_cosine']['median']:.4f}")
    print(f"  p25 / p75    : {out['pred_cosine']['p25']:.4f} / "
          f"{out['pred_cosine']['p75']:.4f}")
    print(f"  --- Cosine(source profile, ground-truth y profile) — reference ---")
    print(f"  evaluated on : {out['gt_cosine']['n']} pairs")
    print(f"  mean / median: {out['gt_cosine']['mean']:.4f} / "
          f"{out['gt_cosine']['median']:.4f}")

    if save_to:
        with open(save_to, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {save_to}")

    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=str, required=True,
                   help="Path to test_predictions_*.json")
    p.add_argument("--edges", type=str, required=True,
                   help="Path to flavorgraph_edges.csv (must have edge_type column)")
    p.add_argument("--save", type=str, default=None,
                   help="Optional output JSON. Default: <predictions>_flavor.json")
    args = p.parse_args()

    save_to = args.save
    if save_to is None:
        base, _ = os.path.splitext(args.predictions)
        save_to = f"{base}_flavor.json"

    evaluate_flavor(args.predictions, args.edges, save_to=save_to)
