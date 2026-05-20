"""GISMo + post-hoc health filter baseline.

Loads a baseline checkpoint (trained with train_v1.py --mode baseline)
and applies a post-hoc filter at test time to bias predictions toward
healthier substitutions. This is the strongest non-health-aware baseline
we need to beat: it shows what you can do *without* retraining.

Two filter modes:
  hard  : drop candidates that don't reduce the targeted nutrient.
          For pair (s, y) with g[k]=1, set score = -inf for any
          candidate v where nutrient_raw[v, k] >= nutrient_raw[s, k].
  soft  : re-rank with score_new = score_base + alpha * sum_k g[k] *
          (n_source[k] - n_cand[k]) using log1p+z-score normalized
          nutrients (so alpha is scale-invariant across sugar/sodium).

Both modes respect the goal vector g — if g_sugar=1 only, only sugar
filters/re-ranks. Same g_override options as training scripts.

REFACTORED for sparse ingredient ID support — see train_v1.py header. The
score tensor is now `[B, N_ing]` with columns indexed by ingredient
*position* (sorted index into `ingredient_ids`), not raw node id. All
nutrient arrays are position-indexed for direct column lookup.

Usage:
    python eval_filter_baseline.py \\
        --checkpoint ./outputs/best_baseline.pt \\
        --data_dir ./data --output_dir ./outputs/filter_baseline \\
        --filter_mode both --alpha 0.1 0.5 1.0 2.0 \\
        --test_g_overrides auto 1_0 0_1 1_1

After running, feed the saved test_predictions_filter_*.json into
evaluate_health.py to get Δsugar/Δsodium/satisfaction-rate metrics.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (HEALTH_NUTRIENT_KEYS, SubstitutionDataset,
                     build_valid_targets_map, compute_thresholds,
                     load_graph, load_node_ids, load_nutrient_tensor,
                     _safe_float)
from models_v1 import GISMo
from train_v1 import (GOAL_DIM, parse_g_label, build_id_to_pos,
                      ids_to_positions)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_test_baseline(model, loader, edge_index, edge_weight,
                        ingredient_ids, device, eval_chunk=256):
    """Compute baseline scores for the entire test set.

    Returns numpy arrays indexed by test pair:
      sources, targets, recipe_ids   : raw node ids
      source_pos                     : positions of source in ingredient_ids
      g_auto                          : auto-derived goal vectors
      scores                          : [n_test, N_ing], columns = ingredient
                                        positions; source column is -inf.
    """
    model.eval()
    h = model.encode_graph(edge_index, edge_weight)
    N_ing = ingredient_ids.shape[0]

    all_sources, all_targets, all_recipes = [], [], []
    all_source_pos, all_g, all_scores = [], [], []

    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        recipe_id = batch["recipe_id"].to(device)
        recipe_ings = batch["recipe_ings"].to(device)
        recipe_mask = batch["recipe_mask"].to(device)
        g = batch["g"].to(device)

        B = source.shape[0]
        scores = torch.empty((B, N_ing), device=device)
        for start in range(0, N_ing, eval_chunk):
            end = min(start + eval_chunk, N_ing)
            chunk = ingredient_ids[start:end].unsqueeze(0).expand(B, -1)
            # Baseline doesn't use g → pass None.
            chunk_scores = model(h, source, chunk, recipe_ings, recipe_mask, g=None)
            scores[:, start:end] = chunk_scores

        # Mask source column (can't substitute itself).
        source_pos = ids_to_positions(source, ingredient_ids)
        scores.scatter_(1, source_pos.unsqueeze(1), float("-inf"))

        all_sources.append(source.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_recipes.append(recipe_id.cpu().numpy())
        all_source_pos.append(source_pos.cpu().numpy())
        all_g.append(g.cpu().numpy())
        all_scores.append(scores.cpu().numpy())

    return {
        "sources": np.concatenate(all_sources),
        "targets": np.concatenate(all_targets),
        "recipe_ids": np.concatenate(all_recipes),
        "source_pos": np.concatenate(all_source_pos),
        "g_auto": np.concatenate(all_g),
        "scores": np.concatenate(all_scores, axis=0),
    }


# ---------------------------------------------------------------------------
# Filters (all position-indexed)
# ---------------------------------------------------------------------------

def apply_hard_filter(scores, source_pos, g, nutrient_raw_pos):
    """For each pair (b, *): set score[b, v] = -inf if ANY nutrient k with
    g[b, k]=1 satisfies nutrient_raw_pos[v, k] >= nutrient_raw_pos[source_pos[b], k]
    (i.e., candidate didn't strictly reduce that nutrient).

    Args:
      scores:           [B, N_ing] numpy   (columns = ingredient positions)
      source_pos:       [B]        numpy int  (positions, not raw ids)
      g:                [B, G]     numpy {0,1}
      nutrient_raw_pos: [N_ing, G] numpy float  (raw values, position-indexed.
                                                 sign of delta is preserved
                                                 through standardization, so
                                                 raw works fine for hard filter)
    """
    n_source = nutrient_raw_pos[source_pos]              # [B, G]
    n_cand = nutrient_raw_pos[None, :, :]                # [1, V, G]
    delta = n_source[:, None, :] - n_cand                # [B, V, G]

    g_b = g[:, None, :]                                  # [B, 1, G]
    no_reduction = (delta <= 0) & (g_b > 0.5)            # active goal & not reduced
    reject = no_reduction.any(axis=-1)                   # [B, V]

    out = scores.copy()
    out[reject] = float("-inf")
    return out


def apply_soft_filter(scores, source_pos, g, nutrient_norm_pos, alpha):
    """Re-rank: score_new = score + alpha * sum_k g[k] * (n_source[k] - n_cand[k]).

    Uses NORMALIZED nutrients (log1p + z-score) so alpha is on a stable
    scale across sugar/sodium (raw sodium in mg vastly dominates sugar in g).
    """
    n_source = nutrient_norm_pos[source_pos]             # [B, G]
    g_dot_source = (g * n_source).sum(axis=-1, keepdims=True)  # [B, 1]
    g_dot_cand = g @ nutrient_norm_pos.T                 # [B, V]
    bonus = g_dot_source - g_dot_cand                    # [B, V]
    return scores + alpha * bonus


# ---------------------------------------------------------------------------
# Valid-target masking (numpy version, position-indexed)
# ---------------------------------------------------------------------------

def mask_valid_targets_np(scores, sources, targets, recipe_ids,
                          valid_targets_map, ingredient_id_to_pos):
    """Set score column = -inf for alternate valid targets.

    sources / targets are RAW node ids; the score tensor columns are
    ingredient POSITIONS, so we translate alt raw ids via
    ingredient_id_to_pos before masking.
    """
    out = scores.copy()
    for i in range(len(sources)):
        s, r, t = int(sources[i]), int(recipe_ids[i]), int(targets[i])
        alts = valid_targets_map.get((s, r), frozenset())
        for a in alts:
            if a != t:
                pos = ingredient_id_to_pos.get(a)
                if pos is not None:
                    out[i, pos] = float("-inf")
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(scores_for_rank, scores_for_top1, target_pos,
                    ingredient_ids_np):
    """MRR, Hit@k, top-1 predictions.

    Args:
      scores_for_rank: [B, N_ing] with valid-target masking applied
                       (for fair rank).
      scores_for_top1: [B, N_ing] without valid-target masking
                       (true predicted top-1).
      target_pos:      [B] target positions in ingredient_ids.
      ingredient_ids_np: [N_ing] numpy array of raw ids; used to map
                         top-1 position back to raw node id for the JSON dump.
    """
    B = scores_for_rank.shape[0]
    top1_pos = scores_for_top1.argmax(axis=1)
    top1 = ingredient_ids_np[top1_pos]                    # raw ids

    target_scores = scores_for_rank[np.arange(B), target_pos]
    rank = (scores_for_rank > target_scores[:, None]).sum(axis=1) + 1

    return {
        "MRR": float((1.0 / rank).mean() * 100),
        "Hit@1": float((rank <= 1).mean() * 100),
        "Hit@3": float((rank <= 3).mean() * 100),
        "Hit@10": float((rank <= 10).mean() * 100),
        "ranks": rank.tolist(),
        "top1": top1.tolist(),
    }


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to best_baseline.pt (from train_v1.py --mode baseline)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs/filter_baseline")

    # DEPRECATED — auto-detected from nodes_filtered.csv.
    p.add_argument("--num_total_nodes", type=int, default=None,
                   help="DEPRECATED — auto-detected from nodes_filtered.csv")
    p.add_argument("--num_ingredients", type=int, default=None,
                   help="DEPRECATED — auto-detected from nodes_filtered.csv")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_recipe_len", type=int, default=20)
    p.add_argument("--eval_chunk", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--tau_percentile", type=float, default=50.0,
                   help="Used only for the 'auto' g override (derives g from training-distribution tau)")
    p.add_argument("--filter_mode", type=str, default="both",
                   choices=["hard", "soft", "both"])
    p.add_argument("--alpha", type=float, nargs="*",
                   default=[0.1, 0.5, 1.0, 2.0],
                   help="Soft filter strength values to sweep")
    p.add_argument("--test_g_overrides", nargs="*",
                   default=["auto", "1_0", "0_1", "1_1"])
    p.add_argument("--no_multi_valid", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_total_nodes is not None or args.num_ingredients is not None:
        print("[deprecated] --num_total_nodes / --num_ingredients are ignored; "
              "auto-detected from nodes_filtered.csv")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[config] filter baseline | device={device}")
    print(f"[config] full args: {json.dumps(vars(args), indent=2, default=str)}")

    for label in args.test_g_overrides:
        parse_g_label(label)

    # --- Load node ids (source of truth) ---
    ingredient_ids, num_total_nodes, all_node_ids = load_node_ids(
        os.path.join(args.data_dir, "nodes_filtered.csv"),
    )
    ingredient_id_to_pos = build_id_to_pos(ingredient_ids)
    ingredient_ids_np = ingredient_ids.numpy()
    ingredient_ids = ingredient_ids.to(device)
    N_ing = ingredient_ids.shape[0]

    tau_sugar, tau_sodium = compute_thresholds(
        pairs_csv=os.path.join(args.data_dir, "pairs_train.csv"),
        usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
        percentile=args.tau_percentile,
    )
    print(f"[tau] sugar={tau_sugar:.4f}  sodium={tau_sodium:.4f} "
          f"(used for 'auto' g only)")

    # Test dataset — use_health_goal=True so we get auto-derived g per pair
    # (model itself doesn't consume g; only the filter does).
    test_ds = SubstitutionDataset(
        pairs_csv=os.path.join(args.data_dir, "pairs_test.csv"),
        recipes_json=os.path.join(args.data_dir, "recipes.json"),
        usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
        use_health_goal=True,
        tau_sugar=tau_sugar,
        tau_sodium=tau_sodium,
        max_recipe_len=args.max_recipe_len,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    print(f"[data] test pairs: {len(test_ds)}")

    valid_targets_map = None
    if not args.no_multi_valid:
        valid_targets_map = build_valid_targets_map(
            os.path.join(args.data_dir, "pairs_train.csv"),
            os.path.join(args.data_dir, "pairs_val.csv"),
            os.path.join(args.data_dir, "pairs_test.csv"),
        )
        print(f"[eval] valid-target map: {len(valid_targets_map)} keys")

    # --- Graph with invalid-edge filtering (all_node_ids from load_node_ids above) ---
    edge_index, edge_weight, max_node = load_graph(
        os.path.join(args.data_dir, "flavorgraph_edges.csv"),
        valid_node_ids=all_node_ids,
    )
    if max_node > num_total_nodes:
        raise ValueError(
            f"Edge file references node {max_node - 1}, but "
            f"num_total_nodes={num_total_nodes}."
        )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    print(f"[graph] {edge_index.shape[1]} edges (after filtering)")

    # --- Nutrient arrays, position-indexed ---
    #   raw : used by hard filter (only sign of delta matters)
    #   norm: used by soft filter (alpha is scale-invariant)
    with open(os.path.join(args.data_dir, "usda_mapping.json")) as f:
        usda_raw = json.load(f)

    nutrient_raw_full = np.zeros((num_total_nodes, 2), dtype=np.float32)
    for k_str, nuts in usda_raw.items():
        ing_id = int(k_str)
        if 0 <= ing_id < num_total_nodes:
            nutrient_raw_full[ing_id, 0] = _safe_float(nuts.get("sugar_g"))
            nutrient_raw_full[ing_id, 1] = _safe_float(nuts.get("sodium_mg"))
    # Re-index to position space.
    nutrient_raw_pos = nutrient_raw_full[ingredient_ids_np]   # [N_ing, 2]

    nutrient_norm_full = load_nutrient_tensor(
        os.path.join(args.data_dir, "usda_mapping.json"),
        num_total_nodes=num_total_nodes,
        nutrient_keys=HEALTH_NUTRIENT_KEYS,
    ).numpy()
    nutrient_norm_pos = nutrient_norm_full[ingredient_ids_np]  # [N_ing, 2]

    # --- Load baseline model ---
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    embed_dim = saved_args.get("embed_dim", 300)
    num_gin_layers = saved_args.get("num_gin_layers", 2)
    saved_ntot = ckpt.get("num_total_nodes", None)
    if saved_ntot is not None and saved_ntot != num_total_nodes:
        print(f"[warn] checkpoint trained with num_total_nodes={saved_ntot} but "
              f"current data has {num_total_nodes}. Loading anyway — verify "
              f"this is the right checkpoint.")

    model = GISMo(
        num_nodes=num_total_nodes,
        embed_dim=embed_dim,
        hidden_dim=embed_dim,
        num_gin_layers=num_gin_layers,
        dropout=0.0,                    # eval-time only
        use_health_goal=False,           # baseline doesn't use g
        goal_dim=GOAL_DIM,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[model] loaded baseline from {args.checkpoint} (epoch {ckpt['epoch']})")

    # Score test set ONCE (filters are post-hoc reweighting; baseline scores
    # don't depend on the filter mode or g).
    print(f"[score] running baseline forward pass on test set...")
    cached = score_test_baseline(
        model, test_loader, edge_index, edge_weight,
        ingredient_ids, device, eval_chunk=args.eval_chunk,
    )
    base_scores = cached["scores"]          # [n_test, N_ing], position-indexed cols
    sources = cached["sources"]              # raw ids
    targets = cached["targets"]              # raw ids
    recipe_ids = cached["recipe_ids"]
    source_pos = cached["source_pos"]        # positions
    g_auto = cached["g_auto"]
    print(f"[score] {len(sources)} test pairs scored")

    # Target positions for rank lookup.
    target_pos = np.array(
        [ingredient_id_to_pos[int(t)] for t in targets], dtype=np.int64
    )

    # Apply valid-target masking once (filter doesn't change which targets
    # are "alternate valid"  it just changes the score landscape).
    if valid_targets_map is not None:
        base_scores_for_rank = mask_valid_targets_np(
            base_scores, sources, targets, recipe_ids,
            valid_targets_map, ingredient_id_to_pos,
        )
    else:
        base_scores_for_rank = base_scores

    # Build (filter_mode, alpha) configs
    configs = []
    if args.filter_mode in ("hard", "both"):
        configs.append(("hard", None))
    if args.filter_mode in ("soft", "both"):
        for a in args.alpha:
            configs.append(("soft", a))

    # Sweep
    summary = {}
    for label in args.test_g_overrides:
        override = parse_g_label(label)
        if override is None:
            g = g_auto
        else:
            g = np.broadcast_to(override.numpy()[None, :], g_auto.shape).copy()

        for fmode, alpha in configs:
            tag = "hard" if alpha is None else f"soft_a{alpha:g}"
            cfg = f"{tag}__{label}"
            print(f"\n=== [{cfg}] ===")

            if fmode == "hard":
                filt_top1 = apply_hard_filter(
                    base_scores, source_pos, g, nutrient_raw_pos,
                )
                filt_rank = apply_hard_filter(
                    base_scores_for_rank, source_pos, g, nutrient_raw_pos,
                )
            else:
                filt_top1 = apply_soft_filter(
                    base_scores, source_pos, g, nutrient_norm_pos, alpha,
                )
                filt_rank = apply_soft_filter(
                    base_scores_for_rank, source_pos, g, nutrient_norm_pos, alpha,
                )

            metrics = compute_metrics(filt_rank, filt_top1, target_pos,
                                       ingredient_ids_np)
            summary[cfg] = metrics

            print(f"  MRR    {metrics['MRR']:.2f}")
            print(f"  Hit@1  {metrics['Hit@1']:.2f}")
            print(f"  Hit@3  {metrics['Hit@3']:.2f}")
            print(f"  Hit@10 {metrics['Hit@10']:.2f}")

            out = {
                "mode": f"filter_{tag}",
                "g_label": label,
                "alpha": alpha,
                "filter_mode": fmode,
                "best_epoch": int(ckpt["epoch"]),
                "metrics": {k: metrics[k] for k in ("MRR", "Hit@1", "Hit@3", "Hit@10")},
                "ranks": metrics["ranks"],
                "top1": metrics["top1"],
                "sources": sources.tolist(),
                "targets": targets.tolist(),
                "goals": g.tolist(),
            }
            pred_path = os.path.join(
                args.output_dir,
                f"test_predictions_filter_{tag}_{label}.json",
            )
            with open(pred_path, "w") as f:
                json.dump(out, f)
            print(f"[saved] {pred_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'config':<35} {'MRR':>7} {'Hit@1':>7} {'Hit@3':>7} {'Hit@10':>7}")
    print("-" * 75)
    for k in sorted(summary.keys()):
        m = summary[k]
        print(f"{k:<35} {m['MRR']:>7.2f} {m['Hit@1']:>7.2f} "
              f"{m['Hit@3']:>7.2f} {m['Hit@10']:>7.2f}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "top1"}
                   for k, v in summary.items()}, f, indent=2)
    print(f"\n[saved] {summary_path}")


if __name__ == "__main__":
    main()
