"""V3: GC-GISMo with heterogeneous graph (nutrient hub nodes + I-N edges).

Difference from v1 MVP:
- Add K nutrient hub nodes (K = len(hub_keys), default 7) — new node ids
  num_total_nodes .. num_total_nodes + K - 1.
- Add bidirectional I-N edges connecting each ingredient to each nutrient
  hub it has data for, with weight = log1p + per-nutrient min-max normalized.
- Model: identical to v1 MVP (models_v1.GISMo with use_health_goal=True).
  Hub embeddings are random-init learnable, propagated via WeightedGINConv,
  same treatment as F (flavor) nodes.

REFACTORED for sparse ingredient ID support — see train_v1.py header for
details. Hub edges now built from explicit ingredient_id set (not arange).

Outputs:
  - best_v3.pt
  - test_predictions_v3_{auto,1_0,0_1,1_1}.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import (HEALTH_NUTRIENT_KEYS, SubstitutionDataset,
                     build_valid_targets_map, compute_thresholds,
                     load_graph, load_node_ids, load_nutrient_tensor,
                     _safe_float)
from models_v1 import GISMo
from train_v1 import (GOAL_DIM, parse_g_label, train_one_epoch, evaluate,
                      build_id_to_pos,
                      save_checkpoint, maybe_resume, cleanup_last_ckpt,
                      maybe_skip_completed)


DEFAULT_HUB_NUTRIENT_KEYS = (
    "calories_kcal",
    "fat_g",
    "saturated_fat_g",
    "carbohydrate_g",
    "sugar_g",
    "protein_g",
    "sodium_mg",
)


def build_hub_edges(usda_json, valid_ingredient_set, num_existing_nodes,
                    hub_keys, weight_scale=1.0):
    """Build bidirectional I-N edges.

    Args:
        usda_json: path to usda_mapping.json
        valid_ingredient_set: set of valid ingredient node ids (sparse).
            Only ingredients in this set contribute hub edges.
        num_existing_nodes: total node count in the base graph (= num_total_nodes).
            Hub node ids start here.
        hub_keys: tuple of nutrient keys, one hub per key.
        weight_scale: multiplier on edge weights.

    Returns:
        hub_edge_index: [2, E_hub]
        hub_edge_weight: [E_hub] in (0.01 * weight_scale, weight_scale]
        K: number of hub nodes (= len(hub_keys))
    """
    with open(usda_json) as f:
        usda = {int(k): v for k, v in json.load(f).items()}

    K = len(hub_keys)
    rows, cols, raw_values = [], [], []
    missing_per_key = {k: 0 for k in hub_keys}

    for ing_id, nuts in usda.items():
        if ing_id not in valid_ingredient_set:
            continue
        for k_idx, key in enumerate(hub_keys):
            v = _safe_float(nuts.get(key))
            if v > 0:
                rows.append(ing_id)
                cols.append(k_idx)
                raw_values.append(v)
            else:
                missing_per_key[key] += 1

    if not rows:
        raise ValueError(
            f"No I-N edges built. None of {hub_keys} had positive values."
        )

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    raw_values = np.array(raw_values, dtype=np.float32)

    # log1p + per-nutrient min-max to [0, 1], then shift to (0.01, 1]
    log_values = np.log1p(raw_values)
    weights = np.zeros_like(log_values)
    for k_idx in range(K):
        mask = (cols == k_idx)
        if mask.any():
            v = log_values[mask]
            v_min, v_max = v.min(), v.max()
            if v_max > v_min:
                weights[mask] = (v - v_min) / (v_max - v_min)
            else:
                weights[mask] = 1.0
    weights = (weights * 0.99 + 0.01) * weight_scale

    # Bidirectional edges
    hub_node_ids = num_existing_nodes + cols
    fwd = np.stack([rows, hub_node_ids], axis=0)
    bwd = np.stack([hub_node_ids, rows], axis=0)
    hub_ei = np.concatenate([fwd, bwd], axis=1)
    hub_ew = np.concatenate([weights, weights])

    print(f"[v3] hub edges: {len(rows)} I-N pairs x 2 = {hub_ei.shape[1]} edges")
    print(f"[v3] hub nodes: {K} added (ids {num_existing_nodes}..{num_existing_nodes + K - 1})")
    print(f"[v3] hub edge weight range: [{hub_ew.min():.4f}, {hub_ew.max():.4f}]  (scale={weight_scale})")
    for key, count in missing_per_key.items():
        if count > 0:
            print(f"[v3] note: '{key}' missing/zero for {count} ingredients")

    return (torch.from_numpy(hub_ei),
            torch.from_numpy(hub_ew.astype(np.float32)),
            K)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")

    # DEPRECATED — auto-detected from nodes_filtered.csv.
    p.add_argument("--num_total_nodes", type=int, default=None,
                   help="DEPRECATED — auto-detected from nodes_filtered.csv "
                        "(base count, NOT counting v3 hubs)")
    p.add_argument("--num_ingredients", type=int, default=None,
                   help="DEPRECATED — auto-detected from nodes_filtered.csv")

    p.add_argument("--embed_dim", type=int, default=300)
    p.add_argument("--num_gin_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)

    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_neg", type=int, default=10)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max_recipe_len", type=int, default=20)
    p.add_argument("--eval_chunk", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--resume", type=str, default=None,
                   help="Explicit checkpoint path. If omitted, auto-resume from "
                        "<output_dir>/last_v3.pt when present.")
    p.add_argument("--no_resume", action="store_true",
                   help="Ignore any existing last_v3.pt and start fresh.")
    p.add_argument("--force_retrain", action="store_true",
                   help="Retrain even if best_v3.pt already exists (backs it "
                        "up to .bak). Default: skip training and re-evaluate "
                        "the saved best.")
    p.add_argument("--no_multi_valid", action="store_true")
    p.add_argument("--ablation_no_compound", action="store_true",
                   help="Drop I-F / I-D edges from the BASE graph (keep I-I + hub edges). "
                        "'w/o flavor compound' ablation — note v3's hub edges are kept.")

    p.add_argument("--lambda_h", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--tau_percentile", type=float, default=50.0)

    p.add_argument("--hub_nutrient_keys", nargs="*",
                   default=list(DEFAULT_HUB_NUTRIENT_KEYS),
                   help="Nutrient keys to use for hub nodes (one hub per key).")
    p.add_argument("--hub_weight_scale", type=float, default=1.0,
                   help="Multiply hub edge weights (tune if base FlavorGraph weights "
                        "are on a different scale).")

    p.add_argument("--test_g_overrides", nargs="*",
                   default=["auto", "1_0", "0_1", "1_1"])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_total_nodes is not None or args.num_ingredients is not None:
        print("[deprecated] --num_total_nodes / --num_ingredients are ignored; "
              "auto-detected from nodes_filtered.csv")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[config] mode=v3 (heterogeneous graph + nutrient hubs)  device={device}")
    print(f"[config] hub_nutrient_keys: {args.hub_nutrient_keys}")
    print(f"[config] full args: {json.dumps(vars(args), indent=2, default=str)}")

    for label in args.test_g_overrides:
        parse_g_label(label)

    # --- Load node ids (source of truth for embedding size & candidate pool) ---
    ingredient_ids, num_total_nodes, all_node_ids = load_node_ids(
        os.path.join(args.data_dir, "nodes_filtered.csv"),
    )
    ingredient_id_to_pos = build_id_to_pos(ingredient_ids)
    valid_ingredient_set = set(ingredient_ids.tolist())
    ingredient_ids = ingredient_ids.to(device)

    # --- thresholds ---
    tau_sugar, tau_sodium = compute_thresholds(
        pairs_csv=os.path.join(args.data_dir, "pairs_train.csv"),
        usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
        percentile=args.tau_percentile,
    )
    print(f"[v3] tau_sugar={tau_sugar:.4f}  tau_sodium={tau_sodium:.4f}")

    def make_ds(split):
        return SubstitutionDataset(
            pairs_csv=os.path.join(args.data_dir, f"pairs_{split}.csv"),
            recipes_json=os.path.join(args.data_dir, "recipes.json"),
            usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
            use_health_goal=True,
            tau_sugar=tau_sugar,
            tau_sodium=tau_sodium,
            max_recipe_len=args.max_recipe_len,
        )

    train_ds, val_ds, test_ds = make_ds("train"), make_ds("val"), make_ds("test")
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    valid_targets_map = None
    if not args.no_multi_valid:
        valid_targets_map = build_valid_targets_map(
            os.path.join(args.data_dir, "pairs_train.csv"),
            os.path.join(args.data_dir, "pairs_val.csv"),
            os.path.join(args.data_dir, "pairs_test.csv"),
        )
        n_alt = sum(1 for v in valid_targets_map.values() if len(v) > 1)
        print(f"[eval] valid-target map: {len(valid_targets_map)} keys, {n_alt} have >=2")

    # --- Base graph with invalid-edge filtering (all_node_ids from load_node_ids above) ---
    edge_index_base, edge_weight_base, max_node = load_graph(
        os.path.join(args.data_dir, "flavorgraph_edges.csv"),
        valid_node_ids=all_node_ids,
        edge_types=("I-I",) if args.ablation_no_compound else None,
    )
    if max_node > num_total_nodes:
        raise ValueError(
            f"Edge file refs node {max_node - 1}, but num_total_nodes={num_total_nodes}."
        )
    print(f"[graph] base: {edge_index_base.shape[1]} edges, max node {max_node - 1}")
    print(f"[graph] base edge weight range: [{edge_weight_base.min().item():.4f}, "
          f"{edge_weight_base.max().item():.4f}]")

    # --- Hub edges (built from valid ingredient set, NOT arange) ---
    hub_edge_index, hub_edge_weight, K_hubs = build_hub_edges(
        usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
        valid_ingredient_set=valid_ingredient_set,
        num_existing_nodes=num_total_nodes,
        hub_keys=tuple(args.hub_nutrient_keys),
        weight_scale=args.hub_weight_scale,
    )

    # --- Augmented graph ---
    edge_index = torch.cat([edge_index_base, hub_edge_index], dim=1).to(device)
    edge_weight = torch.cat([edge_weight_base, hub_edge_weight]).to(device)
    num_total_v3 = num_total_nodes + K_hubs
    print(f"[graph] v3 augmented: {edge_index.shape[1]} edges, {num_total_v3} nodes")

    # --- Nutrient tensor for L_health (sized num_total_v3 so hub-id lookups are
    #     safe too, though they're never used since source/target are always
    #     ingredient ids in pairs_*.csv).
    nutrient_health = load_nutrient_tensor(
        os.path.join(args.data_dir, "usda_mapping.json"),
        num_total_nodes=num_total_v3,
        nutrient_keys=HEALTH_NUTRIENT_KEYS,
    ).to(device)

    # --- Model (v1 GISMo, MVP mode, embedding sized for num_total_v3) ---
    model = GISMo(
        num_nodes=num_total_v3,
        embed_dim=args.embed_dim,
        hidden_dim=args.embed_dim,
        num_gin_layers=args.num_gin_layers,
        dropout=args.dropout,
        use_health_goal=True,
        goal_dim=GOAL_DIM,
    ).to(device)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ckpt_path = os.path.join(args.output_dir, "best_v3.pt")
    last_ckpt_path = os.path.join(args.output_dir, "last_v3.pt")
    ckpt_extra = {
        "num_total_nodes": num_total_nodes,
        "num_total_v3": num_total_v3,
        "K_hubs": K_hubs,
    }

    start_epoch, best_mrr, epochs_no_improve = maybe_resume(
        args, "v3", args.output_dir, model, optimizer, device,
    )
    start_epoch = maybe_skip_completed(args, best_ckpt_path, start_epoch)

    if start_epoch > args.max_epochs:
        print(f"[resume] start_epoch={start_epoch} > max_epochs={args.max_epochs}; "
              f"training loop will be skipped.")
        if not os.path.exists(best_ckpt_path):
            print(f"[resume] ERROR: no {best_ckpt_path} either. "
                  f"Pass --no_resume to start fresh, or raise --max_epochs.")
            return

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, edge_index, edge_weight,
            ingredient_ids, args.num_neg, True,
            nutrient_health, args.lambda_h, args.margin, device,
        )
        val_metrics = evaluate(
            model, val_loader, edge_index, edge_weight, ingredient_ids,
            ingredient_id_to_pos, True, device,
            eval_chunk=args.eval_chunk, valid_targets_map=valid_targets_map,
        )
        elapsed = time.time() - t0

        print(f"[epoch {epoch:3d}] loss={train_metrics['loss']:.4f} "
              f"(sub={train_metrics['sub']:.4f}, h={train_metrics['h']:.4f}) | "
              f"val MRR={val_metrics['MRR']:.2f} Hit@1={val_metrics['Hit@1']:.2f} "
              f"Hit@10={val_metrics['Hit@10']:.2f} | {elapsed:.1f}s")

        improved = val_metrics["MRR"] > best_mrr
        if improved:
            best_mrr = val_metrics["MRR"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save best BEFORE last — see train_v1.py for rationale.
        ckpt_state = dict(
            model=model, optimizer=optimizer, epoch=epoch,
            best_mrr=best_mrr, epochs_no_improve=epochs_no_improve,
            args=args, val_metrics=val_metrics, extra=ckpt_extra,
        )
        if improved:
            save_checkpoint(best_ckpt_path, **ckpt_state)
        save_checkpoint(last_ckpt_path, **ckpt_state)

        if epochs_no_improve >= args.patience:
            print(f"[early-stop] no MRR improvement for {args.patience} epochs.")
            break

    cleanup_last_ckpt(args.output_dir, "v3")

    if not os.path.exists(best_ckpt_path):
        print("[warning] no checkpoint saved.")
        return

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    for label in args.test_g_overrides:
        override = parse_g_label(label)
        suffix = f"_{label}"
        print(f"\n=== TEST [v3{suffix}] (best epoch {ckpt['epoch']}) ===")
        if override is not None:
            print(f"  g override: {override.tolist()}")
        test_metrics = evaluate(
            model, test_loader, edge_index, edge_weight, ingredient_ids,
            ingredient_id_to_pos, True, device,
            eval_chunk=args.eval_chunk, override_g=override,
            valid_targets_map=valid_targets_map,
        )
        print(f"  MRR    {test_metrics['MRR']:.2f}")
        print(f"  Hit@1  {test_metrics['Hit@1']:.2f}")
        print(f"  Hit@3  {test_metrics['Hit@3']:.2f}")
        print(f"  Hit@10 {test_metrics['Hit@10']:.2f}")

        out = {
            "mode": "v3",
            "g_label": label,
            "best_epoch": int(ckpt["epoch"]),
            "metrics": {k: test_metrics[k] for k in ("MRR", "Hit@1", "Hit@3", "Hit@10")},
            "ranks": test_metrics["ranks"],
            "top1": test_metrics["top1"],
            "sources": test_metrics["sources"],
            "targets": test_metrics["targets"],
            "goals": test_metrics["goals"],
        }
        pred_path = os.path.join(args.output_dir, f"test_predictions_v3{suffix}.json")
        with open(pred_path, "w") as f:
            json.dump(out, f)
        print(f"[saved] {pred_path}")


if __name__ == "__main__":
    main()
