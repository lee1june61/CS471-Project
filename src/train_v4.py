"""V4 training script: GISMo with decoder-level nutrient injection.

Same data pipeline and training loop as v1 MVP / v2 / v3:
  - 2-dim goal vector g (sugar, sodium), derived from USDA Delta vs tau.
  - L_health hinge loss (same as v1).
  - v1 encoder unchanged (no nutrient_node tensor, no hub nodes).

The only model change: SubstitutionDecoderWithNutrient additionally
takes raw (log1p + z-score) per-ingredient nutrient features for source
and each candidate. See models_v4.py for rationale.

REFACTORED for sparse ingredient ID support — see train_v1.py header for
details. nutrient_tensor is now sized for num_total_nodes so
`nutrient_tensor[source_ids]` and `nutrient_tensor[candidate_ids]` work
directly with sparse ingredient ids.

Outputs:
  - checkpoint:   best_v4.pt
  - predictions:  test_predictions_v4{_<g_label>}.json
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import (HEALTH_NUTRIENT_KEYS, SubstitutionDataset,
                     build_valid_targets_map, compute_thresholds,
                     load_graph, load_node_ids, load_nutrient_tensor)
from models_v4 import GISMo
from train_v1 import (GOAL_DIM, parse_g_label, sample_negatives,
                      health_loss_fn, _build_valid_target_mask,
                      ids_to_positions, build_id_to_pos,
                      save_checkpoint, maybe_resume, cleanup_last_ckpt,
                      maybe_skip_completed)


# ---------------------------------------------------------------------------
# Train / eval loops (nutrient_tensor threaded through model forward)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, edge_index, edge_weight,
                    nutrient_tensor, ingredient_ids, num_neg,
                    lambda_h, margin, device):
    model.train()
    sums = {"loss": 0.0, "sub": 0.0, "h": 0.0}
    n = 0

    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        recipe_ings = batch["recipe_ings"].to(device)
        recipe_mask = batch["recipe_mask"].to(device)
        g = batch["g"].to(device)

        negs = sample_negatives(target, source, ingredient_ids, num_neg)
        candidates = torch.cat([target.unsqueeze(1), negs], dim=1)

        h = model.encode_graph(edge_index, edge_weight)
        scores = model(h, source, candidates, recipe_ings, recipe_mask,
                       g, nutrient_tensor)

        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
        L_sub = F.cross_entropy(scores, labels)
        L_h = health_loss_fn(g, source, candidates, scores,
                              nutrient_tensor, margin)
        loss = L_sub + lambda_h * L_h

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sums["loss"] += loss.item()
        sums["sub"] += L_sub.item()
        sums["h"] += L_h.item()
        n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, edge_index, edge_weight, nutrient_tensor,
             ingredient_ids, ingredient_id_to_pos, device,
             eval_chunk=256, override_g=None, valid_targets_map=None):
    model.eval()
    h = model.encode_graph(edge_index, edge_weight)
    N_ing = ingredient_ids.shape[0]

    ranks = []
    top1_list, sources_list, targets_list, goals_list = [], [], [], []

    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        recipe_id = batch["recipe_id"].to(device)
        recipe_ings = batch["recipe_ings"].to(device)
        recipe_mask = batch["recipe_mask"].to(device)

        if override_g is not None:
            g = override_g.to(device).unsqueeze(0).expand(source.shape[0], -1)
        else:
            g = batch["g"].to(device)

        B = source.shape[0]
        scores = torch.empty((B, N_ing), device=device)
        for start in range(0, N_ing, eval_chunk):
            end = min(start + eval_chunk, N_ing)
            chunk = ingredient_ids[start:end].unsqueeze(0).expand(B, -1)
            chunk_scores = model(h, source, chunk, recipe_ings, recipe_mask,
                                 g, nutrient_tensor)
            scores[:, start:end] = chunk_scores

        source_pos = ids_to_positions(source, ingredient_ids)
        scores.scatter_(1, source_pos.unsqueeze(1), float("-inf"))

        top1_pos = scores.argmax(dim=1)
        top1 = ingredient_ids[top1_pos]

        if valid_targets_map is not None:
            alt_mask = _build_valid_target_mask(
                source, target, recipe_id, ingredient_id_to_pos,
                N_ing, valid_targets_map, device,
            )
            scores_for_rank = scores.masked_fill(alt_mask, float("-inf"))
        else:
            scores_for_rank = scores

        target_pos = ids_to_positions(target, ingredient_ids)
        target_scores = scores_for_rank.gather(1, target_pos.unsqueeze(1)).squeeze(1)
        rank = (scores_for_rank > target_scores.unsqueeze(1)).sum(dim=1) + 1

        ranks.append(rank.cpu().numpy())
        top1_list.append(top1.cpu().numpy())
        sources_list.append(source.cpu().numpy())
        targets_list.append(target.cpu().numpy())
        goals_list.append(g.cpu().numpy())

    ranks = np.concatenate(ranks)
    return {
        "MRR": float((1.0 / ranks).mean() * 100),
        "Hit@1": float((ranks <= 1).mean() * 100),
        "Hit@3": float((ranks <= 3).mean() * 100),
        "Hit@10": float((ranks <= 10).mean() * 100),
        "n_eval": int(len(ranks)),
        "ranks": ranks.tolist(),
        "top1": np.concatenate(top1_list).tolist(),
        "sources": np.concatenate(sources_list).tolist(),
        "targets": np.concatenate(targets_list).tolist(),
        "goals": np.concatenate(goals_list).tolist(),
    }


# ---------------------------------------------------------------------------
# Args + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")

    # DEPRECATED — auto-detected from nodes_filtered.csv.
    p.add_argument("--num_total_nodes", type=int, default=None,
                   help="DEPRECATED — auto-detected from nodes_filtered.csv")
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
                        "<output_dir>/last_v4.pt when present.")
    p.add_argument("--no_resume", action="store_true",
                   help="Ignore any existing last_v4.pt and start fresh.")
    p.add_argument("--force_retrain", action="store_true",
                   help="Retrain even if best_v4.pt already exists (backs it "
                        "up to .bak). Default: skip training and re-evaluate "
                        "the saved best.")
    p.add_argument("--no_multi_valid", action="store_true")
    p.add_argument("--ablation_no_compound", action="store_true",
                   help="Drop I-F / I-D edges (keep only I-I). 'w/o flavor compound' ablation.")

    p.add_argument("--lambda_h", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--tau_percentile", type=float, default=50.0)

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

    print(f"[config] mode=v4 (decoder nutrient injection)  device={device}")
    print(f"[config] full args: {json.dumps(vars(args), indent=2, default=str)}")

    for label in args.test_g_overrides:
        parse_g_label(label)

    # --- Load node ids (source of truth) ---
    ingredient_ids, num_total_nodes, all_node_ids = load_node_ids(
        os.path.join(args.data_dir, "nodes_filtered.csv"),
    )
    ingredient_id_to_pos = build_id_to_pos(ingredient_ids)
    ingredient_ids = ingredient_ids.to(device)

    tau_sugar, tau_sodium = compute_thresholds(
        pairs_csv=os.path.join(args.data_dir, "pairs_train.csv"),
        usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
        percentile=args.tau_percentile,
    )
    print(f"[v4] tau_sugar={tau_sugar:.4f}  tau_sodium={tau_sodium:.4f}")

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
        print(f"[eval] valid-target map: {len(valid_targets_map)} keys, "
              f"{n_alt} have >=2")

    # --- Graph with invalid-edge filtering (all_node_ids from load_node_ids above) ---
    edge_index, edge_weight, max_node = load_graph(
        os.path.join(args.data_dir, "flavorgraph_edges.csv"),
        valid_node_ids=all_node_ids,
        edge_types=("I-I",) if args.ablation_no_compound else None,
    )
    if max_node > num_total_nodes:
        raise ValueError(
            f"Edge file references node {max_node - 1}, but "
            f"num_total_nodes={num_total_nodes}."
        )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    print(f"[graph] {edge_index.shape[1]} edges (after filtering)")

    # Single nutrient tensor used BOTH for decoder concat AND L_health.
    # Sized for num_total_nodes so raw-id indexing in model.forward works.
    nutrient_tensor = load_nutrient_tensor(
        os.path.join(args.data_dir, "usda_mapping.json"),
        num_total_nodes=num_total_nodes,
        nutrient_keys=HEALTH_NUTRIENT_KEYS,
    ).to(device)
    print(f"[v4] nutrient tensor: {tuple(nutrient_tensor.shape)} "
          f"(decoder concat + L_health both)")

    model = GISMo(
        num_nodes=num_total_nodes,
        nutrient_dim=nutrient_tensor.shape[1],
        embed_dim=args.embed_dim,
        hidden_dim=args.embed_dim,
        num_gin_layers=args.num_gin_layers,
        dropout=args.dropout,
        goal_dim=GOAL_DIM,
    ).to(device)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ckpt_path = os.path.join(args.output_dir, "best_v4.pt")
    last_ckpt_path = os.path.join(args.output_dir, "last_v4.pt")
    ckpt_extra = {"num_total_nodes": num_total_nodes}

    start_epoch, best_mrr, epochs_no_improve = maybe_resume(
        args, "v4", args.output_dir, model, optimizer, device,
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
            nutrient_tensor, ingredient_ids, args.num_neg,
            args.lambda_h, args.margin, device,
        )
        val_metrics = evaluate(
            model, val_loader, edge_index, edge_weight, nutrient_tensor,
            ingredient_ids, ingredient_id_to_pos, device,
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

    cleanup_last_ckpt(args.output_dir, "v4")

    if not os.path.exists(best_ckpt_path):
        print("[warning] no checkpoint saved.")
        return

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    for label in args.test_g_overrides:
        override = parse_g_label(label)
        suffix = f"_{label}"
        print(f"\n=== TEST [v4{suffix}] (best epoch {ckpt['epoch']}) ===")
        if override is not None:
            print(f"  g override: {override.tolist()}")
        test_metrics = evaluate(
            model, test_loader, edge_index, edge_weight, nutrient_tensor,
            ingredient_ids, ingredient_id_to_pos, device,
            eval_chunk=args.eval_chunk, override_g=override,
            valid_targets_map=valid_targets_map,
        )
        print(f"  MRR    {test_metrics['MRR']:.2f}")
        print(f"  Hit@1  {test_metrics['Hit@1']:.2f}")
        print(f"  Hit@3  {test_metrics['Hit@3']:.2f}")
        print(f"  Hit@10 {test_metrics['Hit@10']:.2f}")

        out = {
            "mode": "v4",
            "g_label": label,
            "best_epoch": int(ckpt["epoch"]),
            "metrics": {k: test_metrics[k] for k in ("MRR", "Hit@1", "Hit@3", "Hit@10")},
            "ranks": test_metrics["ranks"],
            "top1": test_metrics["top1"],
            "sources": test_metrics["sources"],
            "targets": test_metrics["targets"],
            "goals": test_metrics["goals"],
        }
        pred_path = os.path.join(args.output_dir, f"test_predictions_v4{suffix}.json")
        with open(pred_path, "w") as f:
            json.dump(out, f)
        print(f"[saved] {pred_path}")


if __name__ == "__main__":
    main()
