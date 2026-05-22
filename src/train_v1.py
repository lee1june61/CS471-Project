"""Training script for Baseline (vanilla GISMo) and MVP (GC-GISMo).

REFACTORED for sparse ingredient ID support:
  - 6,313 ingredient ids are spread sparsely over [0, 7101] (789 gaps).
  - `torch.arange(num_ingredients)` is WRONG — would include non-ingredient
    ids in the candidate pool.
  - All eval / negative-sampling logic now uses an `ingredient_ids` tensor
    (sorted, dense list of valid ingredient node ids).
  - `num_total_nodes` is read from nodes_filtered.csv max(id)+1 (= 8748).
  - Embedding table is sized for num_total_nodes so source/target lookups
    by raw id work directly.
  - load_graph is called with valid_node_ids to drop the ~530 edges that
    reference deleted ingredient ids.

Multi-valid-target evaluation:
  Built from train+val+test pairs. When ranking, other valid substitutions
  for the same (source, recipe) are masked out so they don't penalize MRR/Hit@k.

Usage:
    python train_v1.py --mode baseline --data_dir ./data --output_dir ./outputs
    python train_v1.py --mode mvp      --data_dir ./data --output_dir ./outputs \\
                       --test_g_overrides auto 1_0 0_1
"""

import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import (HEALTH_NUTRIENT_KEYS, SubstitutionDataset,
                     build_valid_targets_map, compute_thresholds,
                     load_graph, load_node_ids, load_nutrient_tensor)
from models_v1 import GISMo


GOAL_DIM = 2


# ---------------------------------------------------------------------------
# Checkpoint helpers (shared with train_v2/v3/v4)
# ---------------------------------------------------------------------------

def _rng_snapshot():
    snap = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        snap["torch_cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap):
    # NOTE: load_checkpoint uses torch.load(map_location=device); when device
    # is CUDA, every tensor in the unpickled dict (including these RNG
    # ByteTensors) lands on CUDA. torch.set_rng_state and
    # torch.cuda.set_rng_state_all both require CPU tensors, so we move
    # them back explicitly before restoring.
    if snap is None:
        return
    if snap.get("torch") is not None:
        torch.set_rng_state(snap["torch"].cpu())
    if snap.get("numpy") is not None:
        np.random.set_state(snap["numpy"])
    if snap.get("python") is not None:
        random.setstate(snap["python"])
    cuda_state = snap.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            cuda_state = [t.cpu() for t in cuda_state]
            torch.cuda.set_rng_state_all(cuda_state)
        except Exception as e:
            print(f"[resume] warning: could not restore CUDA RNG: {e}")


def save_checkpoint(path, *, model, optimizer, epoch, best_mrr,
                    epochs_no_improve, args, val_metrics, extra=None):
    """Atomic checkpoint save with full resumable state.

    Writes to <path>.tmp.<pid> then os.replace so a crash mid-save leaves
    the previous good checkpoint intact, and concurrent writers to the
    same output_dir don't collide on the tmp filename.
    """
    state = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_mrr": float(best_mrr),
        "epochs_no_improve": int(epochs_no_improve),
        "args": vars(args) if not isinstance(args, dict) else args,
        "val_metrics": {k: v for k, v in val_metrics.items()
                         if k in ("MRR", "Hit@1", "Hit@3", "Hit@10")},
        "rng": _rng_snapshot(),
    }
    if extra:
        state.update(extra)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, device, restore_rng=True):
    """Load checkpoint; restore model/optimizer/RNG.

    Returns (start_epoch, best_mrr, epochs_no_improve, raw_ckpt_dict).
    Tolerates older checkpoints (missing best_mrr / epochs_no_improve / rng).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt["epoch"]) + 1
    if "best_mrr" in ckpt:
        best_mrr = float(ckpt["best_mrr"])
    else:
        best_mrr = float(ckpt.get("val_metrics", {}).get("MRR", -1.0))
    epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
    if restore_rng:
        _rng_restore(ckpt.get("rng"))
    return start_epoch, best_mrr, epochs_no_improve, ckpt


# Args whose mismatch between current run and resumed checkpoint
# usually means shape mismatch or different experiment — warn loudly.
_SHAPE_AFFECTING_ARGS = (
    "mode", "embed_dim", "num_gin_layers", "ablation_no_compound",
    "hub_nutrient_keys", "encoder_nutrient_keys", "lambda_h",
    "tau_percentile",
)


def _warn_arg_mismatches(saved_args, current_args):
    """Print actionable diff between saved checkpoint args and current run."""
    mismatches = []
    for k in _SHAPE_AFFECTING_ARGS:
        if k not in saved_args:
            continue
        cur = getattr(current_args, k, None)
        saved = saved_args[k]
        if cur != saved:
            mismatches.append((k, saved, cur))
    if mismatches:
        print(f"[resume] WARNING: {len(mismatches)} arg mismatch(es) vs checkpoint:")
        for k, saved, cur in mismatches:
            print(f"  {k}: ckpt={saved!r}  current={cur!r}")
        print(f"[resume] If model shapes differ, load_state_dict will fail.")
        print(f"[resume] Pass --no_resume to start fresh.")


def maybe_resume(args, mode, output_dir, model, optimizer, device):
    """Decide which checkpoint (if any) to resume from.

    Priority:
      1. Explicit --resume <path>
      2. <output_dir>/last_<mode>.pt (auto-resume on crash recovery)
      3. None (fresh start)

    `--no_resume` skips (2). Conflicting --resume + --no_resume raises.
    On resume, warns about shape-affecting arg mismatches between the
    checkpoint and the current invocation. When --no_resume is set and
    best_<mode>.pt exists, the existing file is backed up to .bak so the
    first improvement of the new run doesn't silently overwrite a prior
    champion.
    Returns (start_epoch, best_mrr, epochs_no_improve).
    """
    if args.resume and args.no_resume:
        raise ValueError("--resume and --no_resume are mutually exclusive.")

    last_path = os.path.join(output_dir, f"last_{mode}.pt")
    best_path = os.path.join(output_dir, f"best_{mode}.pt")

    resume_path = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(args.resume)
        resume_path = args.resume
        print(f"[resume] using explicit --resume {resume_path}")
    elif args.no_resume:
        if os.path.exists(last_path):
            print(f"[resume] --no_resume set; ignoring {last_path}")
        if os.path.exists(best_path):
            bak = best_path + ".bak"
            shutil.copyfile(best_path, bak)
            print(f"[resume] --no_resume set; backed up existing "
                  f"{best_path} -> {bak} so first improvement doesn't clobber it")
    elif os.path.exists(last_path):
        resume_path = last_path
        print(f"[resume] auto-resuming from {resume_path} "
              f"(pass --no_resume to start fresh)")

    if resume_path is None:
        return 1, -1.0, 0

    start_epoch, best_mrr, epochs_no_improve, ckpt = load_checkpoint(
        resume_path, model, optimizer, device,
    )
    saved_args = ckpt.get("args", {})
    if saved_args:
        _warn_arg_mismatches(saved_args, args)
    print(f"[resume] continuing at epoch {start_epoch} "
          f"(best_mrr={best_mrr:.2f}, epochs_no_improve={epochs_no_improve})")
    return start_epoch, best_mrr, epochs_no_improve


def cleanup_last_ckpt(output_dir, mode):
    """Remove last_<mode>.pt after a clean training finish so the next
    invocation doesn't auto-resume from a stale checkpoint."""
    last_path = os.path.join(output_dir, f"last_{mode}.pt")
    if os.path.exists(last_path):
        try:
            os.remove(last_path)
            print(f"[cleanup] removed {last_path}")
        except OSError as e:
            print(f"[cleanup] could not remove {last_path}: {e}")


# ---------------------------------------------------------------------------
# Helpers (sparse id support)
# ---------------------------------------------------------------------------

def build_id_to_pos(ingredient_ids):
    """Map raw ingredient node id -> position in the ingredient_ids tensor.
    Used for converting source/target raw ids to score-tensor column indices.
    """
    return {int(v): i for i, v in enumerate(ingredient_ids.tolist())}


def ids_to_positions(ids, sorted_ingredient_ids):
    """GPU-friendly: raw ids -> positions in sorted_ingredient_ids via searchsorted.

    Assumes every id is present in sorted_ingredient_ids (true for source/target
    in pairs_*.csv, which only contain valid ingredient ids).
    """
    pos = torch.searchsorted(sorted_ingredient_ids, ids)
    # Cheap sanity: catches if a pair csv accidentally references a non-ingredient
    if not (sorted_ingredient_ids[pos] == ids).all():
        bad = ids[sorted_ingredient_ids[pos] != ids][:5].tolist()
        raise ValueError(f"ids_to_positions: ids not in ingredient set: {bad} ...")
    return pos


# ---------------------------------------------------------------------------
# Negative sampling and losses
# ---------------------------------------------------------------------------

def sample_negatives(positive_targets, source_ids, ingredient_ids, num_neg):
    """Sample `num_neg` random ingredient negatives per query.

    ingredient_ids: [N_ing] sorted tensor of valid ingredient node ids (sparse).
    Returns raw node ids (not positions).
    """
    B = positive_targets.shape[0]
    device = positive_targets.device
    N_ing = ingredient_ids.shape[0]

    idx = torch.randint(0, N_ing, (B, num_neg), device=device)
    negs = ingredient_ids[idx]
    for _ in range(3):
        bad = (negs == positive_targets.unsqueeze(1)) | (negs == source_ids.unsqueeze(1))
        if not bad.any():
            break
        new_idx = torch.randint(0, N_ing, (int(bad.sum()),), device=device)
        negs[bad] = ingredient_ids[new_idx]
    return negs


def health_loss_fn(g, source_ids, candidate_ids, scores,
                   nutrient_tensor, margin=0.5):
    """L_health on the model's predicted candidate distribution.

    For each query, take expected nutrient delta under softmax(scores)
    across all candidates (the positive target + sampled negatives), then
    hinge-penalize when the expected reduction is below `margin`.

    Why this form (and not the prior `delta(source, ground_truth_y)`):
    nutrient_tensor is a fixed buffer and the previous formulation
    contained no learnable params  it was a per-pair constant with zero
    gradient. The current form depends on `scores` (from the model),
    so optimizer.step() actually pushes the distribution toward
    health-compatible candidates. Same link-prediction framework as L_sub.

    Args:
        g:             [B, G] goal one-hot, G = len(HEALTH_NUTRIENT_KEYS)
        source_ids:    [B]
        candidate_ids: [B, K] (positive at index 0, then negs)
        scores:        [B, K] from model(...)
        nutrient_tensor: [num_total_nodes, K_nut] (>=G)
        margin:        hinge margin in standardized-nutrient units
    """
    if g is None or nutrient_tensor is None:
        return torch.tensor(0.0, device=source_ids.device)

    probs = F.softmax(scores, dim=1)                               # [B, K]
    n_s = nutrient_tensor[source_ids].unsqueeze(1)                 # [B, 1, K_nut]
    n_cand = nutrient_tensor[candidate_ids]                         # [B, K, K_nut]
    G = g.shape[-1]
    delta = (n_s - n_cand)[:, :, :G]                                # [B, K, G]
    expected_delta = (probs.unsqueeze(-1) * delta).sum(dim=1)       # [B, G]

    per_nut = torch.clamp(margin - expected_delta, min=0.0)         # [B, G]
    weighted = g * per_nut
    active = g.sum() + 1e-6
    return weighted.sum() / active


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, edge_index, edge_weight,
                    ingredient_ids, num_neg, use_health_goal,
                    nutrient_tensor, lambda_h, margin, device):
    model.train()
    sums = {"loss": 0.0, "sub": 0.0, "h": 0.0}
    n = 0

    for batch in loader:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        recipe_ings = batch["recipe_ings"].to(device)
        recipe_mask = batch["recipe_mask"].to(device)
        g = batch["g"].to(device) if use_health_goal else None

        negs = sample_negatives(target, source, ingredient_ids, num_neg)
        candidates = torch.cat([target.unsqueeze(1), negs], dim=1)

        h = model.encode_graph(edge_index, edge_weight)
        scores = model(h, source, candidates, recipe_ings, recipe_mask, g)

        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
        L_sub = F.cross_entropy(scores, labels)

        if use_health_goal and nutrient_tensor is not None:
            L_h = health_loss_fn(g, source, candidates, scores,
                                  nutrient_tensor, margin)
            loss = L_sub + lambda_h * L_h
        else:
            L_h = torch.tensor(0.0, device=device)
            loss = L_sub

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sums["loss"] += loss.item()
        sums["sub"] += L_sub.item()
        sums["h"] += L_h.item()
        n += 1

    return {k: v / max(n, 1) for k, v in sums.items()}


def _build_valid_target_mask(source, target, recipe_id, ingredient_id_to_pos,
                              num_ing, valid_targets_map, device):
    """[B, num_ing] bool mask: True for alternate valid targets to exclude.

    num_ing is len(ingredient_ids). Mask positions correspond to columns of
    the score tensor (not raw node ids).
    """
    sources_cpu = source.cpu().numpy()
    targets_cpu = target.cpu().numpy()
    recipes_cpu = recipe_id.cpu().numpy()
    B = source.shape[0]

    mask_np = np.zeros((B, num_ing), dtype=bool)
    for i in range(B):
        s_i = int(sources_cpu[i])
        r_i = int(recipes_cpu[i])
        y_i = int(targets_cpu[i])
        valid_set = valid_targets_map.get((s_i, r_i))
        if valid_set is None:
            continue
        for v in valid_set:
            if v != y_i:
                pos = ingredient_id_to_pos.get(v)
                if pos is not None:
                    mask_np[i, pos] = True
    return torch.from_numpy(mask_np).to(device)


@torch.no_grad()
def evaluate(model, loader, edge_index, edge_weight, ingredient_ids,
             ingredient_id_to_pos, use_health_goal, device, eval_chunk=256,
             override_g=None, valid_targets_map=None):
    """MRR / Hit@k by ranking over valid ingredient candidates (sparse-id safe).

    ingredient_ids:        sorted [N_ing] LongTensor on device.
    ingredient_id_to_pos:  dict {raw_id: position} for masking & lookups.
    """
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

        if use_health_goal:
            if override_g is not None:
                g = override_g.to(device).unsqueeze(0).expand(source.shape[0], -1)
            else:
                g = batch["g"].to(device)
        else:
            g = None

        B = source.shape[0]
        scores = torch.empty((B, N_ing), device=device)
        for start in range(0, N_ing, eval_chunk):
            end = min(start + eval_chunk, N_ing)
            chunk = ingredient_ids[start:end].unsqueeze(0).expand(B, -1)
            chunk_scores = model(h, source, chunk, recipe_ings, recipe_mask, g)
            scores[:, start:end] = chunk_scores

        # Source can't substitute itself. Convert source raw-id -> position.
        source_pos = ids_to_positions(source, ingredient_ids)
        scores.scatter_(1, source_pos.unsqueeze(1), float("-inf"))

        # top-1 (raw node id, not position)
        top1_pos = scores.argmax(dim=1)
        top1 = ingredient_ids[top1_pos]

        # Rank computation: optionally also mask alt valid targets.
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
        if g is not None:
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
        "goals": (np.concatenate(goals_list).tolist() if goals_list else None),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "mvp"], required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")

    # Note: nodes_filtered.csv now defines num_total_nodes and ingredient ids.
    # These CLI args are no longer needed — kept only for backward CLI compat
    # (ignored; data is the source of truth).
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
                   help="Explicit checkpoint path to resume from. If omitted, "
                        "auto-resume from <output_dir>/last_<mode>.pt when present.")
    p.add_argument("--no_resume", action="store_true",
                   help="Ignore any existing last_<mode>.pt and start fresh.")
    p.add_argument("--no_multi_valid", action="store_true")
    p.add_argument("--ablation_no_compound", action="store_true",
                   help="Drop I-F / I-D edges (keep only I-I). Used for the "
                        "'w/o flavor compound' ablation.")

    p.add_argument("--lambda_h", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--tau_percentile", type=float, default=50.0)

    p.add_argument("--test_g_overrides", nargs="*", default=["auto"])

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_g_label(label, expected_dim=GOAL_DIM):
    if label == "auto":
        return None
    parts = label.split("_")
    if len(parts) != expected_dim:
        raise ValueError(
            f"g label '{label}' has {len(parts)} components, expected {expected_dim}."
        )
    values = [float(x) for x in parts]
    return torch.tensor(values, dtype=torch.float)


def main():
    args = parse_args()
    if args.num_total_nodes is not None or args.num_ingredients is not None:
        print("[deprecated] --num_total_nodes / --num_ingredients are ignored; "
              "auto-detected from nodes_filtered.csv")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_health_goal = (args.mode == "mvp")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[config] mode={args.mode}  device={device}")
    print(f"[config] full args: {json.dumps(vars(args), indent=2, default=str)}")

    if use_health_goal:
        for label in args.test_g_overrides:
            parse_g_label(label)

    # --- Load node ids (source of truth for embedding size & candidate pool) ---
    ingredient_ids, num_total_nodes, all_node_ids = load_node_ids(
        os.path.join(args.data_dir, "nodes_filtered.csv"),
    )
    ingredient_id_to_pos = build_id_to_pos(ingredient_ids)
    ingredient_ids = ingredient_ids.to(device)

    # --- Threshold (MVP only) ---
    tau_sugar, tau_sodium = 0.0, 0.0
    if use_health_goal:
        tau_sugar, tau_sodium = compute_thresholds(
            pairs_csv=os.path.join(args.data_dir, "pairs_train.csv"),
            usda_json=os.path.join(args.data_dir, "usda_mapping.json"),
            percentile=args.tau_percentile,
        )

    def make_ds(split):
        return SubstitutionDataset(
            pairs_csv=os.path.join(args.data_dir, f"pairs_{split}.csv"),
            recipes_json=os.path.join(args.data_dir, "recipes.json"),
            usda_json=(os.path.join(args.data_dir, "usda_mapping.json")
                       if use_health_goal else None),
            use_health_goal=use_health_goal,
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

    # --- Graph with invalid-edge filtering (all_node_ids from load_node_ids above) ---
    edge_index, edge_weight, max_node_in_graph = load_graph(
        os.path.join(args.data_dir, "flavorgraph_edges.csv"),
        valid_node_ids=all_node_ids,
        edge_types=("I-I",) if args.ablation_no_compound else None,
    )
    if max_node_in_graph > num_total_nodes:
        raise ValueError(
            f"Edge file references node id {max_node_in_graph - 1}, but "
            f"num_total_nodes={num_total_nodes}."
        )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    print(f"[graph] {edge_index.shape[1]} edges (after filtering)")

    # --- Nutrient tensor sized for num_total_nodes (allows direct id lookup) ---
    nutrient_tensor = None
    if use_health_goal:
        nutrient_tensor = load_nutrient_tensor(
            os.path.join(args.data_dir, "usda_mapping.json"),
            num_total_nodes=num_total_nodes,
            nutrient_keys=HEALTH_NUTRIENT_KEYS,
        ).to(device)

    # --- Model ---
    model = GISMo(
        num_nodes=num_total_nodes,
        embed_dim=args.embed_dim,
        hidden_dim=args.embed_dim,
        num_gin_layers=args.num_gin_layers,
        dropout=args.dropout,
        use_health_goal=use_health_goal,
        goal_dim=GOAL_DIM,
    ).to(device)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ckpt_path = os.path.join(args.output_dir, f"best_{args.mode}.pt")
    last_ckpt_path = os.path.join(args.output_dir, f"last_{args.mode}.pt")
    ckpt_extra = {"num_total_nodes": num_total_nodes}

    start_epoch, best_mrr, epochs_no_improve = maybe_resume(
        args, args.mode, args.output_dir, model, optimizer, device,
    )

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
            ingredient_ids, args.num_neg, use_health_goal,
            nutrient_tensor, args.lambda_h, args.margin, device,
        )
        val_metrics = evaluate(
            model, val_loader, edge_index, edge_weight, ingredient_ids,
            ingredient_id_to_pos, use_health_goal, device,
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

        # Order matters: save BEST first, then LAST. If we crash between
        # the two, best is up-to-date and last is one epoch behind. On
        # resume, the lagging epoch is re-trained (deterministic via
        # restored RNG) and idempotently rewrites best. The reverse order
        # would leave best stale and let a subsequent worse epoch
        # overwrite the better best that already happened on disk.
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

    # Training loop done (natural end or early-stop). Remove last_*.pt
    # NOW (before test eval) so a test-eval exception doesn't orphan it.
    cleanup_last_ckpt(args.output_dir, args.mode)

    if not os.path.exists(best_ckpt_path):
        print("[warning] no checkpoint saved.")
        return

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if not use_health_goal:
        eval_configs = [(None, "")]
    else:
        eval_configs = [(parse_g_label(lbl), lbl) for lbl in args.test_g_overrides]

    for override, label in eval_configs:
        suffix = f"_{label}" if label else ""
        print(f"\n=== TEST [{args.mode}{suffix}] (best epoch {ckpt['epoch']}) ===")
        if override is not None:
            print(f"  g override: {override.tolist()}")
        test_metrics = evaluate(
            model, test_loader, edge_index, edge_weight, ingredient_ids,
            ingredient_id_to_pos, use_health_goal, device,
            eval_chunk=args.eval_chunk, override_g=override,
            valid_targets_map=valid_targets_map,
        )
        print(f"  MRR    {test_metrics['MRR']:.2f}")
        print(f"  Hit@1  {test_metrics['Hit@1']:.2f}")
        print(f"  Hit@3  {test_metrics['Hit@3']:.2f}")
        print(f"  Hit@10 {test_metrics['Hit@10']:.2f}")

        out = {
            "mode": args.mode,
            "g_label": label if label else None,
            "best_epoch": int(ckpt["epoch"]),
            "metrics": {k: test_metrics[k] for k in ("MRR", "Hit@1", "Hit@3", "Hit@10")},
            "ranks": test_metrics["ranks"],
            "top1": test_metrics["top1"],
            "sources": test_metrics["sources"],
            "targets": test_metrics["targets"],
        }
        if test_metrics["goals"] is not None:
            out["goals"] = test_metrics["goals"]
        pred_path = os.path.join(args.output_dir,
                                 f"test_predictions_{args.mode}{suffix}.json")
        with open(pred_path, "w") as f:
            json.dump(out, f)
        print(f"[saved] {pred_path}")


if __name__ == "__main__":
    main()
