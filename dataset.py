"""Datasets and graph loading."""

import json
from collections import defaultdict
from typing import Dict, FrozenSet, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Standard 2-d health goal nutrient keys. Order must match goal vector
# dimensions (g[0] = sugar, g[1] = sodium).
HEALTH_NUTRIENT_KEYS = ("sugar_g", "sodium_mg")


def _safe_float(value, default=0.0):
    """Convert to float, treating None / non-numeric as `default`.

    USDA mapping sometimes has explicit None values for nutrients that
    aren't reported, so .get(key, 0.0) doesn't help — we need to coerce.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Substitution pair dataset
# ---------------------------------------------------------------------------

class SubstitutionDataset(Dataset):
    """One sample = (source, target, recipe) [+ derived g for MVP].

    Files:
      pairs_csv:    columns [source_id, target_id, recipe_id]
      recipes_json: {recipe_id: [ing_id, ...]}
      usda_json:    {ing_id: {sugar_g, sodium_mg, ...}}  (MVP only)
    """

    def __init__(self, pairs_csv, recipes_json, usda_json=None,
                 use_health_goal=False, tau_sugar=0.0, tau_sodium=0.0,
                 max_recipe_len=20, pad_id=0):
        self.pairs = pd.read_csv(pairs_csv)
        # Cache as numpy for fast __getitem__ access (~10x faster than pandas iloc).
        self.s_arr = self.pairs["source_id"].to_numpy(dtype=np.int64)
        self.y_arr = self.pairs["target_id"].to_numpy(dtype=np.int64)
        self.r_arr = self.pairs["recipe_id"].to_numpy(dtype=np.int64)

        with open(recipes_json) as f:
            raw = json.load(f)
        self.recipes = {int(k): v for k, v in raw.items()}

        self.usda = None
        if use_health_goal and usda_json:
            with open(usda_json) as f:
                self.usda = {int(k): v for k, v in json.load(f).items()}

        self.use_health_goal = use_health_goal
        self.tau_sugar = tau_sugar
        self.tau_sodium = tau_sodium
        self.max_recipe_len = max_recipe_len
        self.pad_id = pad_id

    def __len__(self):
        return len(self.pairs)

    def derive_g(self, source_id, target_id):
        """Return [low_sugar, low_sodium] in {0,1}^2 derived from USDA Δ.

        signed delta = n_source - n_target. goal=1 iff delta > tau
        (tau is the percentile of POSITIVE deltas in training set
         — see compute_thresholds for details).
        Either ingredient missing in USDA → g = [0, 0].
        """
        if self.usda is None:
            return torch.zeros(2, dtype=torch.float)
        s = self.usda.get(source_id)
        y = self.usda.get(target_id)
        if s is None or y is None:
            return torch.zeros(2, dtype=torch.float)
        d_sugar = _safe_float(s.get("sugar_g")) - _safe_float(y.get("sugar_g"))
        d_sodium = _safe_float(s.get("sodium_mg")) - _safe_float(y.get("sodium_mg"))
        return torch.tensor([
            1.0 if d_sugar > self.tau_sugar else 0.0,
            1.0 if d_sodium > self.tau_sodium else 0.0,
        ], dtype=torch.float)

    def __getitem__(self, idx):
        # Fast path: numpy arrays cached in __init__ (was pd.iloc — slow).
        s = int(self.s_arr[idx])
        y = int(self.y_arr[idx])
        r = int(self.r_arr[idx])

        recipe_ings = self.recipes.get(r, [s])  # safe fallback
        recipe_ings = list(recipe_ings)[: self.max_recipe_len]
        L = len(recipe_ings)

        ing_pad = recipe_ings + [self.pad_id] * (self.max_recipe_len - L)
        mask = [1.0] * L + [0.0] * (self.max_recipe_len - L)

        item = {
            "source": torch.tensor(s, dtype=torch.long),
            "target": torch.tensor(y, dtype=torch.long),
            "recipe_id": torch.tensor(r, dtype=torch.long),
            "recipe_ings": torch.tensor(ing_pad, dtype=torch.long),
            "recipe_mask": torch.tensor(mask, dtype=torch.float),
            "pair_idx": torch.tensor(idx, dtype=torch.long),
        }
        if self.use_health_goal:
            item["g"] = self.derive_g(s, y)
        return item


# ---------------------------------------------------------------------------
# Multi-valid-target map (matches GISMo's eval treatment)
# ---------------------------------------------------------------------------

def build_valid_targets_map(*pairs_csvs: str
                            ) -> Dict[Tuple[int, int], FrozenSet[int]]:
    """Combine pairs across splits (train/val/test) and return:
        {(source_id, recipe_id): frozenset of valid target_ids}

    Used by the evaluator to avoid penalizing the model when it ranks
    a *different* but also-valid substitution above the row's target.
    Matches GISMo paper:
        "we avoid penalizing for such cases through their ranking by
         accepting all valid target ingredients as correct answers"
    """
    valid: Dict[Tuple[int, int], set] = defaultdict(set)
    for csv in pairs_csvs:
        df = pd.read_csv(csv)
        for s, y, r in zip(df["source_id"], df["target_id"], df["recipe_id"]):
            valid[(int(s), int(r))].add(int(y))
    return {k: frozenset(v) for k, v in valid.items()}


# ---------------------------------------------------------------------------
# Threshold computation (percentile-th percentile of POSITIVE Δ only)
# ---------------------------------------------------------------------------

def compute_thresholds(pairs_csv, usda_json, percentile=50) -> Tuple[float, float]:
    """τ = `percentile`-th percentile of POSITIVE Δ (source - target > 0)
    over training pairs that have USDA mapping for both s and y.

    Rationale: the goal label fires when (n_source - n_target) > τ — i.e.
    only when the substitution *reduces* the nutrient. So τ should be
    calibrated on the distribution of reductions, not on |Δ| (which would
    mix in substitutions that *increase* the nutrient).

    Special case: percentile == 0 returns (0.0, 0.0). With τ=0 any positive
    Δ activates the goal (every reduction counts).
    """
    if percentile == 0:
        return 0.0, 0.0

    pairs = pd.read_csv(pairs_csv)
    with open(usda_json) as f:
        usda = {int(k): v for k, v in json.load(f).items()}

    pos_sugar, pos_sodium = [], []
    n_total = 0
    for _, row in pairs.iterrows():
        s, y = int(row["source_id"]), int(row["target_id"])
        if s not in usda or y not in usda:
            continue
        n_total += 1
        d_sugar = (_safe_float(usda[s].get("sugar_g"))
                   - _safe_float(usda[y].get("sugar_g")))
        d_sodium = (_safe_float(usda[s].get("sodium_mg"))
                    - _safe_float(usda[y].get("sodium_mg")))
        if d_sugar > 0:
            pos_sugar.append(d_sugar)
        if d_sodium > 0:
            pos_sodium.append(d_sodium)

    tau_sugar = float(np.percentile(pos_sugar, percentile)) if pos_sugar else 0.0
    tau_sodium = float(np.percentile(pos_sodium, percentile)) if pos_sodium else 0.0

    if n_total > 0:
        print(f"[thresholds] {n_total} mapped pairs | "
              f"sugar: {len(pos_sugar)} reducers "
              f"({len(pos_sugar) / n_total * 100:.1f}%), "
              f"tau={tau_sugar:.4f} (p{percentile}) | "
              f"sodium: {len(pos_sodium)} reducers "
              f"({len(pos_sodium) / n_total * 100:.1f}%), "
              f"tau={tau_sodium:.4f} (p{percentile})")
    return tau_sugar, tau_sodium


# ---------------------------------------------------------------------------
# Node id loading (handles sparse ingredient id space)
# ---------------------------------------------------------------------------

def load_node_ids(nodes_csv) -> Tuple[torch.Tensor, int, list]:
    """Load valid ingredient node ids from nodes_filtered.csv.

    Returns:
        ingredient_ids:    sorted [N_ingr] LongTensor of valid ingredient node ids.
                           IMPORTANT: ids are SPARSE  6,313 valid ids spread over
                           [0, 7101] with 789 gaps. Do NOT use torch.arange to
                           enumerate ingredients; iterate this tensor instead.
        num_total_nodes:   max(node_id) + 1 over ALL node types (ingredient +
                           compound), used for sizing the embedding table.
        all_node_ids:      Python list of all valid node ids (ingredient + compound),
                           ready to pass as `valid_node_ids` to `load_graph`.
    """
    df = pd.read_csv(nodes_csv)
    ingredient_ids = torch.tensor(
        sorted(df[df["node_type"] == "ingredient"]["node_id"].tolist()),
        dtype=torch.long,
    )
    all_node_ids = df["node_id"].tolist()
    num_total_nodes = int(df["node_id"].max()) + 1
    print(f"[load_node_ids] {len(ingredient_ids)} ingredients "
          f"(id range [{int(ingredient_ids.min())}, {int(ingredient_ids.max())}]), "
          f"{len(all_node_ids)} total valid nodes, num_total_nodes={num_total_nodes}")
    return ingredient_ids, num_total_nodes, all_node_ids


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(edges_csv, valid_node_ids=None, edge_types=None
               ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Load FlavorGraph edges into (edge_index, edge_weight, max_node_id+1).

    Edges are treated as undirected  both (s, t) and (t, s) added.
    Duplicate edges are dropped (e.g. if data already lists both directions).
    NaN/inf weights are replaced with 1.0  matches GISMo's handling of
    I-F / I-D edges that have no NPMI score.

    Args:
        valid_node_ids: optional iterable / tensor of valid node ids. If
            provided, edges referencing any other node id are dropped. Use
            this with `load_node_ids` to filter out the ~530 edges that
            reference ingredient ids that were removed during data filtering.
        edge_types: optional iterable of edge_type strings to keep
            (e.g. ("I-I",) drops I-F / I-D edges — used by the
            "w/o flavor compound" ablation). None = keep all edges.
            Requires the CSV to have an "edge_type" column.

    Returns the max referenced node id + 1 (after filtering, if applied),
    useful as a sanity check for num_total_nodes.
    """
    df = pd.read_csv(edges_csv)
    df = df.drop_duplicates(subset=["src_id", "dst_id"]).reset_index(drop=True)

    if edge_types is not None:
        if "edge_type" not in df.columns:
            raise ValueError(
                "edge_types filter requested but CSV has no 'edge_type' column."
            )
        before = len(df)
        keep = tuple(edge_types)
        df = df[df["edge_type"].isin(keep)].reset_index(drop=True)
        print(f"[load_graph] edge_type filter {keep}: kept {len(df)}/{before} edges")

    if valid_node_ids is not None:
        if torch.is_tensor(valid_node_ids):
            valid_set = set(valid_node_ids.tolist())
        else:
            valid_set = set(valid_node_ids)
        before = len(df)
        df = df[df["src_id"].isin(valid_set) & df["dst_id"].isin(valid_set)]
        df = df.reset_index(drop=True)
        if before > len(df):
            print(f"[load_graph] dropped {before - len(df)} edges referencing "
                  f"invalid (non-existent) node ids")

    src = df["src_id"].to_numpy(dtype=np.int64)
    dst = df["dst_id"].to_numpy(dtype=np.int64)
    if "weight" in df.columns:
        w = df["weight"].to_numpy(dtype=np.float32)
        bad = np.isnan(w) | np.isinf(w)
        if bad.any():
            print(f"[load_graph] {bad.sum()} edges had NaN/inf weights → set to 1.0")
            w[bad] = 1.0
    else:
        w = np.ones(len(df), dtype=np.float32)

    max_node = int(max(src.max(), dst.max())) + 1

    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    w_sym = np.concatenate([w, w])

    edge_index = torch.tensor(np.stack([src_sym, dst_sym]), dtype=torch.long)
    edge_weight = torch.tensor(w_sym, dtype=torch.float)
    return edge_index, edge_weight, max_node


def load_nutrient_tensor(usda_json, num_total_nodes,
                         nutrient_keys=("sugar_g", "sodium_mg")
                        ) -> torch.Tensor:
    """Build a [num_total_nodes, len(nutrient_keys)] tensor of
    log1p + per-nutrient z-score values.

    Note: sized for num_total_nodes (not num_ingredients) so that
    `tensor[source_id]` works directly for any valid ingredient id  even
    in the sparse id space [0, 7101]. Non-ingredient rows (compounds and
    gaps) get zero, which after z-score == the mean  acts as a neutral
    default if the model ever accidentally looks them up.

    Order of `nutrient_keys` MUST match goal vector dimensions for L_health.
    """
    with open(usda_json) as f:
        usda = {int(k): v for k, v in json.load(f).items()}

    arr = np.zeros((num_total_nodes, len(nutrient_keys)), dtype=np.float32)
    has_data = np.zeros(num_total_nodes, dtype=bool)
    for ing_id, nuts in usda.items():
        if 0 <= ing_id < num_total_nodes:
            for k_idx, k in enumerate(nutrient_keys):
                arr[ing_id, k_idx] = _safe_float(nuts.get(k))
            has_data[ing_id] = True

    arr = np.log1p(np.maximum(arr, 0.0))
    if has_data.any():
        mu = arr[has_data].mean(axis=0, keepdims=True)
        sd = arr[has_data].std(axis=0, keepdims=True) + 1e-6
    else:
        mu = arr.mean(axis=0, keepdims=True)
        sd = arr.std(axis=0, keepdims=True) + 1e-6
    arr = (arr - mu) / sd
    arr[~has_data] = 0.0

    return torch.tensor(arr, dtype=torch.float)
