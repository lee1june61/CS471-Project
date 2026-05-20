"""Convert GISMo's preprocessed Recipe1MSubs (.pkl) + graph team's data
into the format our model expects.

Inputs (assumed in `data_raw/`):
  - train_comments_subs.pkl, val_comments_subs.pkl, test_comments_subs.pkl
  - vocab_ingrs.pkl
  - flavorgraph_edges.csv  (from graph team)
  - nodes_filtered.csv     (from graph team)
  - usda_mapping.json      (from graph team)

Outputs (written to `data/`):
  - pairs_train.csv, pairs_val.csv, pairs_test.csv
  - recipes.json
  - flavorgraph_edges.csv  (copied)
  - usda_mapping.json      (copied)
  - ingredients.csv        (subset of nodes_filtered, ingredient nodes only)
  - data_meta.json         (num_total_nodes, num_ingredients, sizes)

Usage:
    python convert_data.py --data_raw ./data_raw --out_dir ./data
"""

import argparse
import json
import pickle
import shutil
import sys
import types
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Stub Vocabulary class so pickle.load can find it without the GISMo repo.
# pickle only restores __dict__ attributes (word2idx, idx2word, idx), so a
# bare class body is sufficient for read-only use.
# ---------------------------------------------------------------------------

class Vocabulary:
    """Read-only stub matching inv_cooking.datasets.vocabulary.Vocabulary."""
    pass


def _register_vocabulary_stub():
    mod_inv = types.ModuleType("inv_cooking")
    mod_datasets = types.ModuleType("inv_cooking.datasets")
    mod_vocab = types.ModuleType("inv_cooking.datasets.vocabulary")
    mod_vocab.Vocabulary = Vocabulary
    mod_inv.datasets = mod_datasets
    mod_datasets.vocabulary = mod_vocab
    sys.modules.setdefault("inv_cooking", mod_inv)
    sys.modules.setdefault("inv_cooking.datasets", mod_datasets)
    sys.modules.setdefault("inv_cooking.datasets.vocabulary", mod_vocab)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_raw",
        type=str,
        required=True,
        help="Folder with the .pkl files + nodes_filtered.csv + flavorgraph_edges.csv + usda_mapping.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Where to write the converted data files",
    )
    args = parser.parse_args()

    DATA_RAW = Path(args.data_raw)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    # --- Required files check ---
    required = [
        "train_comments_subs.pkl",
        "val_comments_subs.pkl",
        "test_comments_subs.pkl",
        "vocab_ingrs.pkl",
        "flavorgraph_edges.csv",
        "nodes_filtered.csv",
        "usda_mapping.json",
    ]
    missing = [f for f in required if not (DATA_RAW / f).exists()]
    if missing:
        print(f"[error] Missing files in {DATA_RAW}:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    print("=" * 70)
    print("STEP 1: Load all inputs")
    print("=" * 70)

    _register_vocabulary_stub()

    with open(DATA_RAW / "vocab_ingrs.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(DATA_RAW / "train_comments_subs.pkl", "rb") as f:
        train_subs = pickle.load(f)
    with open(DATA_RAW / "val_comments_subs.pkl", "rb") as f:
        val_subs = pickle.load(f)
    with open(DATA_RAW / "test_comments_subs.pkl", "rb") as f:
        test_subs = pickle.load(f)

    nodes = pd.read_csv(DATA_RAW / "nodes_filtered.csv")

    print(f"vocab.word2idx size: {len(vocab.word2idx)}")
    print(f"subs entries:        train={len(train_subs)}  "
          f"val={len(val_subs)}  test={len(test_subs)}")
    print(f"nodes total:         {len(nodes)}")
    print(f"nodes by type:\n{nodes['node_type'].value_counts().to_string()}")
    print(f"node_id range:       [{nodes['node_id'].min()}, {nodes['node_id'].max()}]")

    ing_ids = sorted(nodes[nodes["node_type"] == "ingredient"]["node_id"].tolist())
    other_ids = sorted(nodes[nodes["node_type"] != "ingredient"]["node_id"].tolist())
    if other_ids:
        max_ing = max(ing_ids)
        min_other = min(other_ids)
        if min_other <= max_ing:
            print(f"  [warn] ingredient ids overlap with non-ingredient ids "
                  f"(max_ing={max_ing}, min_other={min_other})")
        else:
            print(f"  [ok] ingredient ids 0..{max_ing} clean; "
                  f"non-ing ids start at {min_other}")

    # =========================================================================
    print()
    print("=" * 70)
    print("STEP 2: Build name → node_id mapping")
    print("=" * 70)

    name_to_node_id = dict(zip(nodes["name"], nodes["node_id"]))
    ingredient_node_ids = set(
        nodes[nodes["node_type"] == "ingredient"]["node_id"].tolist()
    )
    print(f"Distinct names in nodes_filtered: {len(name_to_node_id)}")
    print(f"Ingredient node_ids: {len(ingredient_node_ids)}")

    def to_node_id(name):
        """Map an ingredient name (possibly synonym) to our node_id, or None."""
        if name in name_to_node_id:
            return int(name_to_node_id[name])
        if name in vocab.word2idx:
            for cand in vocab.idx2word[vocab.word2idx[name]]:
                if cand in name_to_node_id:
                    return int(name_to_node_id[cand])
        return None

    # =========================================================================
    print()
    print("=" * 70)
    print("STEP 3: Convert each split")
    print("=" * 70)

    def convert_split(subs_list, split_name):
        kept_pairs = []
        kept_recipes = {}
        n_total = len(subs_list)
        n_dropped_subs = 0
        n_dropped_recipe = 0
        unmapped_subs_names = Counter()

        for entry in subs_list:
            recipe_id_str = entry["id"]
            s_name, t_name = entry["subs"]

            s_id = to_node_id(s_name)
            t_id = to_node_id(t_name)
            if s_id is None or t_id is None:
                n_dropped_subs += 1
                if s_id is None:
                    unmapped_subs_names[s_name] += 1
                if t_id is None:
                    unmapped_subs_names[t_name] += 1
                continue
            if s_id not in ingredient_node_ids or t_id not in ingredient_node_ids:
                n_dropped_subs += 1
                continue

            ing_ids = []
            for synonym_list in entry["ingredients"]:
                primary = synonym_list[0]
                nid = to_node_id(primary)
                if nid is not None and nid in ingredient_node_ids:
                    ing_ids.append(nid)
            if len(ing_ids) == 0:
                n_dropped_recipe += 1
                continue
            if s_id not in ing_ids:
                ing_ids = [s_id] + ing_ids

            kept_recipes[recipe_id_str] = ing_ids
            kept_pairs.append({
                "source_id": s_id,
                "target_id": t_id,
                "recipe_id": recipe_id_str,
            })

        rate = 100.0 * len(kept_pairs) / max(n_total, 1)
        print(f"\n[{split_name}] kept {len(kept_pairs)}/{n_total} ({rate:.1f}%); "
              f"dropped: {n_dropped_subs} sub-unmapped, {n_dropped_recipe} no-recipe")
        if unmapped_subs_names:
            print(f"  Top 10 unmapped sub names:")
            for name, cnt in unmapped_subs_names.most_common(10):
                print(f"    {name!r}: {cnt}")
        return kept_pairs, kept_recipes

    train_pairs, train_recipes = convert_split(train_subs, "train")
    val_pairs, val_recipes = convert_split(val_subs, "val")
    test_pairs, test_recipes = convert_split(test_subs, "test")

    # =========================================================================
    print()
    print("=" * 70)
    print("STEP 4: Convert string recipe_ids → int counters")
    print("=" * 70)

    all_string_ids = set(train_recipes) | set(val_recipes) | set(test_recipes)
    string_to_int = {sid: i for i, sid in enumerate(sorted(all_string_ids))}
    print(f"Total distinct recipes across splits: {len(string_to_int)}")

    for pairs in (train_pairs, val_pairs, test_pairs):
        for row in pairs:
            row["recipe_id"] = string_to_int[row["recipe_id"]]

    recipes_int = {
        string_to_int[k]: v
        for k, v in {**train_recipes, **val_recipes, **test_recipes}.items()
    }

    # =========================================================================
    print()
    print("=" * 70)
    print("STEP 5: Save outputs")
    print("=" * 70)

    pd.DataFrame(train_pairs)[["source_id", "target_id", "recipe_id"]].to_csv(
        OUT_DIR / "pairs_train.csv", index=False)
    pd.DataFrame(val_pairs)[["source_id", "target_id", "recipe_id"]].to_csv(
        OUT_DIR / "pairs_val.csv", index=False)
    pd.DataFrame(test_pairs)[["source_id", "target_id", "recipe_id"]].to_csv(
        OUT_DIR / "pairs_test.csv", index=False)
    with open(OUT_DIR / "recipes.json", "w") as f:
        json.dump({str(k): v for k, v in recipes_int.items()}, f)

    shutil.copy(DATA_RAW / "flavorgraph_edges.csv", OUT_DIR / "flavorgraph_edges.csv")
    shutil.copy(DATA_RAW / "usda_mapping.json", OUT_DIR / "usda_mapping.json")

    ing_df = nodes[nodes["node_type"] == "ingredient"][["node_id", "name"]].copy()
    ing_df.columns = ["id", "name"]
    ing_df.to_csv(OUT_DIR / "ingredients.csv", index=False)

    # =========================================================================
    print()
    print("=" * 70)
    print("STEP 6: Compute model args + write data_meta.json")
    print("=" * 70)

    max_node_id = int(nodes["node_id"].max())
    max_ing_id = int(max(ingredient_node_ids))
    num_total_nodes = max_node_id + 1
    num_ingredients = max_ing_id + 1

    meta = {
        "num_total_nodes": num_total_nodes,
        "num_ingredients": num_ingredients,
        "n_pairs_train": len(train_pairs),
        "n_pairs_val": len(val_pairs),
        "n_pairs_test": len(test_pairs),
        "n_recipes": len(recipes_int),
        "n_ingredients_actual": len(ingredient_node_ids),
        "n_nodes_actual": len(nodes),
    }
    with open(OUT_DIR / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote to {OUT_DIR}:")
    print(f"  pairs_train.csv:        {len(train_pairs)} rows")
    print(f"  pairs_val.csv:          {len(val_pairs)} rows")
    print(f"  pairs_test.csv:         {len(test_pairs)} rows")
    print(f"  recipes.json:           {len(recipes_int)} recipes")
    print(f"  flavorgraph_edges.csv:  copied")
    print(f"  usda_mapping.json:      copied")
    print(f"  ingredients.csv:        {len(ing_df)} ingredients")
    print(f"  data_meta.json:         {meta}")

    print()
    print("Use these flags for train.py:")
    print(f"  --num_total_nodes {num_total_nodes}")
    print(f"  --num_ingredients {num_ingredients}")
    if num_ingredients < num_total_nodes:
        print(f"  (F/D nodes occupy ids {num_ingredients}..{num_total_nodes - 1}, "
              f"used for graph propagation only)")


if __name__ == "__main__":
    main()
