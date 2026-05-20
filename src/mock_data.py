"""Mock data generator (for pipeline smoke tests without real data).

Output structure mirrors the real spec:
  ingredient ids:        0 .. N_ing - 1
  flavor compound ids:   N_ing .. N_ing + N_flav - 1

Files written:
  flavorgraph_edges.csv  (I-I + I-F edges with edge_type column)
  nodes_filtered.csv     (node_id, name, id, node_type, is_hub columns)
  recipes.json
  pairs_{train,val,test}.csv
  usda_mapping.json
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_ingredients", type=int, default=200)
    parser.add_argument("--num_flavor_compounds", type=int, default=50)
    parser.add_argument("--num_recipes", type=int, default=500)
    parser.add_argument("--num_pairs_train", type=int, default=2000)
    parser.add_argument("--num_pairs_val", type=int, default=300)
    parser.add_argument("--num_pairs_test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    N_ing = args.num_ingredients
    N_flav = args.num_flavor_compounds
    N_total = N_ing + N_flav

    # I-I edges
    n_ii = N_ing * 3
    src_ii = np.random.randint(0, N_ing, size=n_ii)
    dst_ii = np.random.randint(0, N_ing, size=n_ii)
    w_ii = np.random.uniform(0.25, 1.0, size=n_ii).astype(np.float32)
    keep = src_ii != dst_ii
    src_ii, dst_ii, w_ii = src_ii[keep], dst_ii[keep], w_ii[keep]

    ii_rows = list(zip(src_ii.tolist(), dst_ii.tolist(),
                       w_ii.tolist(), ["I-I"] * len(src_ii)))

    # I-F edges (each ingredient gets 2-5 flavor compounds)
    if_rows = []
    for ing in range(N_ing):
        n_compounds = random.randint(2, 5)
        compounds = random.sample(range(N_ing, N_total), min(n_compounds, N_flav))
        for cmp in compounds:
            if_rows.append((ing, cmp, 1.0, "I-F"))

    all_rows = ii_rows + if_rows
    pd.DataFrame(all_rows, columns=["src_id", "dst_id", "weight", "edge_type"]
                ).to_csv(os.path.join(args.out_dir, "flavorgraph_edges.csv"), index=False)

    # Recipes (ingredients only)
    recipes = {}
    for r in range(args.num_recipes):
        size = random.randint(3, 10)
        ings = random.sample(range(N_ing), size)
        recipes[r] = ings
    with open(os.path.join(args.out_dir, "recipes.json"), "w") as f:
        json.dump({str(k): v for k, v in recipes.items()}, f)

    def make_pairs(n_pairs):
        rows = []
        for _ in range(n_pairs):
            r = random.randint(0, args.num_recipes - 1)
            ings = recipes[r]
            if len(ings) < 1:
                continue
            s = random.choice(ings)
            y = random.randint(0, N_ing - 1)
            while y == s:
                y = random.randint(0, N_ing - 1)
            rows.append({"source_id": s, "target_id": y, "recipe_id": r})
        return pd.DataFrame(rows)

    make_pairs(args.num_pairs_train).to_csv(
        os.path.join(args.out_dir, "pairs_train.csv"), index=False)
    make_pairs(args.num_pairs_val).to_csv(
        os.path.join(args.out_dir, "pairs_val.csv"), index=False)
    make_pairs(args.num_pairs_test).to_csv(
        os.path.join(args.out_dir, "pairs_test.csv"), index=False)

    # USDA (ingredients only — F/D nodes don't have nutrient values)
    usda = {}
    for ing in range(N_ing):
        if random.random() < 0.1:
            continue
        usda[str(ing)] = {
            "sugar_g": float(np.clip(np.random.exponential(5.0), 0.0, 80.0)),
            "sodium_mg": float(np.clip(np.random.exponential(200.0), 0.0, 4000.0)),
            "calories_kcal": float(np.clip(np.random.normal(150, 100), 0.0, 900.0)),
            "protein_g": float(np.clip(np.random.normal(8, 5), 0.0, 80.0)),
        }
    with open(os.path.join(args.out_dir, "usda_mapping.json"), "w") as f:
        json.dump(usda, f)

    # nodes_filtered.csv: all graph nodes with type tag. Required by
    # dataset.load_node_ids — drives both num_total_nodes and the
    # candidate-pool (ingredient-only) tensor.
    node_rows = [
        {"node_id": i, "name": f"mock_ing_{i:04d}", "id": "",
         "node_type": "ingredient", "is_hub": "hub"}
        for i in range(N_ing)
    ] + [
        {"node_id": i, "name": f"mock_flavor_{i - N_ing:04d}", "id": "",
         "node_type": "compound", "is_hub": ""}
        for i in range(N_ing, N_total)
    ]
    pd.DataFrame(node_rows).to_csv(
        os.path.join(args.out_dir, "nodes_filtered.csv"), index=False)

    print(f"[mock_data] Wrote to {args.out_dir}")
    print(f"  ingredients:        {N_ing}")
    print(f"  flavor compounds:   {N_flav}")
    print(f"  total graph nodes:  {N_total}")
    print(f"  I-I edges:          {len(ii_rows)}")
    print(f"  I-F edges:          {len(if_rows)}")
    print(f"  recipes:            {args.num_recipes}")
    print(f"  pairs (tr/va/te):   {args.num_pairs_train} / {args.num_pairs_val} / {args.num_pairs_test}")
    print(f"  USDA mapped:        {len(usda)} / {N_ing}")


if __name__ == "__main__":
    main()
