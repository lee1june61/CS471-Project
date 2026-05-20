# GC-GISMo: Health-Aware Ingredient Substitution

KAIST CS471 / CS407 Team Project (Spring 2026)

Goal-conditioned extension of [GISMo](https://github.com/facebookresearch/gismo) (Pellegrini et al.) that recommends ingredient substitutes which preserve recipe flavor **and** satisfy a user-specified health goal (e.g., reduce sugar, reduce sodium). Built on top of [FlavorGraph](https://github.com/lamypark/FlavorGraph) ingredient–compound structure with added USDA nutrient information.

---

## Idea in one figure

```
Recipe context r --+
                   |     +--- v2: encoder feature injection (7 nutrients)
Source s --[GIN]---+---->|--- v3: structural injection (7 nutrition hub nodes + I-N edges)
Candidate v -------+     +--- v4: decoder concat (raw n_s, n_v)
                   |
Health goal g -----+--> Decoder MLP --> score(s, v, r, g)

Training:  L = L_substitution (GISMo InfoNCE)  +  λ · L_health
                                                    ↑
                                  expected nutrient improvement under
                                  the model's predicted candidate
                                  distribution (gradient-coupled with L_sub)
```

`g ∈ {0,1}²` — `[low_sugar, low_sodium]`. At test time you can override `g` to ask the trained model for sugar-only, sodium-only, or both-targeting recommendations from the same query.

---

## Variants compared in this repo

| Tag | Where nutrient signal enters | Goal vector `g` | Role |
|---|---|---|---|
| `v1 baseline` | nowhere | (none) | Vanilla GISMo reference |
| `v1 MVP` | only via `L_health` gradient | decoder concat | Ablation 1 — w/o nutrition inject |
| `v2` | encoder (additive `nutrient_proj` before GIN) | decoder concat | Encoder injection |
| `v3` | graph structure (7 hub nodes + I-N edges) | decoder concat | **Structural injection (PDF Full model)** |
| `v4` | decoder (raw `n_s`, `n_v` concat) | decoder concat | Decoder injection |
| `GISMo + Filter` | post-hoc only (hard / soft on score tensor) | applied to filter | Strong no-train baseline |

---

## Repository layout

```
.
├── README.md                       # this file
├── requirements.txt
├── docs/
│   └── data_format.md              # input/output schema spec
├── src/                            # all source code (run scripts from project root: `python src/X.py`)
│   ├── dataset.py                  # data loaders + HEALTH_NUTRIENT_KEYS
│   ├── models_v1.py                # WeightedGINConv, encoder, decoder, GISMo
│   ├── models_v2.py                # encoder with nutrient_proj injection
│   ├── models_v4.py                # decoder with nutrient concat
│   ├── train_v1.py                 # baseline + MVP
│   ├── train_v2.py                 # encoder injection
│   ├── train_v3.py                 # structural hub injection
│   ├── train_v4.py                 # decoder concat injection
│   ├── eval_filter_baseline.py     # post-hoc hard / soft filter
│   ├── evaluate_health.py          # Δ sugar / sodium + satisfaction rate
│   ├── evaluate_flavor.py          # I-F cosine (taste preservation)
│   ├── evaluate_id_ood.py          # MRR split by whether (s, y) in train
│   ├── run_lambda_sweep.py         # orchestrate λ sweep + collect metrics
│   ├── convert_data.py             # GISMo .pkl + graph data → our format
│   └── mock_data.py                # synthetic data for smoke testing
└── notebooks/
    ├── 01_setup_data.ipynb         # data prep (Colab)
    ├── 06_eval_results.ipynb       # Main + Ablation tables + Pareto + case study
    └── _gen_eval_notebook.py       # generator for the eval notebook
```

Outputs (training scripts auto-create subdirs under `--output_dir`):

```
out/
├── baseline/                  # train_v1.py --mode baseline
│   ├── best_baseline.pt
│   └── test_predictions_baseline.json
├── filter_baseline/           # eval_filter_baseline.py
│   ├── test_predictions_filter_hard_auto.json
│   ├── test_predictions_filter_soft_a*_auto.json
│   └── summary.json
├── v3_lam{X}/                 # per-λ artifacts from run_lambda_sweep.py
│   ├── best_v3.pt
│   └── test_predictions_v3_auto.json
├── sweep_summary_v3.csv       # plot-ready Pareto data
└── sweep_summary_v3.json      # same + raw metric dicts
```

---

## Data

- **FlavorGraph** ingredients + I-I (NPMI) + I-F (FlavorDB) edges → `flavorgraph_edges.csv`, `nodes_filtered.csv`
- **Recipe1MSubs** substitution pairs → `pairs_{train,val,test}.csv` + `recipes.json`
- **USDA** nutrient mapping (sugar, sodium, calories, fat, sat. fat, protein, carbohydrate) → `usda_mapping.json`

After running `convert_data.py`, `data/` should look like:

```
data/
├── flavorgraph_edges.csv      # src_id, dst_id, weight, edge_type ∈ {I-I, I-F, I-D}
├── nodes_filtered.csv         # node_id, name, node_type ∈ {ingredient, compound}, is_hub
├── pairs_train.csv            # source_id, target_id, recipe_id
├── pairs_val.csv
├── pairs_test.csv
├── recipes.json               # {recipe_id: [ing_id, ...]}
└── usda_mapping.json          # {ingredient_id: {sugar_g, sodium_mg, ...}}
```

Key counts (post-filter): 6,313 ingredients (sparse ids in [0, 7101]), 1,645 compounds, 8,748 total nodes, ~49k train / 10.7k val / 10.7k test pairs.

---

## Setup

```bash
pip install -r requirements.txt
# torch_geometric installation depends on your torch / CUDA version, see
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

---

## Reproducing results

### Step 0 — Smoke test (1 epoch each, ~5–10 min)

```bash
python src/train_v1.py --mode baseline --max_epochs 1 --patience 1 \
    --data_dir ./data --output_dir ./out/smoke/baseline
python src/train_v1.py --mode mvp      --max_epochs 1 --patience 1 \
    --data_dir ./data --output_dir ./out/smoke/mvp
python src/train_v2.py                  --max_epochs 1 --patience 1 \
    --data_dir ./data --output_dir ./out/smoke/v2
python src/train_v3.py                  --max_epochs 1 --patience 1 \
    --data_dir ./data --output_dir ./out/smoke/v3
python src/train_v4.py                  --max_epochs 1 --patience 1 \
    --data_dir ./data --output_dir ./out/smoke/v4
```

Each script's first log line should read `[load_node_ids] 6313 ingredients ... num_total_nodes=8748`.

### Step 1 — L_health sanity check

Confirm that `L_health` actually drives training (run 3 epochs with two extreme λ values; val MRR should clearly differ):

```bash
python src/train_v3.py --max_epochs 3 --lambda_h 0   --tau_percentile 0 \
    --data_dir ./data --output_dir ./out/lh_check_l0
python src/train_v3.py --max_epochs 3 --lambda_h 10  --tau_percentile 0 \
    --data_dir ./data --output_dir ./out/lh_check_l10
```

### Step 2 — Main training

```bash
# Vanilla GISMo (required for filter baseline)
python src/train_v1.py --mode baseline --data_dir ./data --output_dir ./out/baseline

# v3 λ sweep (PDF Full model)
python src/run_lambda_sweep.py --variant v3 --data_dir ./data --output_dir ./out \
    --tau_percentile 0

# (optional) Cross-variant sweeps for full Pareto comparison
python src/run_lambda_sweep.py --variant v4 --data_dir ./data --output_dir ./out \
    --tau_percentile 0
python src/run_lambda_sweep.py --variant v2 --data_dir ./data --output_dir ./out \
    --tau_percentile 0

# Ablation 1: w/o nutrition inject (with all g overrides for case study)
python src/train_v1.py --mode mvp --tau_percentile 0 \
    --test_g_overrides auto 1_0 0_1 1_1 \
    --data_dir ./data --output_dir ./out/v1mvp

# Ablation 3: w/o flavor compound
python src/train_v3.py --tau_percentile 0 --ablation_no_compound \
    --data_dir ./data --output_dir ./out/v3_no_compound

# (Ablation 2 = w/o L_health is automatically covered by λ=0 in the sweep)
```

### Step 3 — Filter baselines

```bash
python src/eval_filter_baseline.py \
    --checkpoint ./out/baseline/best_baseline.pt \
    --data_dir ./data --output_dir ./out/filter_baseline \
    --filter_mode both --alpha 0.5 1.0
```

### Step 4 — Result tables

1. After the sweep, look at `out/sweep_summary_v3.csv` to choose `BEST_LAMBDA`.
2. (For `g`-override case study) re-run the best-λ v3 with all four overrides:
   ```bash
   python src/train_v3.py --lambda_h {BEST} --tau_percentile 0 \
       --test_g_overrides auto 1_0 0_1 1_1 \
       --data_dir ./data --output_dir ./out/v3_lam{BEST}
   ```
   Using the same `--output_dir` resumes from the saved checkpoint, so only evaluation runs.
3. Open `notebooks/06_eval_results.ipynb`, set `BEST_LAMBDA` at the top, run all cells. Outputs:
   - `out/table1_main_results.csv` — Main results (baselines + ours)
   - `out/table2_ablation.csv` — Ablation study + Δ-from-Full view
   - `out/pareto_v3.png` — MRR ↔ health-satisfaction Pareto curves
   - g-override check + case-study samples printed inline

---

## Hyperparameters

Inherited from GISMo unless noted.

| Parameter | Value | Source |
|---|---|---|
| Embedding dim | 300 | GISMo |
| GIN layers | 2 | GISMo |
| MLP decoder layers | 3 | GISMo |
| Optimizer | Adam | GISMo |
| Learning rate | 5e-5 | GISMo |
| Weight decay | 1e-4 | GISMo |
| Dropout | 0.25 | GISMo |
| Batch size | 64 | ours |
| Negative samples (`K`) | 10 | ours |
| `λ_h` | sweep `{0, 0.1, 1, 5, 10}` | ours |
| Margin (hinge) | 0.5 | ours |
| τ (goal threshold) | percentile of positive Δ in train (default 0) | ours |
| `g_dim` | 2 (sugar, sodium) | ours |
| Hub nutrient keys (v2/v3) | 7 (calories, fat, sat-fat, carb, sugar, protein, sodium) | ours |

---

## Loss

$$\mathcal{L}_{\text{total}} = \underbrace{- \log \frac{e^{\phi(s, y, r, g)}}{\sum_{v \in \mathcal{C}} e^{\phi(s, v, r, g)}}}_{\mathcal{L}_{\text{substitution}} \text{ (GISMo InfoNCE)}} + \lambda \cdot \underbrace{\sum_k g_k \cdot \max\!\left(0,\; m - \mathbb{E}_{v \sim p_\theta}[(n_s - n_v)_k]\right)\Big/ \sum_k g_k}_{\mathcal{L}_{\text{health}}}$$

where the candidate set `C = {target, neg_1, ..., neg_K}` and the expectation is over the model's predicted softmax distribution.

---

## Evaluation metrics

| Metric | What it measures | File |
|---|---|---|
| MRR / Hit@1/3/10 | Substitution accuracy (GISMo) | inside each train script's `evaluate` |
| Δ sugar (g), Δ sodium (mg) | Avg nutrient change of predicted top-1 vs source | `evaluate_health.py` |
| Sugar / sodium satisfaction rate (%) | Fraction of predictions that strictly reduce the target nutrient | `evaluate_health.py` |
| Flavor cosine | I-F profile cosine(source, top-1) on hub-only pairs (= taste preservation) | `evaluate_flavor.py` |
| ID / OOD MRR | MRR split by whether `(s, y)` appeared in train | `evaluate_id_ood.py` |
| g-override sensitivity | Δ satisfaction rate between `g=[1,0]` and `g=[0,1]` (tests whether model uses `g`) | notebook cell 17 |

---

## References

- **GISMo**: Pellegrini, C., Özsoy, E., Wintergerst, M., & Groh, G. (2021). *Exploiting Food Embeddings for Ingredient Substitution.* HEALTHINF. [paper](https://arxiv.org/abs/2105.02927) · [code](https://github.com/facebookresearch/gismo)
- **FlavorGraph**: Park, D., Kim, K., Park, Y., Shin, J., Kang, J. (2021). *FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings.* Scientific Reports. [paper](https://www.nature.com/articles/s41598-020-79422-8)
- **Recipe1MSubs**: substitution pair dataset crawled from Recipe1M comments (released with GISMo).
- **USDA FoodData Central**: standardized nutrient information used for `usda_mapping.json`.

---

KAIST CS471 / CS407 team project (Spring 2026).
