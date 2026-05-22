# GC-GISMo: Health-Aware Ingredient Substitution

> **KAIST CS471 Team Project — Spring 2026**

Goal-conditioned extension of [GISMo](https://github.com/facebookresearch/gismo) (Pellegrini et al.) that recommends ingredient substitutes which preserve recipe flavor **and** satisfy a user-specified health goal (e.g., reduce sugar, reduce sodium). Built on top of the [FlavorGraph](https://github.com/lamypark/FlavorGraph) ingredient–compound structure with added USDA nutrient information.

**TL;DR** — open `notebooks/03_train_v3_sweep.ipynb` on Colab, point `PROJECT_ROOT` at your Drive, run all. That reproduces the main result.

## Contents

- [Idea](#idea-in-one-figure)
- [Variants](#variants-compared-in-this-repo)
- [Repository layout](#repository-layout)
- [Data](#data)
- [Setup](#setup)
- [Reproducing on Colab](#reproducing-on-colab-recommended) ← recommended path
- [Reproducing locally (CLI)](#reproducing-locally-cli)
- [Hyperparameters](#hyperparameters)
- [Loss](#loss)
- [Evaluation metrics](#evaluation-metrics)
- [References](#references)

---

## Idea in one figure

```
Recipe context r ─┐
                  │     ┌── v2: encoder feature injection (7 nutrients)
Source s ──[GIN]──┼────►│── v3: structural injection (7 nutrition hubs + I–N edges)  ◄ Full model
Candidate v ──────┤     └── v4: decoder concat (raw n_s, n_v)
                  │
Health goal g ────┴──► Decoder MLP ──► score(s, v, r, g)

Training:  L = L_substitution (GISMo InfoNCE)  +  λ · L_health
                                                   ▲
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
└── notebooks/                      # Colab-ready, one notebook per training task
    ├── 01_setup_data.ipynb         # data prep (only if you have raw GISMo files)
    ├── 02_train_baseline.ipynb     # vanilla GISMo  (~30 min on T4)
    ├── 03_train_v3_sweep.ipynb     # v3 lambda sweep -- main contribution  (~2.5 hr)
    ├── 03b_train_v4_sweep.ipynb    # v4 lambda sweep -- decoder injection  (~2.5 hr)
    ├── 03c_train_v2_sweep.ipynb    # v2 lambda sweep -- encoder injection  (~2.5 hr)
    ├── 04_train_ablations.ipynb    # v1 MVP + v3 no-compound  (~1 hr)
    ├── 05_filter_baseline.ipynb    # post-hoc filter on baseline ckpt  (~10 min)
    ├── 06_eval_results.ipynb       # final Tables 1 / 2 + Pareto + case study
    ├── _gen_train_notebooks.py     # generator for 02-05
    └── _gen_eval_notebook.py       # generator for 06
```

<details>
<summary><b>Output directory layout</b> (training scripts auto-create subdirs under <code>--output_dir</code>)</summary>

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

</details>

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

## Reproducing on Colab (recommended)

Each training task is one notebook so it fits a single free-T4 session (~12 h limit). Open them on Colab in order:

| # | Notebook | What it does | ~Time |
|---|---|---|---|
| 02 | `02_train_baseline.ipynb` | Vanilla GISMo. Required for the filter baseline (05) and as the MRR reference. | 30 min |
| 03 | `03_train_v3_sweep.ipynb` | v3 lambda sweep `{0, 0.1, 1, 5, 10}` -- the main result. | 2.5 hr |
| 03b | `03b_train_v4_sweep.ipynb` | v4 lambda sweep (decoder concat) -- Pareto comparison. | 2.5 hr |
| 03c | `03c_train_v2_sweep.ipynb` | v2 lambda sweep (encoder injection) -- Pareto comparison. | 2.5 hr |
| 04 | `04_train_ablations.ipynb` | v1 MVP (w/o nutrition inject) + v3 `--ablation_no_compound`. | 1 hr |
| 05 | `05_filter_baseline.ipynb` | GISMo + post-hoc hard / soft filter (uses ckpt from 02). | 10 min |
| 06 | `06_eval_results.ipynb` | Builds Table 1 (main), Table 2 (ablation), Pareto plots, g-override check, case study. | 5 min |

Each notebook expects your Drive layout to be:

```
MyDrive/CS471_project/
├── code/                # this repo's `code/` tree, uploaded to Drive
├── data/                # the dataset (see "Data" section above)
└── outputs/             # auto-created, persists ckpts and predictions
```

Edit `PROJECT_ROOT` in cell 4 if your layout differs. The notebooks `os.chdir` into `code/` on Drive (no git clone), so upload this `code/` directory to your Drive before running.

GPU: free T4 is fine. CPU-only also works but each sweep takes ~10x longer.

---

## Reproducing locally (CLI)

### Step 0 — Smoke test (1 epoch each, ~5–10 min)

`--no_resume` is important here: with `--max_epochs 1`, any stale
`last_*.pt` from a previously-interrupted smoke would make subsequent
smokes a silent no-op (start_epoch=2 > max_epochs=1).

```bash
python src/train_v1.py --mode baseline --max_epochs 1 --patience 1 --no_resume \
    --data_dir ./data --output_dir ./out/smoke/baseline
python src/train_v1.py --mode mvp      --max_epochs 1 --patience 1 --no_resume \
    --data_dir ./data --output_dir ./out/smoke/mvp
python src/train_v2.py                  --max_epochs 1 --patience 1 --no_resume \
    --data_dir ./data --output_dir ./out/smoke/v2
python src/train_v3.py                  --max_epochs 1 --patience 1 --no_resume \
    --data_dir ./data --output_dir ./out/smoke/v3
python src/train_v4.py                  --max_epochs 1 --patience 1 --no_resume \
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

<details>
<summary>Full table — inherited from GISMo unless marked <i>ours</i></summary>

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

</details>

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

KAIST CS471 team project — Spring 2026.
