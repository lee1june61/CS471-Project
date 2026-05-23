# Final-submission notebooks

These are the notebooks for the TA to reproduce our reported numbers. Each one
does a **single train + test at the chosen optimal hyper-parameters** (no sweep)
and prints the full metric set for that run. The exploratory lambda-sweep
notebooks live in `../notebooks/` and are kept as-is.

| # | Notebook | Model | Needs | ~Time |
|---|---|---|---|---|
| 01 | `01_baseline.ipynb` | Vanilla GISMo (no health awareness) | — | 30 min |
| 02 | `02_filter_baseline.ipynb` | GISMo + post-hoc health filter | 01's checkpoint | 10 min |
| 03 | `03_v2.ipynb` | v2 — encoder feature injection | — | 30 min |
| 04 | `04_v3.ipynb` | v3 — structural hub injection (**Full model**) | — | 30 min |
| 05 | `05_v4.ipynb` | v4 — decoder concat injection | — | 30 min |
| 06 | `06_ablations.ipynb` | Table 2: w/o nutrition inject / w/o L_health / w/o flavor compound | 04's checkpoint (Full row) | 1.5 hr |

**Run order**: 01 → 02 (02 loads 01's `best_baseline.pt`). 03/04/05 are
independent of each other and of the baselines. Run 06 after 04 so its
ablations can be compared against the Full v3 model (the Full row is skipped
gracefully if 04 hasn't run).

**Re-running is safe**: if a model's `best_*.pt` already exists, the train cell
**skips training and just re-evaluates** that checkpoint (it won't retrain from
scratch or overwrite it). Pass `--force_retrain` on the training command to
train again from scratch.

## Before you run — prepare the data

The training scripts need **seven** files in `DATA_DIR`:

```
flavorgraph_edges.csv  nodes_filtered.csv  usda_mapping.json   # graph + nutrients
pairs_train.csv  pairs_val.csv  pairs_test.csv  recipes.json   # substitution pairs + recipes
```

If you only have the three graph/nutrient files, generate the pairs + recipes
first with `src/convert_data.py` (or run `../notebooks/01_setup_data.ipynb`).
Each notebook has a data-check cell right after the Drive mount that **stops with
a clear `Missing data files…` error** if any of the seven are absent, so you
won't get a cryptic failure deep inside training.

## The only knobs

`λ_h` is **fixed at `1.0` for every trained model** (and `τ` at `0`). The
**Config** cell near the top of each notebook holds the knob(s) that notebook uses:

| Notebook | Config cell |
|---|---|
| 01 baseline | none (vanilla GISMo has no `λ`/`τ`) |
| 02 filter | `TAU_PERCENTILE` only (post-hoc filter has no `L_health`) |
| 03 / 04 / 05 / 06 | `LAMBDA_H = 1.0` **and** `TAU_PERCENTILE = 0` |

```python
LAMBDA_H       = 1.0   # health-loss weight — same for all models
TAU_PERCENTILE = 0     # goal threshold percentile (0 = any reduction)
```

Edit those lines if you want to try other values; nothing else needs touching.
(The exploratory `λ` sweep that motivated `1.0` lives in `../notebooks/`.)

## Drive layout (Colab)

```
MyDrive/CS471_project/
├── code/                 # this repo's code/ tree
├── data/                 # the dataset (all 7 files above)
└── outputs/final/        # auto-created; final-run ckpts + predictions
```

Final runs write under `outputs/final/` so they never clobber the exploratory
sweep outputs in `outputs/`. Set `PROJECT_ROOT` in the mount cell if your layout
differs, then **Runtime > Run all**.
