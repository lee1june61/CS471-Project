"""Generate 02-05 training notebooks (Colab-ready).

Run once after editing this file:
    python _gen_train_notebooks.py

Each notebook:
- Installs deps, mounts Drive (code + data both live in Drive)
- Runs a 1-epoch smoke test (skip-friendly)
- Runs the actual training command(s)
- Prints summary of artifacts

The TA only needs to:
1. Open the notebook in Colab (free T4 OK)
2. Set PROJECT_ROOT (cell 4) if your Drive layout differs
3. Runtime -> Run all
"""

import json
import uuid


def md(src):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": src if isinstance(src, list) else src.splitlines(keepends=True),
    }


def code(src):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": src if isinstance(src, list) else src.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def common_setup_cells(title, summary, est_time):
    return [
        md(
            f"# {title}\n"
            f"\n"
            f"{summary}\n"
            f"\n"
            f"**Runtime**: Colab free T4 GPU, ~{est_time}.\n"
            f"\n"
            f"**Steps**:\n"
            f"1. Runtime > Change runtime type > T4 GPU\n"
            f"2. Set `PROJECT_ROOT` (cell 4) if your Drive layout differs\n"
            f"3. Runtime > Run all"
        ),
        md("## 1. GPU + dependencies"),
        code("!nvidia-smi"),
        code(
            "# torch_geometric needs to be matched with the installed torch wheel.\n"
            "# Colab's default torch is recent enough that the generic install works.\n"
            "!pip install -q torch_geometric pandas numpy matplotlib"
        ),
        md("## 2. Mount Drive (code + data both live in Drive)"),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive')"
        ),
        code(
            "import os\n"
            "PROJECT_ROOT = '/content/drive/MyDrive/CS471_project'\n"
            "CODE_DIR     = f'{PROJECT_ROOT}/code'\n"
            "DATA_DIR     = f'{PROJECT_ROOT}/data'\n"
            "OUTPUT_DIR   = f'{PROJECT_ROOT}/outputs'\n"
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\n"
            "os.chdir(CODE_DIR)\n"
            "print(f'CWD        = {os.getcwd()}')\n"
            "print(f'DATA_DIR   = {DATA_DIR}')\n"
            "print(f'OUTPUT_DIR = {OUTPUT_DIR}')"
        ),
    ]


def smoke_test_cell(variant, mode_flag=""):
    """1-epoch smoke that confirms the script runs end-to-end.

    `--no_resume` is critical here: max_epochs=1 means a single stale
    `last_*.pt` from a previously-interrupted smoke would make every
    subsequent smoke a silent no-op (start_epoch=2 > max_epochs=1).
    """
    mode = f"--mode {mode_flag} " if mode_flag else ""
    return code(
        f"# 1-epoch smoke test (~30 sec on T4). Skip by changing False below.\n"
        f"RUN_SMOKE = True\n"
        f"if RUN_SMOKE:\n"
        f"    !python src/train_{variant}.py {mode}--max_epochs 1 --patience 1 --no_resume \\\n"
        f"      --data_dir {{DATA_DIR}} --output_dir {{OUTPUT_DIR}}/smoke_{variant}{mode_flag and ('_' + mode_flag) or ''}\n"
        f"    print('\\n[smoke] OK')"
    )


# ---------------------------------------------------------------------------
# 02 — baseline
# ---------------------------------------------------------------------------

cells = common_setup_cells(
    title="02 — Train Vanilla GISMo (baseline)",
    summary=(
        "Trains the no-health-awareness baseline (vanilla GISMo). The checkpoint "
        "is the prerequisite for the `eval_filter_baseline.py` post-hoc filter "
        "and the reference for our MRR comparison."
    ),
    est_time="30 min",
)
cells.append(md("## 3. Smoke test (optional)"))
cells.append(smoke_test_cell("v1", mode_flag="baseline"))
cells.append(md("## 4. Train vanilla GISMo"))
cells.append(code(
    "!python src/train_v1.py --mode baseline \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR}/baseline\n"
    "\n"
    "print('\\n[done] checkpoint:', f'{OUTPUT_DIR}/baseline/best_baseline.pt')"
))
cells.append(md("## 5. Quick check"))
cells.append(code(
    "import json\n"
    "with open(f'{OUTPUT_DIR}/baseline/test_predictions_baseline.json') as f:\n"
    "    pred = json.load(f)\n"
    "print('Vanilla GISMo test metrics:')\n"
    "for k, v in pred['metrics'].items():\n"
    "    print(f'  {k}: {v:.2f}')"
))


def write_nb(cells_list, filename):
    nb = {
        "cells": cells_list,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"wrote {filename} ({len(cells_list)} cells)")


write_nb(cells, "02_train_baseline.ipynb")


# ---------------------------------------------------------------------------
# 03 — v3 sweep (main contribution)
# ---------------------------------------------------------------------------

cells = common_setup_cells(
    title="03 — v3 lambda Sweep (structural hub, PDF Full model)",
    summary=(
        "Trains v3 (nutrition hub nodes + I-N edges) across "
        "lambda in {0, 0.1, 1, 5, 10}. lambda=0 is automatically the "
        "`w/o L_health` ablation. Produces `sweep_summary_v3.csv` for the "
        "Pareto curve in `06_eval_results.ipynb`."
    ),
    est_time="2.5 hr",
)
cells.append(md("## 3. Smoke test (optional)"))
cells.append(smoke_test_cell("v3"))
cells.append(md("## 4. Run lambda sweep"))
cells.append(code(
    "# Sweep produces per-lambda subdir + sweep_summary_v3.csv\n"
    "!python src/run_lambda_sweep.py --variant v3 \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR} \\\n"
    "  --tau_percentile 0 \\\n"
    "  --lambdas 0 0.1 1 5 10"
))
cells.append(md("## 5. Quick look at the sweep summary"))
cells.append(code(
    "import pandas as pd\n"
    "df = pd.read_csv(f'{OUTPUT_DIR}/sweep_summary_v3.csv')\n"
    "show = ['lambda_h', 'MRR', 'Hit@10', 'sugar_sat_pct',\n"
    "         'sodium_sat_pct', 'flavor_cos_mean']\n"
    "print(df[[c for c in show if c in df.columns]].round(2).to_string(index=False))"
))
cells.append(md(
    "## 6. (Optional) Re-evaluate best-lambda v3 with all 4 g overrides\n"
    "\n"
    "Needed for the g-override sensitivity check and case study in "
    "`06_eval_results.ipynb`. Re-uses the saved checkpoint (no retraining)."
))
cells.append(code(
    "BEST_LAMBDA = 1.0  # change after looking at the sweep table above\n"
    "best_tag = str(BEST_LAMBDA).replace('.', '_')\n"
    "best_dir = f'{OUTPUT_DIR}/v3_lam{best_tag}'\n"
    "\n"
    "!python src/train_v3.py --lambda_h {BEST_LAMBDA} --tau_percentile 0 \\\n"
    "  --test_g_overrides auto 1_0 0_1 1_1 \\\n"
    "  --data_dir {DATA_DIR} --output_dir {best_dir}\n"
    "\n"
    "print(f'\\n[done] g-override predictions saved under {best_dir}/')"
))

write_nb(cells, "03_train_v3_sweep.ipynb")


# ---------------------------------------------------------------------------
# 03b — v4 sweep
# ---------------------------------------------------------------------------

cells = common_setup_cells(
    title="03b — v4 lambda Sweep (decoder concat)",
    summary=(
        "Trains v4 (decoder-level nutrient concat) across the same lambda "
        "values as v3. Used for Pareto comparison across injection points."
    ),
    est_time="2.5 hr",
)
cells.append(md("## 3. Smoke test (optional)"))
cells.append(smoke_test_cell("v4"))
cells.append(md("## 4. Run lambda sweep"))
cells.append(code(
    "!python src/run_lambda_sweep.py --variant v4 \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR} \\\n"
    "  --tau_percentile 0 \\\n"
    "  --lambdas 0 0.1 1 5 10"
))
cells.append(md("## 5. Quick look"))
cells.append(code(
    "import pandas as pd\n"
    "df = pd.read_csv(f'{OUTPUT_DIR}/sweep_summary_v4.csv')\n"
    "show = ['lambda_h', 'MRR', 'Hit@10', 'sugar_sat_pct',\n"
    "         'sodium_sat_pct', 'flavor_cos_mean']\n"
    "print(df[[c for c in show if c in df.columns]].round(2).to_string(index=False))"
))

write_nb(cells, "03b_train_v4_sweep.ipynb")


# ---------------------------------------------------------------------------
# 03c — v2 sweep
# ---------------------------------------------------------------------------

cells = common_setup_cells(
    title="03c — v2 lambda Sweep (encoder feature injection)",
    summary=(
        "Trains v2 (encoder feature injection, 7 nutrients added to ingredient "
        "embeddings before GIN) across the lambda values used by v3 / v4."
    ),
    est_time="2.5 hr",
)
cells.append(md("## 3. Smoke test (optional)"))
cells.append(smoke_test_cell("v2"))
cells.append(md("## 4. Run lambda sweep"))
cells.append(code(
    "!python src/run_lambda_sweep.py --variant v2 \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR} \\\n"
    "  --tau_percentile 0 \\\n"
    "  --lambdas 0 0.1 1 5 10"
))
cells.append(md("## 5. Quick look"))
cells.append(code(
    "import pandas as pd\n"
    "df = pd.read_csv(f'{OUTPUT_DIR}/sweep_summary_v2.csv')\n"
    "show = ['lambda_h', 'MRR', 'Hit@10', 'sugar_sat_pct',\n"
    "         'sodium_sat_pct', 'flavor_cos_mean']\n"
    "print(df[[c for c in show if c in df.columns]].round(2).to_string(index=False))"
))

write_nb(cells, "03c_train_v2_sweep.ipynb")


# ---------------------------------------------------------------------------
# 04 — ablations (v1 MVP + v3 no_compound)
# ---------------------------------------------------------------------------

cells = common_setup_cells(
    title="04 — Ablations (w/o nutrition inject, w/o flavor compound)",
    summary=(
        "Two ablation rows for Table 2 of the report:\n"
        "- **v1 MVP**: keeps `g` + `L_health` but no architectural nutrient "
        "injection (Ablation 1, w/o nutrition edge).\n"
        "- **v3 with `--ablation_no_compound`**: structural hub kept but I-F / "
        "I-D edges dropped (Ablation 3, w/o flavor compound).\n"
        "\n"
        "Ablation 2 (w/o L_health) is automatically covered by the lambda=0 "
        "row of the v3 sweep -- nothing extra to run here."
    ),
    est_time="1 hr",
)
cells.append(md("## 3. Smoke tests"))
cells.append(smoke_test_cell("v1", mode_flag="mvp"))
cells.append(smoke_test_cell("v3"))
cells.append(md("## 4. Train v1 MVP (Ablation 1)"))
cells.append(code(
    "# All g overrides included so the case study cell in 06 works.\n"
    "!python src/train_v1.py --mode mvp --tau_percentile 0 \\\n"
    "  --test_g_overrides auto 1_0 0_1 1_1 \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR}/v1mvp"
))
cells.append(md("## 5. Train v3 with no compound edges (Ablation 3)"))
cells.append(code(
    "!python src/train_v3.py --tau_percentile 0 --ablation_no_compound \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR}/v3_no_compound"
))
cells.append(md("## 6. Quick check"))
cells.append(code(
    "import json\n"
    "for label, path in [\n"
    "    ('v1 MVP', f'{OUTPUT_DIR}/v1mvp/test_predictions_mvp_auto.json'),\n"
    "    ('v3 no_compound', f'{OUTPUT_DIR}/v3_no_compound/test_predictions_v3_auto.json'),\n"
    "]:\n"
    "    with open(path) as f:\n"
    "        m = json.load(f)['metrics']\n"
    "    print(f'{label:<20s} MRR={m[\"MRR\"]:.2f}  Hit@10={m[\"Hit@10\"]:.2f}')"
))

write_nb(cells, "04_train_ablations.ipynb")


# ---------------------------------------------------------------------------
# 05 — filter baseline (no training, post-hoc only)
# ---------------------------------------------------------------------------

cells = common_setup_cells(
    title="05 — GISMo + Filter Baseline (post-hoc, no training)",
    summary=(
        "Loads the vanilla GISMo checkpoint trained in `02_train_baseline.ipynb` "
        "and applies post-hoc health filters (hard / soft with alpha sweep). "
        "This is the strongest non-trained baseline -- our trained methods need "
        "to beat both vanilla GISMo and this filtered variant.\n"
        "\n"
        "**Prerequisite**: `02_train_baseline.ipynb` must have been run "
        "(produces `best_baseline.pt`)."
    ),
    est_time="10 min",
)
cells.append(md("## 3. Run post-hoc filters"))
cells.append(code(
    "ckpt_path = f'{OUTPUT_DIR}/baseline/best_baseline.pt'\n"
    "if not os.path.exists(ckpt_path):\n"
    "    raise FileNotFoundError(\n"
    "        f'{ckpt_path} not found. Run 02_train_baseline.ipynb first.')\n"
    "\n"
    "!python src/eval_filter_baseline.py \\\n"
    "  --checkpoint {ckpt_path} \\\n"
    "  --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR}/filter_baseline \\\n"
    "  --filter_mode both --alpha 0.5 1.0"
))
cells.append(md("## 4. Summary"))
cells.append(code(
    "import json\n"
    "with open(f'{OUTPUT_DIR}/filter_baseline/summary.json') as f:\n"
    "    summary = json.load(f)\n"
    "print(f'{\"config\":<35} {\"MRR\":>7} {\"Hit@1\":>7} {\"Hit@10\":>7}')\n"
    "print('-' * 65)\n"
    "for k in sorted(summary.keys()):\n"
    "    m = summary[k]\n"
    "    print(f'{k:<35} {m[\"MRR\"]:>7.2f} {m[\"Hit@1\"]:>7.2f} {m[\"Hit@10\"]:>7.2f}')"
))

write_nb(cells, "05_filter_baseline.ipynb")

print("\nAll 5 training notebooks generated.")
