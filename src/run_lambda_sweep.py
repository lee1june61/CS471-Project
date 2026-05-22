"""Automated λ_h sweep + cross-metric comparison.

For each λ_h in `--lambdas`:
  1. Train the selected variant (v2 / v3 / v4) at that λ_h
     (skipped if a checkpoint already exists, unless --force_retrain).
  2. Run the three post-hoc evaluators on the resulting test_predictions:
       evaluate_health  -> sugar / sodium Δ and satisfaction rate
       evaluate_flavor  -> flavor-compound cosine (taste preservation)
       evaluate_id_ood  -> MRR split by whether (s,y) was in train
  3. Append a row with substitution + health + flavor + ID/OOD numbers.

Aggregated outputs:
  {output_dir}/sweep_summary_{variant}.csv    flat table, plot-ready
  {output_dir}/sweep_summary_{variant}.json   same rows + raw metric dicts
  {output_dir}/{variant}_lam{X}/...            per-run training artifacts

Usage:
  # full sweep on v3 (PDF Full model)
  python run_lambda_sweep.py --variant v3 --data_dir ./data --output_dir ./out

  # custom λ list, custom τ (data is source of truth -- pass with --tau_percentile)
  python run_lambda_sweep.py --variant v3 --lambdas 0 0.5 1 2 \\
      --tau_percentile 25 --data_dir ./data --output_dir ./out

  # re-evaluate only (checkpoints + predictions already on disk)
  python run_lambda_sweep.py --variant v3 --data_dir ./data \\
      --output_dir ./out --skip_train

Resolves sibling `train_{variant}.py` paths from its own location,
so it works from any working directory.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

from evaluate_flavor import evaluate_flavor
from evaluate_health import compute_health_metrics
from evaluate_id_ood import evaluate_id_ood


def lam_to_str(lam):
    """Filesystem-safe representation: 0.1 -> '0_1', 0 -> '0', 10 -> '10'."""
    return str(lam).replace(".", "_").replace("-", "neg")


def out_dir_for(base, variant, lam):
    return os.path.join(base, f"{variant}_lam{lam_to_str(lam)}")


def pred_path_for(out_dir, variant, g_label):
    return os.path.join(out_dir, f"test_predictions_{variant}_{g_label}.json")


def run_train(variant, lam, args):
    out_dir = out_dir_for(args.output_dir, variant, lam)
    ckpt = os.path.join(out_dir, f"best_{variant}.pt")
    if os.path.exists(ckpt) and not args.force_retrain:
        print(f"[sweep] λ={lam}: ckpt exists -> skip training")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"train_{variant}.py")
    cmd = [
        sys.executable, script_path,
        "--data_dir", args.data_dir,
        "--output_dir", out_dir,
        "--lambda_h", str(lam),
        "--tau_percentile", str(args.tau_percentile),
        "--test_g_overrides", args.g_label,
        "--seed", str(args.seed),
    ]
    if args.max_epochs is not None:
        cmd += ["--max_epochs", str(args.max_epochs)]
    if args.patience is not None:
        cmd += ["--patience", str(args.patience)]
    # --force_retrain means start from scratch for this lambda; otherwise
    # a stale last_<variant>.pt in out_dir (from a previously-interrupted
    # run) would trigger auto-resume with the OLD lambda's optimizer state.
    if args.force_retrain:
        cmd.append("--no_resume")
    print(f"\n[sweep] λ={lam}: train -> {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[sweep] λ={lam}: trained in {(time.time() - t0) / 60:.1f} min")
    return out_dir


def collect_metrics(variant, lam, out_dir, args):
    pred_path = pred_path_for(out_dir, variant, args.g_label)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"missing predictions: {pred_path}. "
            f"If you used --skip_train, ensure this file was produced by a "
            f"prior training run."
        )

    with open(pred_path) as f:
        pred = json.load(f)

    health = compute_health_metrics(
        pred_path,
        os.path.join(args.data_dir, "usda_mapping.json"),
        save_to=None,
    )
    flavor = evaluate_flavor(
        pred_path,
        os.path.join(args.data_dir, "flavorgraph_edges.csv"),
        save_to=None,
    )
    idood = evaluate_id_ood(
        pred_path,
        os.path.join(args.data_dir, "pairs_train.csv"),
        save_to=None,
    )

    row = {
        "variant": variant,
        "lambda_h": float(lam),
        "MRR": pred["metrics"]["MRR"],
        "Hit@1": pred["metrics"]["Hit@1"],
        "Hit@3": pred["metrics"]["Hit@3"],
        "Hit@10": pred["metrics"]["Hit@10"],
        "ID_MRR": idood["id"]["MRR"],
        "OOD_MRR": idood["ood"]["MRR"],
        "n_id": idood["n_id"],
        "n_ood": idood["n_ood"],
        "sugar_avg_delta_g": health["pred"]["sugar_g"]["avg_delta"],
        "sodium_avg_delta_mg": health["pred"]["sodium_mg"]["avg_delta"],
        "sugar_sat_pct": health["pred"]["sugar_g"]["satisfaction_rate"],
        "sodium_sat_pct": health["pred"]["sodium_mg"]["satisfaction_rate"],
        "flavor_cos_mean": flavor["pred_cosine"]["mean"],
        "flavor_cos_median": flavor["pred_cosine"]["median"],
        "n_flavor_eval": flavor["pred_cosine"]["n"],
    }
    raw = {"health": health, "flavor": flavor, "idood": idood}
    return row, raw


def print_summary_table(df, variant, g_label):
    print(f"\n=== λ_h sweep summary ({variant}, g={g_label}) ===")
    cols = ["lambda_h", "MRR", "Hit@1", "Hit@10",
            "sugar_sat_pct", "sodium_sat_pct",
            "flavor_cos_mean", "ID_MRR", "OOD_MRR"]
    print(df[cols].to_string(index=False, float_format="%.3f"))


def print_delta_from_baseline(df):
    """Show how each λ differs from the lowest-λ row (architecture-only proxy)."""
    if len(df) < 2:
        return
    base = df.iloc[0]
    print(f"\n=== Δ from λ={base['lambda_h']:g} (lowest-λ row, architecture-only proxy) ===")
    fmt = "  λ={lam:>5g}  ΔMRR={dmrr:+6.2f}pp  Δsugar_sat={dsu:+6.2f}pp  " \
          "Δsodium_sat={dso:+6.2f}pp  Δflavor_cos={dfc:+.3f}"
    for _, row in df.iloc[1:].iterrows():
        print(fmt.format(
            lam=row["lambda_h"],
            dmrr=row["MRR"] - base["MRR"],
            dsu=row["sugar_sat_pct"] - base["sugar_sat_pct"],
            dso=row["sodium_sat_pct"] - base["sodium_sat_pct"],
            dfc=row["flavor_cos_mean"] - base["flavor_cos_mean"],
        ))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["v2", "v3", "v4"], default="v3",
                   help="Which architecture to sweep (default v3 = PDF Full model)")
    p.add_argument("--lambdas", type=float, nargs="+", default=[0, 0.1, 1, 5, 10],
                   help="λ_h values to sweep. Default: 0 0.1 1 5 10 "
                        "(log-ish spacing + λ=0 = w/o L_health ablation)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./out")
    p.add_argument("--tau_percentile", type=float, default=0,
                   help="τ percentile for g derivation; default 0 = any reduction counts")
    p.add_argument("--g_label", type=str, default="auto",
                   choices=["auto", "1_0", "0_1", "1_1"],
                   help="Which test_predictions file to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override variant default (200) -- useful for quick scans")
    p.add_argument("--patience", type=int, default=None,
                   help="Override variant default (10)")
    p.add_argument("--force_retrain", action="store_true",
                   help="Retrain even if checkpoint exists")
    p.add_argument("--skip_train", action="store_true",
                   help="Skip training entirely; assume ckpts + predictions exist")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[sweep] variant={args.variant}  lambdas={args.lambdas}  "
          f"tau_p={args.tau_percentile}  g={args.g_label}")
    if args.skip_train:
        print("[sweep] --skip_train: evaluating existing predictions only")

    rows, raws, failed = [], {}, []
    for lam in args.lambdas:
        try:
            if args.skip_train:
                out_dir = out_dir_for(args.output_dir, args.variant, lam)
            else:
                out_dir = run_train(args.variant, lam, args)
            row, raw = collect_metrics(args.variant, lam, out_dir, args)
            rows.append(row)
            raws[str(lam)] = raw
        except subprocess.CalledProcessError as e:
            print(f"[sweep] λ={lam} TRAIN FAILED (exit {e.returncode})")
            failed.append((lam, f"training exit {e.returncode}"))
        except Exception as e:
            print(f"[sweep] λ={lam} EVAL FAILED: {type(e).__name__}: {e}")
            failed.append((lam, f"{type(e).__name__}: {e}"))

    if not rows:
        print("\n[sweep] no successful runs.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("lambda_h").reset_index(drop=True)

    csv_path = os.path.join(args.output_dir, f"sweep_summary_{args.variant}.csv")
    json_path = os.path.join(args.output_dir, f"sweep_summary_{args.variant}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"rows": rows, "raw": raws,
                   "config": {"variant": args.variant,
                              "lambdas": list(args.lambdas),
                              "tau_percentile": args.tau_percentile,
                              "g_label": args.g_label}},
                  f, indent=2, default=str)

    print_summary_table(df, args.variant, args.g_label)
    print_delta_from_baseline(df)

    if failed:
        print(f"\n[sweep] {len(failed)} run(s) failed:")
        for lam, msg in failed:
            print(f"  λ={lam}: {msg}")

    print(f"\n[saved] {csv_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
