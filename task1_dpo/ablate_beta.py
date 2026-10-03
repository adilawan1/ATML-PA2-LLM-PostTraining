"""Task 1 Step 2: short-run DPO forks for beta in {0.03, 0.10, 0.30} from the original initialization.

Every fork uses the same first `short_ablation_examples` training pairs, optimizer, seed, LoRA config and
shuffle order; only beta changes. Each fork is evaluated with the common protocol in task1_dpo.evaluate.
Re-running resumes: finished forks/evaluations are skipped, interrupted training resumes from checkpoint.

Writes results/task1_dpo/summary_beta.{csv,json}, including the standard one-epoch run (if evaluated) as a
separately labelled budget.
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_adapter, summary_row
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="+", help="override cfg betas (e.g. to run one fork per session)")
    ap.add_argument("--max-gen-prompts", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    results_dir = repo_path(cfg["results_dir"])
    out_root = repo_path(cfg["standard_output"]).parent

    rows = []
    for beta in betas:
        tag = beta_tag(beta)
        out = out_root / tag
        print(f"=== beta={beta}: train {n} pairs -> {out}", flush=True)
        run_training(args.config, tag, output_path=str(out), beta=beta, max_examples=n)
        clear_gpu()
        metrics_path = results_dir / f"eval_{tag}" / "metrics.json"
        if metrics_path.is_file() and not args.max_gen_prompts:
            metrics = load_json(metrics_path)
        else:
            metrics = evaluate_adapter(args.config, str(out), tag, beta=beta, max_gen_prompts=args.max_gen_prompts)
        clear_gpu()
        rows.append(summary_row(metrics, budget=f"short ({n} pairs, 1 pass)", train_examples=n))

    # Include every already-evaluated beta fork (other sessions) plus the standard run for context.
    seen = {r["name"] for r in rows}
    for beta in [float(b) for b in cfg["betas"]]:
        p = results_dir / f"eval_{beta_tag(beta)}" / "metrics.json"
        if beta_tag(beta) not in seen and p.is_file():
            rows.append(summary_row(load_json(p), budget=f"short ({n} pairs, 1 pass)", train_examples=n))
    std = results_dir / "eval_standard" / "metrics.json"
    if std.is_file():
        n_std = load_json(results_dir / "train_standard_summary.json")["n_examples"] if (results_dir / "train_standard_summary.json").is_file() else None
        rows.append(summary_row(load_json(std), budget="standard (1 epoch, full set)", train_examples=n_std))

    df = pd.DataFrame(rows).sort_values(["budget", "beta"])
    df.to_csv(results_dir / "summary_beta.csv", index=False)
    save_json(results_dir / "summary_beta.json", df.to_dict(orient="records"))
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
