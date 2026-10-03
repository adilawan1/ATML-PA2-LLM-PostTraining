"""Task 3 Step 3: canonical GRPO (1/T_k) vs Dr. GRPO-style (1/max_completion_length) normalization.

Two matched short continuations (`fork_updates`) from the identical supplied midpoint: same prompt schedule,
generation seeds, K, reward, beta, epsilon, generation cap and update budget; only `loss_type` differs. Each is
evaluated with the common held-out protocol.

Length-conditioned gradient statistics (from the per-completion training records):
  - completions are binned once into length tertiles (short / medium / long) using the pooled response lengths of
    both forks, counting only completions that receive gradient (non-zero advantage, not masked);
  - per bin: mean per-completion policy-gradient norm (measured by autograd on that completion's exact additive
    term of the loss), mean |advantage|, and the analytic per-completion weight on its summed score-function
    gradient (|A| for grpo since each of T tokens gets |A|/T; |A| T / L_max for dr_grpo);
  - long/short ratio of mean gradient norm and the Pearson correlation of gradient norm with length.
Writes results/task3_grpo/normalization_{summary.json,forks.csv,length_bins.csv,trajectories.csv}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from common.metrics import safe_corr
from common.models import clear_gpu
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_adapter

LOSS_TYPES = ("grpo", "dr_grpo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--loss-types", nargs="+", choices=LOSS_TYPES, help="run a subset (e.g. one fork per session)")
    ap.add_argument("--max-eval-prompts", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    results_dir = repo_path(cfg["results_dir"])
    max_len = int(cfg["max_completion_length"])

    for lt in args.loss_types or LOSS_TYPES:
        name = f"fork_{lt}"
        out = repo_path(cfg["output"]).parent / name
        print(f"=== GRPO fork {name}: {cfg['fork_updates']} updates -> {out}", flush=True)
        run_grpo(args.config, output=str(out), updates=int(cfg["fork_updates"]), loss_type=lt, run_name=name)
        clear_gpu()
        evaluate_adapter(args.config, str(out), name, max_prompts=args.max_eval_prompts)
        clear_gpu()

    done = [lt for lt in LOSS_TYPES if (results_dir / f"eval_fork_{lt}" / "metrics.json").is_file()]
    if len(done) < len(LOSS_TYPES):
        print(f"Only {done} finished; run the remaining fork to produce the comparison.", flush=True)
        return

    comps, trajs, rows = [], [], []
    for lt in LOSS_TYPES:
        name = f"fork_{lt}"
        for r in read_jsonl(results_dir / f"completions_{name}.jsonl"):
            comps.append({**r, "loss_type": lt})
        log = read_jsonl(results_dir / f"train_{name}.jsonl")
        trajs += [{"loss_type": lt, **{k: r[k] for k in ("update", "reward_mean", "kl_token_mean", "entropy", "response_length",
                                                          "grad_norm", "group_reward_std_mean", "uninformative_group_fraction")}} for r in log]
        m = json.loads((results_dir / f"eval_{name}" / "metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "loss_type": lt, "heldout_reward": m["reward_mean"], "heldout_reward_std": m["reward_std"],
            "heldout_kl_token_mean": m["kl_token_mean"], "heldout_kl_seq_mean": m["kl_seq_mean"],
            "heldout_entropy": m["entropy_token_mean"], "heldout_len_mean": m["response_length"]["mean"],
            "heldout_len_std": m["response_length"]["std"], "heldout_len_iqr": m["response_length"]["iqr"],
            "heldout_truncation_rate": m["truncation_rate"],
            "train_generated_tokens": float(sum(c["response_tokens"] for c in comps if c["loss_type"] == lt)),
            "train_len_first_update": float(np.mean([c["response_tokens"] for c in comps if c["loss_type"] == lt and c["update"] == 1])),
            "train_len_last_update": float(np.mean([c["response_tokens"] for c in comps if c["loss_type"] == lt and c["update"] == len(log)])),
        })

    df = pd.DataFrame(comps)
    active = df[(df["loss_tokens"] > 0) & (df["advantage"] != 0) & df["policy_grad_norm"].notna()].copy()
    if active.empty:
        raise SystemExit("No gradient-receiving completions (all masked or zero-advantage); nothing to compare.")
    t1, t2 = np.quantile(active["response_tokens"], [1 / 3, 2 / 3])
    active["length_bin"] = np.where(active["response_tokens"] <= t1, "short", np.where(active["response_tokens"] <= t2, "medium", "long"))
    active["analytic_weight"] = np.where(active["loss_type"] == "grpo", active["advantage"].abs(),
                                         active["advantage"].abs() * active["loss_tokens"] / max_len)
    bins = (active.groupby(["loss_type", "length_bin"])
            .agg(n=("policy_grad_norm", "size"), grad_norm_mean=("policy_grad_norm", "mean"),
                 abs_adv_mean=("advantage", lambda a: float(np.abs(a).mean())), analytic_weight_mean=("analytic_weight", "mean"),
                 length_mean=("response_tokens", "mean"))
            .reset_index())
    stats = {}
    for lt in LOSS_TYPES:
        sub = active[active["loss_type"] == lt]
        b = bins[bins["loss_type"] == lt].set_index("length_bin")
        stats[lt] = {
            "n_active_completions": int(len(sub)),
            "grad_norm_long_over_short": float(b.loc["long", "grad_norm_mean"] / b.loc["short", "grad_norm_mean"])
            if {"long", "short"} <= set(b.index) and b.loc["short", "grad_norm_mean"] > 0 else None,
            "corr_grad_norm_vs_length": safe_corr(sub["response_tokens"], sub["policy_grad_norm"]),
            "corr_analytic_weight_vs_length": safe_corr(sub["response_tokens"], sub["analytic_weight"]),
        }

    summary = {
        "length_bin_rule": "tertiles of response_tokens over pooled gradient-receiving completions of both forks",
        "length_bin_thresholds": [float(t1), float(t2)],
        "max_completion_length": max_len,
        "forks": rows, "length_stats": stats, "length_bins": bins.to_dict(orient="records"),
    }
    save_json(results_dir / "normalization_summary.json", summary)
    pd.DataFrame(rows).to_csv(results_dir / "normalization_forks.csv", index=False)
    bins.to_csv(results_dir / "normalization_length_bins.csv", index=False)
    pd.DataFrame(trajs).to_csv(results_dir / "normalization_trajectories.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(pd.DataFrame(rows).to_string(index=False))
        print(bins.to_string(index=False))
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
