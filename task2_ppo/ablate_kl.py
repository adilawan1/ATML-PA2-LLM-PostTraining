"""Task 2 Step 3: reward-overoptimization / KL-pressure study.

Matched short continuations (task2_ppo.forks) from the identical supplied midpoint for kl_beta in
{0, 0.10, 0.20} at the config clip_epsilon, each evaluated with the common held-out protocol. The per-update
trajectories (reward, KL, entropy, length, losses) are saved alongside the held-out results so the report can
show which quantity moves first when KL pressure is weakened.

Writes results/task2_ppo/kl_forks.{json,csv} and kl_trajectories.csv.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task2_ppo.forks import eval_row, run_fork


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--kl-betas", type=float, nargs="+", help="override cfg kl_values (e.g. one fork per session)")
    ap.add_argument("--max-eval-prompts", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    results_dir = repo_path(cfg["results_dir"])
    eps = float(cfg["clip_epsilon"])

    forks = [run_fork(args.config, eps, float(b), args.max_eval_prompts) for b in (args.kl_betas or cfg["kl_values"])]
    path = results_dir / "kl_forks.json"
    prev = {f["name"]: f for f in (json.loads(path.read_text(encoding="utf-8")) if path.is_file() else [])}
    prev.update({f["name"]: f for f in forks})
    merged = sorted(prev.values(), key=lambda f: f["kl_beta"])
    save_json(path, merged)

    df = pd.DataFrame([eval_row(f) for f in merged])
    df.to_csv(results_dir / "kl_forks.csv", index=False)
    traj = pd.DataFrame([{"name": f["name"], "kl_beta": f["kl_beta"], **t} for f in merged for t in f["trajectory"]])
    traj.to_csv(results_dir / "kl_trajectories.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
