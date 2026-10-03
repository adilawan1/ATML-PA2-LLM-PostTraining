"""Matched short PPO forks shared by the clipping (Task 2 Step 2) and KL-pressure (Step 3) studies.

Every fork starts from the identical supplied midpoint policy/critic, uses the same prompt schedule, generation
seeds, update budget (`fork_updates`) and response cap, and differs only in clip_epsilon and/or kl_beta. The
fork (eps=0.20, kl_beta=0.10) is the same configuration in both studies and is trained/evaluated once.
"""
from __future__ import annotations

import json

import numpy as np

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_adapter


def fork_name(eps: float, kl_beta: float) -> str:
    return f"fork_eps{eps:g}_kl{kl_beta:g}"


def stability_stats(log_path) -> dict:
    """Stability statistics from a fork's per-update log (definitions are part of the reported result).

    - policy_loss_std: std of the per-update policy loss across updates
    - max_approx_kl_old_new: largest per-epoch approximate KL between the rollout policy and the updated policy
    - max_abs_log_ratio: largest |log rho| on any valid token in any epoch (via ratio_min/ratio_max)
    - final_epoch_clip_fraction: mean clip fraction in the last PPO epoch of each update (epoch 0 is on-policy)
    - grad_norm_max / grad_norm_cv: largest and coefficient of variation of the per-update policy gradient norm
    - nonfinite_steps: optimizer steps skipped by the fp16 GradScaler
    """
    rows = [json.loads(l) for l in open(log_path, encoding="utf-8") if l.strip()]
    if not rows:
        return {}
    epochs = [e for r in rows for e in r["epochs"]]
    gn = np.array([r["policy_grad_norm"] for r in rows])
    return {
        "updates": len(rows),
        "policy_loss_std": float(np.std([r["policy_loss"] for r in rows])),
        "max_approx_kl_old_new": float(max(e["approx_kl_old_new"] for e in epochs)),
        "max_abs_log_ratio": float(max(max(abs(np.log(e["ratio_max"])), abs(np.log(max(e["ratio_min"], 1e-12)))) for e in epochs)),
        "final_epoch_clip_fraction": float(np.mean([r["epochs"][-1]["clip_fraction"] for r in rows])),
        "grad_norm_max": float(gn.max()),
        "grad_norm_cv": float(gn.std() / gn.mean()) if gn.mean() > 0 else float("nan"),
        "value_loss_last": float(rows[-1]["value_loss"]),
        "nonfinite_steps": int(sum(e["skipped_nonfinite"] for e in epochs)),
        "train_reward_mean": float(np.mean([r["reward_raw"] for r in rows])),
        "train_kl_last": float(rows[-1]["kl_token_mean"]),
        "generated_tokens": float(sum(r["response_length"] for r in rows)),
    }


def trajectories(log_path) -> list[dict]:
    keys = ("update", "reward_raw", "reward_effective", "kl_token_mean", "kl_seq_mean", "entropy", "response_length",
            "policy_loss", "value_loss", "clip_fraction", "policy_grad_norm", "approx_kl_old_new", "value_explained_variance")
    return [{k: r.get(k) for k in keys} for r in (json.loads(l) for l in open(log_path, encoding="utf-8") if l.strip())]


def run_fork(config_path: str, eps: float, kl_beta: float, max_eval_prompts: int | None = None) -> dict:
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl_beta)
    out = repo_path(cfg["output"]).parent / name
    print(f"=== PPO fork {name}: {cfg['fork_updates']} updates -> {out}", flush=True)
    run_ppo(config_path, output=str(out), updates=int(cfg["fork_updates"]), clip_epsilon=eps, kl_beta=kl_beta, run_name=name)
    clear_gpu()
    metrics = evaluate_adapter(config_path, str(out), name, max_prompts=max_eval_prompts)
    clear_gpu()
    log_path = repo_path(cfg["results_dir"]) / f"train_{name}.jsonl"
    return {"name": name, "clip_epsilon": eps, "kl_beta": kl_beta, "eval": metrics,
            "stability": stability_stats(log_path), "trajectory": trajectories(log_path)}


def eval_row(fork: dict) -> dict:
    m, s = fork["eval"], fork["stability"]
    return {
        "name": fork["name"], "clip_epsilon": fork["clip_epsilon"], "kl_beta": fork["kl_beta"],
        "heldout_reward": m["reward_mean"], "heldout_reward_std": m["reward_std"],
        "heldout_effective_reward": m["effective_reward_mean"],
        "heldout_kl_token_mean": m["kl_token_mean"], "heldout_kl_seq_mean": m["kl_seq_mean"],
        "heldout_entropy": m["entropy_token_mean"],
        "heldout_len_mean": m["response_length"]["mean"], "heldout_len_std": m["response_length"]["std"],
        "heldout_len_iqr": m["response_length"]["iqr"], "heldout_eos_rate": m["eos_rate"],
        **{f"stab_{k}": v for k, v in s.items()},
    }
