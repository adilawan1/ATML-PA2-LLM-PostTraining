"""Task 2 Step 2: clipping study.

Part A - supplied cached rollout batch (results/task2_ppo/clipping_cache.{json,csv}):
  1. Rebuild each cached rollout's token sequence (prompt from the RL prompt pools by prompt_id, response re-tokenized;
     rows whose re-tokenized length does not match the cached log-prob length are reported and excluded).
  2. Convention check: recompute midpoint-policy and reference log-probs at temperature 1 and at the sampling
     temperature and report the mean |difference| from the cached old/ref log-probs. The better-matching temperature
     is used for all ratios below (and reported).
  3. Advantages/returns from the cache only: effective terminal reward + sampled-token KL shaping (kl_beta from the
     config) with the cached old/ref log-probs, GAE over the cached critic values, then whitening per rollout
     (each cached rollout is one update's batch, prompts_per_update = 1).
  4. Static geometry: ratio rho = pi_midpoint / pi_old(cached) on every valid token, then for each epsilon the
     clipped surrogate, unclipped surrogate, affected-token fraction (rho outside [1-eps, 1+eps]) and the
     clip-active fraction (tokens whose gradient the min() removes: A>0 & rho>1+eps or A<0 & rho<1-eps).
  5. Dynamic geometry: for each epsilon, restart from the midpoint and run ppo_epochs passes of policy-only PPO
     steps over the 32 cached rollouts (fixed seeded order, one rollout per step, ratios always against the cached
     old log-probs). Record the same quantities before every step, and finally evaluate the resulting policy's
     ratios on the whole batch under all three epsilons.
Part B - matched short continuation forks (task2_ppo.forks) for each epsilon at the config kl_beta, evaluated with
  the common held-out protocol, with stability statistics (results/task2_ppo/clipping_forks.{json,csv}).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.checkpointing import load_trainable_state_dict, trainable_state_dict
from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json, set_seed
from common.models import clear_gpu, load_policy, load_tokenizer, reference_mode, resolve_dtype, trainable_parameters
from common.rl_utils import disable_dropout, response_logprobs
from task2_ppo.forks import eval_row, run_fork
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def build_cached_batch(cfg, tokenizer, rows, device):
    pool = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            pool.setdefault(r["prompt_id"], r)
    items, excluded = [], []
    for idx, row in enumerate(rows):
        prow = pool.get(row.get("prompt_id"))
        if prow is None:
            excluded.append({"row": idx, "source_index": row["source_index"], "reason": "prompt_id not in RL prompt pools"})
            continue
        prompt_ids = tokenizer.apply_chat_template(prompt_messages(prow), tokenize=True, add_generation_prompt=True)
        prompt_ids = prompt_ids[-int(cfg["max_prompt_length"]):]
        resp_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        if bool(row.get("terminated_with_eos")):
            resp_ids = resp_ids + [tokenizer.eos_token_id]
        n = int(row["old_logprobs"].numel())
        if len(resp_ids) != n:
            excluded.append({"row": idx, "source_index": row["source_index"], "reason": f"re-tokenized length {len(resp_ids)} != cached {n}"})
            continue
        seq = torch.tensor([prompt_ids + resp_ids], device=device)
        items.append({
            "row": idx, "source_index": row["source_index"], "prompt_id": row["prompt_id"],
            "sequences": seq, "attention_mask": torch.ones_like(seq), "prompt_width": len(prompt_ids),
            "response_ids": torch.tensor([resp_ids], device=device),
            "mask": torch.ones(1, n, device=device),
            "old": row["old_logprobs"].float().view(1, -1).to(device),
            "ref": row["ref_logprobs"].float().view(1, -1).to(device),
            "values": row["values"].float().view(1, -1).to(device),
            "reward": float(row.get("effective_terminal_reward", row.get("raw_terminal_reward"))),
        })
    return items, excluded


def geometry(new_logp_list, items, eps):
    """Pooled token-level clip statistics for one epsilon (valid response tokens only)."""
    ratios = torch.cat([torch.exp(lp - it["old"]).flatten() for lp, it in zip(new_logp_list, items)])
    adv = torch.cat([it["adv"].flatten() for it in items])
    surr1 = ratios * adv
    surr2 = ratios.clamp(1 - eps, 1 + eps) * adv
    affected = (ratios < 1 - eps) | (ratios > 1 + eps)
    active = ((adv > 0) & (ratios > 1 + eps)) | ((adv < 0) & (ratios < 1 - eps))
    return {
        "epsilon": eps,
        "n_tokens": int(ratios.numel()),
        "affected_token_fraction": float(affected.float().mean()),
        "clip_active_fraction": float(active.float().mean()),
        "clipped_surrogate": float(torch.minimum(surr1, surr2).mean()),
        "unclipped_surrogate": float(surr1.mean()),
        "ratio_mean": float(ratios.mean()),
        "ratio_std": float(ratios.std(unbiased=False)),
        "mean_abs_log_ratio": float(torch.log(ratios).abs().mean()),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
    }


@torch.no_grad()
def batch_logprobs(model, items, temp, reference=False):
    out = []
    for it in items:
        args = (it["sequences"], it["attention_mask"], it["prompt_width"], it["response_ids"], temp)
        if reference:
            with reference_mode(model):
                lp, _ = response_logprobs(model, *args)
        else:
            lp, _ = response_logprobs(model, *args)
        out.append(lp.float())
    return out


def cache_study(cfg, epsilons):
    seed = int(cfg["seed"])
    set_seed(seed)
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    disable_dropout(policy)
    device = next(policy.parameters()).device
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    items, excluded = build_cached_batch(cfg, tokenizer, rows, device)
    print(f"[cache] {len(items)}/{len(rows)} cached rollouts rebuilt; excluded: {excluded}", flush=True)

    # Convention check.
    policy.eval()
    conv = {}
    for temp in sorted({1.0, float(cfg["generation"]["temperature"])}):
        pol = batch_logprobs(policy, items, temp)
        ref = batch_logprobs(policy, items, temp, reference=True)
        conv[str(temp)] = {
            "mean_abs_diff_old": float(np.mean([(a - it["old"]).abs().mean().item() for a, it in zip(pol, items)])),
            "mean_abs_diff_ref": float(np.mean([(a - it["ref"]).abs().mean().item() for a, it in zip(ref, items)])),
        }
    temp = float(min(conv, key=lambda t: conv[t]["mean_abs_diff_old"]))
    print(f"[cache] log-prob convention check: {conv} -> using temperature {temp}", flush=True)

    beta_kl = float(cfg["kl_beta"])
    for it in items:
        rewards = shaped_rewards(torch.tensor([it["reward"]], device=device), it["old"], it["ref"], it["mask"], beta_kl)
        adv, ret = compute_gae(rewards, it["values"], it["mask"], float(cfg["gamma"]), float(cfg["gae_lambda"]))
        it["adv"] = normalize_advantages(adv, it["mask"])
        it["returns"] = ret

    mid_logp = batch_logprobs(policy, items, temp)
    static = [geometry(mid_logp, items, eps) for eps in epsilons]

    init_state = trainable_state_dict(policy)
    params = trainable_parameters(policy)
    use_scaler = device.type == "cuda" and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    dynamic_steps, final_cross = [], []
    for eps in epsilons:
        load_trainable_state_dict(policy, init_state)
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        order = torch.randperm(len(items), generator=torch.Generator().manual_seed(seed)).tolist()
        policy.train()
        step = 0
        for epoch in range(int(cfg["ppo_epochs"])):
            for i in order:
                it = items[i]
                new_logp, _ = response_logprobs(policy, it["sequences"], it["attention_mask"], it["prompt_width"], it["response_ids"], temp)
                loss, ratio, clip_frac = ppo_policy_loss(new_logp, it["old"], it["adv"], it["mask"], eps)
                g = geometry([new_logp.detach().float()], [it], eps)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                dynamic_steps.append({"train_epsilon": eps, "step": step, "epoch": epoch, "row": it["row"],
                                      "loss": float(loss.detach()), "clip_fraction_helper": float(clip_frac),
                                      "grad_norm": gn, **{k: v for k, v in g.items() if k != "epsilon"}})
        policy.eval()
        final_logp = batch_logprobs(policy, items, temp)
        for eval_eps in epsilons:
            final_cross.append({"train_epsilon": eps, **geometry(final_logp, items, eval_eps)})
        print(f"[cache] eps={eps}: final affected fraction at own eps = "
              f"{[r for r in final_cross if r['train_epsilon'] == eps and r['epsilon'] == eps][0]['affected_token_fraction']:.4f}", flush=True)

    del policy
    clear_gpu()
    dyn = pd.DataFrame(dynamic_steps)
    dyn_summary = dyn.groupby("train_epsilon").agg(
        affected_token_fraction_mean=("affected_token_fraction", "mean"),
        clip_active_fraction_mean=("clip_active_fraction", "mean"),
        clipped_surrogate_mean=("clipped_surrogate", "mean"),
        unclipped_surrogate_mean=("unclipped_surrogate", "mean"),
        mean_abs_log_ratio_last=("mean_abs_log_ratio", "last"),
        grad_norm_mean=("grad_norm", "mean"),
    ).reset_index().to_dict(orient="records")
    return {
        "n_rollouts_used": len(items), "excluded_rows": excluded, "convention_check": conv, "logprob_temperature_used": temp,
        "kl_beta_for_shaping": beta_kl, "advantage_whitening": "per rollout",
        "static_midpoint_vs_cached_old": static, "dynamic_summary": dyn_summary,
        "dynamic_final_cross": final_cross, "dynamic_steps": dynamic_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--epsilons", type=float, nargs="+", help="override cfg clip_values for the forks (e.g. one per session)")
    ap.add_argument("--cache-only", action="store_true", help="run Part A only")
    ap.add_argument("--forks-only", action="store_true", help="run Part B only")
    ap.add_argument("--max-eval-prompts", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    all_eps = [float(e) for e in cfg["clip_values"]]

    if not args.forks_only:
        res = cache_study(cfg, all_eps)
        save_json(results_dir / "clipping_cache.json", res)
        pd.DataFrame(res["static_midpoint_vs_cached_old"]).to_csv(results_dir / "clipping_cache_static.csv", index=False)
        pd.DataFrame(res["dynamic_final_cross"]).to_csv(results_dir / "clipping_cache_dynamic_final.csv", index=False)
        pd.DataFrame(res["dynamic_steps"]).to_csv(results_dir / "clipping_cache_dynamic_steps.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(pd.DataFrame(res["static_midpoint_vs_cached_old"]).to_string(index=False))
            print(pd.DataFrame(res["dynamic_summary"]).to_string(index=False))

    if not args.cache_only:
        forks = [run_fork(args.config, eps, float(cfg["kl_beta"]), args.max_eval_prompts) for eps in (args.epsilons or all_eps)]
        existing = results_dir / "clipping_forks.json"
        prev = {f["name"]: f for f in (json.loads(existing.read_text(encoding="utf-8")) if existing.is_file() else [])}
        prev.update({f["name"]: f for f in forks})
        merged = sorted(prev.values(), key=lambda f: f["clip_epsilon"])
        save_json(existing, merged)
        df = pd.DataFrame([eval_row(f) for f in merged])
        df.to_csv(results_dir / "clipping_forks.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
