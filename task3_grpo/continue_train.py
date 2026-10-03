"""Task 3: GRPO continuation from the supplied midpoint adapter.

One update = `prompts_per_update` prompts from a fixed seeded schedule (shared by every fork), K =
`num_generations` sampled completions each (max_completion_length), scored by the frozen reward model.
Completions that hit the cap without EOS are masked from the loss (mask_truncated_completions) but keep their
reward in the group statistics. Advantages are group-relative (task3_grpo.grpo.group_relative_advantages) and
the clipped objective + beta * KL(k3) to the frozen reference is applied with the selected normalization
(`loss_type` grpo = 1/T_k per completion, dr_grpo = 1/max_completion_length).

Logged per update (results/task3_grpo/train_<run>.jsonl): reward, KL (course helper token mean of
log pi - log pi_ref, and the k3 term inside the loss), within-group reward std, uninformative-group fraction,
policy loss, gradient norm, entropy, clip fraction, response length, truncation; plus per-completion records
(length, advantage, per-completion policy-gradient norm) for the length-normalization analysis.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.optim import AdamW

from common.checkpointing import (
    COMPLETE_MARKER,
    is_complete,
    mark_complete,
    resume_if_available,
    save_checkpoint,
    truncate_jsonl_after,
)
from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, resolve_dtype, trainable_parameters
from common.rl_utils import (
    chunked_logprobs,
    disable_dropout,
    generate,
    logprob_temperature,
    prompt_schedule,
    response_logprobs,
    reward_scores,
)
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

ZERO_STD_TOL = 1e-6  # same tolerance as the eps in group_relative_advantages


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    disable_dropout(policy)
    policy.generation_config.use_cache = True
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def per_completion_policy_terms(new_logp, old_logp, seq_adv, token_mask, eps, loss_type, max_len):
    """Each completion's additive contribution to the policy term of grpo_policy_loss (sums to it exactly)."""
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    obj = torch.minimum(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv)
    token_sum = (obj * token_mask).sum(-1)
    denom = token_mask.sum(-1).clamp_min(1.0) if loss_type == "grpo" else torch.full_like(token_sum, float(max_len))
    return -(token_sum / denom) / token_sum.shape[0]


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard"):
    base_cfg = load_yaml(config_path)
    out = repo_path(output or base_cfg["output"])
    if is_complete(out):
        print(f"[grpo] {run_name}: already complete at {out}; skipping.", flush=True)
        return load_json(out / COMPLETE_MARKER)

    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, tokenizer, opt = bundle["policy"], bundle["tokenizer"], bundle["optimizer"]
    reward = (bundle["reward_model"], bundle["reward_tokenizer"])
    device = next(policy.parameters()).device
    n_updates, k = int(cfg["updates"]), int(cfg["num_generations"])
    per_update, epochs = int(cfg["prompts_per_update"]), int(cfg["policy_epochs"])
    eps, beta, max_len = float(cfg["clip_epsilon"]), float(cfg["kl_beta"]), int(cfg["max_completion_length"])
    temp = logprob_temperature(cfg)
    seed = int(cfg["seed"])
    grad_diag = bool(cfg.get("grad_diagnostics", True))
    schedule = prompt_schedule(bundle["prompt_rows"], seed, n_updates, per_update)
    params = trainable_parameters(policy)

    use_scaler = device.type == "cuda" and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    results_dir = repo_path(cfg["results_dir"])
    run_config = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "prompts_per_update": per_update,
        "num_generations": k, "policy_epochs": epochs, "learning_rate": float(cfg["learning_rate"]),
        "clip_epsilon": eps, "kl_beta": beta, "max_prompt_length": int(cfg["max_prompt_length"]),
        "max_completion_length": max_len, "mask_truncated_completions": bool(cfg["mask_truncated_completions"]),
        "max_grad_norm": float(cfg["max_grad_norm"]), "generation": cfg["generation"], "logprob_temperature": temp,
        "seed": seed, "dropout_disabled": True, "grad_scaler": use_scaler, "zero_std_tolerance": ZERO_STD_TOL,
        "policy_init": cfg["paths"]["grpo_midpoint_policy"], "reward_model": cfg["reward_model"],
        "reference": "base model (policy adapter disabled)", "grad_diagnostics": grad_diag,
        "prompt_schedule": [[r.get("prompt_id") for r in batch] for batch in schedule],
    }
    save_json(results_dir / f"train_{run_name}_config.json", run_config)
    log_path = results_dir / f"train_{run_name}.jsonl"
    comp_path = results_dir / f"completions_{run_name}.jsonl"

    models, optimizers = {"policy": policy}, {"optimizer": opt, "scaler": scaler}
    state = resume_if_available(out, models, optimizers)
    start = state["step"] if state else 0
    elapsed_before = float(state["extra"].get("elapsed_s", 0.0)) if state else 0.0
    peak_before = int(state["extra"].get("peak_vram_bytes", 0)) if state else 0
    for p in (log_path, comp_path):
        if state:
            truncate_jsonl_after(p, start, key="update")
        elif p.exists():
            p.unlink()

    print(f"[grpo] {run_name}: {n_updates} updates, K={k}, loss_type={loss_type}, starting at update {start}", flush=True)
    timer = wall_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for u in range(start, n_updates):
        batch = schedule[u]
        msgs = [prompt_messages(r) for r in batch for _ in range(k)]
        group_ids = torch.tensor([g for g in range(len(batch)) for _ in range(k)], device=device)

        # ---- rollout ----
        set_seed(seed + u)
        policy.eval()
        gen = generate(policy, tokenizer, msgs, cfg, max_len)
        seq, attn, pw, resp = gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
        full_mask = gen["response_mask"].to(device)
        token_mask = mask_truncated_sequences(full_mask, gen["truncated"]) if cfg["mask_truncated_completions"] else full_mask
        ref_logp, _ = chunked_logprobs(policy, gen, temp, reference=True)
        raw = reward_scores(reward, msgs, gen["responses"], cfg)
        rewards = torch.tensor(raw, device=device, dtype=torch.float32)
        adv = group_relative_advantages(rewards, group_ids)
        group_std = torch.stack([rewards[group_ids == g].std(unbiased=False) for g in range(len(batch))])

        # ---- policy update(s) ----
        policy.train()
        old_logp = None
        ep_logs = []
        for epoch in range(epochs):
            new_logp, ent = response_logprobs(policy, seq, attn, pw, resp, temp, with_entropy=(epoch == 0))
            if old_logp is None:
                old_logp = new_logp.detach()  # on-policy first epoch (dropout disabled): rho = 1 exactly
            loss, diag = grpo_policy_loss(new_logp, old_logp, adv, token_mask, ref_logp, eps, beta, loss_type, max_len)
            per_grad = []
            if grad_diag and epoch == 0:
                terms = per_completion_policy_terms(new_logp, old_logp, adv, token_mask, eps, loss_type, max_len)
                scale = float(scaler.get_scale()) if use_scaler else 1.0
                for i in range(terms.shape[0]):
                    if token_mask[i].sum() == 0 or adv[i] == 0:
                        per_grad.append(0.0)
                        continue
                    grads = torch.autograd.grad(terms[i] * scale, params, retain_graph=True, allow_unused=True)
                    sq = sum(float((g.float() / scale).pow(2).sum()) for g in grads if g is not None)
                    per_grad.append(sq ** 0.5)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ep_logs.append({
                "loss": float(loss.detach()), "policy_term": float(diag["policy_term"]), "k3_kl": float(diag["sampled_kl"]),
                "clip_fraction": float(diag["clip_fraction"]), "grad_norm": gn,
                "skipped_nonfinite": bool(use_scaler and scaler.get_scale() < scale_before),
            })
            if epoch == 0:
                entropy = float(masked_mean(ent, full_mask))
                per_completion_grad = per_grad

        elapsed = elapsed_before + timer()
        peak = max(peak_before, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        lengths = full_mask.sum(-1)
        record = {
            "update": u + 1,
            "prompt_ids": [r.get("prompt_id") for r in batch],
            "reward_mean": float(rewards.mean()),
            "reward_std_all": float(rewards.std(unbiased=False)),
            "group_reward_std_mean": float(group_std.mean()),
            "uninformative_group_fraction": float((group_std <= ZERO_STD_TOL).float().mean()),
            "kl_token_mean": float(sampled_kl(old_logp, ref_logp, full_mask)),
            "k3_kl": ep_logs[0]["k3_kl"],
            "policy_loss": float(np.mean([e["loss"] for e in ep_logs])),
            "policy_term": float(np.mean([e["policy_term"] for e in ep_logs])),
            "grad_norm": float(np.mean([e["grad_norm"] for e in ep_logs])),
            "clip_fraction": float(np.mean([e["clip_fraction"] for e in ep_logs])),
            "entropy": entropy,
            "response_length": float(lengths.float().mean()),
            "truncated_rate": float(np.mean(gen["truncated"])),
            "masked_completions": int((token_mask.sum(-1) == 0).sum()),
            "epochs": ep_logs,
            "elapsed_s": elapsed,
            "peak_vram_gib": peak / 2**30,
        }
        append_jsonl(log_path, record)
        for i in range(len(msgs)):
            append_jsonl(comp_path, {
                "update": u + 1, "group": int(group_ids[i]), "prompt_id": batch[int(group_ids[i])].get("prompt_id"),
                "response": gen["responses"][i], "response_tokens": int(lengths[i]),
                "loss_tokens": int(token_mask[i].sum()), "truncated": gen["truncated"][i],
                "reward": raw[i], "advantage": float(adv[i]),
                "policy_grad_norm": per_completion_grad[i] if per_completion_grad else None,
            })
        print(
            f"[grpo] {run_name} update {u + 1}/{n_updates} reward={record['reward_mean']:.3f} "
            f"grp_std={record['group_reward_std_mean']:.3f} uninf={record['uninformative_group_fraction']:.2f} "
            f"kl={record['kl_token_mean']:.4f} ent={entropy:.3f} loss={record['policy_loss']:.4f} "
            f"gnorm={record['grad_norm']:.3f} len={record['response_length']:.0f} {elapsed / 60:.1f}min",
            flush=True,
        )
        if u + 1 < n_updates:
            save_checkpoint(out, u + 1, models, optimizers, extra={"elapsed_s": elapsed, "peak_vram_bytes": peak})

    policy.save_pretrained(str(out))
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        **{k_: v for k_, v in run_config.items() if k_ != "prompt_schedule"},
        "output": str(out),
        "wall_clock_s": elapsed_before + timer(),
        "peak_vram_gib": max(peak_before, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / 2**30,
        "mean_over_updates": {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("reward_mean", "group_reward_std_mean", "uninformative_group_fraction", "kl_token_mean", "k3_kl",
                        "policy_loss", "grad_norm", "entropy", "response_length", "truncated_rate")
        },
        "generated_tokens": float(sum(r["response_length"] * len(r["prompt_ids"]) * k for r in rows)),
        "final_update": {key: v for key, v in rows[-1].items() if key != "epochs"} if rows else None,
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    mark_complete(out, summary)
    print(f"[grpo] {run_name}: saved adapter to {out}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
