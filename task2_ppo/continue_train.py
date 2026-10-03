"""Task 2: PPO continuation from the supplied midpoint policy/value checkpoints.

One update = sample `prompts_per_update` prompts from a fixed seeded schedule (shared by every fork), generate
one response each (max_response_length), score with the frozen reward model (missing-EOS penalty), build
sampled-token KL-shaped rewards against the frozen reference (adapter disabled), compute GAE with the critic,
whiten advantages over valid tokens, then run `ppo_epochs` clipped policy + value steps on that rollout.

Logged per update (results/task2_ppo/train_<run>.jsonl): learned reward (raw and effective), KL to the reference
(course helper token mean and per-sequence sum), policy loss, value loss, entropy, policy/value gradient norms,
clip fraction (per epoch; epoch 0 is exactly 0 because the rollout is on-policy and dropout is disabled),
approximate old->new KL, ratio range, critic explained variance, response length, EOS, wall-clock, peak VRAM.
Each update is checkpointed, so re-running the same command after a disconnect resumes.
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
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    resolve_dtype,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl_utils import (
    chunked_logprobs,
    disable_dropout,
    effective_rewards,
    generate,
    logprob_temperature,
    prompt_schedule,
    response_logprobs,
    reward_scores,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    # Trainable parameters in fp32 (base weights stay fp16) so AdamW/GradScaler update full-precision copies.
    for model in (policy, value_model):
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        disable_dropout(model)
    policy.generation_config.use_cache = True
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def value_estimates(value_model, sequences, attention_mask, prompt_width, n_response):
    """V(s_t) for each response token t: the critic's scalar head at the position *before* token t.

    Like common.models.token_values, but it calls the transformer backbone itself (LoRA layers included) rather
    than the sequence-classification wrapper, so the trainable fp32 head copy is applied only once, in fp32.
    """
    base = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    backbone = getattr(base, base.base_model_prefix)
    hidden = backbone(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state
    head = base.score if hasattr(base, "score") else base.classifier
    values = head(hidden[:, prompt_width - 1 : prompt_width - 1 + n_response].float()).squeeze(-1)
    return values


def explained_variance(pred, target, mask) -> float:
    m = mask.bool()
    y, p = target[m].float(), pred[m].float()
    var = y.var(unbiased=False)
    return float(1.0 - (y - p).var(unbiased=False) / var) if var > 0 else float("nan")


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    base_cfg = load_yaml(config_path)
    out = repo_path(output or base_cfg["output"])
    if is_complete(out):
        print(f"[ppo] {run_name}: already complete at {out}; skipping.", flush=True)
        return load_json(out / COMPLETE_MARKER)

    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, value_model, tokenizer = bundle["policy"], bundle["value_model"], bundle["tokenizer"]
    popt, vopt = bundle["policy_optimizer"], bundle["value_optimizer"]
    reward = (bundle["reward_model"], bundle["reward_tokenizer"])
    device = next(policy.parameters()).device
    n_updates, eps, beta_kl = int(cfg["updates"]), float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    per_update, ppo_epochs = int(cfg["prompts_per_update"]), int(cfg["ppo_epochs"])
    temp = logprob_temperature(cfg)
    seed = int(cfg["seed"])
    schedule = prompt_schedule(bundle["prompt_rows"], seed, n_updates, per_update)
    policy_params, value_params = trainable_parameters(policy), trainable_parameters(value_model)

    use_scaler = device.type == "cuda" and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    results_dir = repo_path(cfg["results_dir"])
    run_config = {
        "run_name": run_name, "updates": n_updates, "prompts_per_update": per_update, "ppo_epochs": ppo_epochs,
        "clip_epsilon": eps, "kl_beta": beta_kl, "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]),
        "value_coef": float(cfg["value_coef"]), "missing_eos_penalty": float(cfg["missing_eos_penalty"]),
        "policy_learning_rate": float(cfg["policy_learning_rate"]), "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]), "value_train_mode": cfg.get("value_train_mode"),
        "max_prompt_length": int(cfg["max_prompt_length"]), "max_response_length": int(cfg["max_response_length"]),
        "reward_max_length": int(cfg["reward_max_length"]), "max_grad_norm": float(cfg["max_grad_norm"]),
        "generation": cfg["generation"], "logprob_temperature": temp, "seed": seed, "dropout_disabled": True,
        "grad_scaler": use_scaler, "policy_init": cfg["paths"]["ppo_midpoint_policy"], "value_init": cfg["paths"]["ppo_midpoint_value"],
        "reward_model": cfg["reward_model"], "reference": "base model (policy adapter disabled)",
        "prompt_schedule": [[r.get("prompt_id") for r in batch] for batch in schedule],
    }
    save_json(results_dir / f"train_{run_name}_config.json", run_config)
    log_path = results_dir / f"train_{run_name}.jsonl"
    rollout_path = results_dir / f"rollouts_{run_name}.jsonl"

    models = {"policy": policy, "value": value_model}
    optimizers = {"policy_opt": popt, "value_opt": vopt, "scaler": scaler}
    state = resume_if_available(out, models, optimizers)
    start = state["step"] if state else 0
    elapsed_before = float(state["extra"].get("elapsed_s", 0.0)) if state else 0.0
    peak_before = int(state["extra"].get("peak_vram_bytes", 0)) if state else 0
    for p in (log_path, rollout_path):
        if state:
            truncate_jsonl_after(p, start, key="update")
        elif p.exists():
            p.unlink()

    print(f"[ppo] {run_name}: {n_updates} updates, eps={eps}, kl_beta={beta_kl}, starting at update {start}", flush=True)
    timer = wall_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for u in range(start, n_updates):
        batch = schedule[u]
        msgs = [prompt_messages(r) for r in batch]

        # ---- rollout (no grad; generation RNG keyed to the update index so forks/resumes match) ----
        set_seed(seed + u)
        policy.eval()
        value_model.eval()
        gen = generate(policy, tokenizer, msgs, cfg, cfg["max_response_length"])
        seq, attn, pw, resp = gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"]
        mask = gen["response_mask"].to(device)
        old_logp, _ = chunked_logprobs(policy, gen, temp)
        ref_logp, _ = chunked_logprobs(policy, gen, temp, reference=True)
        with torch.no_grad():
            old_values = value_estimates(value_model, seq, attn, pw, resp.shape[1]).float() * mask
        raw = reward_scores(reward, msgs, gen["responses"], cfg)
        eff = effective_rewards(raw, gen["terminated_with_eos"], cfg)
        task_reward = torch.tensor(eff, device=device, dtype=torch.float32)
        rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, beta_kl)
        adv, returns = compute_gae(rewards, old_values, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        adv_n = normalize_advantages(adv, mask)
        ev_before = explained_variance(old_values, returns, mask)

        # ---- PPO epochs on this rollout ----
        policy.train()
        value_model.train()
        epochs = []
        for epoch in range(ppo_epochs):
            new_logp, ent = response_logprobs(policy, seq, attn, pw, resp, temp, with_entropy=(epoch == 0))
            p_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_n, mask, eps)
            values = value_estimates(value_model, seq, attn, pw, resp.shape[1])
            v_loss = value_mse_loss(values, returns, mask)
            loss = p_loss + float(cfg["value_coef"]) * v_loss
            scaler.scale(loss).backward()
            scaler.unscale_(popt)
            scaler.unscale_(vopt)
            pg = float(torch.nn.utils.clip_grad_norm_(policy_params, float(cfg["max_grad_norm"])))
            vg = float(torch.nn.utils.clip_grad_norm_(value_params, float(cfg["max_grad_norm"])))
            scale_before = scaler.get_scale()
            scaler.step(popt)
            scaler.step(vopt)
            scaler.update()
            popt.zero_grad(set_to_none=True)
            vopt.zero_grad(set_to_none=True)
            with torch.no_grad():
                log_r = (new_logp.detach() - old_logp)
                m = mask.bool()
                approx_kl = masked_mean(torch.exp(log_r) - 1.0 - log_r, mask)
                epochs.append({
                    "policy_loss": float(p_loss.detach()), "value_loss": float(v_loss.detach()),
                    "clip_fraction": float(clip_frac), "approx_kl_old_new": float(approx_kl),
                    "ratio_min": float(ratio[m].min()) if m.any() else 1.0, "ratio_max": float(ratio[m].max()) if m.any() else 1.0,
                    "policy_grad_norm": pg, "value_grad_norm": vg,
                    "skipped_nonfinite": bool(use_scaler and scaler.get_scale() < scale_before),
                })
                if epoch == 0:
                    entropy = float(masked_mean(ent, mask))

        elapsed = elapsed_before + timer()
        peak = max(peak_before, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        lengths = mask.sum(-1)
        record = {
            "update": u + 1,
            "prompt_ids": [r.get("prompt_id") for r in batch],
            "reward_raw": float(np.mean(raw)),
            "reward_effective": float(np.mean(eff)),
            "kl_token_mean": float(sampled_kl(old_logp, ref_logp, mask)),
            "kl_seq_mean": float(((old_logp - ref_logp) * mask).sum(-1).mean()),
            "entropy": entropy,
            "response_length": float(lengths.float().mean()),
            "eos_rate": float(np.mean(gen["terminated_with_eos"])),
            "truncated_rate": float(np.mean(gen["truncated"])),
            "value_explained_variance": ev_before,
            "value_mean": float(masked_mean(old_values, mask)),
            "return_mean": float(masked_mean(returns, mask)),
            "adv_raw_std": float(adv[mask.bool()].std(unbiased=False)) if mask.sum() > 1 else 0.0,
            "policy_loss": float(np.mean([e["policy_loss"] for e in epochs])),
            "value_loss": float(np.mean([e["value_loss"] for e in epochs])),
            "clip_fraction": float(np.mean([e["clip_fraction"] for e in epochs])),
            "policy_grad_norm": float(np.mean([e["policy_grad_norm"] for e in epochs])),
            "value_grad_norm": float(np.mean([e["value_grad_norm"] for e in epochs])),
            "approx_kl_old_new": float(np.mean([e["approx_kl_old_new"] for e in epochs])),
            "epochs": epochs,
            "elapsed_s": elapsed,
            "peak_vram_gib": peak / 2**30,
        }
        append_jsonl(log_path, record)
        for j, r in enumerate(batch):
            append_jsonl(rollout_path, {
                "update": u + 1, "prompt_id": r.get("prompt_id"), "source_index": r.get("source_index"),
                "response": gen["responses"][j], "response_tokens": int(lengths[j]), "reward_raw": raw[j],
                "reward_effective": eff[j], "terminated_with_eos": gen["terminated_with_eos"][j],
            })
        print(
            f"[ppo] {run_name} update {u + 1}/{n_updates} reward={record['reward_raw']:.3f} "
            f"kl={record['kl_token_mean']:.4f} ent={entropy:.3f} pl={record['policy_loss']:.4f} "
            f"vl={record['value_loss']:.3f} clip={record['clip_fraction']:.3f} len={record['response_length']:.0f} "
            f"{elapsed / 60:.1f}min",
            flush=True,
        )
        if u + 1 < n_updates:
            save_checkpoint(out, u + 1, models, optimizers, extra={"elapsed_s": elapsed, "peak_vram_bytes": peak})

    policy.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "value"))
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = {
        **{k: v for k, v in run_config.items() if k != "prompt_schedule"},
        "output": str(out),
        "wall_clock_s": elapsed_before + timer(),
        "peak_vram_gib": max(peak_before, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / 2**30,
        "mean_over_updates": {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("reward_raw", "kl_token_mean", "entropy", "policy_loss", "value_loss", "clip_fraction", "policy_grad_norm", "response_length", "approx_kl_old_new")
        },
        "final_update": {k: v for k, v in rows[-1].items() if k != "epochs"} if rows else None,
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    mark_complete(out, summary)
    print(f"[ppo] {run_name}: saved policy adapter to {out} (critic in {out / 'value'})", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
