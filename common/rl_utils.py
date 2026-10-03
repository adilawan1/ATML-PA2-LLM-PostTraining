"""Rollout, scoring and evaluation utilities shared by the PPO (Task 2) and GRPO (Task 3) continuations.

Conventions (identical for every PPO/GRPO condition):
  - Token log-probs are computed from logits divided by `logprob_temperature` (default: the sampling
    temperature, as in TRL's PPO/GRPO trainers), for the policy, the frozen reference (adapter disabled) and
    the old policy alike, so ratios and the KL estimate refer to the same distribution that was sampled.
  - Entropy is the full-distribution entropy at that temperature, averaged over valid response tokens.
  - Effective terminal reward = reward-model score - missing_eos_penalty if the response hit the generation
    cap without EOS (matches `effective_terminal_reward` in the supplied PPO cache).
  - Dropout is disabled in RL policies so that rho = 1 exactly at the first update of fresh rollouts.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from common.data import prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.models import reference_mode


def logprob_temperature(cfg: dict) -> float:
    return float(cfg.get("logprob_temperature", cfg["generation"]["temperature"]))


def disable_dropout(model) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "n": int(a.size)}


def response_logprobs(model, sequences, attention_mask, prompt_width, response_ids, temperature, with_entropy=False):
    """Per-token log-probs (and optionally entropy) of `response_ids` given the left-padded prompt.

    Works row by row on the fp32 logits so that only one vocab-sized fp32 slice is live at a time.
    Gradients flow through the log-probs when called under grad; entropy is always detached.
    """
    out = model(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = out.logits[:, prompt_width - 1 : -1, :][:, : response_ids.shape[1], :]
    logps, ents = [], []
    for i in range(logits.shape[0]):
        lf = logits[i].float() / temperature
        lse = torch.logsumexp(lf, dim=-1)
        logps.append(lf.gather(-1, response_ids[i].unsqueeze(-1)).squeeze(-1) - lse)
        if with_entropy:
            with torch.no_grad():
                lfd, lsed = lf.detach(), lse.detach()
                ents.append(lsed - (torch.softmax(lfd, dim=-1) * lfd).sum(-1))
    logp = torch.stack(logps)
    return (logp, torch.stack(ents)) if with_entropy else (logp, None)


@torch.no_grad()
def chunked_logprobs(model, gen, temperature, chunk=4, reference=False, with_entropy=False):
    """No-grad log-probs for a generation batch in chunks (optionally under the reference policy)."""
    lps, ents = [], []
    for s in range(0, gen["sequences"].shape[0], chunk):
        sl = slice(s, s + chunk)
        args = (gen["sequences"][sl], gen["attention_mask"][sl], gen["prompt_width"], gen["response_ids"][sl], temperature, with_entropy)
        if reference:
            with reference_mode(model):
                lp, ent = response_logprobs(model, *args)
        else:
            lp, ent = response_logprobs(model, *args)
        lps.append(lp)
        if with_entropy:
            ents.append(ent)
    return torch.cat(lps), (torch.cat(ents) if with_entropy else None)


def generate(model, tokenizer, messages, cfg, max_new_tokens):
    g = cfg["generation"]
    gen = batch_generate(
        model, tokenizer, messages,
        max_prompt_length=int(cfg.get("max_prompt_length", 256)),
        max_new_tokens=int(max_new_tokens),
        temperature=float(g["temperature"]), top_p=float(g["top_p"]), do_sample=bool(g["do_sample"]),
    )
    # batch_generate runs under torch.inference_mode; clone so the tensors can enter autograd-tracked updates.
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in gen.items()}


@torch.no_grad()
def reward_scores(reward, messages, responses, cfg, chunk=4):
    rm_model, rm_tok = reward
    out = []
    for s in range(0, len(responses), chunk):
        out += score_reward_pairs(rm_model, rm_tok, messages[s : s + chunk], responses[s : s + chunk],
                                  max_length=int(cfg.get("reward_max_length", 1280))).tolist()
    return out


def effective_rewards(raw, terminated, cfg):
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    return [r - (0.0 if t else penalty) for r, t in zip(raw, terminated)]


def prompt_schedule(rows, seed: int, n: int, per_update: int):
    """Fixed, seeded prompt order shared by every fork: update u uses rows order[u*per_update:(u+1)*per_update]."""
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(rows), generator=g).tolist()
    need = n * per_update
    order = (order * (need // len(order) + 1))[:need]
    return [[rows[i] for i in order[u * per_update : (u + 1) * per_update]] for u in range(n)]


def evaluate_policy(model, tokenizer, reward, cfg, name, results_dir, max_new_tokens, max_prompts=None):
    """Common held-out protocol for RL policies on the fixed RL eval prompts.

    One sampled response per prompt (seeded, identical decoding for every condition). Saves
    `<results_dir>/eval_<name>/{generations.jsonl,metrics.json}`.
    """
    timer = wall_timer()
    tokenizer.truncation_side = "left"
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_prompts:
        rows = rows[:max_prompts]
    temp = logprob_temperature(cfg)
    bs, chunk = int(cfg.get("eval_batch_size", 16)), int(cfg.get("eval_logprob_chunk", 2))
    out_dir = repo_path(results_dir) / f"eval_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.generation_config.use_cache = True
    set_seed(int(cfg["seed"]))

    records = []
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        msgs = [prompt_messages(r) for r in batch]
        gen = generate(model, tokenizer, msgs, cfg, max_new_tokens)
        mask = gen["response_mask"]
        pol, ent = chunked_logprobs(model, gen, temp, chunk, with_entropy=True)
        ref, _ = chunked_logprobs(model, gen, temp, chunk, reference=True)
        raw = reward_scores(reward, msgs, gen["responses"], cfg)
        eff = effective_rewards(raw, gen["terminated_with_eos"], cfg)
        for j, row in enumerate(batch):
            m = mask[j]
            records.append({
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "prompt": msgs[j][-1]["content"],
                "response": gen["responses"][j],
                "response_tokens": int(m.sum()),
                "terminated_with_eos": gen["terminated_with_eos"][j],
                "truncated": gen["truncated"][j],
                "reward": raw[j],
                "effective_reward": eff[j],
                "kl_seq": float(((pol[j] - ref[j]) * m).sum()),
                "entropy_sum": float((ent[j] * m).sum()),
                "valid_tokens": float(m.sum()),
            })
        print(f"[eval] {name}: {min(i + bs, len(rows))}/{len(rows)}", flush=True)

    with (out_dir / "generations.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tokens = max(sum(r["valid_tokens"] for r in records), 1.0)
    metrics = {
        "name": name,
        "n_prompts": len(records),
        "reward_mean": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records])),
        "effective_reward_mean": float(np.mean([r["effective_reward"] for r in records])),
        "kl_token_mean": float(sum(r["kl_seq"] for r in records) / tokens),
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])),
        "entropy_token_mean": float(sum(r["entropy_sum"] for r in records) / tokens),
        "response_length": length_stats([r["response_tokens"] for r in records]),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "protocol": {
            "seed": int(cfg["seed"]),
            "generation": cfg["generation"],
            "logprob_temperature": temp,
            "max_new_tokens": int(max_new_tokens),
            "max_prompt_length": int(cfg.get("max_prompt_length", 256)),
            "prompts_file": cfg["paths"]["rl_prompt_eval"],
            "missing_eos_penalty": float(cfg.get("missing_eos_penalty", 0.0)),
        },
        "eval_wall_clock_s": timer(),
    }
    save_json(out_dir / "metrics.json", metrics)
    print(
        f"[eval] {name}: reward={metrics['reward_mean']:.3f} kl_tok={metrics['kl_token_mean']:.4f} "
        f"entropy={metrics['entropy_token_mean']:.3f} len={metrics['response_length']['mean']:.1f} eos={metrics['eos_rate']:.2f}",
        flush=True,
    )
    return metrics
