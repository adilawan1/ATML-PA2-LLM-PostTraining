"""Common Task 1 evaluation protocol, applied identically to every DPO condition (and the SFT baseline).

For one adapter it writes `results/task1_dpo/eval_<name>/`:
  metrics.json             all aggregate metrics below
  pairs_heldout.jsonl      per-pair policy/reference log-probs on the fixed held-out preference pairs
  pairs_stratified.jsonl   the same on the length-stratified held-out set (with length_stratum)
  generations.jsonl        one sampled response per held-out prompt: reward, KL, length
  word_limit.jsonl         sampled responses to the common word-limit prompts with compliance

Metric conventions (assignment "Common Metric Definitions"):
  - preference accuracy: fraction of pairs with reference-adjusted margin m_theta > 0, summed response log-probs
  - KL: sampled-response estimator log pi_theta - log pi_ref on generated response tokens. `kl_token_mean`
    is the course helper (common.metrics.sampled_kl: mean over all valid tokens); `kl_seq_mean` is the
    per-response sum averaged over responses. Both are reported for every condition.
  - entropy: course helper sample_entropy (mean -log pi_theta of sampled tokens)
  - response length: generated response tokens excluding padding, with mean/std/median/IQR
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate


def load_evaluation_bundle(config_path: str, adapter: str | None):
    cfg = load_yaml(config_path)
    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"  # keep the generation prompt's tail if a prompt is over the cap
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": tokenizer,
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "n": int(a.size)}


@torch.no_grad()
def score_pairs(model, tokenizer, rows, cfg, batch_size):
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(model.parameters()).device
    records = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        chosen, rejected = (
            {k: v.to(device) for k, v in b.items()} for b in collate(chunk)
        )
        pc, _, mc = response_sequence_logprobs(model, chosen)
        pr, _, mr = response_sequence_logprobs(model, rejected)
        with reference_mode(model):
            rc, _, _ = response_sequence_logprobs(model, chosen)
            rr, _, _ = response_sequence_logprobs(model, rejected)
        for j, row in enumerate(chunk):
            records.append({
                "prompt_id": row.get("prompt_id"),
                "source_index": row.get("source_index"),
                "length_stratum": row.get("length_stratum"),
                "score_chosen": row.get("score_chosen"),
                "score_rejected": row.get("score_rejected"),
                "chosen_tokens": int(mc[j].sum()),
                "rejected_tokens": int(mr[j].sum()),
                "policy_chosen_logp": float(pc[j]),
                "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]),
                "ref_rejected_logp": float(rr[j]),
            })
    return records


def pair_metrics(records, beta) -> dict:
    if not records:
        return {}
    t = {k: torch.tensor([r[k] for r in records], dtype=torch.float64) for k in ("policy_chosen_logp", "policy_rejected_logp", "ref_chosen_logp", "ref_rejected_logp")}
    loss, diag = dpo_loss(t["policy_chosen_logp"], t["policy_rejected_logp"], t["ref_chosen_logp"], t["ref_rejected_logp"], beta)
    chosen_lr = t["policy_chosen_logp"] - t["ref_chosen_logp"]
    rejected_lr = t["policy_rejected_logp"] - t["ref_rejected_logp"]
    margin = chosen_lr - rejected_lr
    return {
        "n_pairs": len(records),
        "dpo_loss": float(loss),
        "preference_accuracy": float(diag["preference_accuracy"]),
        "pref_margin_mean": float(margin.mean()),
        "pref_margin_median": float(margin.median()),
        "chosen_logratio_mean": float(chosen_lr.mean()),
        "rejected_logratio_mean": float(rejected_lr.mean()),
        "implicit_reward_chosen_mean": float(beta * chosen_lr.mean()),
        "implicit_reward_rejected_mean": float(beta * rejected_lr.mean()),
    }


@torch.no_grad()
def policy_ref_token_logprobs(model, gen, chunk):
    """Per-token log-probs of the sampled response under the policy and the reference, in small chunks."""
    pol, ref = [], []
    for s in range(0, gen["sequences"].shape[0], chunk):
        sl = slice(s, s + chunk)
        args = (gen["sequences"][sl], gen["attention_mask"][sl], gen["prompt_width"], gen["response_ids"][sl])
        lp, logits = response_token_logprobs(model, *args)
        del logits
        pol.append(lp)
        with reference_mode(model):
            lr, logits = response_token_logprobs(model, *args)
            del logits
        ref.append(lr)
    return torch.cat(pol), torch.cat(ref)


def generate_records(model, tokenizer, reward, items, cfg, max_new_tokens, seed):
    """Sample one response per (id, messages) item; score reward, KL to reference, entropy and length."""
    rm_model, rm_tok = reward
    gcfg = cfg["generation"]
    bs, chunk = int(cfg.get("eval_batch_size", 16)), int(cfg.get("eval_logprob_chunk", 4))
    set_seed(seed)
    records = []
    for i in range(0, len(items), bs):
        batch = items[i : i + bs]
        msgs = [m for _, m in batch]
        gen = batch_generate(
            model, tokenizer, msgs,
            max_prompt_length=int(cfg.get("max_prompt_length", 512)),
            max_new_tokens=max_new_tokens,
            temperature=float(gcfg["temperature"]), top_p=float(gcfg["top_p"]), do_sample=bool(gcfg["do_sample"]),
        )
        pol, ref = policy_ref_token_logprobs(model, gen, chunk)
        mask = gen["response_mask"]
        diff = (pol - ref) * mask
        rewards = []
        for s in range(0, len(batch), chunk):
            rewards += score_reward_pairs(
                rm_model, rm_tok, msgs[s : s + chunk], gen["responses"][s : s + chunk],
                max_length=int(cfg.get("reward_max_length", 1280)),
            ).tolist()
        for j, (item_id, m) in enumerate(batch):
            n = float(mask[j].sum())
            records.append({
                "id": item_id,
                "prompt": m[-1]["content"] if m else "",
                "response": gen["responses"][j],
                "response_tokens": int(n),
                "terminated_with_eos": gen["terminated_with_eos"][j],
                "truncated": gen["truncated"][j],
                "reward": float(rewards[j]),
                "kl_seq": float(diff[j].sum()),
                "logp_sum": float((pol[j] * mask[j]).sum()),
                "valid_tokens": n,
            })
        print(f"[eval] generated {min(i + bs, len(items))}/{len(items)}", flush=True)
    return records


def generation_metrics(records) -> dict:
    if not records:
        return {}
    tokens = sum(r["valid_tokens"] for r in records)
    return {
        "n_responses": len(records),
        "reward_mean": float(np.mean([r["reward"] for r in records])),
        "reward_std": float(np.std([r["reward"] for r in records])),
        # course helper convention: mean over all valid response tokens in the evaluation set
        "kl_token_mean": float(sum(r["kl_seq"] for r in records) / max(tokens, 1.0)),
        "kl_seq_mean": float(np.mean([r["kl_seq"] for r in records])),
        "entropy_token_mean": float(-sum(r["logp_sum"] for r in records) / max(tokens, 1.0)),
        "response_length": length_stats([r["response_tokens"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
    }


def evaluate_adapter(config_path: str, adapter: str | None, name: str, beta: float | None = None,
                     max_gen_prompts: int | None = None, skip_stratified: bool = False):
    timer = wall_timer()
    bundle = load_evaluation_bundle(config_path, adapter)
    cfg, model, tokenizer = bundle["cfg"], bundle["policy"], bundle["tokenizer"]
    out_dir = repo_path(cfg["results_dir"]) / f"eval_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])

    if beta is None:
        run_summary = repo_path(adapter) / "_COMPLETE.json" if adapter else None
        beta = float(load_json(run_summary)["beta"]) if run_summary and run_summary.is_file() else float(cfg["beta"])
    pair_bs = int(cfg.get("eval_pair_batch_size", 4))

    def dump(fname, rows):
        with (out_dir / fname).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[eval] {name}: scoring {len(bundle['rows'])} held-out pairs", flush=True)
    heldout = score_pairs(model, tokenizer, bundle["rows"], cfg, pair_bs)
    dump("pairs_heldout.jsonl", heldout)

    stratified_metrics = {}
    if not skip_stratified:
        strat_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
        print(f"[eval] {name}: scoring {len(strat_rows)} length-stratified pairs", flush=True)
        stratified = score_pairs(model, tokenizer, strat_rows, cfg, pair_bs)
        dump("pairs_stratified.jsonl", stratified)
        stratified_metrics = {"all": pair_metrics(stratified, beta)}
        for stratum in sorted({r["length_stratum"] for r in stratified}):
            stratified_metrics[stratum] = pair_metrics([r for r in stratified if r["length_stratum"] == stratum], beta)

    gen_rows = bundle["rows"][: max_gen_prompts] if max_gen_prompts else bundle["rows"]
    items = [(r["prompt_id"], prompt_messages_from_preference(r)) for r in gen_rows]
    print(f"[eval] {name}: generating for {len(items)} held-out prompts", flush=True)
    gens = generate_records(model, tokenizer, bundle["reward"], items, cfg, int(cfg["max_generation_tokens"]), seed)
    dump("generations.jsonl", gens)

    wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    k = int(cfg.get("word_limit_samples", 5))
    wl_items = [(f"{r['prompt_id']}#{s}", prompt_messages(r)) for r in wl_rows for s in range(k)]
    wl = generate_records(model, tokenizer, bundle["reward"], wl_items, cfg, int(cfg["max_generation_tokens"]), seed)
    for r in wl:
        r["prompt_id"] = r["id"].split("#")[0]
        r["words"] = word_count(r["response"])
        r["compliant"] = word_limit_compliance(r["prompt"], r["response"])
    dump("word_limit.jsonl", wl)
    compliance = [r["compliant"] for r in wl if r["compliant"] is not None]

    metrics = {
        "name": name,
        "adapter": adapter,
        "beta_for_loss": beta,
        "heldout_pairs": pair_metrics(heldout, beta),
        "stratified_pairs": stratified_metrics,
        "generation": generation_metrics(gens),
        "word_limit": {
            "n_responses": len(wl),
            "samples_per_prompt": k,
            "compliance_rate": float(np.mean(compliance)) if compliance else None,
            "words": length_stats([r["words"] for r in wl]),
            "response_length": length_stats([r["response_tokens"] for r in wl]),
        },
        "protocol": {
            "seed": seed,
            "generation": cfg["generation"],
            "max_generation_tokens": int(cfg["max_generation_tokens"]),
            "max_prompt_length": int(cfg.get("max_prompt_length", 512)),
            "max_sequence_length": int(cfg["max_sequence_length"]),
            "heldout_file": cfg["paths"]["dpo_standard_eval"],
            "stratified_file": None if skip_stratified else cfg["paths"]["dpo_length_eval"],
            "word_limit_file": cfg["paths"]["word_limit_prompts"],
            "n_generation_prompts": len(items),
        },
        "eval_wall_clock_s": timer(),
    }
    save_json(out_dir / "metrics.json", metrics)
    h, g = metrics["heldout_pairs"], metrics["generation"]
    print(
        f"[eval] {name}: loss={h['dpo_loss']:.4f} acc={h['preference_accuracy']:.3f} "
        f"reward={g['reward_mean']:.3f} kl_tok={g['kl_token_mean']:.4f} kl_seq={g['kl_seq_mean']:.2f} "
        f"len={g['response_length']['mean']:.1f} wl_compliance={metrics['word_limit']['compliance_rate']}",
        flush=True,
    )
    del model, bundle
    clear_gpu()
    return metrics


def summary_row(metrics: dict, **extra) -> dict:
    h, g = metrics["heldout_pairs"], metrics["generation"]
    L = g["response_length"]
    return {
        **extra,
        "name": metrics["name"],
        "beta": metrics["beta_for_loss"],
        "heldout_dpo_loss": h.get("dpo_loss"),
        "heldout_pref_acc": h.get("preference_accuracy"),
        "heldout_margin_mean": h.get("pref_margin_mean"),
        "reward_mean": g["reward_mean"],
        "reward_std": g["reward_std"],
        "kl_token_mean": g["kl_token_mean"],
        "kl_seq_mean": g["kl_seq_mean"],
        "entropy_token_mean": g["entropy_token_mean"],
        "len_mean": L["mean"],
        "len_std": L["std"],
        "len_median": L["median"],
        "len_iqr": L["iqr"],
        "truncation_rate": g["truncation_rate"],
        "word_limit_compliance": metrics["word_limit"]["compliance_rate"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the untouched SFT policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the held-out DPO loss (default: the adapter's training beta)")
    ap.add_argument("--max-gen-prompts", type=int, help="smoke tests only: limit generation prompts")
    ap.add_argument("--skip-stratified", action="store_true")
    args = ap.parse_args()
    adapter = None if args.adapter.lower() == "none" else args.adapter
    evaluate_adapter(args.config, adapter, args.name, args.beta, args.max_gen_prompts, args.skip_stratified)


if __name__ == "__main__":
    main()
