"""Cross-task check of which reported comparisons carry signal beyond evaluation noise.

Every condition within a task is evaluated on the same prompts with the same seed, so differences are paired by
prompt. For each comparison this reports the fraction of byte-identical responses and the paired mean difference of
the headline metric with a percentile bootstrap 95% CI (resampling prompts). A CI that excludes 0 is marked
`resolved`; otherwise the comparison is a null result at this sample size. CPU only; reads results/ and writes
results/informativeness_audit.json and results/informativeness_audit.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
RNG = np.random.default_rng(6304)
B = 10000


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def boot(diff):
    diff = np.asarray(diff, dtype=float)
    idx = RNG.integers(0, len(diff), size=(B, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


rows = []


def add(task, comparison, metric, a, b, identical=None):
    """a, b: aligned per-item arrays; reports mean(a - b)."""
    mean, lo, hi = boot(np.asarray(a, float) - np.asarray(b, float))
    rows.append({"task": task, "comparison": comparison, "metric": metric, "n": len(a),
                 "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)), "diff": mean, "ci_lo": lo, "ci_hi": hi,
                 "resolved": bool(lo > 0 or hi < 0), "identical_frac": identical})


def gen_pair(task, d_a, d_b, label, key):
    ga = {r[key]: r for r in read(R / task / d_a / "generations.jsonl")}
    gb = {r[key]: r for r in read(R / task / d_b / "generations.jsonl")}
    ids = [i for i in ga if i in gb]
    ident = float(np.mean([ga[i]["response"] == gb[i]["response"] for i in ids]))
    add(task, label, "reward", [ga[i]["reward"] for i in ids], [gb[i]["reward"] for i in ids], ident)
    add(task, label, "response_tokens", [ga[i]["response_tokens"] for i in ids], [gb[i]["response_tokens"] for i in ids], ident)
    add(task, label, "kl_seq", [ga[i]["kl_seq"] for i in ids], [gb[i]["kl_seq"] for i in ids], ident)


def margin(r):
    return (r["policy_chosen_logp"] - r["ref_chosen_logp"]) - (r["policy_rejected_logp"] - r["ref_rejected_logp"])


def pref_pair(d_a, d_b, label, file="pairs_heldout.jsonl", stratum=None):
    pa, pb = read(R / "task1_dpo" / d_a / file), read(R / "task1_dpo" / d_b / file)
    if stratum:
        pa = [r for r in pa if r["length_stratum"] == stratum]
        pb = [r for r in pb if r["length_stratum"] == stratum]
    suffix = "_" + stratum if stratum else ""
    add("task1_dpo", label, f"pref_acc{suffix}", [margin(r) > 0 for r in pa], [margin(r) > 0 for r in pb])
    add("task1_dpo", label, f"pref_margin{suffix}", [margin(r) for r in pa], [margin(r) for r in pb])


def vs_chance(d, label):
    p = read(R / "task1_dpo" / d / "pairs_heldout.jsonl")
    add("task1_dpo", label, "pref_acc_vs_chance", [margin(r) > 0 for r in p], [0.5] * len(p))


# Task 1
gen_pair("task1_dpo", "eval_standard", "eval_sft", "standard vs sft", "id")
for d in ["eval_standard", "eval_beta_0.03", "eval_beta_0.1", "eval_beta_0.3", "eval_length_balanced"]:
    vs_chance(d, d.removeprefix("eval_"))
for a, b in [("0.03", "0.1"), ("0.3", "0.1"), ("0.3", "0.03")]:
    gen_pair("task1_dpo", f"eval_beta_{a}", f"eval_beta_{b}", f"beta {a} vs {b}", "id")
    pref_pair(f"eval_beta_{a}", f"eval_beta_{b}", f"beta {a} vs {b}")
gen_pair("task1_dpo", "eval_length_balanced", "eval_standard", "length_balanced vs standard", "id")
for s in ["preferred_longer", "length_matched", "rejected_longer"]:
    pref_pair("eval_length_balanced", "eval_standard", "length_balanced vs standard", "pairs_stratified.jsonl", s)

# Task 2
gen_pair("task2_ppo", "eval_standard", "eval_midpoint", "standard vs midpoint", "prompt_id")
gen_pair("task2_ppo", "eval_fork_eps0.05_kl0.1", "eval_fork_eps0.2_kl0.1", "eps 0.05 vs 0.2", "prompt_id")
gen_pair("task2_ppo", "eval_fork_eps0.5_kl0.1", "eval_fork_eps0.2_kl0.1", "eps 0.5 vs 0.2", "prompt_id")
gen_pair("task2_ppo", "eval_fork_eps0.2_kl0", "eval_fork_eps0.2_kl0.1", "kl 0 vs 0.1", "prompt_id")
gen_pair("task2_ppo", "eval_fork_eps0.2_kl0.2", "eval_fork_eps0.2_kl0.1", "kl 0.2 vs 0.1", "prompt_id")
gen_pair("task2_ppo", "eval_fork_eps0.2_kl0", "eval_fork_eps0.2_kl0.2", "kl 0 vs 0.2", "prompt_id")

# Task 3
gen_pair("task3_grpo", "eval_standard", "eval_midpoint", "standard vs midpoint", "prompt_id")
gen_pair("task3_grpo", "eval_fork_dr_grpo", "eval_fork_grpo", "dr_grpo vs grpo", "prompt_id")

# Task 4: per-policy refusal on safe prompts and answering on unsafe prompts, paired with SFT on the same prompt
REFUSE = {"JUSTIFIED_REFUSAL", "OVER_REFUSAL"}
t4 = {p: {r["xstest_id"]: r for r in read(R / "task4_safety" / f"judged_{p}.jsonl")} for p in ["sft", "dpo", "ppo", "grpo"]}
g4 = {p: {r["xstest_id"]: r["response"] for r in read(R / "task4_safety" / f"generated_{p}.jsonl")} for p in t4}
for p in ["dpo", "ppo", "grpo"]:
    for cls, metric, labs in [("SAFE", "safe_any_refusal", REFUSE), ("UNSAFE", "unsafe_any_answer", {"SAFE_ANSWER", "UNSAFE_COMPLIANCE"})]:
        ids = [i for i, r in t4["sft"].items() if r["benchmark_class"] == cls]
        ident = float(np.mean([g4[p][i] == g4["sft"][i] for i in ids]))
        add("task4_safety", f"{p} vs sft", metric, [t4[p][i]["label"] in labs for i in ids],
            [t4["sft"][i]["label"] in labs for i in ids], ident)

# Task 5: exact accuracy paired by problem
for ds in ["gsm", "transfer"]:
    pp = pd.read_csv(R / "task5_feedback" / ds / "per_problem.csv")
    gens = {p: {str(r["source_index"]): r["response"] for r in read(R / "task5_feedback" / ds / f"generated_{p}.jsonl")}
            for p in ["sft", "rlvr", "rlaif"]}
    for a, b in [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]:
        ident = float(np.mean([gens[a][k] == gens[b][k] for k in gens[a]]))
        add("task5_feedback", f"{ds}: {a} vs {b}", "exact_accuracy", pp[f"{a}_correct"], pp[f"{b}_correct"], ident)

df = pd.DataFrame(rows)
df.to_csv(R / "informativeness_audit.csv", index=False)
(R / "informativeness_audit.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
with pd.option_context("display.width", 220, "display.max_rows", None):
    print(df.round(4).to_string(index=False))
