"""Task 3 Step 2: group-size study on the supplied completion/reward cache (no policy training).

Equal-generation regrouping rule (used for every K):
  The cache has 8 completions per prompt (generation_index 0..7) for P prompts, i.e. a fixed budget of
  8P generations. For each K in {2, 4, 8}, every prompt's completions, in generation_index order, are split
  into 8/K disjoint consecutive groups of size K (K=2: {0,1},{2,3},...; K=4: {0..3},{4..7}; K=8: {0..7}).
  Every condition therefore uses exactly the same 8P completions and rewards; only the grouping changes.
  As a robustness check that does not depend on which indices happen to be adjacent, the same statistics are
  also averaged over *all* C(8, K) size-K subsets of each prompt's completions ("all_subsets").

Definitions:
  - group reward std: population std (unbiased=False), as in task3_grpo.grpo.group_relative_advantages
  - uninformative group: std <= tol, tol = 1e-6 (the eps of the released advantage helper). Because the
    reward model is continuous, the script also reports the same rate at looser tolerances as a clearly
    labelled sensitivity analysis.
  - group-relative signal: A = (r - mean_group) / (std_group + eps) from the released helper. Reported:
    variance of A over all completions, variance of the centred reward (r - mean_group), and
    "sign agreement": fraction of completions whose advantage sign under K matches the sign of
    r - mean over all 8 completions of that prompt (how faithfully a K-group ranks a completion
    relative to the prompt's own reward distribution).
  - prompt difficulty: mean reward of the prompt's 8 cached completions; prompts are split into tertile
    bins hard / medium / easy (lowest third of mean reward = hard). Defined once, used for every K.

Writes results/task3_grpo/group_size_{summary,by_difficulty,per_prompt}.{csv,json}.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

TOL = 1e-6
SENSITIVITY_TOLS = [0.05, 0.10, 0.25]
EPS = 1e-6


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's first 8 completions (generation_index order) are cut into 8/k disjoint consecutive groups,
    so every K uses the identical set of 8 * n_prompts completions.
    """
    if 8 % k:
        raise ValueError(f"K={k} must divide 8")
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        for g in range(8 // k):
            groups.append({"prompt": pid, "group": g, "rows": rows[g * k : (g + 1) * k]})
    return groups


def all_subset_groups(by_prompt, k: int):
    return [
        {"prompt": pid, "group": idx, "rows": [rows[i] for i in subset]}
        for pid, rows in by_prompt.items()
        for idx, subset in enumerate(combinations(range(8), k))
    ]


def group_stats(groups, prompt_mean8):
    rewards = torch.tensor([float(r["reward"]) for g in groups for r in g["rows"]], dtype=torch.float64)
    gids = torch.tensor([i for i, g in enumerate(groups) for _ in g["rows"]])
    adv = group_relative_advantages(rewards, gids, eps=EPS).double()
    stds = np.array([np.std([float(r["reward"]) for r in g["rows"]]) for g in groups])
    means = np.array([np.mean([float(r["reward"]) for r in g["rows"]]) for g in groups])
    centred = rewards.numpy() - means[gids.numpy()]
    ref_sign = np.sign(np.array([float(r["reward"]) - prompt_mean8[g["prompt"]] for g in groups for r in g["rows"]]))
    adv_sign = np.sign(adv.numpy())
    nonzero_ref = ref_sign != 0
    out = {
        "n_groups": len(groups),
        "n_completions": int(rewards.numel()),
        "informative_rate": float((stds > TOL).mean()),
        "uninformative_rate": float((stds <= TOL).mean()),
        "mean_within_group_std": float(stds.mean()),
        "median_within_group_std": float(np.median(stds)),
        "adv_variance": float(adv.var(unbiased=False)),
        "adv_mean_abs": float(adv.abs().mean()),
        "centred_reward_variance": float(centred.var()),
        "sign_agreement_with_k8": float((adv_sign[nonzero_ref] == ref_sign[nonzero_ref]).mean()) if nonzero_ref.any() else float("nan"),
    }
    for tol in SENSITIVITY_TOLS:
        out[f"uninformative_rate_tol{tol:g}"] = float((stds <= tol).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    results_dir = repo_path(cfg["results_dir"])
    ks = [int(k) for k in cfg["group_sizes"]]

    prompt_mean8 = {pid: float(np.mean([float(r["reward"]) for r in rows[:8]])) for pid, rows in by_prompt.items()}
    prompt_std8 = {pid: float(np.std([float(r["reward"]) for r in rows[:8]])) for pid, rows in by_prompt.items()}
    pids = sorted(by_prompt, key=lambda p: prompt_mean8[p])
    t1, t2 = np.quantile([prompt_mean8[p] for p in pids], [1 / 3, 2 / 3])
    difficulty = {p: ("hard" if prompt_mean8[p] <= t1 else "medium" if prompt_mean8[p] <= t2 else "easy") for p in pids}
    bins = ["hard", "medium", "easy"]

    summary, by_diff, per_prompt = [], [], []
    for k in ks:
        for scheme, builder in (("disjoint", regroup_equal_generation_budget), ("all_subsets", all_subset_groups)):
            groups = builder(by_prompt, k)
            summary.append({"K": k, "scheme": scheme, "prompts_equivalent": len(groups) * k // 8, **group_stats(groups, prompt_mean8)})
            for b in bins:
                sub = [g for g in groups if difficulty[g["prompt"]] == b]
                by_diff.append({"K": k, "scheme": scheme, "difficulty": b,
                                "n_prompts": sum(difficulty[p] == b for p in pids), **group_stats(sub, prompt_mean8)})
            if scheme == "disjoint":
                for p in pids:
                    sub = [g for g in groups if g["prompt"] == p]
                    stds = [np.std([float(r["reward"]) for r in g["rows"]]) for g in sub]
                    per_prompt.append({
                        "K": k, "source_index": p, "prompt_id": by_prompt[p][0].get("prompt_id"),
                        "difficulty": difficulty[p], "prompt_mean_reward_k8": prompt_mean8[p], "prompt_std_k8": prompt_std8[p],
                        "group_stds": [float(s) for s in stds],
                        "informative_groups": int(sum(s > TOL for s in stds)), "n_groups": len(stds),
                        "truncated_completions": int(sum(bool(r.get("clipped_at_max")) for r in by_prompt[p][:8])),
                    })

    meta = {
        "cache": cfg["group_cache"],
        "n_prompts": len(by_prompt),
        "total_generations": 8 * len(by_prompt),
        "tolerance": TOL,
        "sensitivity_tolerances": SENSITIVITY_TOLS,
        "difficulty_rule": "tertiles of per-prompt mean reward over its 8 cached completions (lowest third = hard)",
        "difficulty_thresholds": [float(t1), float(t2)],
        "difficulty_bins": {b: [p for p in pids if difficulty[p] == b] for b in bins},
        "regrouping_rule": "per prompt, consecutive disjoint groups of size K in generation_index order; all_subsets = average over all C(8,K) subsets",
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("group_size_summary", summary), ("group_size_by_difficulty", by_diff), ("group_size_per_prompt", per_prompt)):
        pd.DataFrame(rows).to_csv(results_dir / f"{name}.csv", index=False)
        save_json(results_dir / f"{name}.json", {"meta": meta, "rows": rows})

    cols = ["K", "scheme", "n_groups", "informative_rate", "mean_within_group_std", "adv_variance", "centred_reward_variance", "sign_agreement_with_k8", "uninformative_rate_tol0.1"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pd.DataFrame(summary)[cols].to_string(index=False))
        print()
        print(pd.DataFrame(by_diff)[["K", "scheme", "difficulty"] + cols[2:]].to_string(index=False))


if __name__ == "__main__":
    main()
