"""Task 5 Step 2: reward sensitivity and robustness on the supplied Controlled Reward Diagnostic Set.

For every problem the five variants are scored by both feedback mechanisms:
  - exact verifier: exact_reward(response, gold_final) in {0, 1}
  - AI judge: all C(5,2) = 10 pairwise comparisons per problem in the fixed variant order below (the earlier variant
    is candidate A; the judge's own hash-based orientation balancing still applies), aggregated per variant into the
    RLAIF normalized win rate (wins + 0.5 ties) / (5 - 1), as in PairwiseAIJudge.group_rewards.

Controlled pairs (diagnostically better response first; each perturbation is compared with clean_correct):
  reasoning  : clean_correct  vs corrupt_reasoning_correct_final  (same correct final, reasoning degraded) -> S_reason
  outcome    : clean_correct  vs good_reasoning_wrong_final       (reasoning ~fixed, designated final changed) -> S_outcome
  filler     : clean_correct  vs persuasive_filler_correct        (same content + irrelevant persuasive filler)
  distractor : clean_correct  vs gold_distractor_wrong_final      (gold number mentioned, wrong designated final)
For each pair and mechanism: better-response rate (prefers the first), tie rate, wrong-preference rate. A verifier tie
on the reasoning and filler pairs is the expected invariance, not an error. Per-variant scores (verifier reward,
judge win rate) cover all five categories. The verifier's scores are also checked against `expected_exact_reward`.

`--verifier-only` needs no GPU. Writes results/task5_feedback/diagnostics/{pairs.csv,variant_scores.csv,
pair_summary.csv,summary.json,judge_comparisons.jsonl}.
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}
ORDER = [
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
]
PAIRS = {
    "reasoning": ("clean_correct", "corrupt_reasoning_correct_final"),
    "outcome": ("clean_correct", "good_reasoning_wrong_final"),
    "filler": ("clean_correct", "persuasive_filler_correct"),
    "distractor": ("clean_correct", "gold_distractor_wrong_final"),
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome_of(pref: int) -> str:
    return {1: "better", 0: "tie", -1: "wrong"}[pref]


def summarize(pairs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mech, pair), sub in pairs_df.groupby(["mechanism", "pair"]):
        n = len(sub)
        rows.append({"mechanism": mech, "pair": pair, "n": n,
                     "better_rate": float((sub["outcome"] == "better").mean()),
                     "tie_rate": float((sub["outcome"] == "tie").mean()),
                     "wrong_preference_rate": float((sub["outcome"] == "wrong").mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--verifier-only", action="store_true", help="skip the AI judge (no GPU needed)")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    ver = {pid: {v: exact_reward(g[v]["response"], g[v]["gold_final"]) for v in ORDER} for pid, g in groups.items()}
    mismatches = [{"problem_id": pid, "variant": v, "verifier": ver[pid][v], "expected": float(groups[pid][v]["expected_exact_reward"])}
                  for pid in groups for v in ORDER if ver[pid][v] != float(groups[pid][v]["expected_exact_reward"])]

    judge_res = {}
    if not args.verifier_only:
        judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
        records = []
        for pid, g in groups.items():
            question = g["clean_correct"]["question"]
            for va, vb in itertools.combinations(ORDER, 2):
                pref = judge.compare(question, g[va]["response"], g[vb]["response"])
                judge_res[(pid, va, vb)] = pref
                records.append({"problem_id": pid, "variant_a": va, "variant_b": vb, "judge": pref})
        pd.DataFrame(records).to_json(outdir / "judge_comparisons.jsonl", orient="records", lines=True)

    pair_rows = []
    for pid in groups:
        for name, (good, bad) in PAIRS.items():
            v = int((ver[pid][good] > ver[pid][bad]) - (ver[pid][good] < ver[pid][bad]))
            pair_rows.append({"problem_id": pid, "pair": name, "better": good, "worse": bad, "mechanism": "verifier",
                              "score_better": ver[pid][good], "score_worse": ver[pid][bad], "outcome": outcome_of(v)})
            if judge_res:
                j = {"A": 1, "TIE": 0, "B": -1}[judge_res[(pid, good, bad)]]  # good always precedes bad in ORDER
                pair_rows.append({"problem_id": pid, "pair": name, "better": good, "worse": bad, "mechanism": "ai_judge",
                                  "score_better": None, "score_worse": None, "outcome": outcome_of(j)})
    pairs_df = pd.DataFrame(pair_rows)
    pair_summary = summarize(pairs_df)

    variant_rows = []
    for pid in groups:
        wins, ties = defaultdict(float), defaultdict(float)
        for (p, va, vb), pref in judge_res.items():
            if p != pid:
                continue
            if pref == "A":
                wins[va] += 1
            elif pref == "B":
                wins[vb] += 1
            else:
                ties[va] += 1
                ties[vb] += 1
        for v in ORDER:
            variant_rows.append({"problem_id": pid, "variant": v, "verifier_reward": ver[pid][v],
                                 "judge_win_rate": (wins[v] + 0.5 * ties[v]) / (len(ORDER) - 1) if judge_res else None})
    variants_df = pd.DataFrame(variant_rows)
    variant_summary = variants_df.groupby("variant").agg(verifier_reward_mean=("verifier_reward", "mean"),
                                                         judge_win_rate_mean=("judge_win_rate", "mean")).reindex(ORDER).reset_index()

    def rate(mech, pair, what):
        r = pair_summary[(pair_summary["mechanism"] == mech) & (pair_summary["pair"] == pair)]
        return float(r[what].iloc[0]) if len(r) else None

    summary = {
        "n_problems": len(groups),
        "verifier_expected_reward_mismatches": mismatches,
        "S_reason": {m: rate(m, "reasoning", "better_rate") for m in ("verifier", "ai_judge")},
        "S_outcome": {m: rate(m, "outcome", "better_rate") for m in ("verifier", "ai_judge")},
        "filler_susceptibility_prefers_filler": {m: rate(m, "filler", "wrong_preference_rate") for m in ("verifier", "ai_judge")},
        "distractor_robustness": {m: rate(m, "distractor", "better_rate") for m in ("verifier", "ai_judge")},
        "pair_summary": pair_summary.to_dict(orient="records"),
        "variant_summary": variant_summary.to_dict(orient="records"),
        "judge_included": bool(judge_res),
        "pair_definitions": {k: {"better": a, "worse": b} for k, (a, b) in PAIRS.items()},
    }
    pairs_df.to_csv(outdir / "pairs.csv", index=False)
    variants_df.to_csv(outdir / "variant_scores.csv", index=False)
    pair_summary.to_csv(outdir / "pair_summary.csv", index=False)
    save_json(outdir / "summary.json", summary)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pair_summary.to_string(index=False))
        print(variant_summary.to_string(index=False))
    print("S_reason:", summary["S_reason"], "S_outcome:", summary["S_outcome"])
    print("verifier vs expected_exact_reward mismatches:", len(mismatches))


if __name__ == "__main__":
    main()
