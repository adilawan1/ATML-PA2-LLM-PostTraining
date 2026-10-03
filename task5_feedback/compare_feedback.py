"""Task 5 final comparison: combine in-domain (GSM8K), controlled-diagnostic and transfer (SVAMP) results.

Reads results/task5_feedback/{gsm,transfer}/metrics.json and diagnostics/summary.json and writes
results/task5_feedback/feedback_comparison.{json,csv}:
  - per policy: exact accuracy, format compliance, mean/std length and failure types in each domain, and the
    transfer drop (transfer minus in-domain) for accuracy and format compliance;
  - per comparison (RLVR/RLAIF vs SFT, RLVR vs RLAIF): AI pairwise win rate, ties and verifier-judge agreement in
    each domain, and the win-rate drop;
  - diagnostic sensitivities (S_reason, S_outcome, filler susceptibility, distractor robustness) per mechanism;
  - feedback cost: measured judge seconds per comparison, comparisons needed per K-sample group for RLAIF
    (K choose 2), versus a regex verifier call per sample.
"""
from __future__ import annotations

import argparse
import json
from math import comb

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import save_json


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, transfer = load(root / "gsm" / "metrics.json"), load(root / "transfer" / "metrics.json")
    diag = load(root / "diagnostics" / "summary.json")
    missing = [n for n, v in (("gsm", gsm), ("transfer", transfer), ("diagnostics", diag)) if v is None]
    if missing:
        print(f"Missing inputs: {missing}; the comparison below covers what is available.", flush=True)

    policy_rows = []
    for p in ("sft", "rlvr", "rlaif"):
        row = {"policy": p}
        for name, m in (("gsm", gsm), ("transfer", transfer)):
            if m is None:
                continue
            pm = m["policies"][p]
            row.update({
                f"{name}_exact_accuracy": pm["exact_accuracy"], f"{name}_format_compliance": pm["format_compliance"],
                f"{name}_len_mean": pm["response_tokens"]["mean"], f"{name}_len_std": pm["response_tokens"]["std"],
                **{f"{name}_fail_{k}": v for k, v in pm["failure_types"].items()},
            })
        if gsm and transfer:
            row["accuracy_drop_transfer_minus_gsm"] = row["transfer_exact_accuracy"] - row["gsm_exact_accuracy"]
            row["format_drop_transfer_minus_gsm"] = row["transfer_format_compliance"] - row["gsm_format_compliance"]
        policy_rows.append(row)

    pair_rows = []
    keys = sorted({k for m in (gsm, transfer) if m and "pairwise" in m for k in m["pairwise"]})
    for k in keys:
        row = {"comparison": k}
        for name, m in (("gsm", gsm), ("transfer", transfer)):
            if m and "pairwise" in m and k in m["pairwise"]:
                pw = m["pairwise"][k]
                row.update({f"{name}_win_rate": pw["win_rate"], f"{name}_ties": pw["ties"],
                            f"{name}_agree_3way": pw["verifier_judge_agreement_3way"],
                            f"{name}_agree_on_decisive": pw["judge_agrees_on_decisive"],
                            f"{name}_judge_decisive_when_verifier_tied": pw["judge_decisive_when_verifier_tied"]})
        if "gsm_win_rate" in row and "transfer_win_rate" in row:
            row["win_rate_drop_transfer_minus_gsm"] = row["transfer_win_rate"] - row["gsm_win_rate"]
        pair_rows.append(row)

    k = int(load_yaml("configs/grpo.yaml")["num_generations"])
    costs = [m["judge_cost"] for m in (gsm, transfer) if m and m.get("judge_cost", {}).get("seconds_per_call")]
    cost = {
        "judge_model": cfg["ai_judge_model"],
        "measured_seconds_per_judge_comparison": (sum(c["seconds"] for c in costs) / sum(c["judge_calls"] for c in costs)) if costs else None,
        "rlaif_comparisons_per_group": comb(k, 2), "group_size_K": k,
        "rlvr_cost": "one regex parse per sample (negligible, no model call)",
    }

    out = {
        "policies": policy_rows,
        "pairwise": pair_rows,
        "diagnostics": {key: diag[key] for key in ("S_reason", "S_outcome", "filler_susceptibility_prefers_filler", "distractor_robustness")} if diag else None,
        "diagnostic_pair_summary": diag["pair_summary"] if diag else None,
        "cost": cost,
    }
    save_json(root / "feedback_comparison.json", out)
    pd.DataFrame(policy_rows).to_csv(root / "feedback_comparison_policies.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(root / "feedback_comparison_pairwise.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(pd.DataFrame(policy_rows).T.to_string())
        if pair_rows:
            print(pd.DataFrame(pair_rows).T.to_string())
    print(json.dumps({"diagnostics": out["diagnostics"], "cost": cost}, indent=2))


if __name__ == "__main__":
    main()
