"""Task 1 Step 3: length-confounding study.

1. Dataset diagnostics (a property of the data, not the policy): how often the preferred response is longer,
   and the chosen-minus-rejected token difference, for the standard and length-balanced training sets.
2. Trains the length-balanced DPO model from the original initialization (same budget/hyperparameters as the
   standard run: one epoch over its 1500 pairs), then evaluates it with the common protocol.
3. Compares standard vs length-balanced DPO on per-stratum held-out preference accuracy
   (preferred_longer / length_matched / rejected_longer), generated response length, and word-limit compliance.

Requires the standard run to be trained (python -m task1_dpo.train --run-name standard). The standard
evaluation is run here if it is missing. Writes results/task1_dpo/summary_length.{json,csv} and
results/task1_dpo/dataset_length_stats.json.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.checkpointing import is_complete
from common.data import load_yaml, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_tokenizer
from task1_dpo.evaluate import evaluate_adapter, length_stats
from task1_dpo.train import run_training

STRATA = ["preferred_longer", "length_matched", "rejected_longer"]


def dataset_length_stats(cfg, path, tokenizer) -> dict:
    rows = read_jsonl(path)
    diffs, chosen, rejected = [], [], []
    for row in rows:
        yc, yr = preference_responses(row)
        lc = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        lr = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        chosen.append(lc)
        rejected.append(lr)
        diffs.append(lc - lr)
    d = np.asarray(diffs)
    out = {
        "file": str(path),
        "n_pairs": len(rows),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_rejected_longer": float((d < 0).mean()),
        "frac_equal": float((d == 0).mean()),
        "chosen_tokens": length_stats(chosen),
        "rejected_tokens": length_stats(rejected),
        "chosen_minus_rejected_tokens": length_stats(diffs),
    }
    if rows and "length_stratum" in rows[0]:
        out["strata_counts"] = {s: sum(r["length_stratum"] == s for r in rows) for s in STRATA}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--max-gen-prompts", type=int, help="smoke tests only")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    tokenizer = load_tokenizer(cfg["base_model"])
    data_stats = {
        "standard_train": dataset_length_stats(cfg, cfg["paths"]["dpo_standard_train"], tokenizer),
        "length_balanced_train": dataset_length_stats(cfg, cfg["paths"]["dpo_length_train"], tokenizer),
        "standard_heldout": dataset_length_stats(cfg, cfg["paths"]["dpo_standard_eval"], tokenizer),
        "length_stratified_heldout": dataset_length_stats(cfg, cfg["paths"]["dpo_length_eval"], tokenizer),
    }
    save_json(results_dir / "dataset_length_stats.json", data_stats)
    for k, v in data_stats.items():
        print(f"[data] {k}: chosen longer {v['frac_chosen_longer']:.3f}, mean diff {v['chosen_minus_rejected_tokens']['mean']:.1f} tokens", flush=True)

    standard_out = repo_path(cfg["standard_output"])
    if not is_complete(standard_out):
        raise SystemExit(f"Standard DPO adapter not found at {standard_out}; run task1_dpo.train --run-name standard first.")

    balanced_out = repo_path(cfg["length_output"])
    run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"], output_path=str(balanced_out))
    clear_gpu()

    metrics = {}
    for name, adapter in (("standard", standard_out), ("length_balanced", balanced_out)):
        p = results_dir / f"eval_{name}" / "metrics.json"
        if p.is_file() and not args.max_gen_prompts:
            metrics[name] = load_json(p)
        else:
            metrics[name] = evaluate_adapter(args.config, str(adapter), name, max_gen_prompts=args.max_gen_prompts)
            clear_gpu()

    rows = []
    for name, m in metrics.items():
        row = {"model": name}
        for stratum in ["all"] + STRATA:
            s = m["stratified_pairs"].get(stratum, {})
            row[f"acc_{stratum}"] = s.get("preference_accuracy")
            row[f"margin_{stratum}"] = s.get("pref_margin_mean")
            row[f"n_{stratum}"] = s.get("n_pairs")
        L = m["generation"]["response_length"]
        row.update({
            "heldout_pref_acc": m["heldout_pairs"]["preference_accuracy"],
            "reward_mean": m["generation"]["reward_mean"],
            "kl_token_mean": m["generation"]["kl_token_mean"],
            "len_mean": L["mean"], "len_std": L["std"], "len_median": L["median"], "len_iqr": L["iqr"],
            "truncation_rate": m["generation"]["truncation_rate"],
            "word_limit_compliance": m["word_limit"]["compliance_rate"],
            "word_limit_words_mean": m["word_limit"]["words"]["mean"],
            "word_limit_words_std": m["word_limit"]["words"]["std"],
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "summary_length.csv", index=False)
    save_json(results_dir / "summary_length.json", {"models": df.to_dict(orient="records"), "dataset": data_stats})
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.T.to_string())


if __name__ == "__main__":
    main()
