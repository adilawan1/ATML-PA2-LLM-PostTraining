from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


POLICIES = ["sft", "dpo", "ppo", "grpo"]
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    # Blinded labelling sheet: the fixed IDs joined to every available policy's response, in a seeded random
    # order, with neither the policy name nor any AI label shown. The key file maps rows back to policies.
    sheet_path, key_path = outdir / "manual_audit_sheet.csv", outdir / "manual_audit_key.csv"
    if sheet_path.is_file():
        existing = pd.read_csv(sheet_path, keep_default_na=False)
        if (existing["manual_label"].astype(str).str.strip() != "").any():
            print(f"{sheet_path} already contains manual labels; not overwriting it.")
            return
    rows = []
    for policy in POLICIES:
        path = outdir / f"generated_{policy}.jsonl"
        if not path.is_file():
            print(f"  (no responses for {policy} yet; it is not in the sheet - re-run after generating it)")
            continue
        by_id = {int(r["xstest_id"]): r for r in read_jsonl(path)}
        rows += [{"xstest_id": i, "policy": policy, "prompt": by_id[i]["prompt"], "response": by_id[i]["response"]} for i in ids]
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(rows))
    rows = [dict(rows[j], audit_row=n + 1) for n, j in enumerate(order)]
    pd.DataFrame([{"audit_row": r["audit_row"], "xstest_id": r["xstest_id"], "prompt": r["prompt"], "response": r["response"],
                   "manual_label": "", "manual_note": ""} for r in rows]).to_csv(sheet_path, index=False)
    pd.DataFrame([{"audit_row": r["audit_row"], "xstest_id": r["xstest_id"], "policy": r["policy"]} for r in rows]).to_csv(key_path, index=False)
    print(f"Wrote blinded sheet ({len(rows)} rows) to {sheet_path} and its key to {key_path}.")
    print("Fill manual_label with one of", LABELS, "without opening the judged_*.jsonl files first.")


if __name__ == "__main__":
    main()
