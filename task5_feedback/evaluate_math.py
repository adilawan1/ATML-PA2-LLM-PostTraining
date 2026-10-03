"""Task 5 Steps 1 and 3: SFT vs RLVR vs RLAIF on the fixed GSM8K subset (--dataset gsm) or the fixed SVAMP
transfer subset (--dataset transfer).

1. One greedy response per problem and policy (identical decoding, math_max_new_tokens) using each row's fixed
   `messages` prompt -> results/task5_feedback/<dataset>/generated_<policy>.jsonl (skipped if complete).
2. Exact verifier: accuracy (designated `#### <number>` matches gold), format compliance (a designated final is
   parsed), response length, and failure type per response: correct / wrong_final / no_final_format /
   truncated_no_final.
3. Fixed pairwise AI judge (task5_feedback.rlaif.PairwiseAIJudge, cached on disk): RLVR vs SFT and RLAIF vs SFT
   (win 1, tie 0.5, loss 0), plus RLVR vs RLAIF head-to-head; tie counts reported separately.
4. Verifier-judge agreement on each pair: the verifier prefers the correct response when exactly one is correct
   and ties otherwise; reported as a 3x3 table, agreement on verifier-decisive pairs, and how often the judge
   separates responses the verifier ties.
Writes results/task5_feedback/<dataset>/{metrics.json,pairwise.jsonl,per_problem.csv}.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import append_jsonl, load_json, save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

POLICIES = ["sft", "rlvr", "rlaif"]
COMPARISONS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def length_stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(med), "iqr": float(q3 - q1), "n": int(a.size)}


def generate_all(cfg, rows, tokenizer, outdir, policies):
    tokenizer.truncation_side = "left"
    bs = int(cfg.get("math_batch_size", 16))
    for name in policies:
        path = outdir / f"generated_{name}.jsonl"
        if path.is_file() and len(read_jsonl(path)) == len(rows):
            print(f"[math] {name}: {path.name} complete; skipping generation", flush=True)
            continue
        adapter = policy_specs(cfg)[name]
        if adapter is not None and not repo_path(adapter).is_dir():
            raise SystemExit(f"[math] adapter for {name} not found at {adapter}")
        model = load_frozen_policy(cfg, name)
        records = []
        for start in range(0, len(rows), bs):
            chunk = rows[start:start + bs]
            gen = batch_generate(
                model, tokenizer, [r["messages"] for r in chunk],
                max_prompt_length=int(cfg.get("math_max_prompt_length", 512)),
                max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False,
            )
            for row, resp, n, trunc in zip(chunk, gen["responses"], gen["response_lengths"], gen["truncated"]):
                records.append({"source_index": str(row["source_index"]), "policy": name, "response": resp,
                                "response_tokens": int(n), "truncated": bool(trunc)})
            print(f"[math] {name}: {min(start + bs, len(rows))}/{len(rows)}", flush=True)
        write_jsonl(path, records)
        del model
        clear_gpu()


def failure_type(response, gold, truncated):
    if extract_designated_final(response) is None:
        return "truncated_no_final" if truncated else "no_final_format"
    return "correct" if exact_reward(response, gold) == 1.0 else "wrong_final"


def judge_pairs(cfg, rows, gens, outdir):
    judge = PairwiseAIJudge(cfg, outdir / "judge_cache.json")
    question = {str(r["source_index"]): r["question"] for r in rows}
    pair_path = outdir / "pairwise.jsonl"
    done = {(r["policy_a"], r["policy_b"], r["source_index"]) for r in read_jsonl(pair_path)} if pair_path.is_file() else set()
    t0, calls = time.perf_counter(), 0
    for a, b in COMPARISONS:
        for row in rows:
            sid = str(row["source_index"])
            if (a, b, sid) in done:
                continue
            pref = judge.compare(question[sid], gens[a][sid]["response"], gens[b][sid]["response"])
            calls += 1
            append_jsonl(pair_path, {"policy_a": a, "policy_b": b, "source_index": sid, "judge": pref})
        print(f"[math] judged {a} vs {b}", flush=True)
    # Accumulate measured judge time across sessions so a resumed run keeps the cost of earlier calls.
    cost_path = outdir / "judge_cost.json"
    prev = load_json(cost_path) if cost_path.is_file() else {"judge_calls": 0, "seconds": 0.0}
    total = {"judge_calls": prev["judge_calls"] + calls, "seconds": prev["seconds"] + ((time.perf_counter() - t0) if calls else 0.0)}
    save_json(cost_path, total)
    pairs = pd.DataFrame(read_jsonl(pair_path)).drop_duplicates(["policy_a", "policy_b", "source_index"], keep="last")
    return pairs, {"judge_calls": total["judge_calls"], "seconds": total["seconds"],
                   "seconds_per_call": total["seconds"] / total["judge_calls"] if total["judge_calls"] else None,
                   "total_cached_comparisons": len(judge.cache)}


def pairwise_metrics(pairs, df):
    correct = {p: dict(zip(df["source_index"], df[f"{p}_correct"])) for p in POLICIES}
    out = {}
    for a, b in COMPARISONS:
        sub = pairs[(pairs["policy_a"] == a) & (pairs["policy_b"] == b)]
        score = sub["judge"].map({"A": 1.0, "TIE": 0.5, "B": 0.0})
        ver = np.array([int(np.sign(correct[a][s] - correct[b][s])) for s in sub["source_index"]])
        jud = sub["judge"].map({"A": 1, "TIE": 0, "B": -1}).to_numpy()
        decisive = ver != 0
        table = pd.crosstab(pd.Categorical(ver, [1, 0, -1]), pd.Categorical(jud, [1, 0, -1]),
                            rownames=["verifier"], colnames=["judge"], dropna=False)
        out[f"{a}_vs_{b}"] = {
            "n": int(len(sub)),
            "win_rate": float(score.mean()),
            "wins": int((sub["judge"] == "A").sum()), "ties": int((sub["judge"] == "TIE").sum()),
            "losses": int((sub["judge"] == "B").sum()),
            "verifier_judge_agreement_3way": float((ver == jud).mean()),
            "decisive_pairs": int(decisive.sum()),
            "judge_agrees_on_decisive": float((ver[decisive] == jud[decisive]).mean()) if decisive.any() else None,
            "verifier_tied_pairs": int((~decisive).sum()),
            "judge_decisive_when_verifier_tied": float((jud[~decisive] != 0).mean()) if (~decisive).any() else None,
            "table_verifier_rows_judge_cols": {str(k): {str(c): int(v) for c, v in r.items()} for k, r in table.iterrows()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--skip-judge", action="store_true", help="exact-verifier metrics only")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)

    generate_all(cfg, rows, tokenizer, outdir, POLICIES)
    gens = {p: {str(r["source_index"]): r for r in read_jsonl(outdir / f"generated_{p}.jsonl")} for p in POLICIES}

    per_problem = []
    for row in rows:
        sid = str(row["source_index"])
        rec = {"source_index": sid, "gold_final": row["gold_final"]}
        for p in POLICIES:
            g = gens[p][sid]
            rec[f"{p}_correct"] = exact_reward(g["response"], row["gold_final"])
            rec[f"{p}_pred"] = extract_designated_final(g["response"])
            rec[f"{p}_format_ok"] = rec[f"{p}_pred"] is not None
            rec[f"{p}_tokens"] = g["response_tokens"]
            rec[f"{p}_failure_type"] = failure_type(g["response"], row["gold_final"], g["truncated"])
        per_problem.append(rec)
    df = pd.DataFrame(per_problem)

    metrics = {"dataset": args.dataset, "n_problems": len(rows), "policies": {}}
    for p in POLICIES:
        metrics["policies"][p] = {
            "exact_accuracy": float(df[f"{p}_correct"].mean()),
            "format_compliance": float(df[f"{p}_format_ok"].mean()),
            "response_tokens": length_stats(df[f"{p}_tokens"]),
            "failure_types": {k: int(v) for k, v in df[f"{p}_failure_type"].value_counts().items()},
        }
    if not args.skip_judge:
        pairs, cost = judge_pairs(cfg, rows, gens, outdir)
        metrics["pairwise"] = pairwise_metrics(pairs, df)
        metrics["judge_cost"] = cost
    metrics["protocol"] = {"decoding": "greedy", "max_new_tokens": int(cfg["math_max_new_tokens"]),
                           "data_file": dataset_path(cfg, args.dataset), "judge_model": cfg["ai_judge_model"],
                           "policies": policy_specs(cfg), "win_rate": "win 1, tie 0.5, loss 0 for policy_a vs policy_b"}
    df.to_csv(outdir / "per_problem.csv", index=False)
    save_json(outdir / "metrics.json", metrics)
    for p in POLICIES:
        m = metrics["policies"][p]
        print(f"[math] {args.dataset} {p}: acc={m['exact_accuracy']:.3f} format={m['format_compliance']:.3f} "
              f"len={m['response_tokens']['mean']:.0f}", flush=True)
    for k, v in metrics.get("pairwise", {}).items():
        print(f"[math] {k}: win_rate={v['win_rate']:.3f} (W/T/L {v['wins']}/{v['ties']}/{v['losses']}) "
              f"agree_on_decisive={v['judge_agrees_on_decisive']}", flush=True)


if __name__ == "__main__":
    main()
