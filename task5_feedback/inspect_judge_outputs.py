"""Task 5 audit of the fixed pairwise AI judge: what text does it actually emit?

The supplied PairwiseAIJudge decodes 4 tokens, searches them for A / B / TIE and falls back to TIE when nothing
matches, and its cache stores only the parsed verdict. This script re-runs the exact same judge model, rubric,
orientation balancing and greedy 4-token decoding on the 80 controlled diagnostic pairs (clean_correct vs each
perturbation) and records the raw decoded text, whether the verdict was parsed or came from the TIE fallback, and a
longer 32-token greedy continuation for context. It does not touch the judge caches or any reported metric.

Writes results/task5_feedback/diagnostics/judge_raw_outputs.jsonl and judge_raw_summary.json.
"""
from __future__ import annotations

import argparse
import collections
import re

import torch

from common.data import load_yaml, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.score_perturbations import PAIRS, load_diagnostic_groups


@torch.no_grad()
def raw_compare(judge: PairwiseAIJudge, problem: str, a: str, b: str) -> dict:
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], return_tensors="pt", add_generation_prompt=True
    ).to(next(judge.model.parameters()).device)
    out = {}
    for n in (4, 32):
        gen = judge.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                                   pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id)
        out[n] = judge.tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
    decoded = out[4].strip().upper()
    m = re.search(r"\b(A|B|TIE)\b", decoded)
    parsed = m.group(1) if m else None
    verdict = parsed or "TIE"
    if swap:
        verdict = {"A": "B", "B": "A", "TIE": "TIE"}[verdict]
    return {"swap": swap, "raw_4": out[4], "raw_32": out[32], "parsed_token": parsed,
            "tie_fallback": parsed is None, "verdict": verdict, "prompt_tokens": int(ids.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "diagnostics"
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    # Scratch cache path: the judge object is used only for its model/tokenizer/key, never for cached verdicts.
    judge = PairwiseAIJudge(cfg, outdir / "_inspect_scratch_cache.json")
    rows = []
    for pid, g in groups.items():
        for pair, (better, worse) in PAIRS.items():
            r = raw_compare(judge, g["clean_correct"]["question"], g[better]["response"], g[worse]["response"])
            rows.append({"problem_id": pid, "pair": pair, **r})
        print(f"[inspect] problem {pid} done", flush=True)
    write_jsonl(outdir / "judge_raw_outputs.jsonl", rows)
    summary = {
        "n": len(rows),
        "tie_fallback_rate": sum(r["tie_fallback"] for r in rows) / len(rows),
        "parsed_token_counts": dict(collections.Counter(str(r["parsed_token"]) for r in rows)),
        "most_common_raw_4": collections.Counter(r["raw_4"].strip() for r in rows).most_common(15),
        "by_pair": {p: {"tie_fallback_rate": sum(r["tie_fallback"] for r in rows if r["pair"] == p) / sum(r["pair"] == p for r in rows),
                        "verdicts": dict(collections.Counter(r["verdict"] for r in rows if r["pair"] == p))}
                    for p in PAIRS},
    }
    save_json(outdir / "judge_raw_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
