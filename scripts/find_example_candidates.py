"""List candidate qualitative examples for the report (Tasks 1-2) by fixed, mechanical criteria.

The criteria only select where to look; whether a response is actually better or worse is the author's judgement
after reading it. Writes results/report_examples/{task1,task2}.md with full prompts/responses side by side.
    python -m scripts.find_example_candidates
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
OUT = R / "report_examples"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def block(title, rows):
    out = [f"### {title}\n"]
    for r in rows:
        out.append(f"#### {r['tag']}\n\n**Prompt:** {r['prompt']}\n")
        for name, resp, meta in r["sides"]:
            out.append(f"**{name}** ({meta}):\n\n```text\n{resp}\n```\n")
    return "\n".join(out)


def paired(task, a, b, key, la, lb):
    ga = {r[key]: r for r in jl(R / task / a / "generations.jsonl")}
    gb = {r[key]: r for r in jl(R / task / b / "generations.jsonl")}
    return [(ga[i], gb[i]) for i in ga if i in gb and ga[i]["response"] != gb[i]["response"]], la, lb


def meta(r):
    return f"reward {r['reward']:.2f}, {r['response_tokens']} tokens{', truncated' if r['truncated'] else ''}"


def side_rows(pairs, la, lb, n, sort_key, tag):
    pairs = sorted(pairs, key=sort_key)[:n]
    return [{"tag": f"{tag} #{k + 1}: reward {lb} {b['reward']:.2f} -> {la} {a['reward']:.2f}", "prompt": a["prompt"],
             "sides": [(lb, b["response"], meta(b)), (la, a["response"], meta(a))]} for k, (a, b) in enumerate(pairs)]


def task1():
    pairs, la, lb = paired("task1_dpo", "eval_standard", "eval_sft", "id", "DPO", "SFT")
    up = side_rows(pairs, la, lb, 4, lambda p: -(p[0]["reward"] - p[1]["reward"]), "Largest reward gain")
    trunc = side_rows([p for p in pairs if p[0]["truncated"] and p[0]["reward"] > p[1]["reward"]], la, lb, 3,
                      lambda p: -p[0]["reward"], "Reward up but DPO response hits the 256-token cap")
    wl = {}
    for name in ["eval_sft", "eval_standard", "eval_length_balanced"]:
        for r in jl(R / "task1_dpo" / name / "word_limit.jsonl"):
            wl.setdefault(r["id"], {})[name] = r
    viol = [v for v in wl.values() if v["eval_standard"]["compliant"] < 1 and v["eval_standard"]["reward"] >= v["eval_sft"]["reward"]]
    viol = sorted(viol, key=lambda v: -v["eval_standard"]["words"])[:4]
    wl_rows = [{"tag": f"Word-limit violation with reward >= SFT ({v['eval_standard']['id']})", "prompt": v["eval_standard"]["prompt"],
                "sides": [(n, v[k]["response"], f"reward {v[k]['reward']:.2f}, {v[k]['words']} words, compliant={int(v[k]['compliant'])}")
                          for n, k in [("SFT", "eval_sft"), ("DPO", "eval_standard"), ("Length-balanced", "eval_length_balanced")]]}
              for v in viol]
    return "# Task 1 example candidates\n\n" + "\n".join([
        block("(i) Preference/reward vs quality: largest reward gains of DPO over SFT (read: is DPO actually better?)", up),
        block("(i) Reward gain on a response cut off by the generation cap", trunc),
        block("(ii) Length / instruction compliance on the word-limit prompts", wl_rows)])


def task2():
    pairs, la, lb = paired("task2_ppo", "eval_standard", "eval_midpoint", "prompt_id", "PPO-20", "Midpoint")
    up = side_rows(pairs, la, lb, 3, lambda p: -(p[0]["reward"] - p[1]["reward"]), "Largest reward gain")
    trunc = side_rows([p for p in pairs if p[0]["truncated"] and p[0]["reward"] > p[1]["reward"]], la, lb, 3,
                      lambda p: -(p[0]["reward"] - p[1]["reward"]), "Reward up while the PPO response is truncated (no EOS)")
    kl0, la0, lb0 = paired("task2_ppo", "eval_fork_eps0.2_kl0", "eval_fork_eps0.2_kl0.1", "prompt_id", "beta_KL=0", "beta_KL=0.1")
    kl = side_rows(kl0, la0, lb0, 3, lambda p: -(p[0]["reward"] - p[1]["reward"]), "Weaker KL pressure, largest reward gain")
    return "# Task 2 example candidates\n\n" + "\n".join([
        block("Reward and quality: largest held-out reward gains after the 20-update continuation", up),
        block("Possible disagreement: reward rises on a truncated response", trunc),
        block("KL-pressure study: largest gains with beta_KL = 0 vs 0.1", kl)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("task1", task1), ("task2", task2)]:
        (OUT / f"{name}.md").write_text(fn(), encoding="utf-8")
        print("wrote", OUT / f"{name}.md")


if __name__ == "__main__":
    main()
