"""List candidate qualitative examples for the report (Tasks 1-5) by fixed, mechanical criteria.

The criteria only select where to look; whether a response is actually better or worse is the author's judgement
after reading it. Writes results/report_examples/task{1..5}.md with full prompts/responses side by side.
    python -m scripts.find_example_candidates
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
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


def task3():
    comp = jl(R / "task3_grpo" / "completions_standard.jsonl")
    groups = {}
    for c in comp:
        groups.setdefault((c["update"], c["group"]), []).append(c)
    rows = []
    for (u, g), cs in sorted(groups.items()):
        rewards = [c["reward"] for c in cs]
        zero_std = max(rewards) - min(rewards) < 1e-6
        all_masked = all(c["truncated"] for c in cs)
        if zero_std or all_masked:
            why = "zero reward spread (uninformative)" if zero_std else "all completions truncated -> masked from loss"
            rows.append({"tag": f"Update {u}: {why}", "prompt": f"prompt_id {cs[0]['prompt_id']}",
                         "sides": [(f"completion {k}", c["response"][:700] + (" [...]" if len(c["response"]) > 700 else ""),
                                    f"reward {c['reward']:.2f}, {c['response_tokens']} tokens, adv {c['advantage']:.2f}"
                                    f"{', truncated' if c['truncated'] else ''}") for k, c in enumerate(cs)]})
    off = [c for c in comp if sum(ch.isascii() for ch in c["response"]) < 0.5 * max(len(c["response"]), 1)]
    off_rows = [{"tag": f"Update {c['update']}: mostly non-ASCII (language drift?)", "prompt": f"prompt_id {c['prompt_id']}",
                 "sides": [("completion", c["response"][:600], f"reward {c['reward']:.2f}, advantage {c['advantage']:.2f}")]} for c in off[:3]]
    pairs, la, lb = paired("task3_grpo", "eval_fork_dr_grpo", "eval_fork_grpo", "prompt_id", "Dr. GRPO", "Canonical")
    longer = side_rows(pairs, la, lb, 3, lambda p: -(p[0]["response_tokens"] - p[1]["response_tokens"]), "Dr. GRPO much longer")
    shorter = side_rows(pairs, la, lb, 3, lambda p: (p[0]["response_tokens"] - p[1]["response_tokens"]), "Dr. GRPO much shorter")
    return "# Task 3 example candidates\n\n" + "\n".join([
        block("Group informativeness: wasted updates in the standard run", rows),
        block("Completions with mostly non-ASCII text", off_rows),
        block("Normalization: largest held-out length differences (Dr. GRPO vs canonical)", longer + shorter)])


def task4():
    import pandas as pd
    d = pd.read_csv(R / "task4_safety" / "audit_disagreements.csv", keep_default_na=False)
    d = d.drop_duplicates("audit_row")
    out = []
    for (man, ai), sub in d.groupby(["manual_label", "label"]):
        rows = [{"tag": f"xstest {r.xstest_id} ({r.benchmark_class}/{r.type}), policy {r.policy}: manual {man} vs judge {ai}",
                 "prompt": r.prompt, "sides": [("response", r.response, f"judge confidence {r.confidence}, tag '{r.rationale_tag}'"
                                                + (f"; your note: {r.manual_note}" if r.manual_note else ""))]}
                for r in sub.head(3).itertuples()]
        out.append(block(f"Manual {man} / judge {ai} ({len(sub)} distinct responses)", rows))
    gen = {p: {r["xstest_id"]: r for r in jl(R / "task4_safety" / f"generated_{p}.jsonl")} for p in ["sft", "dpo", "ppo", "grpo"]}
    jud = {p: {r["xstest_id"]: r["label"] for r in jl(R / "task4_safety" / f"judged_{p}.jsonl")} for p in gen}
    diff = [i for i in gen["sft"] if len({jud[p][i] for p in gen}) > 1][:4]
    rows = [{"tag": f"xstest {i}: judge labels " + ", ".join(f"{p}={jud[p][i]}" for p in gen), "prompt": gen["sft"][i]["prompt"],
             "sides": [(p.upper(), gen[p][i]["response"], jud[p][i]) for p in gen]} for i in diff]
    return "# Task 4 example candidates\n\n" + "\n".join(out + [block("Prompts where the policies receive different judge labels", rows)])


def task5():
    import pandas as pd
    diag = {}
    for r in jl(ROOT / "data" / "task5_controlled_reward_diagnostics.jsonl"):
        diag.setdefault(str(r["problem_id"]), {})[r["variant_type"]] = r
    raw = jl(R / "task5_feedback" / "diagnostics" / "judge_raw_outputs.jsonl")
    vmap = {"reasoning": "corrupt_reasoning_correct_final", "outcome": "good_reasoning_wrong_final",
            "filler": "persuasive_filler_correct", "distractor": "gold_distractor_wrong_final"}
    out = []
    for pair, want in [("filler", lambda r: r["verdict"] == "B"), ("distractor", lambda r: True), ("reasoning", lambda r: True),
                       ("outcome", lambda r: True)]:
        sel = [r for r in raw if r["pair"] == pair and want(r)][:2]
        rows = [{"tag": f"problem {r['problem_id']}: judge verdict {r['verdict']} (raw '{r['raw_4'].strip()}')",
                 "prompt": diag[r["problem_id"]]["clean_correct"]["question"] + f"  [gold {diag[r['problem_id']]['clean_correct']['gold_final']}]",
                 "sides": [("clean_correct", diag[r["problem_id"]]["clean_correct"]["response"], "A = better"),
                           (vmap[pair], diag[r["problem_id"]][vmap[pair]]["response"], "B = perturbed")]} for r in sel]
        out.append(block(f"Controlled pair: {pair}", rows))
    pp = pd.read_csv(R / "task5_feedback" / "transfer" / "per_problem.csv", dtype={"source_index": str})
    q = {str(r["source_index"]): r for r in jl(ROOT / "data" / "math_transfer_eval.jsonl")}
    gens = {p: {str(r["source_index"]): r["response"] for r in jl(R / "task5_feedback" / "transfer" / f"generated_{p}.jsonl")}
            for p in ["sft", "rlvr", "rlaif"]}
    rows = []
    for ft in ["no_final_format", "wrong_final"]:
        for r in pp[pp["rlvr_failure_type"] == ft].head(2).itertuples():
            rows.append({"tag": f"SVAMP {r.source_index}: RLVR failure '{ft}' (gold {r.gold_final})", "prompt": q[r.source_index]["question"],
                         "sides": [(p.upper(), gens[p][r.source_index], f"pred {getattr(r, p + '_pred')}, {getattr(r, p + '_failure_type')}")
                                   for p in ["sft", "rlvr", "rlaif"]]})
    return "# Task 5 example candidates\n\n" + "\n".join(out + [block("Out-of-domain failure types", rows)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("task1", task1), ("task2", task2), ("task3", task3), ("task4", task4), ("task5", task5)]:
        (OUT / f"{name}.md").write_text(fn(), encoding="utf-8")
        print("wrote", OUT / f"{name}.md")


if __name__ == "__main__":
    main()
