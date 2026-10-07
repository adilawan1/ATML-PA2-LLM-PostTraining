"""Build every report figure (report/figures/*.pdf) and table (report/tables/*.tex) from results/.

CPU only, no model code. Tables are booktabs tabulars without captions (captions and all prose live in
report/main.tex). Re-run after any result changes:  python -m scripts.make_report_assets
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"

# Validated categorical palette (first three slots pass all-pairs CVD checks), neutral for ties, ink tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL, INK, INK2, GRID = "#b4b2ab", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.4, "pdf.fonttype": 42, "font.family": "serif",
})
TEXTWIDTH = 5.5  # NeurIPS text width in inches


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def js(path):
    return json.load(open(path, encoding="utf-8"))


def save_fig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure", name)


def save_tab(name, body):
    (TAB / f"{name}.tex").write_text(body, encoding="utf-8")
    print("table ", name)


def f(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ---------------------------------------------------------------- Task 1
def task1():
    t = R / "task1_dpo"
    rows = [("SFT (reference)", "--", "eval_sft"), ("Standard DPO, $\\beta{=}0.10$", "1 ep, 1500", "eval_standard"),
            ("Short, $\\beta{=}0.03$", "600", "eval_beta_0.03"), ("Short, $\\beta{=}0.10$", "600", "eval_beta_0.1"),
            ("Short, $\\beta{=}0.30$", "600", "eval_beta_0.3"), ("Length-balanced", "1 ep, 1500", "eval_length_balanced")]
    lines = []
    for label, budget, d in rows:
        m = js(t / d / "metrics.json")
        h, g, w = m["heldout_pairs"], m["generation"], m["word_limit"]
        lines.append(f"{label} & {budget} & {f(h['dpo_loss'])} & {f(h['preference_accuracy'])} & {f(h['pref_margin_mean'])} & "
                     f"{f(g['kl_token_mean'] * 1e3, 2)} & {f(g['reward_mean'])} & {g['response_length']['mean']:.0f} $\\pm$ "
                     f"{g['response_length']['std']:.0f} & {f(w['compliance_rate'], 2)} \\\\")
    save_tab("t1_dpo", "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
             "Model & Pairs & Loss & Acc & Margin & KL$_\\text{tok}$ ($10^{-3}$) & RM score & Length & Word lim. \\\\\n\\midrule\n"
             + "\n".join(lines[:2]) + "\n\\midrule\n" + "\n".join(lines[2:5]) + "\n\\midrule\n" + lines[5]
             + "\n\\bottomrule\n\\end{tabular}\n")

    s = pd.read_csv(t / "summary_length.csv").set_index("model")
    strata = ["preferred_longer", "length_matched", "rejected_longer"]
    names = ["Preferred\nlonger", "Length\nmatched", "Rejected\nlonger"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.7))
    x, wdt = np.arange(3), 0.36
    for ax, metric, title in [(axes[0], "acc", "Held-out preference accuracy"), (axes[1], "margin", "Mean preference margin $m_\\theta$")]:
        for k, (model, col, lab) in enumerate([("standard", BLUE, "Standard"), ("length_balanced", ORANGE, "Length-balanced")]):
            v = [s.loc[model, f"{metric}_{st}"] for st in strata]
            ax.bar(x + (k - 0.5) * wdt, v, wdt * 0.92, color=col, label=lab)
            for xi, vi in zip(x + (k - 0.5) * wdt, v):
                ax.text(xi, vi + (0.01 if metric == "acc" else (0.04 if vi >= 0 else -0.04)), f"{vi:.2f}", ha="center",
                        va="bottom" if vi >= 0 else "top", fontsize=5.5, color=INK2)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.axhline(0.5 if metric == "acc" else 0.0, color=INK2, lw=0.7, ls="--" if metric == "acc" else "-")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(-0.8, 1.7)
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "t1_length_strata")


# ---------------------------------------------------------------- Task 2 / 3 trajectories
def trajectories(path, panels, name):
    t = pd.DataFrame(jl(path))
    fig, axes = plt.subplots(2, 4, figsize=(TEXTWIDTH, 2.5), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(t["update"], t[col], color=BLUE, marker="o", ms=2.2)
        ax.set_title(title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.yaxis.get_offset_text().set_fontsize(6)
    for ax in axes[1]:
        ax.set_xlabel("Update")
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    save_fig(fig, name)


def task2():
    t = R / "task2_ppo"
    trajectories(t / "train_standard.jsonl",
                 [("reward_raw", "RM reward"), ("kl_token_mean", "KL to reference"), ("policy_loss", "Policy loss"),
                  ("value_loss", "Value loss"), ("entropy", "Entropy"), ("clip_fraction", "Clip fraction"),
                  ("policy_grad_norm", "Policy grad norm"), ("response_length", "Response length")], "t2_ppo_trajectories")
    cache = pd.read_csv(t / "clipping_cache_static.csv").set_index("epsilon")
    forks = pd.concat([pd.read_csv(t / "clipping_forks.csv"), pd.read_csv(t / "kl_forks.csv")]).drop_duplicates("name")
    order = [("fork_eps0.05_kl0.1", "$\\epsilon{=}0.05$", 0.05), ("fork_eps0.2_kl0.1", "$\\epsilon{=}0.20$", 0.2),
             ("fork_eps0.5_kl0.1", "$\\epsilon{=}0.50$", 0.5), ("fork_eps0.2_kl0", "$\\beta_\\text{KL}{=}0$", None),
             ("fork_eps0.2_kl0.2", "$\\beta_\\text{KL}{=}0.20$", None)]
    fk = forks.set_index("name")
    lines = []
    for i, (n, lab, eps) in enumerate(order):
        r = fk.loc[n]
        aff = f(cache.loc[eps, "affected_token_fraction"]) if eps is not None else "--"
        lines.append(f"{lab} & {r['clip_epsilon']:.2f} & {r['kl_beta']:.2f} & {aff} & {f(r['heldout_reward'])} & "
                     f"{f(r['heldout_kl_seq_mean'], 4)} & {f(r['heldout_entropy'])} & {r['heldout_len_mean']:.0f} & "
                     f"{f(r['stab_grad_norm_cv'], 2)} & {f(r['stab_final_epoch_clip_fraction'], 3)} \\\\")
        if i == 2:
            lines.append("\\midrule")
    mid, std = js(t / "eval_midpoint" / "metrics.json"), js(t / "eval_standard" / "metrics.json")
    ref = [f"{lab} & 0.20 & 0.10 & -- & {f(m['reward_mean'])} & {f(m['kl_seq_mean'], 4)} & {f(m['entropy_token_mean'])} & "
           f"{m['response_length']['mean']:.0f} & -- & -- \\\\" for lab, m in [("Midpoint (start)", mid), ("Standard, 20 upd.", std)]]
    save_tab("t2_ppo", "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
             "Condition & $\\epsilon$ & $\\beta_\\text{KL}$ & Cache aff. & RM score & KL$_\\text{seq}$ & Entropy & Length & "
             "Grad CV & Clip frac. \\\\\n\\midrule\n" + "\n".join(ref) + "\n\\midrule\n" + "\n".join(lines)
             + "\n\\bottomrule\n\\end{tabular}\n")


def task3():
    t = R / "task3_grpo"
    trajectories(t / "train_standard.jsonl",
                 [("reward_mean", "RM reward"), ("kl_token_mean", "KL to reference"), ("group_reward_std_mean", "Within-group reward std"),
                  ("uninformative_group_fraction", "Uninformative groups"), ("policy_loss", "Policy loss"), ("grad_norm", "Grad norm"),
                  ("entropy", "Entropy"), ("response_length", "Response length")], "t3_grpo_trajectories")

    gs = pd.read_csv(t / "group_size_summary.csv")
    gd = pd.read_csv(t / "group_size_by_difficulty.csv")
    lines = []
    for k in [2, 4, 8]:
        o = gs[(gs.K == k) & (gs.scheme == "disjoint")].iloc[0]
        cells = [f"{o['informative_rate']:.3f}", f"{o['mean_within_group_std']:.3f}", f"{o['adv_variance']:.3f}"]
        for diff in ["easy", "medium", "hard"]:
            d = gd[(gd.K == k) & (gd.scheme == "disjoint") & (gd.difficulty == diff)].iloc[0]
            cells += [f"{d['informative_rate']:.3f}", f"{d['mean_within_group_std']:.3f}"]
        lines.append(f"{k} & {int(o['n_groups'])} & " + " & ".join(cells) + " \\\\")
    save_tab("t3_group_size", "\\begin{tabular}{rr rrr rr rr rr}\n\\toprule\n"
             " & & \\multicolumn{3}{c}{All prompts} & \\multicolumn{2}{c}{Easy} & \\multicolumn{2}{c}{Medium} & \\multicolumn{2}{c}{Hard} \\\\\n"
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n"
             "$K$ & Groups & Inform. & Std & Var$(A)$ & Inform. & Std & Inform. & Std & Inform. & Std \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    nf = pd.read_csv(t / "normalization_forks.csv").set_index("loss_type")
    bins = pd.read_csv(t / "normalization_length_bins.csv")
    lines = []
    for lt, lab in [("grpo", "Canonical ($1/T_k$)"), ("dr_grpo", "Dr.~GRPO")]:
        r = nf.loc[lt]
        g = {b: bins[(bins.loss_type == lt) & (bins.length_bin == b)].iloc[0]["grad_norm_mean"] for b in ["short", "medium", "long"]}
        lines.append(f"{lab} & {f(r['heldout_reward'])} & {f(r['heldout_kl_seq_mean'], 4)} & {r['heldout_len_mean']:.0f} & "
                     f"{f(r['heldout_truncation_rate'], 3)} & {f(g['short'])} & {f(g['medium'])} & {f(g['long'])} \\\\")
    save_tab("t3_normalization", "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
             " & \\multicolumn{4}{c}{Held-out} & \\multicolumn{3}{c}{Per-completion grad norm} \\\\\n"
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\n"
             "Normalization & RM score & KL$_\\text{seq}$ & Length & Trunc. & Short & Medium & Long \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.7))
    shades = {"easy": "#9ec5f0", "medium": "#5b9be4", "hard": "#1d5aa6"}  # sequential single hue: ordinal difficulty
    for diff in ["easy", "medium", "hard"]:
        d = gd[(gd.scheme == "disjoint") & (gd.difficulty == diff)].sort_values("K")
        axes[0].plot(d.K, d.mean_within_group_std, color=shades[diff], marker="o", ms=3, label=diff.capitalize())
    axes[0].set_xticks([2, 4, 8])
    axes[0].set_xlabel("Group size $K$ (192 generations total)")
    axes[0].set_title("Mean within-group reward std")
    axes[0].legend(frameon=False, loc="upper left", ncol=3)
    axes[0].set_ylim(0.2, 0.9)
    x, wdt = np.arange(3), 0.36
    for k, (lt, col, lab) in enumerate([("grpo", BLUE, "Canonical"), ("dr_grpo", ORANGE, "Dr. GRPO")]):
        v = [bins[(bins.loss_type == lt) & (bins.length_bin == b)].iloc[0]["grad_norm_mean"] for b in ["short", "medium", "long"]]
        axes[1].bar(x + (k - 0.5) * wdt, v, wdt * 0.92, color=col, label=lab)
    axes[1].set_xticks(x, ["Short", "Medium", "Long"])
    axes[1].set_xlabel("Completion length tercile")
    axes[1].set_title("Mean per-completion gradient norm")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "t3_groupsize_normalization")


# ---------------------------------------------------------------- Task 4
def task4():
    t = R / "task4_safety"
    s = js(t / "safety_summary.json")
    pol = [("sft", "SFT"), ("dpo", "DPO"), ("ppo", "PPO"), ("grpo", "GRPO")]
    lines = []
    for p, lab in pol:
        r = s["per_policy"][p]
        lines.append(f"{lab} & {f(r['safe_answer_rate'])} & {f(r['safe_over_refusal_rate'])} & {f(r['safe_any_refusal_rate'])} & "
                     f"{f(r['unsafe_compliance_rate'])} & {f(r['justified_refusal_rate'])} & {f(r['ambiguous_rate'])} & "
                     f"{r['response_tokens_mean']:.0f} \\\\")
    save_tab("t4_safety_rates", "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
             " & \\multicolumn{3}{c}{Safe prompts ($n{=}250$)} & \\multicolumn{2}{c}{Unsafe prompts ($n{=}200$)} & & \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
             "Policy & Answer & Over-ref. & Any ref.$^\\dagger$ & Unsafe comp. & Just. ref. & Ambig. & Length \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

    a = s["manual_audit"]
    labs = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
    short = {"SAFE_ANSWER": "Safe ans.", "JUSTIFIED_REFUSAL": "Just. ref.", "UNSAFE_COMPLIANCE": "Unsafe comp.",
             "OVER_REFUSAL": "Over-ref.", "AMBIGUOUS": "Ambig."}
    conf = pd.read_csv(t / "audit_confusion.csv", index_col=0)
    lines = [f"{short[m]} & " + " & ".join(str(int(conf.loc[m, c])) for c in labs) + f" & {int(conf.loc[m].sum())} \\\\" for m in labs]
    save_tab("t4_audit_confusion", "\\begin{tabular}{l rrrrr r}\n\\toprule\n"
             "Manual $\\downarrow$ / AI judge $\\rightarrow$ & " + " & ".join(short[c] for c in labs) + " & Total \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\midrule\n"
             f"\\multicolumn{{7}}{{l}}{{Agreement {a['agreement']:.3f}, Cohen's $\\kappa$ {a['cohen_kappa']:.3f}; safe {a['agreement_by_class']['SAFE']:.3f}, "
             f"unsafe {a['agreement_by_class']['UNSAFE']:.3f}; $n{{=}}{a['n_labelled']}$ pairs ({a['n_distinct_responses_labelled']} distinct)}} \\\\\n"
             "\\bottomrule\n\\end{tabular}\n")

    c = pd.read_csv(t / "category_label_fractions.csv")
    c["refusal"] = c["JUSTIFIED_REFUSAL"] + c["OVER_REFUSAL"]
    piv = c.pivot_table(index=["benchmark_class", "type"], columns="policy", values="refusal")[[p for p, _ in pol]]
    fig, ax = plt.subplots(figsize=(2.9, 3.6))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [l for _, l in pol])
    ax.set_yticks(range(len(piv)), [f"{'S' if cls == 'SAFE' else 'U'}: {ty.replace('contrast_', '').replace('_', ' ')}" for cls, ty in piv.index])
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.3, color="white" if v > 0.55 else INK)
    n_safe = int((piv.index.get_level_values(0) == "SAFE").sum())
    ax.axhline(n_safe - 0.5, color=INK, lw=0.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("AI-judge refusal rate (just. + over-refusal)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout()
    save_fig(fig, "t4_category_refusal")


# ---------------------------------------------------------------- Task 5
def task5():
    t = R / "task5_feedback"
    pc = pd.read_csv(t / "feedback_comparison_policies.csv").set_index("policy")
    pw = pd.read_csv(t / "feedback_comparison_pairwise.csv").set_index("comparison")
    gsm, tr = js(t / "gsm" / "metrics.json")["pairwise"], js(t / "transfer" / "metrics.json")["pairwise"]
    lines = []
    for p, lab in [("sft", "SFT"), ("rlvr", "RLVR"), ("rlaif", "RLAIF")]:
        r = pc.loc[p]
        key = f"{p}_vs_sft"
        win_g = f"{gsm[key]['win_rate']:.3f} ({gsm[key]['ties']})" if key in gsm else "--"
        win_t = f"{tr[key]['win_rate']:.3f} ({tr[key]['ties']})" if key in tr else "--"
        lines.append(f"{lab} & {f(r['gsm_exact_accuracy'])} & {f(r['gsm_format_compliance'])} & {r['gsm_len_mean']:.0f} & {win_g} & "
                     f"{f(r['transfer_exact_accuracy'])} & {f(r['transfer_format_compliance'])} & {r['transfer_len_mean']:.0f} & {win_t} & "
                     f"{r['transfer_exact_accuracy'] - r['gsm_exact_accuracy']:+.3f} \\\\")
    agree = []
    for comp, lab in [("rlvr_vs_sft", "RLVR vs SFT"), ("rlaif_vs_sft", "RLAIF vs SFT"), ("rlvr_vs_rlaif", "RLVR vs RLAIF")]:
        g, x = gsm[comp], tr[comp]
        agree.append(f"\\multicolumn{{11}}{{l}}{{\\quad {lab}: verifier--judge agreement {g['verifier_judge_agreement_3way']:.2f} (GSM8K), "
                     f"{x['verifier_judge_agreement_3way']:.2f} (SVAMP); decisive pairs {g['decisive_pairs']} / {x['decisive_pairs']}, "
                     f"judge agrees on {f(g['judge_agrees_on_decisive'], 2)} / {f(x['judge_agrees_on_decisive'], 2)}}} \\\\")
    save_tab("t5_feedback", "\\begin{tabular}{l rrrr rrrr r}\n\\toprule\n"
             " & \\multicolumn{4}{c}{GSM8K ($n{=}300$)} & \\multicolumn{4}{c}{SVAMP transfer ($n{=}100$)} & \\\\\n"
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
             "Policy & Acc & Format & Length & Win vs SFT (ties) & Acc & Format & Length & Win vs SFT (ties) & $\\Delta$Acc \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\midrule\n" + "\n".join(agree) + "\n\\bottomrule\n\\end{tabular}\n")

    ps = pd.read_csv(t / "diagnostics" / "pair_summary.csv")
    summ = js(t / "diagnostics" / "summary.json")
    raw = js(t / "diagnostics" / "judge_raw_summary.json")
    lines = []
    for pair, lab in [("reasoning", "Reasoning corrupted"), ("outcome", "Wrong final answer"), ("filler", "Persuasive filler"),
                      ("distractor", "Gold-number distractor")]:
        cells = []
        for mech in ["verifier", "ai_judge"]:
            r = ps[(ps.mechanism == mech) & (ps.pair == pair)].iloc[0]
            cells += [f"{r['better_rate']:.2f}", f"{r['tie_rate']:.2f}", f"{r['wrong_preference_rate']:.2f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    save_tab("t5_diagnostics", "\\begin{tabular}{l rrr rrr}\n\\toprule\n"
             " & \\multicolumn{3}{c}{Exact verifier} & \\multicolumn{3}{c}{AI pairwise judge} \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
             "Clean correct vs \\ldots & Better & Tie & Wrong & Better & Tie & Wrong \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\midrule\n"
             f"\\multicolumn{{7}}{{l}}{{$S_\\text{{reason}}$: verifier {summ['S_reason']['verifier']:.2f}, judge {summ['S_reason']['ai_judge']:.2f}; "
             f"$S_\\text{{outcome}}$: verifier {summ['S_outcome']['verifier']:.2f}, judge {summ['S_outcome']['ai_judge']:.2f}; "
             f"judge parse fallback {raw['tie_fallback_rate']:.2f} ($n{{=}}{raw['n']}$)}} \\\\\n"
             "\\bottomrule\n\\end{tabular}\n")

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 1.75))
    order = [("reasoning", "Reasoning corrupted"), ("outcome", "Wrong final answer"), ("filler", "Persuasive filler"),
             ("distractor", "Gold-number distractor")]
    ylab, y = [], 0
    for pair, lab in order:
        for mech, mlab in [("verifier", "verifier"), ("ai_judge", "AI judge")]:
            r = ps[(ps.mechanism == mech) & (ps.pair == pair)].iloc[0]
            left = 0
            for val, col in [(r["better_rate"], BLUE), (r["tie_rate"], NEUTRAL), (r["wrong_preference_rate"], ORANGE)]:
                if val > 0:
                    ax.barh(y, val, left=left, color=col, height=0.72, edgecolor="white", linewidth=1)
                    if val >= 0.12:
                        ax.text(left + val / 2, y, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                                color="white" if col != NEUTRAL else INK)
                left += val
            ylab.append(f"{lab} — {mlab}")
            y += 1
        y += 0.4
    ys = [i + 0.4 * (i // 2) for i in range(len(ylab))]
    ax.set_yticks(ys, ylab)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of the 20 controlled pairs")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Prefers better response"), Patch(color=NEUTRAL, label="Tie"),
                       Patch(color=ORANGE, label="Prefers worse response")], frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.4, 1.0))
    fig.tight_layout()
    save_fig(fig, "t5_diagnostics")


# ---------------------------------------------------------------- compute
def compute():
    rows = [("DPO standard (1 epoch)", R / "task1_dpo" / "train_standard_summary.json"),
            ("PPO continuation (20 updates)", R / "task2_ppo" / "train_standard_summary.json"),
            ("GRPO continuation (20 updates)", R / "task3_grpo" / "train_standard_summary.json")]
    lines = [f"{lab} & {js(p)['wall_clock_s'] / 60:.1f} & {js(p)['peak_vram_gib']:.2f} \\\\" for lab, p in rows]
    save_tab("compute", "\\begin{tabular}{lrr}\n\\toprule\nRun (Colab T4) & Wall clock (min) & Peak VRAM (GiB) \\\\\n\\midrule\n"
             + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    task1()
    task2()
    task3()
    task4()
    task5()
    compute()


if __name__ == "__main__":
    main()
