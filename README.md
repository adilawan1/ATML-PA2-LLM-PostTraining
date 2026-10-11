# ATML PA2 - LLM Post-Training

DPO, PPO and GRPO on `Qwen/Qwen2.5-1.5B-Instruct` (LoRA), safety calibration on XSTest, and RLVR vs RLAIF feedback
on GSM8K/SVAMP. Built on the course starter repository; every experiment is a Python module run from the repo root,
and every reported number is written to a JSON/CSV file under `results/`.

## 1. Setup

```bash
git clone https://github.com/adilawan1/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets      # pinned course assets (HF dataset AbDu11aHHH/ATML-PA2-assets @ 0b350481)
python -m scripts.validate_assets
python -m scripts.check_environment
python -m pytest tests -q              # CPU checks of the corrected objectives and checkpoint/resume
```

Public base, reward and judge models are downloaded from Hugging Face at runtime. Pinned versions:
Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, Tokenizers 0.22.1.

**Google Colab.** `colab/pa2_colab_launcher.ipynb` is a thin launcher (setup cells + one `run(...)` line per command
below). It keeps `outputs/` on Google Drive, mirrors `results/` to Drive every 5 minutes, and every training script
checkpoints periodically, so re-running the same line after a disconnect resumes where it stopped.

## 2. Corrected objective defects

Each of Tasks 1-3 shipped with one deliberate defect in its core objective. All three are fixed and covered by
`tests/test_objectives.py`:

| Task | File | Defect | Fix |
|---|---|---|---|
| DPO | `task1_dpo/dpo.py` | logits used `policy_margin + ref_margin`; accuracy ignored the reference | `beta * (policy_margin - ref_margin)`; accuracy is `m_theta > 0` |
| PPO | `task2_ppo/ppo.py` | clipped surrogate used `max` of the two terms (no trust region) | `min(rho*A, clip(rho)*A)` |
| GRPO | `task3_grpo/grpo.py` | advantages normalized over the whole batch | mean/std computed within each prompt group |

## 3. Reproducing every experiment

Run the commands in this order. Each script skips work that is already complete (finished training runs, existing
evaluations, cached judge calls) and resumes interrupted training from its latest checkpoint.

### Task 1 - DPO (`configs/dpo.yaml`, results in `results/task1_dpo/`)

```bash
python -m task1_dpo.train    --config configs/dpo.yaml --run-name standard                    # 1 epoch, 1500 pairs, beta=0.10
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter none --name sft             # untouched SFT reference
python -m task1_dpo.ablate_beta    --config configs/dpo.yaml                                  # beta in {0.03,0.10,0.30}, first 600 pairs
python -m task1_dpo.analyze_length --config configs/dpo.yaml                                  # length-balanced run + strata comparison
```

Outputs: `train_<run>.jsonl` (per-step loss, accuracy, margins, grad norm), `eval_<name>/metrics.json` (held-out DPO
loss and preference accuracy, per-stratum accuracy, reward-model score, sampled-response KL, entropy, length
statistics, word-limit compliance) with per-example records, `summary_beta.csv`, `summary_length.csv`,
`dataset_length_stats.json`.

### Task 2 - PPO (`configs/ppo.yaml`, results in `results/task2_ppo/`)

```bash
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --cache-only      # cached-batch clip geometry; run first (see note)
python -m task2_ppo.continue_train   --config configs/ppo.yaml --run-name standard  # 20 updates from the supplied midpoint
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml --forks-only      # eps in {0.05,0.20,0.50}, 8-update forks
python -m task2_ppo.ablate_kl        --config configs/ppo.yaml                   # kl_beta in {0,0.10,0.20}, 8-update forks
```

Note: the cache study reports which log-prob temperature (1.0 or the sampling temperature 0.7) reproduces the
supplied cached log-probs (`clipping_cache.json -> convention_check`). If it selects 1.0, set
`logprob_temperature: 1.0` in `configs/ppo.yaml` before running the continuation and forks. The fork with
eps=0.20, kl_beta=0.10 is shared by both ablations and is trained once.

### Task 3 - GRPO (`configs/grpo.yaml`, results in `results/task3_grpo/`)

```bash
python -m task3_grpo.analyze_group_size    --config configs/grpo.yaml           # K in {2,4,8}, CPU only (results committed)
python -m task3_grpo.continue_train        --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.compare_normalization --config configs/grpo.yaml           # canonical vs Dr. GRPO, 8-update forks
```

### Task 4 - Safety calibration (`configs/feedback.yaml`, results in `results/task4_safety/`)

Requires the standard DPO, PPO and GRPO adapters from Tasks 1-3.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml   # greedy, 450 XSTest prompts x 4 policies
python -m task4_safety.judge_responses    --config configs/feedback.yaml   # fixed categorical judge, resumable per row
python -m task4_safety.make_audit_sheet   --config configs/feedback.yaml   # fixed 60 IDs + blinded labelling sheet
python -m task4_safety.label_audit                                        # label the sheet by hand (policy and AI labels hidden)
python -m task4_safety.evaluate_safety    --config configs/feedback.yaml   # rates, category tables, audit agreement
```

The audit covers all four policies on the fixed 60 IDs (240 id-policy pairs). Byte-identical responses to the same
prompt share one sheet row (116 distinct responses), and `manual_audit_key.csv` maps each row back to every policy
that produced it, so agreement is computed over all 240 pairs. Because the fixed judge never emits `OVER_REFUSAL`
(it labels refusals of safe prompts `JUSTIFIED_REFUSAL`), `evaluate_safety` also reports the derived
`safe_any_refusal_rate` and `unsafe_any_answer_rate` next to the required rates.

### Task 5 - RLVR vs RLAIF (`configs/feedback.yaml`, results in `results/task5_feedback/`)

Uses only the supplied RLVR/RLAIF adapters, so it can run before Tasks 1-4.

```bash
python -m task5_feedback.evaluate_math       --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml   # --verifier-only needs no GPU
python -m task5_feedback.evaluate_math       --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback    --config configs/feedback.yaml
python -m task5_feedback.inspect_judge_outputs --config configs/feedback.yaml # raw judge text on the 80 controlled pairs
```

`inspect_judge_outputs` re-runs the fixed judge with its own rubric and 4-token decoding on the controlled pairs and
saves the raw text it emits, to separate genuine `TIE` answers from the parser's TIE fallback. It changes no metric.

### Report assets

```bash
python -m scripts.audit_informativeness   # paired bootstrap CIs for every comparison -> results/informativeness_audit.csv
python -m scripts.make_report_assets      # report/figures/*.pdf and report/tables/*.tex from results/
python -m scripts.find_example_candidates # candidate qualitative examples -> results/report_examples/task{1..5}.md
python -m scripts.check_report_numbers    # flags decimals in report/main.tex that do not appear in results/
```

`report/main.tex` (NeurIPS 2025 style) includes these files; upload the `report/` folder to Overleaf to compile.

## 4. Implementation conventions

- **Fixed data and seeds.** All runs use seed 6304, the released data files, and the released prompt IDs. Training
  prompt order, generation seeds per update, and evaluation decoding are identical across compared conditions.
  Each run saves its full configuration (including the prompt IDs/schedule) as `train_<run>_config.json`.
- **Added configuration keys** are operational only (checkpoint interval, evaluation batch sizes, prompt caps,
  word-limit samples, evaluation cap for GRPO). They are listed under "Student additions" in each config file;
  no released constant was changed.
- **KL** is the sampled-response estimator `log pi_theta - log pi_ref` on generated response tokens, reported both as
  the course helper's token mean (`kl_token_mean`) and as a per-sequence sum (`kl_seq_mean`). The reference is the
  base model with the adapter disabled.
- **PPO/GRPO.** Dropout is disabled in RL policies so that the importance ratio is exactly 1 at the first epoch on
  fresh rollouts. Trainable LoRA/value-head parameters are kept in fp32 and updated with an fp16 GradScaler. PPO
  applies the configured missing-EOS penalty. GRPO masks truncated completions from the loss while keeping their
  rewards in the group statistics, and logs exact per-completion gradient norms for the normalization study.
- **Judges.** Task 4 uses the supplied `judge_one` (`judge_batch_size: 1`). Task 5 uses the supplied
  `PairwiseAIJudge`, whose decisions are cached in `results/task5_feedback/*/judge_cache.json`.

## 5. Repository layout

```text
common/          shared data, model, generation, metric, checkpointing and RL rollout/evaluation helpers
configs/         base + per-task configurations
task1_dpo/ ... task5_feedback/   task scripts (one module per experiment step)
tests/           CPU unit tests (objectives, checkpoint/resume)
colab/           thin Colab launcher notebook
results/         machine-readable logs/metrics committed for the report
data/ cached/ checkpoints/ manifests/   course assets (downloaded, not committed)
outputs/         trained adapters (not committed)
```

## 6. Attribution

- Starter code, configurations, fixed data, cached diagnostics and supplied checkpoints:
  course release (https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining,
  https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets).
- Libraries: PyTorch, Hugging Face Transformers, PEFT, TRL, bitsandbytes, pandas, NumPy, scikit-learn.
- Coding assistance: parts of the implementation (training/evaluation loops, analysis scripts, tests, Colab
  launcher) were written with the help of an LLM coding assistant (Anthropic Claude); the author is responsible for
  all submitted code.
