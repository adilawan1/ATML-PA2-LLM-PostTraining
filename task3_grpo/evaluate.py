"""Common held-out GRPO evaluation: one sampled response per fixed RL eval prompt at the frozen-evaluation cap
(`eval_max_response_length`), scored for learned reward, KL to the reference, entropy and length.

Also used for the supplied midpoint (`--adapter checkpoints/grpo_midpoint_policy --name midpoint`) and both normalization forks.
Writes results/task3_grpo/eval_<name>/{metrics.json,generations.jsonl}.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from common.rl_utils import evaluate_policy


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_adapter(config_path: str, adapter: str, name: str, max_prompts: int | None = None, reuse: bool = True):
    cfg = load_yaml(config_path)
    metrics_path = repo_path(cfg["results_dir"]) / f"eval_{name}" / "metrics.json"
    if reuse and metrics_path.is_file() and not max_prompts:
        print(f"[eval] {name}: reusing {metrics_path}", flush=True)
        return load_json(metrics_path)
    bundle = load_evaluation_bundle(config_path, adapter)
    metrics = evaluate_policy(
        bundle["policy"], bundle["tokenizer"], bundle["reward"], bundle["cfg"], name,
        bundle["cfg"]["results_dir"], int(bundle["cfg"]["eval_max_response_length"]), max_prompts,
    )
    metrics["adapter"] = adapter
    del bundle
    clear_gpu()
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, help="smoke tests only")
    ap.add_argument("--force", action="store_true", help="re-run even if metrics already exist")
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name, args.max_prompts, reuse=not args.force)


if __name__ == "__main__":
    main()
