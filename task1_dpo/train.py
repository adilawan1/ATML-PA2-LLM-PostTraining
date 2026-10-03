from __future__ import annotations

import argparse
import math
from collections import defaultdict
from itertools import islice

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.checkpointing import (
    COMPLETE_MARKER,
    is_complete,
    mark_complete,
    resume_if_available,
    save_checkpoint,
    truncate_jsonl_after,
)
from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, resolve_dtype, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
        # Seeded generator: the shuffle order is identical in a fresh process, so a resumed run
        # can skip exactly the batches it already consumed.
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def pair_logprobs(model, chosen: dict, rejected: dict):
    """Summed response-token log-probs of chosen/rejected under the policy (with grad) and the reference."""
    with torch.no_grad(), reference_mode(model):
        ref_c, _, _ = response_sequence_logprobs(model, chosen)
        ref_r, _, _ = response_sequence_logprobs(model, rejected)
    pol_c, _, mask_c = response_sequence_logprobs(model, chosen)
    pol_r, _, mask_r = response_sequence_logprobs(model, rejected)
    return pol_c, pol_r, ref_c, ref_r, mask_c.sum(-1), mask_r.sum(-1)


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    output = repo_path(output_path or load_yaml(config_path)["standard_output"])
    if is_complete(output):
        print(f"[dpo] {run_name}: already complete at {output}; skipping training.", flush=True)
        return load_json(output / COMPLETE_MARKER)

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, loader, optimizer = bundle["model"], bundle["tokenizer"], bundle["loader"], bundle["optimizer"]
    beta = bundle["beta"]
    device = next(model.parameters()).device

    accum = int(cfg["grad_accum_steps"])
    n_batches = len(loader)
    total_steps = math.ceil(n_batches / accum)
    save_every = int(cfg.get("checkpoint_every_steps", 10))
    params = trainable_parameters(model)

    # fp16 base weights: scale the loss so small DPO gradients do not underflow in fp16 activations.
    use_scaler = device.type == "cuda" and resolve_dtype(cfg.get("dtype", "float16")) == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    run_config = {
        "run_name": run_name,
        "dataset": bundle["dataset_path"],
        "n_examples": len(bundle["rows"]),
        "beta": beta,
        "epochs": 1,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "optimizer_steps": total_steps,
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "max_grad_norm": float(cfg["max_grad_norm"]),
        "seed": int(cfg["seed"]),
        "lora": cfg["lora"],
        "base_model": cfg["base_model"],
        "dtype": cfg.get("dtype"),
        "grad_scaler": use_scaler,
        "prompt_ids": [r.get("prompt_id") for r in bundle["rows"]],
    }
    results_dir = repo_path(cfg["results_dir"])
    save_json(results_dir / f"train_{run_name}_config.json", run_config)
    log_path = results_dir / f"train_{run_name}.jsonl"

    models, optimizers = {"policy": model}, {"optimizer": optimizer, "scaler": scaler}
    state = resume_if_available(output, models, optimizers)
    step = state["step"] if state else 0
    start_step = step
    elapsed_before = float(state["extra"].get("elapsed_s", 0.0)) if state else 0.0
    peak_before = int(state["extra"].get("peak_vram_bytes", 0)) if state else 0
    if state:
        truncate_jsonl_after(log_path, step)
    elif log_path.exists():
        log_path.unlink()  # fresh run: drop rows from an attempt that died before its first checkpoint

    print(
        f"[dpo] {run_name}: {len(bundle['rows'])} pairs, beta={beta}, {total_steps} optimizer steps "
        f"(batch {cfg['batch_size']} x accum {accum}), starting at step {step}",
        flush=True,
    )
    timer = wall_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    batches = islice(iter(loader), step * accum, None)  # resume: skip already-consumed batches
    while step < total_steps:
        window = min(accum, n_batches - step * accum)
        agg = defaultdict(float)
        for _ in range(window):
            chosen, rejected = next(batches)
            chosen, rejected = _to_device(chosen, device), _to_device(rejected, device)
            pol_c, pol_r, ref_c, ref_r, len_c, len_r = pair_logprobs(model, chosen, rejected)
            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            scaler.scale(loss / window).backward()

            agg["loss"] += float(loss.detach())
            agg["preference_accuracy"] += float(diag["preference_accuracy"])
            agg["pref_margin"] += float(diag["pref_margin_mean"])
            agg["chosen_logratio"] += float((pol_c - ref_c).detach().mean())
            agg["rejected_logratio"] += float((pol_r - ref_r).detach().mean())
            agg["chosen_tokens"] += float(len_c.float().mean())
            agg["rejected_tokens"] += float(len_r.float().mean())

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"])))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        elapsed = elapsed_before + timer()
        peak = max(peak_before, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0)
        record = {"step": step, **{k: v / window for k, v in agg.items()}}
        record.update({
            "grad_norm": grad_norm,
            "step_skipped_nonfinite": bool(use_scaler and scaler.get_scale() < scale_before),
            "examples_seen": min(step * accum, n_batches) * int(cfg["batch_size"]),
            "elapsed_s": elapsed,
            "peak_vram_gib": peak / 2**30,
        })
        append_jsonl(log_path, record)
        eta = (total_steps - step) * timer() / (step - start_step)
        print(
            f"[dpo] {run_name} step {step}/{total_steps} loss={record['loss']:.4f} "
            f"acc={record['preference_accuracy']:.3f} margin={record['pref_margin']:.3f} "
            f"gnorm={grad_norm:.3f} {elapsed / 60:.1f}min eta={eta / 60:.1f}min",
            flush=True,
        )
        if step % save_every == 0 and step < total_steps:
            save_checkpoint(output, step, models, optimizers, extra={"elapsed_s": elapsed, "peak_vram_bytes": peak})

    model.save_pretrained(str(output))
    summary = {
        **{k: v for k, v in run_config.items() if k != "prompt_ids"},
        "output": str(output),
        "wall_clock_s": elapsed_before + timer(),
        "peak_vram_gib": max(peak_before, torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / 2**30,
        "final_train_record": record if total_steps else None,
    }
    save_json(results_dir / f"train_{run_name}_summary.json", summary)
    mark_complete(output, summary)
    print(f"[dpo] {run_name}: saved adapter to {output}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
