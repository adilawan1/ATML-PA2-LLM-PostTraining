"""Crash-safe periodic checkpoints so a Colab disconnect only loses work since the last save.

A checkpoint is one file `<run_dir>/_checkpoints/step_XXXXXX.pt` holding the trainable parameters of
each model (LoRA adapters / value head only, so files stay small), optimizer states, Python/NumPy/
torch RNG states, and an arbitrary `extra` dict (e.g. data position, schedule state). Files are
written to a temporary name and atomically renamed, so a crash mid-save never leaves a corrupt
"latest" checkpoint.

Typical loop:

    state = resume_if_available(run_dir, {"policy": model}, {"policy": opt})
    start = state["step"] if state else 0
    if state: truncate_jsonl_after(log_path, start)
    for step in range(start, total):
        ...
        if (step + 1) % save_every == 0:
            save_checkpoint(run_dir, step + 1, {"policy": model}, {"policy": opt}, extra={...})
    model.save_pretrained(run_dir); mark_complete(run_dir, {...})
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.data import repo_path

_CKPT_RE = re.compile(r"^step_(\d+)\.pt$")
COMPLETE_MARKER = "_COMPLETE.json"


def checkpoint_dir(run_dir: str | Path) -> Path:
    return repo_path(run_dir) / "_checkpoints"


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_trainable_state_dict(model, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    missing = [name for name in state if name not in params]
    if missing:
        raise KeyError(f"Checkpoint parameters not found in model: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(device=params[name].device, dtype=params[name].dtype))


def save_checkpoint(
    run_dir: str | Path,
    step: int,
    models: dict[str, Any],
    optimizers: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    keep_last: int = 2,
) -> Path:
    ckpt_dir = checkpoint_dir(run_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "models": {name: trainable_state_dict(m) for name, m in models.items()},
        "optimizers": {name: opt.state_dict() for name, opt in (optimizers or {}).items()},
        "rng": _rng_state(),
        "extra": extra or {},
    }
    final = ckpt_dir / f"step_{int(step):06d}.pt"
    tmp = final.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, final)

    for old in list_checkpoints(run_dir)[:-keep_last] if keep_last > 0 else []:
        old.unlink(missing_ok=True)
    return final


def list_checkpoints(run_dir: str | Path) -> list[Path]:
    ckpt_dir = checkpoint_dir(run_dir)
    if not ckpt_dir.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in ckpt_dir.iterdir() if (m := _CKPT_RE.match(p.name))]
    return [p for _, p in sorted(found)]


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: str | Path,
    models: dict[str, Any],
    optimizers: dict[str, Any] | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    # weights_only=False: the file is written by save_checkpoint and contains NumPy RNG state.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name, model in models.items():
        load_trainable_state_dict(model, payload["models"][name])
    for name, opt in (optimizers or {}).items():
        opt.load_state_dict(payload["optimizers"][name])
    if restore_rng:
        _set_rng_state(payload["rng"])
    return {"step": payload["step"], "extra": payload["extra"], "path": str(path)}


def resume_if_available(
    run_dir: str | Path,
    models: dict[str, Any],
    optimizers: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Load the newest checkpoint for this run, or return None to start from scratch."""
    path = latest_checkpoint(run_dir)
    if path is None:
        return None
    state = load_checkpoint(path, models, optimizers)
    print(f"[checkpoint] resumed {run_dir} at step {state['step']} from {path.name}", flush=True)
    return state


def mark_complete(run_dir: str | Path, info: dict[str, Any] | None = None, clean_checkpoints: bool = True) -> None:
    """Record that a run finished (final adapter already saved) and drop its intermediate checkpoints."""
    run = repo_path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / COMPLETE_MARKER).write_text(json.dumps(info or {}, indent=2), encoding="utf-8")
    if clean_checkpoints:
        for p in list_checkpoints(run_dir):
            p.unlink(missing_ok=True)


def is_complete(run_dir: str | Path) -> bool:
    return (repo_path(run_dir) / COMPLETE_MARKER).is_file()


def truncate_jsonl_after(path: str | Path, step: int, key: str = "step") -> int:
    """Drop log rows with `key > step` so a resumed run does not duplicate rows it will rewrite.

    Returns the number of rows removed.
    """
    p = repo_path(path)
    if not p.is_file():
        return 0
    kept, dropped = [], 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if key in row and int(row[key]) > int(step):
            dropped += 1
        else:
            kept.append(line)
    if dropped:
        p.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return dropped
