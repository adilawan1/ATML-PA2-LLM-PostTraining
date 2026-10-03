"""CPU-only checks that periodic checkpoints restore a run exactly after an interruption.

Run: python -m pytest tests -q
"""
from __future__ import annotations

import json

import torch
from torch import nn

from common.checkpointing import (
    is_complete,
    latest_checkpoint,
    list_checkpoints,
    mark_complete,
    resume_if_available,
    save_checkpoint,
    truncate_jsonl_after,
)


def _model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    m[0].requires_grad_(False)  # stands in for the frozen base model; only m[1] is "LoRA"
    return m


def _train(model, opt, steps):
    for _ in range(steps):
        x = torch.randn(8, 4)
        loss = model(x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_resume_matches_uninterrupted_run(tmp_path):
    # Uninterrupted reference: 6 steps.
    ref = _model()
    ref_opt = torch.optim.AdamW([p for p in ref.parameters() if p.requires_grad], lr=1e-2)
    torch.manual_seed(123)
    _train(ref, ref_opt, 6)

    # Interrupted run: 3 steps, checkpoint, then a fresh process resumes for 3 more.
    run = _model()
    opt = torch.optim.AdamW([p for p in run.parameters() if p.requires_grad], lr=1e-2)
    torch.manual_seed(123)
    _train(run, opt, 3)
    save_checkpoint(tmp_path, 3, {"policy": run}, {"policy": opt}, extra={"batches_done": 3})
    torch.manual_seed(999)  # simulate a new session with different RNG state

    resumed = _model()
    resumed_opt = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-2)
    state = resume_if_available(tmp_path, {"policy": resumed}, {"policy": resumed_opt})
    assert state["step"] == 3 and state["extra"] == {"batches_done": 3}
    _train(resumed, resumed_opt, 3)

    for a, b in zip(ref.state_dict().values(), resumed.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_stores_only_trainable_parameters(tmp_path):
    m = _model()
    path = save_checkpoint(tmp_path, 1, {"policy": m})
    payload = torch.load(path, weights_only=False)
    assert set(payload["models"]["policy"]) == {"1.weight", "1.bias"}


def test_no_checkpoint_means_fresh_start(tmp_path):
    assert resume_if_available(tmp_path, {"policy": _model()}) is None


def test_keep_last_prunes_old_checkpoints(tmp_path):
    m = _model()
    for step in (2, 4, 6, 8):
        save_checkpoint(tmp_path, step, {"policy": m}, keep_last=2)
    assert [p.name for p in list_checkpoints(tmp_path)] == ["step_000006.pt", "step_000008.pt"]
    assert latest_checkpoint(tmp_path).name == "step_000008.pt"


def test_mark_complete_cleans_checkpoints(tmp_path):
    save_checkpoint(tmp_path, 1, {"policy": _model()})
    assert not is_complete(tmp_path)
    mark_complete(tmp_path, {"steps": 1})
    assert is_complete(tmp_path) and list_checkpoints(tmp_path) == []


def test_truncate_jsonl_after_drops_rows_past_resume_step(tmp_path):
    log = tmp_path / "train.jsonl"
    log.write_text("".join(json.dumps({"step": s, "loss": 0.1 * s}) + "\n" for s in range(1, 7)))
    assert truncate_jsonl_after(log, 4) == 2
    assert [json.loads(l)["step"] for l in log.read_text().splitlines()] == [1, 2, 3, 4]
