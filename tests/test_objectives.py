"""CPU-only checks of the Task 1-3 objectives against the equations in the assignment manual.

Run: python -m pytest tests -q
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import compute_gae, ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss


# ---------------- Task 1: DPO ----------------

def test_dpo_matches_manual_equation():
    torch.manual_seed(0)
    pc, pr, rc, rr = (torch.randn(16) * 5 for _ in range(4))
    beta = 0.1
    loss, diag = dpo_loss(pc, pr, rc, rr, beta)
    m = (pc - rc) - (pr - rr)
    expected = -F.logsigmoid(beta * m).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert math.isclose(float(diag["preference_accuracy"]), float((m > 0).float().mean()))


def test_dpo_policy_equal_to_reference_gives_log2():
    x = torch.randn(8)
    y = torch.randn(8)
    loss, diag = dpo_loss(x, y, x, y, beta=0.3)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-6)
    assert float(diag["preference_accuracy"]) == 0.0


def test_dpo_gradient_raises_chosen_lowers_rejected():
    pc = torch.zeros(4, requires_grad=True)
    pr = torch.zeros(4, requires_grad=True)
    loss, _ = dpo_loss(pc, pr, torch.zeros(4), torch.zeros(4), beta=0.1)
    loss.backward()
    assert (pc.grad < 0).all() and (pr.grad > 0).all()


# ---------------- Task 2: PPO ----------------

def test_ppo_clipped_surrogate_is_pessimistic_min():
    new = torch.log(torch.tensor([[1.5, 0.5, 1.05, 0.95]]))
    old = torch.zeros_like(new)
    mask = torch.ones_like(new)
    eps = 0.2
    for a in (1.0, -1.0):
        adv = torch.full_like(new, a)
        loss, ratio, clip_frac = ppo_policy_loss(new, old, adv, mask, eps)
        r = torch.exp(new - old)
        obj = torch.minimum(r * adv, r.clamp(1 - eps, 1 + eps) * adv)
        assert torch.allclose(loss, -obj.mean(), atol=1e-6)
        assert math.isclose(float(clip_frac), 0.5)


def test_ppo_no_gradient_beyond_clip_for_positive_advantage():
    new = torch.log(torch.tensor([[1.5]])).requires_grad_(True)
    loss, _, _ = ppo_policy_loss(new, torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1), eps=0.2)
    loss.backward()
    assert float(new.grad.abs().sum()) == 0.0


def test_gae_lambda1_gamma1_is_return_minus_value():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 0.4, 0.6, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv[0, :3], torch.tensor([0.8, 0.6, 0.4]), atol=1e-6)
    assert float(adv[0, 3]) == 0.0


# ---------------- Task 3: GRPO ----------------

def test_grpo_advantages_are_per_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0])
    gids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    adv = group_relative_advantages(rewards, gids)
    g0 = rewards[:4]
    expected = (g0 - g0.mean()) / (g0.std(unbiased=False) + 1e-6)
    assert torch.allclose(adv[:4], expected, atol=1e-5)
    assert torch.allclose(adv[4:], torch.zeros(4))  # uninformative group -> zero signal


def test_grpo_advantages_handle_interleaved_ids():
    rewards = torch.tensor([0.0, 5.0, 1.0, 5.0])
    gids = torch.tensor([7, 3, 7, 3])
    adv = group_relative_advantages(rewards, gids)
    assert adv[0] < 0 < adv[2]
    assert torch.allclose(adv[[1, 3]], torch.zeros(2))


def test_grpo_normalization_variants():
    new = torch.zeros(2, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    adv = torch.tensor([1.0, 1.0])
    loss_g, _ = grpo_policy_loss(new, new, adv, mask, new, eps=0.2, beta=0.0, loss_type="grpo")
    loss_d, _ = grpo_policy_loss(new, new, adv, mask, new, eps=0.2, beta=0.0,
                                 loss_type="dr_grpo", max_completion_length=8)
    assert math.isclose(float(loss_g), -1.0)              # each sequence averaged by its own length
    assert math.isclose(float(loss_d), -(2 + 4) / 8 / 2)  # constant denominator
