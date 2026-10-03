from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual before using it.
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # FIX: the DPO implicit-reward margin is the policy log-ratio margin MINUS the reference
    # margin, i.e. [log pi(y+)/ref(y+)] - [log pi(y-)/ref(y-)]. The starter added ref_margin.
    pref_margin = policy_margin - ref_margin
    logits = beta * pref_margin

    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "pref_margin_mean": pref_margin.detach().mean(),
        # Manual definition: accuracy is m_theta > 0 with m_theta measured relative to the reference.
        "preference_accuracy": (pref_margin > 0).float().mean().detach(),
    }
