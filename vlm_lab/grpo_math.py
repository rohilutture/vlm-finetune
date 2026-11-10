"""Pure-tensor GRPO math (same functions as the text-to-SQL GRPO project, re-tested here), kept free of model code so
it is unit-testable on CPU."""
from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, scale_std: bool = True, eps: float = 1e-4) -> torch.Tensor:
    """rewards: [n_prompts, G]. A group where every reward is equal gets zero advantage (no learning signal)."""
    adv = rewards - rewards.mean(dim=1, keepdim=True)
    if scale_std:
        adv = adv / (rewards.std(dim=1, keepdim=True) + eps)
    return adv


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Schulman k3 estimator of KL(policy || ref), per token, always >= 0."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


def grpo_loss(logp, old_logp, ref_logp, adv, mask, clip: float = 0.2, beta: float = 0.02, length_norm: str = "seq"):
    """logp/old_logp/ref_logp/mask: [B, T]; adv: [B] sequence-level advantage."""
    ratio = torch.exp(logp - old_logp)
    a = adv.unsqueeze(1)
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    per_tok = -(surr - beta * kl_k3(logp, ref_logp))
    m = mask.to(per_tok.dtype)
    if length_norm == "seq":
        per_seq = (per_tok * m).sum(1) / m.sum(1).clamp(min=1)
    elif length_norm == "const":
        per_seq = (per_tok * m).sum(1) / m.shape[1]
    else:
        raise ValueError(length_norm)
    with torch.no_grad():
        denom = m.sum().clamp(min=1)
        stats = {"kl": ((kl_k3(logp, ref_logp) * m).sum() / denom).item(),
                 "clip_frac": (((ratio - 1).abs() > clip).to(m.dtype) * m).sum().div(denom).item()}
    return per_seq.mean(), stats
