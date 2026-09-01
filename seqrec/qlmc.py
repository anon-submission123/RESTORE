"""Inference-time QLMC score correction used by Ours."""

from __future__ import annotations

import torch


def qlmc_scores(
    base_scores: torch.Tensor,
    full_scores: torch.Tensor,
    seen: torch.Tensor,
    *,
    local_top_l: int = 100,
    lambda_q: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if base_scores.shape != full_scores.shape or seen.shape != (base_scores.shape[1],):
        raise ValueError("QLMC inputs have inconsistent shapes")
    if local_top_l <= 0 or lambda_q < 0:
        raise ValueError("invalid QLMC parameters")
    k = min(local_top_l, int(seen.sum()))
    if k == 0:
        return full_scores.clone(), torch.zeros(base_scores.shape[0], device=base_scores.device)
    top = base_scores.masked_fill(~seen.unsqueeze(0), -torch.inf).topk(k, dim=1).indices
    base_local = torch.gather(base_scores, 1, top)
    full_local = torch.gather(full_scores, 1, top)
    delta = (torch.logsumexp(full_local, 1) - torch.logsumexp(base_local, 1)).clamp_min(0)
    corrected = full_scores - lambda_q * delta.unsqueeze(1) * seen.to(full_scores.dtype).unsqueeze(0)
    return corrected, delta
