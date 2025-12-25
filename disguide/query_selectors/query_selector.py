from typing import Any, Dict, Tuple

import torch

from scoring.informativeness import compute_boundary, compute_epistemic


def _compute_stats(scores: torch.Tensor, percentiles) -> Dict[str, float]:
    stats = {
        "mean": scores.mean().item(),
        "max": scores.max().item(),
        "min": scores.min().item(),
    }
    for p in percentiles:
        stats[f"p{int(p*100)}"] = torch.quantile(scores, p).item()
    return stats


def select(x_cand: torch.Tensor, ensemble, cfg: Any) -> Tuple[torch.Tensor, Dict]:
    """Posterior selector using MC-dropout + boundary scores."""
    meta: Dict[str, Any] = {
        "num_candidates": x_cand.shape[0],
        "mc": {},
        "boundary": {},
    }
    device = x_cand.device
    indices = torch.arange(x_cand.shape[0], device=device)

    working = x_cand
    working_indices = indices

    if getattr(cfg, "use_mc_uncertainty", False) and working.numel() > 0:
        epistemic = compute_epistemic(working, ensemble, cfg)
        percentiles = getattr(cfg, "percentiles", [0.5, 0.9])
        meta["mc"]["stats"] = _compute_stats(epistemic, percentiles)
        threshold_q = getattr(cfg, "mc_threshold", None)
        if threshold_q is not None:
            threshold = torch.quantile(epistemic, threshold_q)
            meta["mc"]["threshold"] = threshold.item()
            mask = epistemic >= threshold
            working = working[mask]
            working_indices = working_indices[mask]
            meta["mc"]["kept"] = int(mask.sum().item())
            meta["mc"]["keep_ratio"] = float(mask.float().mean().item())

    if getattr(cfg, "use_boundary_sampling", False) and working.numel() > 0:
        boundary_scores = compute_boundary(working, ensemble, cfg)
        percentiles = getattr(cfg, "percentiles", [0.5, 0.9])
        meta["boundary"]["stats"] = _compute_stats(boundary_scores, percentiles)
        ratio = getattr(cfg, "boundary_sample_ratio", None)
        if ratio is None:
            raise NotImplementedError("TODO: set boundary_sample_ratio in cfg for selector")
        k = max(1, int(ratio * working_scores_size(boundary_scores)))
        topk = torch.topk(boundary_scores, k)
        working = working[topk.indices]
        working_indices = working_indices[topk.indices]
        meta["boundary"]["k"] = k
        meta["boundary"]["keep_ratio"] = k / boundary_scores.shape[0]

    meta["selected_indices"] = working_indices
    meta["selected_ratio_total"] = working.shape[0] / max(1, x_cand.shape[0])
    return working, meta


def working_scores_size(scores: torch.Tensor) -> int:
    """Helper to decouple size logic for easy overrides."""
    return scores.shape[0]
