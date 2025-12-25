import contextlib
from typing import Any, Dict, Tuple

import torch


def _resolve_differentiable(cfg: Any) -> bool:
    """Return whether scoring should keep gradients."""
    return bool(getattr(cfg, "differentiable_version", False))


@contextlib.contextmanager
def _maybe_no_grad(cfg: Any):
    """Disable grads by default; allow opt-in via cfg."""
    differentiable = _resolve_differentiable(cfg)
    with torch.enable_grad() if differentiable else torch.no_grad():
        yield


def _toggle_dropout(ensemble) -> Tuple[list, list]:
    """Enable dropout layers while keeping the ensemble in eval mode."""
    original_train_flags = []
    dropout_modules = []
    for module in ensemble.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            dropout_modules.append(module)
            original_train_flags.append(module.training)
            module.train()
    return dropout_modules, original_train_flags


def _restore_dropout(dropout_modules, original_train_flags):
    for module, flag in zip(dropout_modules, original_train_flags):
        module.train(flag)


def compute_epistemic(x: torch.Tensor, ensemble, cfg: Any) -> torch.Tensor:
    """Wrapper around Ensemble.get_epistemic_uncertainty."""
    with _maybe_no_grad(cfg):
        prev_mode = ensemble.training
        ensemble.eval()
        dropout_modules, flags = _toggle_dropout(ensemble)
        try:
            mc_samples = getattr(cfg, "mc_samples", None)
            epistemic = ensemble.get_epistemic_uncertainty(x, mc_samples=mc_samples)
        finally:
            _restore_dropout(dropout_modules, flags)
            ensemble.train(prev_mode)
    return epistemic


def compute_boundary(x: torch.Tensor, ensemble, cfg: Any) -> torch.Tensor:
    """Wrapper around Ensemble.get_boundary_score."""
    with _maybe_no_grad(cfg):
        prev_mode = ensemble.training
        ensemble.eval()
        boundary = ensemble.get_boundary_score(x)
        ensemble.train(prev_mode)
    return boundary


def combine(epistemic: torch.Tensor, boundary: torch.Tensor, cfg: Any) -> torch.Tensor:
    """Combine epistemic + boundary into informativeness score.

    The actual combination recipe is fully controlled by cfg and should be tuned externally.
    """
    strategy = getattr(cfg, "combine_strategy", "weighted_sum")
    if strategy == "weighted_sum":
        weight_cfg = getattr(cfg, "weighted_sum_cfg", None)
        if weight_cfg is None:
            raise NotImplementedError("TODO: provide cfg.weighted_sum_cfg with weights")
        w_e = getattr(weight_cfg, "w_epistemic", None)
        w_b = getattr(weight_cfg, "w_boundary", None)
        if w_e is None or w_b is None:
            raise NotImplementedError("TODO: set w_epistemic and w_boundary in cfg.weighted_sum_cfg")
        return w_e * epistemic + w_b * boundary

    if strategy == "gate_then_rank":
        gate_cfg = getattr(cfg, "gate_then_rank_cfg", None)
        if gate_cfg is None:
            raise NotImplementedError("TODO: provide cfg.gate_then_rank_cfg with gating values")
        gate_quantile = getattr(gate_cfg, "epistemic_gate", None)
        topk_ratio = getattr(gate_cfg, "boundary_topk_ratio", None)
        if gate_quantile is None or topk_ratio is None:
            raise NotImplementedError("TODO: set epistemic_gate and boundary_topk_ratio for gate_then_rank")
        threshold = torch.quantile(epistemic, gate_quantile)
        mask = epistemic >= threshold
        gated_boundary = boundary.clone()
        gated_boundary[~mask] = float("-inf")
        k = max(1, int(topk_ratio * boundary.shape[0]))
        _, indices = torch.topk(gated_boundary, k=k)
        score = torch.full_like(boundary, fill_value=float("-inf"))
        score[indices] = boundary[indices]
        return score

    raise NotImplementedError(f"Unknown combine strategy: {strategy}")
