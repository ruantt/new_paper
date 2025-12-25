from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from scoring.informativeness import combine, compute_boundary, compute_epistemic


def _disguide_loss(fake: torch.Tensor, student_ensemble, args) -> Tuple[torch.Tensor, Dict]:
    preds = []
    for idx in range(student_ensemble.size()):
        preds.append(student_ensemble(fake, idx=idx))
    preds = torch.stack(preds, dim=1)
    preds = F.softmax(preds, dim=2)
    std = torch.std(preds, dim=1)
    loss = -torch.mean(std)
    aux = {
        "disagreement": loss.detach().item(),
    }
    if args.lambda_div != 0:
        soft_vote_mean = torch.mean(torch.mean(preds + 0.000001, dim=1), dim=0)
        div_loss = torch.sum(soft_vote_mean * torch.log(soft_vote_mean))
        loss = loss + args.lambda_div * div_loss
        aux["diversity"] = div_loss.detach().item()
    return loss, aux


def _reduce_informativeness(score: torch.Tensor, cfg: Any) -> torch.Tensor:
    objective = getattr(cfg, "objective", "mean")
    if objective == "mean":
        return -score.mean()
    if objective == "topk_mean":
        ratio = getattr(cfg, "topk_ratio", None)
        if ratio is None:
            raise NotImplementedError("TODO: set topk_ratio when using topk_mean objective")
        k = max(1, int(ratio * score.shape[0]))
        topk = torch.topk(score, k=k)
        return -topk.values.mean()
    raise NotImplementedError(f"Unknown informativeness objective: {objective}")


def _compute_informativeness_prior(fake: torch.Tensor, student_ensemble, args, global_step: int):
    gen_cfg = getattr(args, "gen_cfg", SimpleNamespace())
    if not getattr(gen_cfg, "use_informativeness_prior", False):
        return None, {}

    scoring_cfg = getattr(args, "informativeness_cfg", None)
    if scoring_cfg is None:
        raise ValueError("Missing args.informativeness_cfg for informativeness prior")

    epistemic = compute_epistemic(fake, student_ensemble, scoring_cfg)
    boundary = compute_boundary(fake, student_ensemble, scoring_cfg)
    combined = combine(epistemic, boundary, scoring_cfg)

    schedule = getattr(gen_cfg, "informativeness_schedule", "none")
    base_weight = getattr(gen_cfg, "informativeness_weight", 0.0)
    warmup = getattr(gen_cfg, "informativeness_warmup", 1)
    if schedule == "linear":
        factor = min(1.0, max(0.0, float(global_step) / float(max(1, warmup))))
    elif schedule == "none":
        factor = 1.0
    else:
        raise NotImplementedError("TODO: add more schedules")
    scaled_weight = base_weight * factor

    if scaled_weight == 0:
        return None, {"prior_weight": scaled_weight}

    reduction = _reduce_informativeness(combined, gen_cfg)
    return scaled_weight * reduction, {
        "prior_weight": scaled_weight,
        "informativeness_mean": combined.mean().item(),
    }


def disguide_generator_step(args, z, generator, student_ensemble, global_step: int):
    fake = generator(z)
    base_loss, aux = _disguide_loss(fake, student_ensemble, args)
    prior_term, prior_meta = _compute_informativeness_prior(fake, student_ensemble, args, global_step)

    total_loss = base_loss
    if prior_term is not None:
        total_loss = total_loss + prior_term
        aux.update(prior_meta)

    total_loss.backward()
    aux["total"] = total_loss.detach().item()
    return total_loss.detach().item(), aux
