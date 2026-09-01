"""Shared loss pieces for the report-generation stages.

Both the single and dual trainers need these: stage 2 swaps the only scan, stage
4 swaps the prior, and the batch-pairing logic is identical either way.
"""
import torch
import torch.nn.functional as F


def make_swap_perm(groups, generator=None):
    """Permutation pairing each sample with one from a different group.

    A counterfactual only teaches anything if substituting the scan should change
    the report. Swapping two "Stable" priors (or two normal scans at stage 2) is
    a near-no-op, so those pairs would push the margin on samples whose true gap
    really is ~0 and inject noise.

    `groups` is `classification` at stage 4 and `pathology` at stage 2.
    BalancedBatchSampler spans all classes per batch, so a differing partner
    almost always exists; samples without one map to themselves and are flagged
    so the caller can drop them.

    Returns (perm [B], valid [B] bool).
    """
    differs = groups.unsqueeze(0) != groups.unsqueeze(1)                  # [B, B]
    has_partner = differs.any(dim=1)
    # Uniform over valid partners: random scores with invalid entries at -inf.
    scores = torch.rand(differs.shape, device=groups.device, generator=generator)
    scores = scores.masked_fill(~differs, float("-inf"))
    perm = scores.argmax(dim=1)
    identity = torch.arange(len(groups), device=groups.device)
    perm[~has_partner] = identity[~has_partner]
    return perm, has_partner


def focal_loss(logits, targets, alpha, gamma=2.0):
    """Class-balanced focal loss for the auxiliary change head.

    `alpha` is a per-class inverse-frequency weight. Without it the head just
    predicts Stable, which is 52% of the S4 split.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal = (1 - log_pt.exp()) ** gamma
    return (alpha[targets] * focal * -log_pt).mean()
