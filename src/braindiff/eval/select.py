"""Checkpoint selection on a generated subset, scored with temporal_score.

Val cross-entropy is the wrong signal: on the old S3 run it bottomed at epoch 5 while the
best RadGraph checkpoint was epoch 10. Generating the full val set to score properly costs
~3 h, which is why it was never wired.

`temporal_score` is the only in-loop signal validated against RadGraph -- Pearson 0.998
within stage 2 over 14 cached checkpoints, picking the same best checkpoint, at 5.4 ms/report
on CPU. The cost is generation, not scoring, so generate a subset.

Subset regret, measured by replaying the cached checkpoints: at 256 pairs the pick often
differs from full-val RadGraph but costs a mean of 0.003 rg_er and at worst 0.014, against a
best-to-second gap of only 0.004. Selection is therefore two-stage: this keeps a shortlist
and drives early stopping, and the offline radgraph_score.py pass makes the final call.
"""
import torch
import torch.distributed as dist

from braindiff.data.report_text import strip_pathology_prefix
from braindiff.eval.temporal_score import score


@torch.no_grad()
def score_subset(raw_model, loader, device, max_new_tokens, n_pairs=256,
                 distributed=False, dual=False, no_repeat_ngram_size=0):
    """Greedy-generate up to `n_pairs` val samples and return triple-F1.

Greedy, not beam: this ranks checkpoints rather than shipping reports, and beam search costs
~3.5x for no benefit to the ranking. The subset is the first n_pairs of the val loader,
deterministic because the val sampler is built with shuffle=False, so the same samples are
scored every epoch.

`no_repeat_ngram_size=0` (default) disables n-gram blocking; stage 2 passes 20 to stop the
decoder looping whole sentences without touching clinical phrases that legitimately recur.
    """
    raw_model.eval()
    world = dist.get_world_size() if distributed else 1
    per_rank = max(1, n_pairs // world)

    hyps, refs = [], []
    for batch in loader:
        if len(hyps) >= per_rank:
            break
        kwargs = dict(
            tokens_main=batch["tokens_main"].to(device),
            coords_main=batch["coords_main"].to(device),
            present_main=batch["present_main"].to(device),
            max_new_tokens=max_new_tokens,
            num_beams=1,                      # greedy
            repetition_penalty=1.0,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if dual:
            kwargs.update(
                tokens_ref=batch["tokens_ref"].to(device),
                coords_ref=batch["coords_ref"].to(device),
                present_ref=batch["present_ref"].to(device),
            )
        if "prompt_ids" in batch:
            kwargs["prompt"] = batch["prompt_ids"].to(device)
            if "prompt_attn" in batch:
                kwargs["prompt_attn"] = batch["prompt_attn"].to(device)

        hyps.extend(raw_model.generate_caption_batch(**kwargs))
        refs.extend(batch["caption"])

    # Stage 2 targets carry a leading "Pathologies: ..." line. Score the report only,
    # or triple_f1 stops being comparable with earlier S2 checkpoints and with the
    # offline RadGraph pass. A no-op on every other stage.
    hyps = [strip_pathology_prefix(h) for h in hyps]
    refs = [strip_pathology_prefix(r) for r in refs]

    hyps, refs = hyps[:per_rank], refs[:per_rank]

    # Score locally, then average the per-rank means. Gathering the strings would
    # need object collectives for no gain -- triple_f1 is a per-pair mean, so the
    # mean of per-rank means over equal-sized shards is the same number.
    local = score(hyps, refs)
    value = torch.tensor([local["triple_f1"], float(local["n"])], device=device)
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value[0].item(), int(value[1].item()) * world
