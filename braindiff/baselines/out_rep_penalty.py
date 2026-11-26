"""HF-semantics repetition penalty for the vLLM baselines: penalises OUTPUT tokens only (vLLM's built-in one also
penalises prompt tokens, which would charge the rows for copying the prior report). HF's rule: logit < 0 -> * p, else / p.
Per request via extra_body {"vllm_xargs": {"out_rep_penalty": 1.1}}; server flag
    --logits-processors out_rep_penalty:OutputRepetitionPenalty   (PYTHONPATH=baselines)"""
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


def _penalise(p, out_ids, logits):
    if not out_ids:
        return logits
    ix = torch.tensor(sorted(set(out_ids)), device=logits.device)
    s = logits[ix]
    logits[ix] = torch.where(s < 0, s * p, s / p)
    return logits


class OutputRepetitionPenalty(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, params: SamplingParams):
        p = (params.extra_args or {}).get("out_rep_penalty")
        if p is not None and float(p) <= 0:
            raise ValueError("out_rep_penalty must be > 0")

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        p = (params.extra_args or {}).get("out_rep_penalty")
        if p is None or float(p) == 1.0:
            return None
        p = float(p)
        return lambda out_ids, logits: _penalise(p, out_ids, logits)
