"""Single place that resolves pretrained weights, always through the HF cache.

Everything downloads once into ~/.cache/huggingface/hub and every later call resolves from
disk. With HF_HUB_OFFLINE=1 the Hub is never contacted, which also stops 4 ranks issuing 8
revision checks that hang if the Hub is unreachable.

Two gated repos (the token in ~/.cache/huggingface/token is used automatically):

  mlinslab/neurovfm-encoder   config.json + pytorch_model.bin        ~273 MiB
  mlinslab/neurovfm-llm       vision_connector.pt, language_model/   ~30 GB

The ViT is loaded from the ENCODER repo, not from neurovfm-llm/vision_encoder.pt. Both hold
the same 136 tensors, but vision_encoder.pt is a bf16 round-trip of the encoder repo's copy
(verified A[k] == bf16(B[k]); the 36 that match are mixer.{qkv,proj}, already bf16 in both).
Forward difference on 2016 real tokens is cosine 0.999945/token.
"""
import os
from functools import lru_cache

ENCODER_REPO = "mlinslab/neurovfm-encoder"
LLM_REPO = "mlinslab/neurovfm-llm"


@lru_cache(maxsize=None)
def llm_root() -> str:
    """Local snapshot dir of mlinslab/neurovfm-llm. Downloads once, then cached.

    Override with BRAINDIFF_LLM_ROOT to point at a local copy.
    """
    override = os.environ.get("BRAINDIFF_LLM_ROOT")
    if override:
        return override
    from huggingface_hub import snapshot_download
    return snapshot_download(LLM_REPO)


def decoder_dir() -> str:
    """<snapshot>/language_model -- the Qwen3-14B weights + tokenizer."""
    return os.path.join(llm_root(), "language_model")


def connector_path() -> str:
    """<snapshot>/vision_connector.pt -- pretrained perceiver + projection MLP."""
    return os.path.join(llm_root(), "vision_connector.pt")


@lru_cache(maxsize=None)
def encoder_files() -> tuple:
    """(config.json, pytorch_model.bin) for the ViT backbone. Downloads once."""
    from huggingface_hub import hf_hub_download
    return (hf_hub_download(ENCODER_REPO, "config.json"),
            hf_hub_download(ENCODER_REPO, "pytorch_model.bin"))
