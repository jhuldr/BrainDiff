"""NeuroVFM vision encoder, pretrained-weight resolution, and the blocks 1-9 / 10-12 split.

NOT a HuggingFace model (`AutoModel` cannot load it) and packed variable-length: no dense [B,C,D,H,W] path. Blocks 1-9
are frozen; blocks 10-12 + final norm are trained in Stage 1-D in fp32 (NeuroVFM ships qkv/proj in bf16, where small
AdamW steps round away) and frozen in Stage 2.

Weights resolve through the HF cache; HF_HUB_OFFLINE=1 never contacts the Hub. Gated repos:
  mlinslab/neurovfm-encoder   config.json + pytorch_model.bin        ~273 MiB
  mlinslab/neurovfm-llm       vision_connector.pt, language_model/   ~30 GB
Load the ViT from the encoder repo, not neurovfm-llm/vision_encoder.pt (a bf16 round-trip of it).
BRAINDIFF_LLM_ROOT points at a local copy of neurovfm-llm instead.
"""
import json
import os
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

ENCODER_REPO = "mlinslab/neurovfm-encoder"
LLM_REPO = "mlinslab/neurovfm-llm"
GRID = (12, 14, 12)
N_TOKENS = GRID[0] * GRID[1] * GRID[2]      # 2016
# MRI normalisation stats from neurovfm/systems/utils.py (index 0 = mri), as upstream's EncoderPipeline.embed.
MRI_MEAN, MRI_STD = 0.3141, 0.2623


@lru_cache(maxsize=None)
def llm_root() -> str:
    """Local snapshot dir of mlinslab/neurovfm-llm (BRAINDIFF_LLM_ROOT overrides). Downloads once, then cached."""
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
    return hf_hub_download(ENCODER_REPO, "config.json"), hf_hub_download(ENCODER_REPO, "pytorch_model.bin")


class NeuroVFMEncoder(nn.Module):
    """The frozen NeuroVFM ViT-B (`self.model`), loaded strictly from the encoder repo; kept out of every optimizer
    except the blocks a model unfreezes itself."""

    def __init__(self):
        super().__init__()
        from neurovfm.models import get_vit_backbone
        cfg_path, w_path = encoder_files()
        config = json.loads(open(cfg_path).read())
        model = get_vit_backbone(**config)
        state = torch.load(w_path, map_location="cpu")
        state = state.get("state_dict", state)
        incompatible = model.load_state_dict(state, strict=False)
        # strict: a renamed checkpoint must not load as a random encoder.
        if incompatible.missing_keys:
            raise RuntimeError(f"{ENCODER_REPO} left {len(incompatible.missing_keys)} keys uninitialised, e.g. "
                               f"{incompatible.missing_keys[:5]}. Refusing to run on partly random weights.")
        self.output_dim = config["params"]["embed_dim"]
        for p in model.parameters():
            p.requires_grad = False
        self.model = model


class _Stop(Exception):
    pass


def run_top(vit, x):
    """blocks 10-12 + final LayerNorm on block-9 streams x [B, 2016, 768]."""
    b, n, d = x.shape
    cu = torch.arange(0, (b + 1) * n, n, dtype=torch.int32, device=x.device)
    h, res = x.reshape(b * n, d), None
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for blk in vit.blocks[9:]:
            h, res, _ = blk(h, cu_seqlens=cu, max_seqlen=n, residual=res, use_flash_attn=True)
    s = h.float() + res.float()
    return F.layer_norm(s, (d,), vit.norm.weight.float(), vit.norm.bias.float(), vit.norm.eps).view(b, n, d)


class TopBlocks:
    """Mixin for a model with `self.encoder` (NeuroVFMEncoder): fp32 top blocks, the token grid, block9 / encode."""

    def init_top_blocks(self):
        vit = self.encoder.model
        for m in list(vit.blocks[9:]) + [vit.norm]:      # fp32 masters, as Stage 1-D trained and saved them
            m.float()
        g = torch.stack(torch.meshgrid(*[torch.arange(k) for k in GRID], indexing="ij"), -1)
        self.register_buffer("coords", g.reshape(-1, 3), persistent=False)

    @torch.no_grad()
    def block9(self, tokens):
        """tokens [V, 2016, 1024] raw -> block-9 residual stream [V, 2016, 768] fp32, frozen."""
        vit, v = self.encoder.model, tokens.shape[0]
        x = (tokens.reshape(v * N_TOKENS, -1).float() - MRI_MEAN) / MRI_STD
        c = self.coords.repeat(v, 1)
        cu = torch.arange(0, (v + 1) * N_TOKENS, N_TOKENS, dtype=torch.int32, device=x.device)
        cap = {}

        def hook(m, args, out):
            cap["b9"] = out[0].float() + out[1].float()
            raise _Stop
        h = vit.blocks[8].register_forward_hook(hook)
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vit(x, c, cu_seqlens=cu, max_seqlen=N_TOKENS, use_flash_attn=True)
        except _Stop:
            pass
        finally:
            h.remove()
        return cap["b9"].view(v, N_TOKENS, -1)

    def encode(self, tokens, chunk: int = 64):
        """tokens [V, 2016, 1024] -> final features [V, 2016, 768] (grad through blocks 10-12 + norm)."""
        x = torch.cat([self.block9(tokens[i:i + chunk]) for i in range(0, len(tokens), chunk)])
        return run_top(self.encoder.model, x)
