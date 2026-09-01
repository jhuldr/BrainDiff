"""NeuroVFM vision encoder.

V-JEPA2 at crop_size=96 gave one token per ~38 mm of a 229 mm FOV, while the reports
describe lesions of median 8 mm. NeuroVFM's 4x16x16 patches at its 1x1x4 mm pretraining
spacing give ~16 mm tokens instead.

NOT a HuggingFace model. The published config is `{"which": "vit", "params": {...}}`, so
`AutoModel` cannot load it, and it is packed variable-length: there is no dense
[B,C,D,H,W] path upstream. Tokenization happens in the dataloader; this module takes
already-tokenized input. Every volume is template-space on the same 12x14x12 = 2016 token
grid, so a batch packs, runs, and reshapes back to the [B, N, D] contract.

LoRA works on attention ONLY. Verified by perturbing lora_B and measuring the output:
mixer.qkv/mixer.proj move it by 0.073, mlp.fc1/fc2 by exactly 0.000000 -- FusedMLP reads
raw weights instead of calling the submodules, so MLP LoRA attaches and silently does
nothing.
"""
import json

import torch
import torch.nn as nn

MODEL_ID = "mlinslab/neurovfm-encoder"

# MRI normalization stats from neurovfm/systems/utils.py (index 0 = mri). Applied
# here rather than in the dataloader to match upstream's EncoderPipeline.embed,
# and so a caller cannot forget them.
MRI_MEAN, MRI_STD = 0.3141, 0.2623

# Attention only. `proj` is deliberately scoped under blocks.* so it cannot also
# match token_embed.proj.
LORA_TARGETS = r".*blocks\.\d+\.mixer\.(qkv|proj)$"


class NeuroVFMEncoder(nn.Module):
    """forward(tokens, coords) -> [B, N, D].

Args:
    tokens: [B, N, 1024] raw patch values (4*16*16), unnormalized
    coords: [B, N, 3] integer patch-grid indices, volume-local

The frozen backbone stays out of every optimizer, which sidesteps its mixed bf16/fp32
parameter dtypes and its unconditional gradient checkpointing (keyed off self.training
upstream, not configurable).
    """

    def __init__(
        self,
        model_name: str = MODEL_ID,
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        normalize: bool = True,
        lora_trainable: bool = True,
    ):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from neurovfm.models import get_vit_backbone

        config = json.loads(open(hf_hub_download(model_name, "config.json")).read())
        model = get_vit_backbone(**config)

        state = torch.load(hf_hub_download(model_name, "pytorch_model.bin"),
                           map_location="cpu")
        state = state.get("state_dict", state)
        incompatible = model.load_state_dict(state, strict=False)
        # Loud rather than silent: strict=False would otherwise let a renamed
        # checkpoint through as a randomly-initialised encoder.
        if incompatible.missing_keys:
            raise RuntimeError(
                f"{model_name} left {len(incompatible.missing_keys)} keys uninitialised, "
                f"e.g. {incompatible.missing_keys[:5]}. Refusing to run on partly "
                f"random weights."
            )

        self.output_dim = config["params"]["embed_dim"]
        self.normalize = normalize
        self.use_lora = use_lora

        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False

        if use_lora:
            from peft import LoraConfig, get_peft_model
            # task_type must be None: FEATURE_EXTRACTION wraps this in
            # PeftModelForFeatureExtraction, which injects input_ids= and breaks
            # the plain nn.Module signature.
            model = get_peft_model(model, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=LORA_TARGETS, bias="none", task_type=None,
            ))
            # Building LoRA and TRAINING LoRA are different decisions, and
            # conflating them silently breaks the curriculum. get_peft_model
            # rewrites every key (blocks.0.mixer.qkv.weight ->
            # ...base_layer.weight plus lora_A/lora_B), so a later stage built
            # without LoRA matches ZERO of an earlier stage's 184 encoder tensors
            # and falls back to the bare HF weights -- discarding all curriculum
            # adaptation without raising. Stages that freeze the encoder must
            # still BUILD it the same way, and set lora_trainable=False.
            if not lora_trainable:
                for p in model.parameters():
                    p.requires_grad = False

        self.model = model

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, tokens: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        x = tokens.reshape(b * n, -1).float()
        if self.normalize:
            x = (x - MRI_MEAN) / MRI_STD
        c = coords.reshape(b * n, 3).long()
        # Uniform lengths, so cu_seqlens is just the multiples of n. flash-attn
        # varlen needs int32 offsets and the max length.
        cu = torch.arange(0, (b + 1) * n, n, dtype=torch.int32, device=tokens.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(x, c, cu_seqlens=cu, max_seqlen=n, use_flash_attn=True)
        return out.float().view(b, n, self.output_dim)
