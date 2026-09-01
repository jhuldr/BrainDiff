"""BrainDiff_NeuroVFM braindiff.models.

flash-attn is a hard dependency and is checked here so the failure is one readable line
instead of an ImportError three frames deep inside neurovfm/models/vit.py, which imports it
at module scope with no try/except and builds every attention qkv/proj as FusedDense.
`use_flash_attn=False` only swaps the attention kernel, not the imports.

Consequence: the ViT cannot run on CPU at all -- flash-attn routes LayerNorm through a
Triton kernel that rejects CPU pointers.
"""
try:
    from flash_attn.ops.fused_dense import FusedDense as _FusedDense
    if _FusedDense is None:
        raise ImportError("flash_attn imported but FusedDense is None")
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "flash-attn is required: neurovfm/models/vit.py imports it at module scope "
        "and builds every attention projection as FusedDense. There is no fallback. "
        f"Install it into this environment (pip install flash-attn). Original: {e}"
    ) from e
