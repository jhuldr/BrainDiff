"""Thin Qwen3-14B wrapper -- everything the captioner needs, nothing it doesn't.

We deliberately do NOT use neurovfm.models.vlm.LanguageModel. It imports fine, but its
__init__ monkey-patches the tokenizer for `outlines` and its generate() forces a JSON
schema; we want neither. What we keep from it is the two conventions that matter:
left padding, and generation from inputs_embeds only.

Weights come from the cached snapshot (models/paths.py) -- downloaded once, never again.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from braindiff.models.paths import decoder_dir

# Qwen3 attention projections. Plain names, because PEFT is applied to the inner `llm`.
# NOT BrainDiff's `.*language_model.*` regex -- there is no such submodule here and no
# vision tower to exclude. q_norm/k_norm (per-head RMSNorm) are intentionally unmatched.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class Qwen3Decoder(nn.Module):
    """Qwen3-14B + tokenizer + the <|image_pad|> id the splice keys on."""

    def __init__(self, model_dir: str = None,
                 attn_implementation: str = "flash_attention_2",
                 use_gradient_checkpointing: bool = True,
                 use_lora: bool = False, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, dtype=torch.bfloat16):
        super().__init__()
        model_dir = model_dir or decoder_dir()

        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=dtype, attn_implementation=attn_implementation,
                low_cpu_mem_usage=True)
        except (ImportError, ValueError) as e:
            if attn_implementation == "flash_attention_2":
                print(f"[decoder] flash_attention_2 unavailable ({type(e).__name__}), "
                      f"falling back to sdpa")
                attn_implementation = "sdpa"
                self.llm = AutoModelForCausalLM.from_pretrained(
                    model_dir, dtype=dtype, attn_implementation=attn_implementation,
                    low_cpu_mem_usage=True)
            else:
                raise

        # padding_side='left' is load-bearing: the prefix/caption split, the
        # logits_to_keep tail, and inputs_embeds generation all assume it.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert self.image_pad_id is not None and self.image_pad_id >= 0, \
            "<|image_pad|> missing -- this is a native Qwen3 token, no vocab resize needed"

        self.hidden_size = self.llm.config.hidden_size
        self.dtype_ = dtype
        self.use_lora = use_lora

        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             bias="none", task_type=TaskType.CAUSAL_LM,
                             target_modules=LORA_TARGETS)
            self.llm = get_peft_model(self.llm, cfg)
            # Without this the checkpointed segment gets no grad-requiring input and
            # every LoRA gradient is silently zero -- no error, just a run that does
            # nothing. Step 7's grad audit exists to catch a regression here.
            self.llm.enable_input_require_grads()
        else:
            for p in self.llm.parameters():
                p.requires_grad = False

        if use_gradient_checkpointing:
            # use_reentrant=False is required: reentrant checkpointing with DDP
            # find_unused_parameters=True deadlocks or double-marks buckets.
            self.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def forward(self, **kw):
        return self.llm(**kw)

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, **kw):
        kw.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kw.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return self.llm.generate(inputs_embeds=inputs_embeds.to(self.dtype_),
                                 attention_mask=attention_mask, **kw)
