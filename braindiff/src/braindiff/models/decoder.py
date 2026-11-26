"""Qwen3-14B decoder with attention LoRA.

Not neurovfm.models.vlm.LanguageModel: its __init__ patches the tokenizer for `outlines` and its generate() forces a JSON
schema. Weights come from the cached neurovfm-llm snapshot (models/encoder.py decoder_dir).
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from braindiff.models.encoder import decoder_dir

# Qwen3 attention projections, plain names (PEFT is applied to the inner `llm`). q_norm/k_norm intentionally unmatched.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


class Qwen3Decoder(nn.Module):
    """Qwen3-14B + tokenizer + the <|image_pad|> id the splice keys on. LoRA r 16 / alpha 32 / dropout 0.05, CAUSAL_LM."""

    def __init__(self, model_dir=None, attn_implementation="flash_attention_2", use_lora=True, lora_r=16,
                 lora_alpha=32, lora_dropout=0.05, dtype=torch.bfloat16):
        super().__init__()
        model_dir = model_dir or decoder_dir()
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype,
                                                            attn_implementation=attn_implementation,
                                                            low_cpu_mem_usage=True)
        except (ImportError, ValueError) as e:
            if attn_implementation != "flash_attention_2":
                raise
            print(f"[decoder] flash_attention_2 unavailable ({type(e).__name__}), falling back to sdpa")
            self.llm = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation="sdpa",
                                                            low_cpu_mem_usage=True)
        # padding_side='left' is load-bearing: the prefix/caption split, the logits_to_keep tail, and inputs_embeds
        # generation all assume it.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert self.image_pad_id is not None and self.image_pad_id >= 0, "<|image_pad|> missing (native Qwen3 token)"
        self.hidden_size = self.llm.config.hidden_size
        self.dtype_ = dtype
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            self.llm = get_peft_model(self.llm, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                                                           bias="none", task_type=TaskType.CAUSAL_LM,
                                                           target_modules=LORA_TARGETS))
            # Without this the checkpointed segment gets no grad-requiring input and every LoRA gradient is silently 0.
            self.llm.enable_input_require_grads()
        else:
            for p in self.llm.parameters():
                p.requires_grad = False
        # use_reentrant=False: reentrant checkpointing with DDP deadlocks or double-marks buckets.
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def embed(self, ids):
        return self.llm.get_input_embeddings()(ids)

    def forward(self, **kw):
        return self.llm(**kw)

    @torch.no_grad()
    def generate(self, inputs_embeds, attention_mask, **kw):
        kw.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        kw.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        return self.llm.generate(inputs_embeds=inputs_embeds.to(self.dtype_), attention_mask=attention_mask, **kw)
