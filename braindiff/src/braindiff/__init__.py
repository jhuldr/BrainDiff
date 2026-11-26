"""BrainDiff: longitudinal comparison-report generation for brain MRI (NeuroVFM encoder + per-scan connector +
Qwen3-14B with LoRA; Stage 1-D then status-first Stage 2).

flash-attn is a hard dependency of the model code (the NeuroVFM ViT cannot run on CPU); build it for your GPU.
braindiff.eval.score and braindiff.eval.temporal_score do not import the model code and run in the scoring environment
(radgraph, transformers < 5).
"""
