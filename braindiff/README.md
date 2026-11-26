# BrainDiff

A prior and a current brain MRI study (T1w, T1ce, T2w, FLAIR in template space) plus the prior report go in; a comparison report comes
out, opening with a status line that is cut before scoring. NeuroVFM encoder (blocks 10-12 trained), per-scan perceiver connector,
Qwen3-14B decoder with LoRA. Trained in two stages: Stage 1-D (status and entity tasks, epoch 5), then status-first report generation
(Stage 2, epoch 20). Decoding: greedy with an output-only repetition penalty of 1.1.

## Install

```bash
pip install -e .                       # trainer env; flash-attn must be built for your GPU (imported by the NeuroVFM ViT)
python scripts/fetch_weights.py        # BrainDiff checkpoint + gated NeuroVFM and Qwen3 bases (accept their terms first)
pip install -r requirements-scoring.txt   # separate scoring env: radgraph, transformers<5
pip install raddelta                   # Rad-Δ, released separately: label building, macro-MCC / agreement, Fig. 2
# also needed separately: neurovfm (github.com/mlinslab/neurovfm) for the model; antspyx and HD-BET for dataset/bind_register.py
```

## Layout

```
src/braindiff/
  models/      encoder.py, connector.py, decoder.py, prompts.py, stage1.py (Stage 1-D), braindiff.py (final model, generation)
  data/        transforms.py (volumes -> NeuroVFM tokens), datasets.py (Stage 1-D and Stage 2)
  training/    train_stage1.py, train_stage2.py
  eval/        generate.py, score.py (Table 2 per-pair metrics), temporal_score.py (F1temp), knows_says.py (Fig. 2)
  configs/     stage1.yaml, stage2.yaml
dataset/       bind_preprocess.py, bind_register.py, bind_qc_split.py, build_stage1.py, build_stage2.py
baselines/     generate.py, run.sh, finetune.py, out_rep_penalty.py (every brain Table 2 baseline)
  cxr/         cxr_generate.py, maira2_finetune.py (MIMIC-CXR rows: MAIRA-2, Libra; transformers 4.51.3 env)
paper/         table2.py, readers.py, reader_brief.txt, table2_golden.json, cache/, figures
scripts/       train.sh, test.sh, fetch_weights.py, strip_release_checkpoint.py
tests/         test_reproduce.py
```

Run order and reproducibility notes: [`../docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).
