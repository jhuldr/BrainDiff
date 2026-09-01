# Reproduction

## From this repository, no GPU

```bash
pip install -e .
python paper/stats/all_cis.py         # -> paper/stats/PAPER_CIS.md
python paper/stats/all_metrics.py     # -> paper/stats/PAPER_METRICS.md
python paper/figures/figure3_confusion/confusion_pair.py --classifier v2
pytest tests/test_reproduce.py
```

## The probes, on your own system

The three grounding tools are model-agnostic and need none of our data:

```bash
python paper/probes/change_classifier.py --reports mine.json --out pred.json
python paper/probes/reversal_probe.py --reports mysystem.json
python paper/probes/factorial_probe.py \
    --present-own real.json --present-other vis_roll.json \
    --withheld-own noreport.json --withheld-other nr_vis_roll.json
```

Input contracts and interpretation: `docs/PROBES.md`.

## With the released weights

```bash
huggingface-cli login                 # NeuroVFM repositories are gated
python scripts/fetch_weights.py
```

`checkpoints/braindiff_production.pt` holds our trained weights only; the backbone and decoder
come from the Hub. A correct load reports the encoder as LoRA-only.

## From scratch

Training needs the MR-RATE, OASIS-3, BraTS and BIND corpora, which are not
redistributable (see `docs/DATA_TERMS.md`), plus `flash-attn` built for your GPU architecture.

```bash
# 1. Train (4 GPUs). The stage name selects the block in curriculum.yaml.
torchrun --standalone --nproc_per_node=4 -m braindiff.training.curriculum \
    --track neurovfm --stage nv_stage1_unified.pt          # then S2, S3, S4

# 2. Generate on the test split (4 shards, then merge)
python -m braindiff.eval.probe_delta_rger \
    --ckpt checkpoints/braindiff_production.pt \
    --diff-checkpoint "" --include-delta 0 \
    --split test --modes real,vis_roll --n-batches 242 \
    --out-dir logs/<arm> --gpu $S --shard-index $S --num-shards 4
python -m braindiff.eval.merge_rger_shards --out-dir logs/<arm> --modes real,vis_roll
#   add --no-report for the withheld-report condition; --modes scan_zero for images zeroed

# 3. Score (separate env; RadGraph-XL pins transformers<5)
python paper/stats/rescore_canonical.py
python paper/stats/rescore_baselines.py

# 4. All intervals and all metrics, one seed
python paper/stats/all_cis.py            # -> paper/stats/PAPER_CIS.md
python paper/stats/all_metrics.py        # -> paper/stats/PAPER_METRICS.md

# 5. Figures and per-table scripts (see paper/README.md)
python paper/figures/figure3_confusion/confusion_pair.py --classifier v2
python paper/tables/table2_baselines/frontier_eval.py    # scoring env: rg_er + BLEU-4 + METEOR
```

Steps 2-3 need the corpora and model weights; steps 4-5 run from the shipped caches alone.

Every hyperparameter lives in `src/braindiff/configs/curriculum.yaml`, one block per stage and
per ablation arm.

Generation and scoring run in two different environments, because RadGraph-XL pins
`transformers<5` and the trainer needs 5.x:

```bash
pip install -r requirements.txt            # trainer
pip install -r requirements-scoring.txt    # RadGraph, separate env
```
