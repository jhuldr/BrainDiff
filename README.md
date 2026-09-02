# BrainDiff

Longitudinal comparison-report generation for multimodal brain MRI. Two studies of the same
patient go in and a report describing what changed between them comes out.

Built on NeuroVFM's vision stack — its ViT-B encoder and perceiver connector — with a Qwen3-14B
decoder, trained through a four-stage curriculum. Source code for *BrainDiff: Longitudinal Report
Generation for Multimodal Brain MRI*.

## Install

```bash
pip install -e .
```

## Reproduce the paper's numbers (no GPU, no data)

```bash
python paper/stats/all_cis.py         # every confidence interval  -> paper/stats/PAPER_CIS.md
python paper/stats/all_metrics.py     # every rg_er / BLEU-4 / METEOR -> paper/stats/PAPER_METRICS.md
python paper/figures/figure3_confusion/confusion_pair.py --classifier v2
pytest tests/test_reproduce.py        # checks all of the above 
```

These read the per-report score caches in `paper/cache/`. See `paper/README.md` for the
artifact-by-artifact map.

**What this repository does and does not support.** It supports *numeric* reproduction: every
interval and metric re-derives from the shipped caches. It does not support *generative*
reproduction (you cannot regenerate the reports those scores came from, because the MR-RATE and
BIND datasets are health-system data under privacy restriction and are not redistributed). Three
BIND rows are absent from `PAPER_CIS.md` for that reason; everything else is present and
matches. See `docs/DATA_TERMS.md`.

## Reusable probes — for any longitudinal report generator

Three of the diagnostics here are standalone tools, not reproduction scripts. They measure whether
a model actually reads the images, they import no model and assume no architecture, and they run
deterministically on CPU. If you build a system that writes comparison reports, you can measure
it with these.

```bash
# What direction of change does a report assert? (rule-based, no model in the loop)
python paper/probes/change_classifier.py --reports mine.json --out pred.json

# Does the asserted direction of change depend on which study is the prior?
python paper/probes/reversal_probe.py --reports mysystem.json

# How much of the score comes from the images, and how much from the prior report?
python paper/probes/factorial_probe.py \
    --present-own   real.json      --present-other   vis_roll.json \
    --withheld-own  noreport.json  --withheld-other  nr_vis_roll.json
```

All three take plain JSON — reports for the classifier, forward/reversed generations for the
reversal probe, four ablation conditions for the factorial. `docs/PROBES.md` has the input contracts, the label mapping, how a flip is decided,
and how to read the output. They score; generating the inputs means running your own model under
the interventions.

## Layout

```
src/braindiff/        the library
  models/             encoder, connector, change map, decoder, prompts, weight resolution
  data/               corpus reading, tokenisation, batching
  training/           the four curriculum stages, losses, freeze spec, checkpoint contract
  eval/               generation, RadGraph/BLEU/METEOR scoring, the interval-change classifier
  configs/            curriculum.yaml — every hyperparameter, one block per stage and arm
paper/                the deliverables: stats/, tables/, figures/, supplement/, cache/
  probes/             reusable tools — change classifier, reversal and factorial probes
dataset/              how the corpora were built: intake/, assembly/, reports/
scripts/              fetch_weights.py, strip_release_checkpoint.py
docs/                 PROBES.md, data terms, reproduction, code manifest
tests/                the golden-file reproduction test
```

## Weights

`checkpoints/braindiff_production.pt` (385 MB) holds 450
tensors, 100,705,793 parameters: encoder LoRA, the perceiver connector, decoder LoRA and the
temporal embeddings.

```bash
huggingface-cli login          # both NeuroVFM repositories are gated; accept their terms
python scripts/fetch_weights.py            # or --encoder for just the ViT (~274 MB)
```

`braindiff.models.paths` resolves everything through the standard HuggingFace cache, so the
download happens once. `HF_HUB_OFFLINE=1` afterwards keeps HuggingFace out of the loop, and
`BRAINDIFF_LLM_ROOT` points at a local copy if you already have one.

A correct load reports the backbone as absent. That is expected, not a partial load:

```
Loaded braindiff_production.pt: 48/48 tensors
    encoder              48/48
      of which LoRA      48/48
    encoder base absent from this checkpoint -- backbone comes from the HuggingFace
```

## Training

`docs/REPRODUCTION.md` has the full order. Training needs the dataset, which are not
redistributable, and `flash-attn` built for your GPU architecture — it is imported unguarded by
the NeuroVFM ViT, so a wheel without SASS for your card fails at the first attention call rather
than at import. RadGraph-XL pins `transformers<5` and runs in a separate environment from the
trainer; `requirements.txt` and `requirements-scoring.txt` are split for that reason.

## Citation

See `CITATION.cff`.

## Intended use

A research artifact. **Not a medical device**, not validated for clinical use, and not to be
used to inform patient care.
