# Grounding probes

Two diagnostics for longitudinal report generators, packaged as standalone tools. They are the
part of this repository most likely to be useful to someone who is not reproducing our paper:
they measure *whether a model reads the images*, and neither one imports a model, assumes an
architecture, or requires our data.

Both consume plain JSON and are deterministic on CPU. Both are exercised against the paper's
published values by `tests/test_probes.py`.

| | question | needs |
|---|---|---|
| `reversal_probe.py` | does the asserted direction of change depend on which study is the prior? | forward and reversed generations |
| `factorial_probe.py` | how much of the score comes from the images versus the prior report? | four ablation conditions |
| `change_classifier.py` | what direction of change does this report assert? | reports, or predictions you already have |

The classifier underpins the other two, and is useful on its own: it turns free-text radiology
reports into interval-change labels without a model in the loop.

**They score; they do not generate.** Producing the inputs means running your own model under
the interventions described below. That is inherent — these measure behaviour under
intervention, so something has to perform the intervention.

---

## 1. Reversal probe

Generate a report for a study pair, generate a second with the two timepoints **swapped**, and
check whether the asserted direction of change flips. A model that reads the images flips. A
model paraphrasing the prior report is indifferent to the swap, and its flip rate stays near the
floor.

This is a held-out behavioural test: nothing optimises it during training.

### Input

One JSON file per system or arm — a list of objects:

```json
[
  {"gt": 3, "fwd": "Findings: ... Impression: Interval progression of the left frontal lesion.",
             "rev": "Findings: ... Impression: Interval decrease in lesion size."},
  {"gt": 4, "fwd": "...", "rev": "..."}
]
```

| field | meaning |
|---|---|
| `gt` | ground-truth change label, as an index (below) |
| `fwd` | report generated with the true (prior, current) ordering |
| `rev` | report generated with the two studies swapped, everything else held fixed |

Label indices are positions in `CHANGE_CLASSES`:

```
0 Stable   1 New lesion   2 Indeterminate   3 Progressed
4 Improved   5 Mixed interval change   6 Resolved
```

If your labels differ, map them onto these; only the four **directional** classes
(`1, 3, 4, 6`) are scored. A reversal has no direction to flip on a Stable, Indeterminate or
Mixed pair, so those rows are excluded by construction.

### Running it

```bash
python paper/probes/reversal_probe.py --reports mysystem.json

python paper/probes/reversal_probe.py \
    --reports armA.json armB.json \
    --labels "no counterfactual" "counterfactual + dropout"
```

Options: `--classifier {v1,v2,v5}` (v2 is the paper's), `--resamples`, `--seed`,
`--flips-out` to write the per-case arrays for your own analysis.

### How a flip is decided

Direction comes from the LLM-free rule classifier in `braindiff.eval.temporal_score`: the
report's Impression is read, an explicit statement of indeterminacy wins, and otherwise a
41-term change lexicon assigns `+1` (progression: *new*, *increased*, *enlargement*, …),
`-1` (improvement: *decreased*, *resolution*, *resolving*, …) or `0` (no direction asserted).

A case counts only if the **forward** report asserts a direction. Among those, a signed flip is
`sign(direction(rev)) != sign(direction(fwd))`. Cases whose forward report asserts nothing stay
in the output array as `NaN` rather than being dropped, so two systems remain index-aligned and
can be compared pair-by-pair.

Because the classifier is rule-based, scoring is reproducible and free: no GPU, no API, no
second model whose own errors would contaminate the measurement.

### Reading the result

```
arm                                       flip rate      n   cases
no counterfactual              0.2246 [0.1754, 0.2737]    285     503
counterfactual + dropout       0.2730 [0.2234, 0.3262]    282     503
```

`cases` is every directional ground-truth pair; `n` is the subset whose forward report asserted
a direction. Higher is better: it means reversing the images changed what the model claimed.
With two arms the probe also prints the unpaired difference (the form our paper reports) and the
paired difference over the cases both arms scored.

---

## 2. Prior-report x image factorial

Cross two interventions and read off where the score comes from:

|  | own scans | another patient's scans |
|---|---|---|
| **prior report present** | the full configuration | images swapped |
| **prior report withheld** | report removed | both removed |

- **image effect** = own − other, at fixed report availability. How much the images are worth.
- **prior-report effect** = present − withheld, at fixed image identity.
- **interaction** = how far the image effect shrinks once the report is available. Strongly
  negative means the report is substituting for the images.

We swap the images rather than zeroing them: zeroing is out of distribution and can collapse a
decoder into degenerate output, which is not a floor and is not comparable across models.

### Input

Four files, one per cell. Each is either of:

```json
{"rg_er": [0.41, 0.29, ...]}                     scores you already have — no GPU, no text
{"hyps": ["..."], "refs": ["..."]}               generations, scored here with RadGraph-XL
```

Rows must be index-aligned across the four: position *i* is the same study in every cell. With
`hyps`/`refs` the probe checks this by comparing reference text and refuses to proceed if they
disagree; with bare score arrays it can only be assumed, so keep your generation order fixed.

Any per-report metric works in the `rg_er` field — it is not required to be RadGraph. If you
supply your own array, the label is just a name.

### Running it

```bash
python paper/probes/factorial_probe.py \
    --present-own   real.json      --present-other   vis_roll.json \
    --withheld-own  noreport.json  --withheld-other  nr_vis_roll.json
```

Scoring from text needs the RadGraph environment (`pip install -r requirements-scoring.txt`)
and honours `RG_CUDA` to pick a device.

### Reading the result

```
                              own scans            other patient
prior report present     0.3837 [0.3769, 0.3902]   0.3450 [0.3382, 0.3519]
prior report withheld    0.2600 [0.2541, 0.2661]   0.2057 [0.2000, 0.2115]

image effect, report present   +0.0387 [+0.0324, +0.0449]
image effect, report withheld  +0.0544 [+0.0480, +0.0610]
interaction                    -0.0157 [-0.0231, -0.0083]
```

Intervals are paired bootstraps over reports, 10,000 resamples, seed 0.

---

## 3. Interval-change classifier

The rule classifier the reversal probe uses to decide direction, exposed as a tool. It reads a
report's Impression, honours an explicit statement of indeterminacy, and otherwise matches a
41-term change lexicon. Deterministic, ~5 ms per report, no GPU and no API — which is the point:
a learned classifier would contaminate whatever you are measuring with its own errors.

Seven classes, optionally collapsed to four by clinical direction:

```
Stable · Improved (+Resolved) · Worsened (Progressed +New lesion) · Mixed/unclear (Mixed +Indeterminate)
```

Three rule sets: `v1` (set-union over the whole report), `v2` (Impression-restricted with an
explicit-indeterminacy override — the paper's), `v5` (v2 plus mixed/unclear cue lists).

### Classify reports

```bash
python paper/probes/change_classifier.py --reports mine.json --out pred.json
python paper/probes/change_classifier.py --csv mine.csv --text-col report --label-col truth
```

`mine.json` is a list of strings, or of `{"report": ..., "label": ..., "id": ...}`. Supply
labels and it also prints per-class recall, balanced accuracy and a confusion matrix.

### Evaluate predictions you already have

```bash
python paper/probes/change_classifier.py --pred pred.json --labels truth.json
```

Both are `{id: class}` maps. This path needs no report text at all.

```
  n = 1444   balanced accuracy = 0.4166

                       Stable      Improved      Worsened  Mixed/unclea    recall
  Stable                  462            91           108            46     0.653
  Improved                 28           119            13            21     0.657
  Worsened                149            60            87            26     0.270
  Mixed/unclear            83            85            46            20     0.085
```

That is Figure 3(a) of the paper, reproduced from the shipped cache — `tests/test_probes.py`
asserts it.

### Validation, and what it is not

Validated against RadGraph-XL at Pearson 0.998 within a stage, and against LLM-assigned labels
on reference reports at 81.1 four-class balanced accuracy (v2, up from 72.2 for v1). It is a
measuring instrument with its own error, tuned to radiology report prose; on very differently
styled output, check it before trusting it. The vocabularies ship as package data
(`braindiff/eval/vocab/`), so the classifier is importable directly:

```python
from braindiff.eval.temporal_score import change_class_v2
change_class_v2("Impression: Interval decrease in lesion size.")   # -> 'Improved'
```

---

## Reproducing our numbers with them

Both probes regenerate the paper's values from the caches in `paper/cache/perreport/`:

```bash
python paper/probes/factorial_probe.py \
    --present-own   paper/cache/perreport/C_braindiff_real.json \
    --present-other paper/cache/perreport/C_braindiff_visroll.json \
    --withheld-own  paper/cache/perreport/C_braindiff_noreport.json \
    --withheld-other paper/cache/perreport/C_braindiff_nr_visroll.json
```

That prints Table 6 exactly. The reversal probe's cached flip arrays
(`paper/cache/perreport/reversal_*_v2.json`) were verified bit-identical to a fresh
recomputation from the generations before that text was removed from the release.

## Limitations

- **The rule classifier is a measuring instrument with its own error.** Validated against
  RadGraph-XL at Pearson 0.998 within a stage, and 4-class balanced accuracy against
  LLM-assigned labels on reference reports is 81.1 (v2). It is not perfect, and it is tuned to
  radiology report prose — on very differently styled output, check it before trusting it.
- **The reversal probe assumes swapping is meaningful.** If your model cannot be given the
  studies in the other order, the probe does not apply.
- The change-decodability probes under `paper/tables/table8_decodability/` are **not** in this
  category: they fit linear probes on frozen NeuroVFM features and are tied to that backbone.
