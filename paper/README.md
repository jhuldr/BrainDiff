# Paper deliverables

One directory per artifact. Every number the paper prints is either regenerated here from the
caches in `cache/`, or documented as requiring data that cannot be redistributed.

## Start here

```bash
python paper/stats/all_cis.py        # -> stats/PAPER_CIS.md      every confidence interval
python paper/stats/all_metrics.py    # -> stats/PAPER_METRICS.md  every rg_er / BLEU-4 / METEOR
python paper/figures/figure3_confusion/confusion_pair.py --classifier v2
```

Those three need no GPU, no network and no corpora. `pytest tests/test_reproduce.py` checks
their output against committed goldens.

## Map

Two different questions, so two columns. *Numbers reproduce* means the paper's values for that
artifact come out of the shipped caches. *Script runs* means the original per-artifact script
executes here — most do not, because they regenerate reports and need the corpora.

| artifact | numbers reproduce? | script runs? | where |
|---|---|---|---|
| All confidence intervals | **yes** — `stats/all_cis.py` | yes | `stats/` |
| All rg_er / BLEU-4 / METEOR | **yes** — `stats/all_metrics.py` | yes | `stats/` |
| Figure 3 — interval-change confusion | **yes** | yes | `figures/figure3_confusion/` |
| Table 4 — counterfactual + reversal probe | **yes** — both flip rates and all CIs, from `cache/perreport/reversal_*_v2.json` | no — `table3_reversal_v2.py` generates reports, so it needs a checkpoint, a GPU and the volumes | `tables/table4_reversal/` |
| Table 6 — prior-report x image 2x2 | **yes** — all four cells, both image effects, the prior-report effects and the interaction | no — `table2_score.py` and `table2_floor.py` re-score report text that is not shipped | `tables/table6_factorial/` |
| Table 2 — baselines and frontier models | partly — the scored rows are in `PAPER_METRICS.md` | no — generation needs API keys and the corpora | `tables/table2_baselines/` |
| Table 3 — BIND external validation | **no** — the three BIND rows are absent | no — BIND is not redistributable | `tables/table3_bind/` |
| Table 8 — change decodability | AUROCs are quoted in `PAPER_CIS.md` | no — needs BraTS masks and a GPU | `tables/table8_decodability/` |
| Supplement — real-Impression validation | no | no — needs report text | `supplement/` |

### The probes are reusable — run them on any system

`paper/probes/` holds the two grounding diagnostics as standalone instruments. Neither imports
a model or assumes BrainDiff: they consume plain JSON, so any generative VLM that writes
longitudinal comparison reports can be measured the same way.

**Reversal probe** — does the asserted direction of change actually depend on which study is the
prior? Input is one file per system, a list of `{"gt": <7-way label index>, "fwd": <report>,
"rev": <report with the timepoints swapped>}`:

```bash
python paper/probes/reversal_probe.py --reports mysystem.json
python paper/probes/reversal_probe.py --reports armA.json armB.json --labels nocf cf+dropout
```

Only ground-truth directional change cases count, and among those only the ones whose forward
report asserts a direction at all. Direction comes from the LLM-free rule classifier, so scoring
is deterministic and needs no GPU. `--classifier {v1,v2,v5}` selects the rule set; v2 is the
paper's.

**Factorial probe** — where does the score come from: the images or the prior report? Input is
four cells, each either `{"rg_er": [...]}` if you already have per-report scores, or
`{"hyps": [...], "refs": [...]}` to be scored here with RadGraph-XL:

```bash
python paper/probes/factorial_probe.py \
    --present-own real.json      --present-other  vis_roll.json \
    --withheld-own noreport.json --withheld-other nr_vis_roll.json
```

Reports the four cells, both image effects, both prior-report effects and the interaction, each
with a paired bootstrap interval.

Both reproduce the paper exactly from the shipped caches — see `tests/test_probes.py`. What is
*not* portable is generating the inputs: producing `fwd`/`rev` pairs, or the four factorial
cells, requires running your own model over your own data. The probes score; they do not
generate.

### Getting Table 4 and Table 6 for BrainDiff


Both come out of the intervals script; there is no separate command:

```bash
python paper/stats/all_cis.py
sed -n '/^## Table 4/,/^## Table 5/p' paper/stats/PAPER_CIS.md   # reversal probe
sed -n '/^## Table 6/,/^## Table 7/p' paper/stats/PAPER_CIS.md   # the 2x2 factorial
```

Table 4 gives the two flip rates with their intervals (n=285 and n=282 of the 503 directional
change cases), the paired and unpaired differences, and the counterfactual gain. Table 6 gives
all four cells of the factorial, both image effects, both prior-report effects and the
interaction.

## Why some of it does not run

This repository supports **numeric reproduction, not generative reproduction**. The per-report
scores in `cache/perreport/` let you re-derive every interval and metric; they do not let you
regenerate the reports those scores came from, because the MR-RATE and BIND corpora are
health-system data under privacy restriction. See `docs/DATA_TERMS.md`.

Three rows are therefore absent from `PAPER_CIS.md` relative to the paper — the BIND rg_er, its
image contribution, and the BIND metric floor. Everything else is present and matches.

## cache/

| | |
|---|---|
| `perreport/` | 48 arms: per-report rg_er, uids, corpus BLEU-4, mean METEOR, source dump |
| `s4_test.csv` | 1,464 test pairs — identifiers and labels, no report text |
| `subset_64.csv` | the frozen 64-study frontier subset |
| `confusion/` | cached classifier predictions behind Figure 3 |
| `outputs/` | baseline system outputs, identifiers and labels only |
