# Data terms

## This repository contains no radiology report text

No MR-RATE source reports, no LLM-written comparison targets, no model generations. What ships
is identifiers, derived labels and numeric scores.

| file | content |
|---|---|
| `assets/s4_test.csv` | 1,464 test pairs: `patient_uid`, `study_uid1`, `study_uid2`, `batch1/2`, `duration`, `pathology1/2`, `classification` |
| `benchmarks/mrrate_proprietary/subset_64.csv` | the frozen 64-study frontier subset, identifiers + `classification` |
| `benchmarks/.../neurovfm_base/fulltest_longitudinal_meta_reports.csv` | `study_uid2`, `pathology1`, `classification` |
| `Validated_Results/perreport/*.json` | `{rg_er: [...], uids: [...], bleu4, meteor, n, source}` — per-report scores keyed by `study_uid2` |
| `benchmarks/mrrate_proprietary/confusion/pred_{v1,v2}.json` | `study_uid2` → predicted change class |

## Reproducing with your own MR-RATE access

Every cache row carries its `study_uid2`. Join on that to recover the reports from your copy of
the corpus and re-derive anything here. `Validated_Results/rescore_canonical.py` does exactly
this — point `S4_TEST_FULL` at the full test table and give it the raw dumps under `logs/`.

## What changed when the text was removed

- **BLEU-4 and METEOR** were recomputed at run time from the cached generations. They are now
  stored in each cache as the scalars that recomputation produced. They are deterministic
  functions of the text, so the stored values are exact.
- **Pairing** was recovered by joining on reference text. It is now a join on `study_uid2` —
  which is what the text was standing in for. All 1,444 rows resolved; the paired bootstrap is
  unchanged.
- **The reversal probe** re-classified forward and reversed generations from
  `logs/reversal_v2/*_reports.json`. The flip arrays were verified identical to that
  recomputation, cached, and the logs dropped.
- **Figure 3** classified generations at run time. The `study_uid2` → class maps are
  precomputed in `confusion/pred_{v1,v2}.json`.

`PAPER_CIS.md` is byte-identical across this change; `PAPER_METRICS.md` differs only in its
provenance note.

## Prompt exemplars

`models/prompts.py` and `benchmarks/mrrate_proprietary/prompts.py` embed two short
format exemplars in the instruction text (chronic demyelinating plaques; subcortical gliotic
foci). They are part of the method — the frontier baseline is not reproducible with a different
prompt — and contain no identifiers. They are retained deliberately.

## What is NOT here

**BIND.** The external-validation cohort (800 pairs, 2 sites) is health-system data under a use
agreement that does not permit redistribution. No BIND-derived text, cache or result row appears
in this repository. Table 3's BIND column is therefore not reproducible from this tree; it
stands on the manuscript alone. The builders (`data_build/process_bind/`) and scorer
(`benchmarks/mrrate_proprietary/bind_score.py`) are included so the method is auditable, but
they require data that is not shipped.

**Imaging volumes.** None, for any dataset.

**NeuroVFM weights.** Gated on HuggingFace (`mlinslab/neurovfm-encoder`, `mlinslab/neurovfm-llm`).

## Terms

Code in this repository is released under the terms in `LICENSE`. The derived label columns
(`classification`, `pathology1/2`) and the pseudonymous identifiers originate in the MR-RATE
corpus; use of them is subject to that corpus's terms. No report text is redistributed.