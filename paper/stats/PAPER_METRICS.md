# rg_er / BLEU-4 / METEOR for the manuscript -- single source

Every value below comes from ONE run of `paper/stats/all_metrics.py`, reading the cached per-report arrays in `paper/cache/perreport/`.

`rg_er` is read from the cache (RadGraph-XL, `reward_level="all"`, index [1] = entity+relation F1). `BLEU-4` is corpus-level with method-1 smoothing and `METEOR` is per-report averaged; both are cached scalars, computed from the generations before the MR-RATE text was removed from this tree. No GPU, no API calls, no regeneration.

Rows marked *diff* are the paired difference between two arms, index-paired with the study uids asserted equal at every position. rg_er is differenced at full precision; BLEU-4 and METEOR are differenced on the 4-decimal values, matching the manuscript's tables so that subtracting two printed cells gives the printed difference. BLEU-4 is corpus-level, so its differences are not per-report paired quantities and carry no interval here.

**BIND is not included.** Table 3's external-cohort rows are omitted: BIND is health-system data under a use agreement that does not permit redistribution, so its per-report caches are not in this tree. Only the internal side of Table 3 is reproducible from these files.

Confidence intervals live in `PAPER_CIS.md`, which owns rg_er intervals only. The two files share the same caches, so rg_er agrees between them by construction.


## Table 2

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| BrainDiff, full test | 0.3837 | 0.2281 | 0.4582 | 1444 | `logs/recompute/pp/rger_real.rank*.json` |
| NeuroVFM + report pipeline, full test | 0.3614 | 0.1907 | 0.4123 | 1444 | `--` |
| Text-only (images zeroed), full test | 0.3217 | 0.1590 | 0.3947 | 1444 | `logs/recompute_sz/rger_scan_zero.rank*.json` |
| Opus 5, 64-subset | 0.2083 | 0.0436 | 0.3823 | 64 | `--` |
| GPT-5.6 Sol, 64-subset | 0.2920 | 0.0958 | 0.3480 | 64 | `--` |
| BrainDiff, 64-subset | 0.3870 | 0.2198 | 0.4459 | 64 | `--` |
| NeuroVFM, 64-subset | 0.3660 | 0.1945 | 0.4081 | 64 | `--` |

## Table 3

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| internal (full test) | 0.3837 | 0.2281 | 0.4582 | 1444 | `logs/recompute/pp/rger_real.rank*.json` |
| internal image contribution | +0.0387 | +0.0214 | +0.0292 | 1444 | index-paired |

## Table 4

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| nocf, own scans | 0.3896 | 0.2279 | 0.4627 | 1444 | `logs/recompute_nocf/pp/rger_real.rank*.json` |
| nocf, other patient's scans | 0.3633 | 0.2148 | 0.4431 | 1444 | `logs/recompute_nocf/pp/rger_vis_roll.rank*.json` |
| cf+dropout, own scans | 0.3837 | 0.2281 | 0.4582 | 1444 | `logs/recompute/pp/rger_real.rank*.json` |
| cf+dropout, other patient's scans | 0.3450 | 0.2067 | 0.4290 | 1444 | `logs/recompute/pp/rger_vis_roll.rank*.json` |
| nocf: image effect | +0.0263 | +0.0131 | +0.0196 | 1444 | index-paired |
| cf+dropout: image effect | +0.0387 | +0.0214 | +0.0292 | 1444 | index-paired |
| quality cost (cf+dropout - nocf) | -0.0059 | +0.0002 | -0.0045 | 1444 | index-paired |

## Table 5

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| none: real / wrong scans | 0.3896 | 0.2306 | 0.4673 | 1444 | `logs/ncnf_test/rger_real.rank*.json` |
| no curriculum: real | 0.3823 | 0.2256 | 0.4508 | 1444 | `logs/recompute_nc/pp/rger_real.rank*.json` |
| S1 only: real | 0.3867 | 0.2269 | 0.4558 | 1444 | `logs/nostage2_test/pp/rger_real.rank*.json` |
| none (no curriculum, no cf, no dropout) | +0.0156 | +0.0074 | +0.0130 | 1444 | index-paired |
| no curriculum (cf + dropout) | +0.0266 | +0.0167 | +0.0210 | 1444 | index-paired |
| S1 only | +0.0299 | +0.0166 | +0.0272 | 1444 | index-paired |
| S1 + S2 | +0.0387 | +0.0214 | +0.0292 | 1444 | index-paired |

## Table 6

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| report present, own scans | 0.3837 | 0.2281 | 0.4582 | 1444 | `logs/recompute/pp/rger_real.rank*.json` |
| report present, other patient's scans | 0.3450 | 0.2067 | 0.4290 | 1444 | `logs/recompute/pp/rger_vis_roll.rank*.json` |
| report withheld, own scans | 0.2600 | 0.1507 | 0.3555 | 1444 | `logs/recompute/nr/rger_real.noreport.rank*.json` |
| report withheld, other patient's scans | 0.2057 | 0.1248 | 0.3131 | 1444 | `logs/recompute/nr/rger_vis_roll.noreport.rank*.json` |
| image effect, report present | +0.0387 | +0.0214 | +0.0292 | 1444 | index-paired |
| image effect, report withheld | +0.0544 | +0.0259 | +0.0424 | 1444 | index-paired |
| effect of prior report, own scans | +0.1236 | +0.0774 | +0.1027 | 1444 | index-paired |

## Table 7

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| delta on | 0.3846 | 0.2231 | 0.4513 | 1444 | `logs/rger_AB/A/rger_real.merged.json` |
| delta off | 0.3837 | 0.2281 | 0.4582 | 1444 | `logs/recompute/pp/rger_real.rank*.json` |
| wrong patient's scans (delta off) | 0.3450 | 0.2067 | 0.4290 | 1444 | `logs/recompute/pp/rger_vis_roll.rank*.json` |
| images zeroed (delta off) | 0.3217 | 0.1590 | 0.3947 | 1444 | `logs/recompute_sz/rger_scan_zero.rank*.json` |
| delta contribution | +0.0009 | -0.0050 | -0.0069 | 1444 | index-paired |
| image contribution vs wrong scans | +0.0387 | +0.0214 | +0.0292 | 1444 | index-paired |

## Table S1

| quantity | rg_er | BLEU-4 | METEOR | n | source |
|---|---:|---:|---:|---:|---|
| nodelta (S1+S2), report available | 0.3837 | 0.2281 | 0.4582 | 1444 | `logs/recompute/pp/rger_real.rank*.json` |
| nostage2 (S1), report available | 0.3867 | 0.2269 | 0.4558 | 1444 | `logs/nostage2_test/pp/rger_real.rank*.json` |
| nocurriculum (none), report available | 0.3823 | 0.2256 | 0.4508 | 1444 | `logs/recompute_nc/pp/rger_real.rank*.json` |
| nodelta (S1+S2), report withheld | 0.2600 | 0.1507 | 0.3555 | 1444 | `logs/recompute/nr/rger_real.noreport.rank*.json` |
| nostage2 (S1), report withheld | 0.2343 | 0.1385 | 0.3265 | 1444 | `logs/nostage2_test/nr/rger_real.noreport.rank*.json` |
| nocurriculum (none), report withheld | 0.2411 | 0.1372 | 0.3305 | 1444 | `logs/recompute_nc/nr/rger_real.noreport.rank*.json` |
| report-reliance, nodelta | +0.1236 | +0.0774 | +0.1027 | 1444 | index-paired |
| report-reliance, nostage2 | +0.1524 | +0.0884 | +0.1293 | 1444 | index-paired |
| report-reliance, nocurriculum | +0.1412 | +0.0884 | +0.1203 | 1444 | index-paired |

## Not covered here

- **Reversal probe** (Table 4's right-hand column) is a signed-flip rate, not a report-quality metric; it lives in `PAPER_CIS.md`.
- **Change-decodability AUROCs** (Table 8) are probe outputs, not generations.
- **Metric floors** are reference-vs-reference scores and carry no BLEU or METEOR.
