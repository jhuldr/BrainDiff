# Bootstrap intervals for the manuscript — single source

Every interval below comes from ONE run of `paper/stats/all_cis.py`: `numpy.random.default_rng(0)`, 10,000 resamples, percentile method. Inputs are the per-report rg_er arrays in `paper/cache/perreport/`, produced by re-scoring the cached generations with RadGraph-XL (`reward_level="all"`, index [1] = entity+relation F1).

Pairing is asserted, never assumed: `index` requires the two dumps' references to match at every position; `uid` joins the BrainDiff dump to the NeuroVFM CSV on `study_uid2`; `ref` joins the 64-study frontier files by reference text. Every rg_er row is n=1444, the full test split.

**BIND is not included.** Table 3's external-cohort rows are omitted here: BIND is health-system data under a use agreement that does not permit redistribution, so its per-report caches are not in this tree. Only the internal side of Table 3 is reproducible from these files.

**Reversal probe.** Direction is assigned with `change_class_v2` — the Impression-restricted classifier with the explicit-indeterminacy rule, the same one behind Fig. 3. The probe ran on the **503 directional change cases** of the test split -- every row labelled New lesion, Progressed, Improved or Resolved (510 of 1,464), less 7 dropped for incomplete imaging -- not the full 1,444. Stable, Mixed and Indeterminate rows are excluded by construction: a reversal has no direction to flip. It is then further restricted to change-cases whose forward report asserts a direction, which is why n is 282-285. Rows marked `[v1 classifier, not used]` are the superseded `change_class` values, kept only so they are not re-derived by mistake.

The comparison below is against **BrainDiff Main.pdf** (2026-08-30 21:03); it is re-read from the PDF on every run.


## Table 2

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| BrainDiff, full test | 0.3837 [0.3769, 0.3902] | 1444 | -- |
| NeuroVFM + report pipeline, full test | 0.3614 [0.3550, 0.3677] | 1444 | -- |
| Text-only (images zeroed), full test | 0.3217 [0.3152, 0.3280] | 1444 | -- |
| ours - NeuroVFM | +0.0222 [+0.0157, +0.0288] | 1444 | uid |
| Opus 5, 64-subset | 0.2083 [0.1926, 0.2246] | 64 | -- |
| GPT-5.6 Sol, 64-subset | 0.2920 [0.2723, 0.3117] | 64 | -- |
| BrainDiff, 64-subset | 0.3870 [0.3495, 0.4242] | 64 | -- |
| NeuroVFM, 64-subset | 0.3660 [0.3398, 0.3926] | 64 | -- |
| ours - Opus 5 (64-subset) | +0.1787 [+0.1441, +0.2133] | 64 | ref |
| ours - GPT-5.6 (64-subset) | +0.0950 [+0.0647, +0.1250] | 64 | ref |
| ours - NeuroVFM (64-subset) | +0.0210 [-0.0090, +0.0514] | 64 | ref |

## Table 3

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| internal rg_er | 0.3837 [0.3769, 0.3902] | 1444 | -- |
| internal image contribution | +0.0387 [+0.0324, +0.0449] | 1444 | index |

## Table 4

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| nocf: image effect | +0.0263 [+0.0202, +0.0322] | 1444 | index |
| cf+dropout: image effect | +0.0387 [+0.0324, +0.0449] | 1444 | index |
| quality cost (cf+dropout - nocf) | -0.0059 [-0.0111, -0.0007] | 1444 | index |
| counterfactual gain (image effect difference) | +0.0124 [+0.0056, +0.0194] | 1444 | index |
| reversal flip rate, nocf | 0.2246 [0.1754, 0.2737] | 285 | own denominator |
| reversal flip rate, cf+dropout | 0.2730 [0.2234, 0.3262] | 282 | own denominator |
| reversal flip rate, nocf (common subset) | 0.1934 [0.1415, 0.2500] | 212 | common subset |
| reversal flip rate, cf+dropout (common subset) | 0.2217 [0.1651, 0.2783] | 212 | common subset |
| reversal difference (paired, common subset) | +0.0283 [-0.0472, +0.1038] | 212 | paired |
| reversal difference (UNPAIRED -- the value the paper reports) | +0.0485 [-0.0221, +0.1190] | 285 | unpaired |
| [v1 classifier, not used] flip rate nocf | 0.2171 [0.1708, 0.2669] | 281 | own denominator |
| [v1 classifier, not used] flip rate cf+dropout | 0.2679 [0.2179, 0.3214] | 280 | own denominator |

## Table 5

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| none (no curriculum, no cf, no dropout) | +0.0156 [+0.0107, +0.0205] | 1444 | index |
| no curriculum (cf + dropout) | +0.0266 [+0.0202, +0.0329] | 1444 | index |
| S1 only | +0.0299 [+0.0240, +0.0360] | 1444 | index |
| S1 + S2 | +0.0387 [+0.0324, +0.0449] | 1444 | index |
| total (S1+S2 - none) | +0.0231 [+0.0160, +0.0301] | 1444 | index |

## Table 6

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| report present, own scans | 0.3837 [0.3769, 0.3902] | 1444 | -- |
| report present, other patient's scans | 0.3450 [0.3382, 0.3519] | 1444 | -- |
| report withheld, own scans | 0.2600 [0.2541, 0.2661] | 1444 | -- |
| report withheld, other patient's scans | 0.2057 [0.2000, 0.2115] | 1444 | -- |
| image effect, report present | +0.0387 [+0.0324, +0.0449] | 1444 | index |
| image effect, report withheld | +0.0544 [+0.0480, +0.0610] | 1444 | index |
| effect of prior report, own scans | +0.1236 [+0.1167, +0.1306] | 1444 | index |
| effect of prior report, other scans | +0.1393 [+0.1323, +0.1464] | 1444 | index |
| interaction (present - withheld image effect) | -0.0157 [-0.0231, -0.0083] | 1444 | index |

## Table 7

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| delta on | 0.3846 [0.3778, 0.3912] | 1444 | -- |
| delta off | 0.3837 [0.3769, 0.3902] | 1444 | -- |
| wrong patient's scans (delta off) | 0.3450 [0.3382, 0.3519] | 1444 | -- |
| images zeroed (delta off) | 0.3217 [0.3152, 0.3280] | 1444 | -- |
| delta contribution | +0.0009 [-0.0048, +0.0065] | 1444 | index |
| image contribution vs wrong scans | +0.0387 [+0.0324, +0.0449] | 1444 | index |
| image contribution vs zeroed images | +0.0619 [+0.0543, +0.0698] | 1444 | index |

## Table S1

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| nodelta (S1+S2), report available | 0.3837 [0.3769, 0.3902] | 1444 | -- |
| nostage2 (S1), report available | 0.3867 [0.3800, 0.3936] | 1444 | -- |
| nocurriculum (none), report available | 0.3823 [0.3755, 0.3891] | 1444 | -- |
| nodelta (S1+S2), report withheld | 0.2600 [0.2541, 0.2661] | 1444 | -- |
| nostage2 (S1), report withheld | 0.2343 [0.2281, 0.2407] | 1444 | -- |
| nocurriculum (none), report withheld | 0.2411 [0.2352, 0.2471] | 1444 | -- |
| report-reliance, nodelta (S1+S2) | +0.1236 [+0.1167, +0.1306] | 1444 | index |
| report-reliance, nostage2 (S1) | +0.1524 [+0.1453, +0.1595] | 1444 | index |
| report-reliance, nocurriculum (none) | +0.1412 [+0.1342, +0.1485] | 1444 | index |
| available: nodelta - nostage2 | -0.0030 [-0.0084, +0.0026] | 1444 | index |
| withheld: nodelta - nostage2 | +0.0258 [+0.0204, +0.0311] | 1444 | index |
| reliance DiD vs nostage2 | -0.0288 [-0.0361, -0.0215] | 1444 | index |
| available: nodelta - nocurriculum | +0.0013 [-0.0042, +0.0070] | 1444 | index |
| withheld: nodelta - nocurriculum | +0.0190 [+0.0135, +0.0245] | 1444 | index |
| reliance DiD vs nocurriculum | -0.0176 [-0.0251, -0.0102] | 1444 | index |

## Table 6/3

| quantity | value [95% CI] | n | pairing |
|---|---|---:|---|
| metric floor, internal (patient-disjoint) | 0.1510 [0.1469, 0.1553] | 1444 | derangement |
| metric floor, internal (published roll-1) | 0.2471 [0.2392, 0.2548] | 1444 | roll-1 |

> PDF comparison skipped: `BrainDiff Main.pdf` not readable.

## Not bootstrapped here

- **Table 8** (change-decodability AUROCs) — read from the probe result JSONs; no interval was computed for them. All four reproduce: 0.766 / 0.601 / 0.702 / 0.997.
- **Real-Impression ranking** (supplement; not in the main PDF) — `impression_validation_v2.py`, `default_rng(0)`, 10,000 resamples: BrainDiff 0.0961, NeuroVFM 0.0885, difference **+0.0075 [+0.0015, +0.0134]**, n=1297.
- **BLEU-4 / METEOR** — corpus-level and per-report respectively; point estimates only, in `paper/stats/PAPER_METRICS.md`.
