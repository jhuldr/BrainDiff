# paper/cache

Per-pair numbers for the Table 2 brain block, written by `python paper/table2.py --build_cache` from the private scorer outputs.

**What is here** (numeric arrays only, one row per evaluation pair, in a fixed opaque order per split; no pair, study or patient IDs, no text):
- `groups_<split>.npz`: `group` = patient index (0..n_patients-1, in sorted-patient order). Used only to resample patients in the bootstrap; the order keeps the draws identical to the original scripts.
- `<system>_<split>.npz` for the 9 systems and both splits (`test` = internal, `external_test`): `rger` (RadGraph entity+relation F1 against the edited target report) and `f1temp` (temporal F1), one value per pair.

**What is not here, and why:** nothing derived from the labels of the reference reports. That excludes the Rad-Δ status flags of the original current reports (needed for macro-MCC), the change / stable split and reference change counts (needed for both agreement columns and for the reader change / stable split), the per-pair agreement scores, the generated reports' status presences, and the reader scores (their cells need the change / stable split). Those cells cannot be recomputed from this cache. `table2.py` (cache mode) recomputes rg_er and F1temp (points, 95% CIs, paired differences) and copies every other cell from `../table2_golden.json`, which holds the original scripts' outputs. It lists the copied cells under `from_golden` and marks them `g`. `table2.py --full` recomputes everything but needs the private scorer outputs and reader answers.
