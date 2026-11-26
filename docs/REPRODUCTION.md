# Reproduction

## Generation is not bitwise repeatable across processes

NeuroVFM's blocks use flash-attn's Triton LayerNorm, which benchmarks several kernel configurations the first time it runs in a process and
keeps the fastest. Different processes can choose different configurations, which changes the order of floating-point additions and so the
vision features in the last few bits. With greedy decoding, a one-token difference can make the rest of a report diverge.

Measured on 64 internal test pairs with the released checkpoint (greedy, output-only repetition penalty 1.1):

| Setup | Reports identical to the paper's generations |
|---|---|
| One process (original code or this release) | 64 / 64 |
| Four processes, one per GPU (original code or this release) | 52 / 64 |

Within one process, this release and the original code give identical outputs. Expect the paper's metrics to reproduce within
run-to-run noise rather than exactly when generating with several processes; scoring the paper's generations reproduces Table 2 exactly.

## Image registration is not repeatable either

`braindiff/dataset/bind_register.py` (ANTs) does not give identical volumes on rerun. On three re-registered sessions, release and
original code fell within the same rerun spread; one clean session varied widely across four reruns (voxel correlation 0.18-0.66). The
released model was trained on one specific registration of each session.

## Run order (needs the data)

| Step | Command | Environment |
|---|---|---|
| BIND tables | `braindiff/dataset/bind_preprocess.py manifest`, `pairs`, `select --no-anatomy-filter`, `select3d`, `imageonly` | any |
| BIND images | `bind_preprocess.py skullstrip`, `register` (then `retry`) | `bind_proc` (HD-BET, ANTs) |
| QC and split | `braindiff/dataset/bind_qc_split.py qc`, `split` | `bind_proc` |
| Stage 1 tables (needs `raddelta` and its student checkpoint, `RADDELTA_CKPT`) | `braindiff/dataset/build_stage1.py dupes`, `probe_pairs`, `coverage`, `label`, `combine`, `pairs` (with `STAGE1_BIND_SNAPSHOT`), `entity_text`, `tasks` | scoring / trainer |
| Stage 2 targets and pairs | `braindiff/dataset/build_stage2.py rewrite` (paid Anthropic batch), `merge`, `pairs` | scoring |
| Train | `braindiff/scripts/train.sh 0,1,2,3` (Stage 1-D, selection, Stage 2, selection) | trainer |
| Test | `braindiff/scripts/test.sh` (greedy, repetition penalty 1.1, scoring) | trainer + scoring |
| Baselines | `braindiff/baselines/run.sh <system>` per Table 2 row | vLLM + scoring |
| Chest X-ray rows | `braindiff/baselines/cxr/cxr_generate.py`, `maira2_finetune.py` (needs Rad-Δ status labels of MIMIC-CXR train) | transformers 4.51.3 |
| Table 2 | `braindiff/paper/table2.py --full` | scoring |

`select --no-anatomy-filter` reproduces the report-pair tables the model was built on; the default adds a later in-plane resolution
filter and gives fewer pairs.
