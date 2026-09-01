"""Derive per-study MULTI-LABEL intracranial pathology from the multi-hot source.

`process.ipynb` cell 5 (`get_category_and_name`) collapsed each study to ONE label
by returning on the first match in a hardcoded priority order. On the 38,086 S2
studies that discarded 33,523 of 71,609 positive labels (46.8%), and the loss was
biased by list position, not by clinical relevance:

    Glioma (index 0)   survived 100.0%
    Cerebral atrophy    20.6%
    Ventriculomegaly     8.7%
    Watershed infarct    1.7%   (177 studies -> 3 rows, which MIN_CLASS_COUNT
                                 then deleted, so only 28 of 29 classes trained)

Worse, the co-occurrences are causal: 2,083 studies carry both a metastasis and
cerebral edema, and the rule always kept the metastasis -- dropping the finding
that describes the visible imaging change.

Nothing needed re-labelling; the multi-hot matrix was intact the whole time. This
script re-derives from it and emits three columns per study:

    pathologies        ", ".join(sorted(labels))   <- the new pcls target
    pathology_bucket   the RAREST label present    <- sampler / split key
    pathology_primary  the old priority-order winner, asserted equal to the
                       notebook's output so this file is provably a superset

`pathology_bucket` is the rarest rather than the first label so that a study
carrying Watershed infarct lands in the Watershed bucket instead of being
absorbed into Gliosis -- that is what stops the sampler reproducing the same
positional bias the labels just escaped.

    python -m process_mrrate.build_multilabel_pathology
"""
import argparse
import os

import pandas as pd

SRC = "/home/data/MR-RATE/pathology_labels/mrrate_labels.csv"
NOTEBOOK_OUT = os.path.join(os.path.dirname(__file__), "condensed_pathology_labels.csv")
S2_CSV = "/home/data/BRAIN_DIFF_S2/single_timepoint_final.csv"
OUT = os.path.join(os.path.dirname(__file__), "pathology_multilabel.csv")

# Intracranial allow-list. Enforced, not inherited: the source is asserted to
# partition exactly into this set plus EXTRACRANIAL below, so a renamed or added
# column fails the run instead of silently entering (or vanishing from) a caption.
#
# Two names are not self-evidently intracranial and will be re-litigated by every
# future reader, so: `Schwannoma` is vestibular/CPA in brain MRI, and `Cavernous
# hemangioma` is an intracranial cavernoma here (the term is also used for orbital
# lesions). `Pituitary adenoma` and `Empty sella syndrome` are sellar --
# intracranial but extra-axial -- and are included as standard.
INTRACRANIAL = [
    "Arachnoid cyst",
    "Cavernous hemangioma",
    "Cerebellar degeneration",
    "Cerebral atrophy",
    "Cerebral edema",
    "Cerebral hemorrhage",
    "Cerebral infarction",
    "Chiari malformation",
    "Choroid plexus cyst",
    "Cyst of pineal gland",
    "Demyelinating disease of central nervous system",
    "Empty sella syndrome",
    "Encephalomalacia",
    "Glioma",
    "Gliosis",
    "Intracranial aneurysm",
    "Intracranial meningioma",
    "Lacunar infarct",
    "Lipoma of brain",
    "Mega cisterna magna",
    "Metastatic malignant neoplasm to brain",
    "Pituitary adenoma",
    "Rathke's pouch cyst",
    "Schwannoma",
    "Silent micro-hemorrhage of brain",
    "Structure of cave of septum pellucidum",
    "Subdural intracranial hemorrhage",
    "Ventriculomegaly",
    "Watershed infarct",
]

# Deliberately excluded: spinal, vertebral, temporal-bone and calvarial findings.
# Named explicitly so the partition assertion can tell "known exclusion" apart
# from "unrecognised new column".
EXTRACRANIAL = [
    "Herniation of nucleus pulposus",
    "Spinal cord compression",
    "Spinal stenosis",
    "Foraminal Spinal Stenosis",
    "Hemangioma of vertebral column",
    "Mastoiditis",
    "Chronic mastoiditis",
    "Hyperostosis of skull",
]

# The notebook's priority order (brain_pathologies_organized, insertion-ordered).
# Kept ONLY to reproduce `pathology_primary` and prove agreement -- it is not
# used to build `pathologies`.
PRIORITY = [
    "Glioma", "Intracranial meningioma", "Metastatic malignant neoplasm to brain",
    "Schwannoma", "Pituitary adenoma", "Lipoma of brain",
    "Cerebral infarction", "Lacunar infarct", "Watershed infarct",
    "Cerebral hemorrhage", "Subdural intracranial hemorrhage",
    "Silent micro-hemorrhage of brain", "Cavernous hemangioma", "Intracranial aneurysm",
    "Gliosis", "Encephalomalacia", "Cerebral edema", "Cerebral atrophy",
    "Demyelinating disease of central nervous system", "Cerebellar degeneration",
    "Arachnoid cyst", "Cyst of pineal gland", "Choroid plexus cyst", "Rathke's pouch cyst",
    "Chiari malformation", "Mega cisterna magna", "Ventriculomegaly",
    "Empty sella syndrome", "Structure of cave of septum pellucidum",
]


def assert_source_schema(df):
    """The source must partition exactly into INTRACRANIAL + EXTRACRANIAL."""
    cols = [c for c in df.columns if c != "study_uid"]
    unknown = sorted(set(cols) - set(INTRACRANIAL) - set(EXTRACRANIAL))
    missing = sorted(set(INTRACRANIAL) - set(cols))
    if unknown or missing:
        raise RuntimeError(
            "mrrate_labels.csv schema drifted -- refusing to build captions from it.\n"
            f"  unrecognised columns (neither allowed nor a known exclusion): {unknown}\n"
            f"  expected intracranial columns that are absent: {missing}\n"
            "Classify each explicitly in INTRACRANIAL or EXTRACRANIAL before re-running."
        )
    assert set(PRIORITY) == set(INTRACRANIAL), "PRIORITY and INTRACRANIAL disagree"


def build(src=SRC, notebook_out=NOTEBOOK_OUT, out=OUT):
    df = pd.read_csv(src)
    assert_source_schema(df)
    assert df["study_uid"].is_unique, "source is not one row per study_uid"

    flags = df[INTRACRANIAL].fillna(0).astype(int).values.astype(bool)
    uids = df["study_uid"].values

    # Global per-label counts drive the bucket choice; computed over every study
    # that has any intracranial finding, so the ranking is stable regardless of
    # which downstream subset consumes it.
    has_any = flags.any(axis=1)
    counts = dict(zip(INTRACRANIAL, flags[has_any].sum(axis=0)))
    rarity = sorted(INTRACRANIAL, key=lambda c: (counts[c], c))
    rank = {c: i for i, c in enumerate(rarity)}
    prio = {c: i for i, c in enumerate(PRIORITY)}

    rows = []
    for uid, row in zip(uids[has_any], flags[has_any]):
        labels = [c for c, on in zip(INTRACRANIAL, row) if on]
        rows.append((
            uid,
            ", ".join(sorted(labels)),                 # alphabetical, comma-space
            min(labels, key=lambda c: rank[c]),        # rarest present
            min(labels, key=lambda c: prio[c]),        # notebook's winner
        ))
    ml = pd.DataFrame(rows, columns=["study_uid", "pathologies",
                                     "pathology_bucket", "pathology_primary"])

    # Prove this is a superset of the notebook rather than a reinterpretation.
    if os.path.exists(notebook_out):
        nb = pd.read_csv(notebook_out)[["study_uid", "true_name"]]
        chk = ml.merge(nb, on="study_uid", how="inner")
        agree = (chk["pathology_primary"] == chk["true_name"]).mean()
        print(f"[check] pathology_primary vs notebook true_name: "
              f"{agree:.4%} over {len(chk):,} shared rows")
        if agree < 1.0:
            bad = chk[chk["pathology_primary"] != chk["true_name"]].head(5)
            raise RuntimeError(f"priority-order reproduction diverged:\n{bad}")
    else:
        print(f"[check] SKIPPED -- {notebook_out} not found")

    ml.to_csv(out, index=False)
    n = ml["pathologies"].str.count(", ") + 1
    print(f"-> {out}: {len(ml):,} studies")
    print(f"   labels/study: mean {n.mean():.2f}  max {n.max()}  multi-label {(n > 1).mean():.1%}")
    print(f"   distinct label strings: {ml['pathologies'].nunique():,}")
    print(f"   buckets: {ml['pathology_bucket'].nunique()}  "
          f"(min {ml['pathology_bucket'].value_counts().min():,})")
    print(f"   labels recovered vs single-label: {int(n.sum()):,} vs {len(ml):,} "
          f"(+{int(n.sum()) - len(ml):,})")
    return ml


def patch_s2(ml, s2_csv=S2_CSV):
    """Add `pathologies` + `pathology_bucket` to the S2 frame, ADDITIVELY.

    `pathology` is deliberately left untouched: mutli_single_dataloader.py:154
    emits it as `group`, and losses.make_swap_perm picks counterfactual partners
    by integer inequality on it. A multi-label group would make "Gliosis" and
    "Gliosis, Cerebral atrophy" count as a valid contrast despite sharing a
    finding, silently weakening the S2 gate. So S2 keeps the single label and is
    bit-for-bit unaffected by this change.
    """
    s2 = pd.read_csv(s2_csv)
    if "pathologies" in s2.columns:
        s2 = s2.drop(columns=[c for c in ("pathologies", "pathology_bucket") if c in s2.columns])

    bak = s2_csv + ".pre_multilabel.bak"
    if not os.path.exists(bak):
        pd.read_csv(s2_csv).to_csv(bak, index=False)
        print(f"[backup] {bak}")

    before = len(s2)
    merged = s2.merge(ml[["study_uid", "pathologies", "pathology_bucket"]],
                      on="study_uid", how="left")
    assert len(merged) == before, f"merge changed row count {before} -> {len(merged)}"
    missing = merged["pathologies"].isna().sum()
    assert missing == 0, f"{missing} studies had no multi-hot row"

    # The single label must be one of the multi labels, or the two disagree.
    contained = [p in set(m.split(", ")) for p, m in
                 zip(merged["pathology"], merged["pathologies"])]
    assert all(contained), f"{len(contained) - sum(contained)} rows: single label not in multi set"

    merged.to_csv(s2_csv, index=False)
    n = merged["pathologies"].str.count(", ") + 1
    print(f"-> {s2_csv}: {len(merged):,} rows, `pathology` unchanged, "
          f"+pathologies/+pathology_bucket")
    print(f"   labels/study: mean {n.mean():.2f}  multi-label {(n > 1).mean():.1%}  "
          f"recovered +{int(n.sum()) - len(merged):,} labels")
    return merged


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=SRC)
    p.add_argument("--notebook_out", default=NOTEBOOK_OUT)
    p.add_argument("--out", default=OUT)
    p.add_argument("--patch-s2", action="store_true",
                   help="additively add the multi-label columns to single_timepoint_final.csv")
    a = p.parse_args()
    ml = build(a.src, a.notebook_out, a.out)
    if a.patch_s2:
        patch_s2(ml)
