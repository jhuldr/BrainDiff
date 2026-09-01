#!/usr/bin/env python3
"""
Assemble the BRAIN_DIFF_S4-shaped manifests for the processed BIND volumes.

The analogue of process_mrrate/revise_dataframes.ipynb: turn selection rows into
absolute output paths, capitalize the modality columns, and keep only pairs whose
images actually landed on disk.

Column names deliberately match /home/data/BRAIN_DIFF_S4/{main,image}.csv exactly,
including `study_uid1`/`study_uid2`, so dataloaders/MultiModal/multi_dual_dataloader.py
consumes this with no code change.

    image.csv  study_uid, T1w, T1ce, T2w, FLAIR, centered_image, site
    main.csv   patient_uid, study_uid1, study_uid2, site, duration, report1, report2, ...
"""

import argparse
from pathlib import Path

import pandas as pd

from paths import MODALITIES, OUTPUT_ROOT, session_dir

DATA_DIR = Path(__file__).parent / "data"
SELECTION_CSV = DATA_DIR / "bind_longitudinal_data.csv"
PAIRS_CSV = DATA_DIR / "bind_longitudinal_meta_paired.csv"

COLUMN_NAMES = {"t1w": "T1w", "t1ce": "T1ce", "t2w": "T2w", "flair": "FLAIR"}


def resolve_outputs(selection, output_root):
    """
    Replace input paths with output paths, keeping only volumes that exist.

    A modality selected upstream can still be missing here: ANTs may have failed
    on that one series, or the session may not have been processed yet. Checking
    the filesystem rather than trusting the selection is what keeps main.csv from
    pointing at files the trainer will fail to open.
    """
    rows = []

    for row in selection.itertuples(index=False):
        out_dir = session_dir(output_root, row.site, row.session_uid)

        record = {
            "study_uid": row.session_uid,
            "site": row.site,
            "patient_uid": row.patient_uid,
        }

        found = 0
        for modality in MODALITIES:
            selected = getattr(row, modality, None)
            written = out_dir / f"{modality}.nii.gz"

            if pd.notna(selected) and selected and written.is_file():
                record[COLUMN_NAMES[modality]] = str(written)
                found += 1
            else:
                record[COLUMN_NAMES[modality]] = None

        if not found:
            continue

        # `centered_image` names the volume the affine was estimated from. That
        # image is written as one of the four modalities, so point at the T1w
        # (always present when a centre exists) rather than the native-space file.
        record["centered_image"] = record["T1w"] or next(
            (record[COLUMN_NAMES[m]] for m in MODALITIES if record[COLUMN_NAMES[m]]), None
        )
        record["t1ce_source"] = row.t1ce_source

        rows.append(record)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION_CSV)
    parser.add_argument("--pairs", type=Path, default=PAIRS_CSV)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    selection = pd.read_csv(
        args.selection, dtype={"session_uid": str, "patient_uid": str}
    )
    pairs = pd.read_csv(args.pairs, dtype={"study_uid1": str, "study_uid2": str})

    print(f"Selected sessions: {len(selection):,}   candidate pairs: {len(pairs):,}")

    image = resolve_outputs(selection, args.output_root)
    print(f"Sessions with volumes on disk: {len(image):,}")

    if image.empty:
        print("Nothing processed yet -- run DataProcessing/process_files_bind.py first.")
        return

    available = set(image["site"] + "/" + image["study_uid"])

    keep = pairs.apply(
        lambda r: f"{r['site']}/{r['study_uid1']}" in available
        and f"{r['site']}/{r['study_uid2']}" in available,
        axis=1,
    )
    main_df = pairs[keep].copy()

    # Drop sessions no surviving pair refers to.
    used = set(main_df["site"] + "/" + main_df["study_uid1"]) | set(
        main_df["site"] + "/" + main_df["study_uid2"]
    )
    image = image[(image["site"] + "/" + image["study_uid"]).isin(used)]

    image_cols = [
        "study_uid", "T1w", "T1ce", "T2w", "FLAIR", "centered_image",
        "site", "patient_uid", "t1ce_source",
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    image[image_cols].to_csv(args.output_root / "image.csv", index=False)
    main_df.to_csv(args.output_root / "main.csv", index=False)

    print(f"\n=== Written to {args.output_root} ===")
    print(f"  image.csv: {len(image):,} sessions")
    print(f"  main.csv : {len(main_df):,} pairs, "
          f"{main_df['patient_uid'].nunique():,} patients")

    print("\nModality coverage:")
    for column in COLUMN_NAMES.values():
        print(f"  {column:6s}: {image[column].notna().sum():,} "
              f"({image[column].notna().mean():.1%})")

    print(f"\nduration days: median {main_df['duration'].median():.0f}, "
          f"mean {main_df['duration'].mean():.0f}")


if __name__ == "__main__":
    main()
