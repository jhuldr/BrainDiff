#!/usr/bin/env python3
"""
Select longitudinal brain MRI series from MR-RATE metadata.

Outputs one row per study_uid with:
    study_uid, t1w, t1ce, t2w, flair, centered_image
"""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from generate_single import *


# ============================================================
# Configuration
# ============================================================

METADATA_DIR = Path("/home/data/MR-RATE/metadata")

LONGITUDINAL_META_CSV = Path(
    "/path/to/code/BrainDiff/process_mrrate/longitudional_meta2.csv"
)

OUTPUT_CSV = Path(
    "/path/to/code/BrainDiff/process_mrrate/data/longitudinal_brain_data.csv"
)

FILE_COL = "series_id"


def filter_timepoint(df):
    """
    Keeps only longitudinal studies.
    """
    longitudinal_df = pd.read_csv(LONGITUDINAL_META_CSV)

    df = df[
        (
            df["study_uid"].isin(longitudinal_df["study_uid1"])
            | df["study_uid"].isin(longitudinal_df["study_uid2"])
        )
    ].copy()

    return df

# ============================================================
# Main
# ============================================================

def main():
    """
    Run the full single-timepoint brain MRI selection pipeline.
    """
    metadata_df = load_metadata(METADATA_DIR)

    print("Keeping longitudinal studies...")
    metadata_df = filter_timepoint(metadata_df)

    df_brain = prepare_metadata(metadata_df)

    selected_brains_df = select_target_brain_rows(df_brain)
    selected_brains_df = order_and_trim_selected_rows(selected_brains_df)

    selected_brains_wide = make_wide_single_timepoint_df(selected_brains_df)

    batch_df = (
        metadata_df[["study_uid", "batch"]]
        .dropna(subset=["study_uid"])
        .drop_duplicates()
        .groupby("study_uid", as_index=False)
        .agg(batch=("batch", "first"))
        )

    selected_brains_wide = pd.merge(
        selected_brains_wide,
        batch_df,
        on="study_uid",
        how="left",
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving output to {OUTPUT_CSV}...")
    selected_brains_wide.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved {len(selected_brains_wide):,} rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()