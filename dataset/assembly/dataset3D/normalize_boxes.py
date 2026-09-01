"""Regenerate the stage-1 box annotations from the raw voxel boxes.

Ported from adjust_bounding_boxes.ipynb so the file training consumes can be
rebuilt in one command. build_unified_dataframe.py already exists for the same
reason -- "the file that training actually consumes could not be regenerated
without hand-running a notebook" -- and this closes the other half of that chain.

    raw {ATLAS,BRATS-MEN,BRATS-MET,ISLES-22}/OUTPUT/boxes.csv   (voxel indices)
      -> boxes_norm.csv          boxes in the NeuroVFM frame
      -> step1_intake_norm.csv   + aref/gcap/caref prompts and captions
      -> build_unified_dataframe.py  + S2 pcls rows -> unified_dataframe_norm.csv

Boxes are normalized with dataloaders/MultiModal/neurovfm_transforms, which is
also what builds the image. Those two MUST stay derived from one source: computing
them separately is how the previous coordinates silently stopped matching their
images, and the V-JEPA->NeuroVFM switch broke them again (axis swap plus two
mirrored axes -- up to 66 percentage points off once off-centre).

    python -m dataset3D.normalize_boxes
"""
# --- notebook cell 0 ---
import os
import sys

import nibabel as nib

# The geometry lives with the image transform, not here. to_neurovfm_grid and
# voxel_box_to_percent must derive from the same formulas -- they were once
# computed separately in separate files, which is how these coordinates silently
# stopped matching the images they annotate. Import, do not reimplement.
#
# Switched from vjepa_transforms to neurovfm_transforms when the encoder changed.
# These are NOT interchangeable: V-JEPA mapped x->D, y->H, z->W with no flips,
# NeuroVFM has D=S and W=R (first and third axes swapped) and mirrors two axes
# for the RAS->RPI reorientation. Boxes normalized under the old convention are
# wrong by up to 66 percentage points once off-centre.
sys.path.insert(0, os.path.abspath(".."))
from dataloaders.MultiModal.neurovfm_transforms import (  # noqa: E402
    voxel_box_to_percent,
)

_SHAPE_CACHE = {}


def _native_shape(image_path: str) -> tuple:
    """Header-only shape lookup, cached -- every row in a study repeats the same path."""
    if image_path not in _SHAPE_CACHE:
        _SHAPE_CACHE[image_path] = nib.load(image_path).shape[:3]
    return _SHAPE_CACHE[image_path]


def normalize_bbox(row) -> tuple:
    """Native-voxel bounding box -> the coordinate space the model sees.

    Output is six integers 0..100: each coordinate's position in the FINAL
    preprocessed image as a percentage of that image's extent, so there is no
    resample affine left for the model to learn.

    Args:
        row: needs "BoundingBox" (string "(x1, x2, y1, y2, z1, z2)" of native
             voxel indices) and "temp_modality" (a NIfTI path, for native shape)
    """
    bbox = tuple(int(x.strip()) for x in str(row["BoundingBox"]).strip("()[]").split(","))
    return voxel_box_to_percent(bbox, _native_shape(row["temp_modality"]))

# --- notebook cell 1 ---
import pandas as pd
df = pd.concat([pd.read_csv("/home/data/BRAIN_DIFF_S1/ATLAS/OUTPUT/boxes.csv"), pd.read_csv("/home/data/BRAIN_DIFF_S1/BRATS-MEN/OUTPUT/boxes.csv"), pd.read_csv("/home/data/BRAIN_DIFF_S1/BRATS-MET/OUTPUT/boxes.csv"), pd.read_csv("/home/data/BRAIN_DIFF_S1/ISLES-22/OUTPUT/boxes.csv")])

# --- notebook cell 2 ---
def fix_paths(df: pd.DataFrame, old_folder: str, new_folder: str) -> pd.DataFrame:
    """
    Replace folder names in all string columns containing file paths.
    
    Args:
        df: Input dataframe
        old_folder: Old folder name to replace
        new_folder: New folder name
    
    Returns:
        Dataframe with corrected paths
    """
    df = df.copy()
    
    for col in df.columns:
        if df[col].dtype == 'str' or df[col].dtype == 'object':  # String columns
            df[col] = df[col].str.replace(old_folder, new_folder, regex=False)
    
    return df

# --- notebook cell 3 ---
df = fix_paths(df, "BRAIN_DIFF", "BRAIN_DIFF_S1")

# --- notebook cell 4 ---
df["temp_modality"] = df[['T1w', 'T1ce', 'T2w', 'FLAIR']].bfill(axis=1).iloc[:, 0]

# --- notebook cell 5 ---
df["BoundingBox"] = df.apply(normalize_bbox, axis=1)

# --- notebook cell 6 ---
df.reset_index(inplace=True)
df[['Caption', 'BoundingBox', 'AnatomicalRegion', 'with_lesion', 'T1w',
       'T1ce', 'T2w', 'FLAIR', 'Pathology']].to_csv(
    "/home/data/BRAIN_DIFF_S1/boxes_norm.csv", index=False)

# --- notebook cell 8 ---
df = pd.read_csv("/home/data/BRAIN_DIFF_S1/boxes_norm.csv")

# --- notebook cell 9 ---
import pandas as pd

# Defined once and reused by every template here AND by
# merge_aref_and_caref_duplicate_boxes below. The two used to carry separate
# copies of this wording, which is how they drift.
COORD_SPEC = (
    "The coordinates are in the format [x1, x2, y1, y2, z1, z2], where each value is "
    "an integer from 0 to 100 giving the position as a percentage along that axis of "
    "the image: (x1, x2) are the min/max along the first axis, (y1, y2) along the "
    "second, and (z1, z2) along the third."
)

FINDING_FORMAT = (
    "'There is a {pathology} infarct in the {anatomical region}.' "
    "or if there is no finding: 'The {anatomical region} is unremarkable.'"
)

AREF_PROMPT = (
    "Given the following finding, provide only the bounding box coordinates. "
    + COORD_SPEC
    + " Output nothing other than the formatted coordinates.\n\nFinding: "
)

GCAP_PROMPT_HEAD = (
    "Describe the finding at the specified bounding box location. "
    "Only output a short description in the format: " + FINDING_FORMAT + "\n\n"
    + COORD_SPEC + "\n\nBounding box: "
)

CAREF_PROMPT_HEAD = (
    "Given the anatomical region name, provide only a description of any "
    "findings and the bounding box coordinates.\n\n"
    "The findings should be output in the format: " + FINDING_FORMAT + "\n\n"
    + COORD_SPEC + "\n\n"
    "The overall output format is:\n"
    "<findings> [x1, x2, y1, y2, z1, z2]\n\nRegion: "
)


def create_location_aware_dataframe(
    df: pd.DataFrame,
    tasks: list[str] | None = None,
    print_distribution: bool = True,
) -> pd.DataFrame:
    """
    Create a new dataframe with prompt/caption/task columns for location-aware pretraining.
    """

    if tasks is None:
        tasks = ["aref", "gcap", "caref"]

    tasks = set(tasks)

    # Work on a copy to avoid modifying original df
    base = df.copy()

    # Normalize required fields
    caption = base["Caption"].astype("string").str.strip()
    bbox = base["BoundingBox"].astype("string").str.strip()

    # Keep only rows with valid caption and bbox
    valid_mask = caption.notna() & caption.ne("") & bbox.notna() & bbox.ne("")
    base = base.loc[valid_mask].copy()

    if base.empty:
        return pd.DataFrame(columns=list(df.columns) + ["prompt", "caption", "task"])

    description = base["Caption"].astype("string").str.strip()
    bbox_clean = (
        base["BoundingBox"]
        .astype("string")
        .str.strip()
        .str.replace("(", "[", regex=False)
        .str.replace(")", "]", regex=False)
    )

    outputs = []

    if "aref" in tasks:
        aref = base.copy()
        aref["prompt"] = AREF_PROMPT + description
        aref["caption"] = bbox_clean
        aref["task"] = "aref"
        outputs.append(aref)

    if "gcap" in tasks:
        gcap = base.copy()
        gcap["prompt"] = GCAP_PROMPT_HEAD + bbox_clean
        gcap["caption"] = description
        gcap["task"] = "gcap"
        outputs.append(gcap)

    if "caref" in tasks:
        region = base["AnatomicalRegion"].astype("string").str.strip()
        caref_mask = region.notna() & region.ne("")

        if caref_mask.any():
            caref = base.loc[caref_mask].copy()
            region_clean = region.loc[caref_mask]
            desc_caref = description.loc[caref_mask]
            bbox_caref = bbox_clean.loc[caref_mask]

            caref["prompt"] = CAREF_PROMPT_HEAD + region_clean
            caref["caption"] = desc_caref + " " + bbox_caref
            caref["task"] = "caref"
            outputs.append(caref)

    if not outputs:
        return pd.DataFrame(columns=list(df.columns) + ["prompt", "caption", "task"])

    result_df = pd.concat(outputs, ignore_index=True)

    if print_distribution:
        print("Task distribution:")
        print(result_df["task"].value_counts())

    return result_df

# --- notebook cell 10 ---
final_df = create_location_aware_dataframe(df)

# --- notebook cell 11 ---
import ast

def parse_bbox(bbox):
    """
    Convert bbox string/list to a list of numbers:
    [x1, x2, y1, y2, z1, z2]
    """
    if isinstance(bbox, list):
        return bbox

    bbox = str(bbox).strip()
    bbox = bbox.replace("(", "[").replace(")", "]")

    return list(ast.literal_eval(bbox))


def format_bbox(bbox):
    """
    Convert bbox list back to standard string format.
    """
    return "[" + ", ".join(str(x) for x in bbox) + "]"


def combine_bboxes(bboxes):
    """
    Combine multiple bounding boxes into one enclosing bounding box.

    Each bbox is expected to be:
    [x1, x2, y1, y2, z1, z2]
    """
    parsed = [parse_bbox(b) for b in bboxes]

    combined = [
        min(b[0] for b in parsed),  # x1
        max(b[1] for b in parsed),  # x2
        min(b[2] for b in parsed),  # y1
        max(b[3] for b in parsed),  # y2
        min(b[4] for b in parsed),  # z1
        max(b[5] for b in parsed),  # z2
    ]

    return format_bbox(combined)

# --- notebook cell 12 ---
def merge_aref_and_caref_duplicate_boxes(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    For task == 'aref' and task == 'caref', merge rows where these columns are equal:

        Caption,
        AnatomicalRegion,
        with_lesion,
        T1w,
        T1ce,
        T2w,
        FLAIR,
        Pathology

    For aref:
        - BoundingBox is updated to the combined bounding box
        - caption is updated to the combined bounding box

    For caref:
        - BoundingBox is updated to the combined bounding box
        - caption is updated to "{Caption} {combined_bbox}"

    Non-aref/caref rows are left unchanged.

    Note the union is taken in normalized (percent) space. That is safe: the
    per-axis transform is monotonically increasing, so a min/max union of
    normalized boxes equals the normalization of the union of the voxel boxes.

    Prompts are rebuilt from the same AREF_PROMPT / CAREF_PROMPT_HEAD constants
    used by create_location_aware_dataframe, rather than restating them here.
    """

    group_cols = [
        "Caption",
        "AnatomicalRegion",
        "with_lesion",
        "T1w",
        "T1ce",
        "T2w",
        "FLAIR",
        "Pathology",
    ]

    required_cols = group_cols + ["BoundingBox", "caption", "task"]
    missing_cols = [col for col in required_cols if col not in final_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    merge_tasks = ["aref", "caref"]

    to_merge_df = final_df[final_df["task"].isin(merge_tasks)].copy()
    untouched_df = final_df[~final_df["task"].isin(merge_tasks)].copy()

    if to_merge_df.empty:
        return final_df.copy()

    merged_rows = []

    for task_name, task_df in to_merge_df.groupby("task", sort=False):
        for _, group in task_df.groupby(group_cols, dropna=False, sort=False):
            new_row = group.iloc[0].copy()

            combined_bbox = combine_bboxes(group["BoundingBox"])

            new_row["BoundingBox"] = combined_bbox

            if task_name == "aref":
                new_row["caption"] = combined_bbox
                new_row["prompt"] = AREF_PROMPT + str(new_row["Caption"]).strip()

            elif task_name == "caref":
                description = str(new_row["Caption"]).strip()
                new_row["caption"] = f"{description} {combined_bbox}"
                new_row["prompt"] = CAREF_PROMPT_HEAD + str(new_row["AnatomicalRegion"]).strip()

            merged_rows.append(new_row)

    merged_df = pd.DataFrame(merged_rows)

    result_df = pd.concat([untouched_df, merged_df], ignore_index=True)

    return result_df

# --- notebook cell 13 ---
final_df = merge_aref_and_caref_duplicate_boxes(final_df)

# --- notebook cell 14 ---
final_df[['BoundingBox', 'AnatomicalRegion',
       'with_lesion', 'T1w', 'T1ce', 'T2w', 'FLAIR', 'Pathology', 'prompt',
       'caption', 'task']].to_csv(
    "/home/data/BRAIN_DIFF_S1/step1_intake_norm.csv", index=False)
