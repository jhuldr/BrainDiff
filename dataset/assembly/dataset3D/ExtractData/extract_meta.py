from pathlib import Path

import nibabel as nib
import numpy as np
import os
import pandas as pd
import random
from collections import Counter

# Reuse the shared label maps / constants from the base dataset package.
from dataset.ExtractData.config import *
import dataset.dataset_utils as utils

NO_LESION_KEEP_RATE = 0.05


# The goal is to make this work for other image types in the future
def open_image(image_path, convert: bool = True) -> np.ndarray:
    if os.path.isfile(image_path):
        data = nib.load(image_path).get_fdata()
        return data
    else:
        raise Exception("File Not Found")


class ExtractMetaData3D:
    """
    3D analogue of dataset.ExtractData.extract_meta.ExtractMetaData.

    Instead of slicing each volume and emitting one row per 2D slice, this
    extractor treats the WHOLE 3D volume as the unit of analysis: every brain
    (or brain pair) produces a single metadata row. The per-region counting,
    territory, laterality, size and lesion-change-scenario logic is identical to
    the 2D pipeline — it operates on label/count arrays and binary masks that
    are dimension-agnostic — so the core logic is preserved unchanged and simply
    fed full-volume data.
    """

    def get_intersection(self, lesion_path, seg_path) -> np.ndarray:
        lesion_data = open_image(lesion_path, False)
        seg_data = open_image(seg_path, False)

        if lesion_data.shape != seg_data.shape:
            raise ValueError("The two NIfTI files do not have the same dimensions.")

        intersect = (lesion_data > 0.5) & (seg_data > 0) 
        seg_data[~intersect] = 0

        return seg_data

    def get_lesion_side(self, left_count, right_count):
        total_lesion = left_count + right_count

        if total_lesion == 0:
            return "none"

        left_ratio = left_count / total_lesion

        if left_ratio >= 0.95:
            return "left"
        elif left_ratio <= 0.05:
            return "right"
        elif left_ratio >= 0.65:
            return "bilateral, left-predominant"
        elif left_ratio <= 0.35:
            return "bilateral, right-predominant"
        else:
            return "bilateral"

    def get_lesion_size(self, total_ratio):
        # NOTE: thresholds were tuned on 2D slice area ratios. On a true 3D
        # volume fraction they may warrant retuning, but per the "keep the core
        # logic the same" requirement they are left unchanged here.
        if total_ratio < .05:
            return "small"
        elif total_ratio < .10:
            return "moderate"
        else:
            return "large"

    def get_territory(self, labels, counts, total_ratio: float):

        if sum(counts) == 0:
            return "None"

        scores = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0
        }

        territory_labels = [DEEP_LABELS, POSTERIOR_LABELS, MCA_LABELS, PCA_LABELS, ACA_LABELS]
        for label, count in zip(labels, counts):
            for index, territory in enumerate(territory_labels):
                if label in territory:
                    scores[str(index)] += count
                    break

        # Lacunar rule
        if total_ratio <= LACUNAR_MAX_RATIO and scores["0"] > 0:
            return "Lacunar"

        if not scores:
            return "Indeterminate"

        # Determine dominant territory; handle ties
        dominant_key = max(scores, key=scores.get)

        if scores[dominant_key] / sum(counts) < 0.60:  # 60 % threshold; adjust as needed
            return "Mixed"

        if dominant_key == "0":
            return "None"
        elif dominant_key == "1":
            return "Vertebrobasilar-predominant"
        elif dominant_key == "2":
            return "MCA-predominant"
        elif dominant_key == "3":
            return "PCA-predominant"
        elif dominant_key == "4":
            return "ACA-predominant"
        else:
            return "None"

    def init_label_meta(self):
        """
        Create a default metadata dictionary for a whole 3D brain volume.

        Compared to the 2D extractor this drops the slice-only fields
        (Orientation, Level, FilePath, LesionSlicePath, DataSplit, Caption).
        """
        meta = {
            "Ventricular": False,
            "Ventricle": [],
            "NoLesion": True,
            "TotalSize": None,
            "LesionSide": None,
            "Territory": None,
            "TissueGroup": [],
            "Modality": None,
        }

        # Dynamically create lesion location/ratio slots
        for i in range(MAX_LENGTH):
            meta[f"LesionLocation_{i}"] = None
            meta[f"LesionRatio_{i}"] = 0.0

        return meta

    def extract_modality(self, mri_path):
        mri_path = Path(mri_path)
        # Get the first part of filename and convert to uppercase for comparison
        modality = mri_path.name.split("_")[0].upper()
        
        # Check against valid modalities (case-insensitive due to .upper())
        if modality in ["T1CE", "T1", "T2", "FLAIR"]:
            return modality  # Already uppercase
        else:
            return "T1"


    def _get_delta_and_scenario(
            self,
            mask1: np.ndarray,
            mask2: np.ndarray,
    ) -> tuple[float | None, str]:
        """
        Compute the DeltaFraction and LesionScenario from two binary masks.

        This is the unchanged scenario logic from the 2D pipeline
        (`_get_slice_delta_and_scenario`). It operates on arbitrary binary
        arrays, so for 3D we simply pass the full-volume lesion masks.

        DeltaFraction
        -------------
        sum(mask2) / sum(mask1) as a raw volume ratio.
        None when mask1 is empty (no T0 lesion), because the denominator is
        undefined — this always coincides with new_lesion or no_lesion.

        LesionScenario
        --------------
        no_lesion                 : neither volume has a lesion
        new_lesion                : only T1 (mask2) has a lesion, or both have
                                    lesions but masks are fully disjoint
        new_lesion_with_regression: both have lesions, masks partially overlap but
                                    neither is contained within the other
        resolved                  : only T0 (mask1) has a lesion
        stable                    : masks are contained (>=95%) and delta within [0.95, 1.05]
        growth                    : masks are contained (>=95%) and delta > 1.05
        shrinkage                 : masks are contained (>=95%) and delta < 0.95

        Containment threshold: 80% to tolerate registration noise.
        """
        CONTAINMENT_THRESHOLD = 0.80

        has1 = np.any(mask1)
        has2 = np.any(mask2)

        # ── no lesion on either volume ─────────────────────────────────────────
        if not has1 and not has2:
            return None, "no_lesion"

        # ── new lesion: T0 clear, T1 has lesion ───────────────────────────────
        if not has1 and has2:
            return None, "new_lesion"

        # ── resolved: T0 has lesion, T1 clear ─────────────────────────────────
        if has1 and not has2:
            return 0.0, "resolved"

        # ── both have lesions ─────────────────────────────────────────────────
        count1 = np.sum(mask1)
        count2 = np.sum(mask2)

        delta = count2 / count1

        overlap = np.sum(mask1 & mask2)

        # Containment checks
        mask2_in_mask1 = (overlap / count2) >= CONTAINMENT_THRESHOLD  # mask2 subset of mask1
        mask1_in_mask2 = (overlap / count1) >= CONTAINMENT_THRESHOLD  # mask1 subset of mask2

        if mask2_in_mask1 or mask1_in_mask2:
            if 0.95 <= delta <= 1.05:
                scenario = "stable"
            elif delta < 0.95:
                scenario = "shrinkage"
            else:
                scenario = "growth"
            return delta, scenario

        # Neither contained within the other
        if overlap > 0:
            # Partial overlap: old lesion partially regressed, new territory appeared
            return delta, "new_lesion_with_regression"
        else:
            # Completely disjoint masks: entirely new lesion location
            return delta, "new_lesion"

    def _extract_single_volume_meta(
            self,
            brain_volume: np.ndarray,
            intersection: np.ndarray,
    ) -> dict:
        """
        Extract all metadata for one whole 3D brain volume.
        Returns a plain dict (one row's worth of label_meta).

        This is intentionally kept close to the original 2D `_extract_single_slice_meta`
        loop body so that the core extraction logic is not altered — the only
        difference is that label/count statistics are computed over the entire
        volume rather than a single slice, and no 2D preview image is saved.
        """
        if random.random() > 0.5:
            MAPPING_DICT = BRAIN_MAPPING
        else:
            MAPPING_DICT = SIMPLE_BRAIN_MAPPING

        label_meta = self.init_label_meta()

        total_brain_count = np.sum(brain_volume > 0.09)

        labels, counts = np.unique(intersection[intersection > 0], return_counts=True)

        #label_meta["Modality"] = modality

        total_ratio = np.sum(counts) / total_brain_count if total_brain_count > 0 else 0.0

        if total_ratio > 0:
            label_meta["NoLesion"] = False

        label_meta["TotalSize"] = self.get_lesion_size(total_ratio)
        label_meta["Territory"] = self.get_territory(labels, counts, total_ratio)

        left_count = 0
        right_count = 0
        recorded_regions = 0
        ratio_dict = Counter()

        for (label, count) in zip(labels[::-1], counts[::-1]):

            if "left" in MAPPING_DICT[label]:
                left_count += count
            elif "right" in MAPPING_DICT[label]:
                right_count += count

            if recorded_regions < MAX_LENGTH:
                if label in [4, 43, 5, 44, 14, 15, 24]:
                    label_meta["Ventricular"] = True
                    if label != 24:
                        label_meta["Ventricle"].append(
                            MAPPING_DICT[label].replace("left", "").replace("right", "").strip()
                        )
                else:
                    ratio_dict[f"LesionRatio_{recorded_regions}"] += count
                    label_meta[f"LesionLocation_{recorded_regions}"] = MAPPING_DICT[label].replace(
                        "left", ""
                    ).replace("right", "").strip()

                    if "TissueGroup" in label_meta:
                        label_meta["TissueGroup"].append(TISSUE_GROUPS.get(label, ""))
                    else:
                        label_meta["TissueGroup"] = [TISSUE_GROUPS.get(label, "")]

                    recorded_regions += 1

        for key in ratio_dict:
            label_meta[key] = (ratio_dict[key] / ratio_dict.total()) * 100

        label_meta["LesionSide"] = self.get_lesion_side(left_count, right_count)

        return label_meta

    def process_single_brain(self, mri_path, lesion_path, seg_path, output_path, pathology,
                             tag: str = "Real") -> pd.DataFrame:
        """
        Extract whole-volume metadata for a single 3D brain.

        Returns a one-row DataFrame. `pathology` (e.g. "stroke", "glioma") is
        recorded verbatim in a `Pathology` column.
        """
        intersection = self.get_intersection(lesion_path, seg_path)
        brain_vol = open_image(mri_path, False)

        label_meta = self._extract_single_volume_meta(brain_vol, intersection)

        label_meta["MRIPath"] = str(Path(mri_path).resolve())
        label_meta["LesionPath"] = str(Path(lesion_path).resolve())
        label_meta["Pathology"] = pathology
        label_meta["Tag"] = tag

        return pd.DataFrame([label_meta])

    def process_brain_pair(
            self,
            brain1_path,
            brain1_lesion,
            brain1_synth,
            brain2_path,
            brain2_lesion,
            brain2_synth,
            output_path,
            pathology,
            tag: str = "Real",
    ) -> pd.DataFrame:
        """
        Process a matched pair of 3D brain scans (e.g. baseline T0 and follow-up
        T1) and return a SINGLE-ROW DataFrame describing the whole-volume change.

        Both brains are assumed to be in the same space (MNI), so the lesion
        masks are directly comparable.

        Pair-level columns
        ------------------
        Brain1_*/Brain2_*   : All per-brain whole-volume metadata columns, prefixed.
        Brain1_MRIPath      : Absolute path to brain1's 3D MRI volume.
        Brain2_MRIPath      : Absolute path to brain2's 3D MRI volume.
        Brain1_LesionPath   : Absolute path to brain1's 3D lesion mask.
        Brain2_LesionPath   : Absolute path to brain2's 3D lesion mask.
        Brain1_HasLesion    : Boolean — whether brain1 carries any lesion.
        Brain2_HasLesion    : Boolean — whether brain2 carries any lesion.
        HasLesionPair       : Boolean — True when at least one brain carries a lesion.
        DeltaFraction       : Whole-volume lesion ratio sum(mask2)/sum(mask1).
                              None when mask1 is empty (new_lesion / no_lesion).
        LesionScenario      : Categorical descriptor of how the lesion changed
                              (see `_get_delta_and_scenario`).
        Pathology           : The pathology type for this run (e.g. "stroke", "glioma").
        PairTag             : The tag string shared by both brains in this pair.
        """
        output_path = Path(output_path)

        # Load all volumes once
        brain1_vol = open_image(brain1_path, False)
        brain2_vol = open_image(brain2_path, False)
        lesion1_vol = open_image(brain1_lesion, False)
        lesion2_vol = open_image(brain2_lesion, False)

        intersection1 = self.get_intersection(brain1_lesion, brain1_synth)
        intersection2 = self.get_intersection(brain2_lesion, brain2_synth)

        modality = self.extract_modality(brain1_path)

        # Whole-volume binary lesion masks
        mask1 = lesion1_vol > 0.1
        mask2 = lesion2_vol > 0.1

        has_lesion1 = bool(np.any(mask1))
        has_lesion2 = bool(np.any(mask2))
        has_lesion_pair = has_lesion1 or has_lesion2

        delta_fraction, lesion_scenario = self._get_delta_and_scenario(mask1, mask2)

        meta1 = self._extract_single_volume_meta(brain1_vol, intersection1)
        meta2 = self._extract_single_volume_meta(brain2_vol, intersection2)

        # Build one merged row: prefix every per-brain key, then add pair-level fields
        paired_row = {}

        for key, val in meta1.items():
            paired_row[f"Brain1_{key}"] = val

        for key, val in meta2.items():
            paired_row[f"Brain2_{key}"] = val

        # 3D source paths — kept at the pair level so each row is fully self-contained
        paired_row["Brain1_MRIPath"] = str(Path(brain1_path).resolve())
        paired_row["Brain2_MRIPath"] = str(Path(brain2_path).resolve())
        paired_row["Brain1_LesionPath"] = str(Path(brain1_lesion).resolve())
        paired_row["Brain2_LesionPath"] = str(Path(brain2_lesion).resolve())

        # Pair-level derived fields
        paired_row["Brain1_HasLesion"] = has_lesion1
        paired_row["Brain2_HasLesion"] = has_lesion2
        paired_row["HasLesionPair"] = has_lesion_pair
        paired_row["DeltaFraction"] = delta_fraction
        paired_row["LesionScenario"] = lesion_scenario
        paired_row["Pathology"] = pathology
        paired_row["Modality"] = modality
        paired_row["PairTag"] = tag

        return pd.DataFrame([paired_row])


if __name__ == "__main__":
    meta_instance = ExtractMetaData3D()
    df = meta_instance.process_single_brain(
        "data/y0_0.nii.gz",
        "data/x0_0.nii.gz",
        "data/y0_0_synthseg.nii.gz",
        "data/output",
        pathology="stroke",
    )
    df.to_csv("output.csv")
