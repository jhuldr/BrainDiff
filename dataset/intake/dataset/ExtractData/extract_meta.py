from pathlib import Path

import nibabel as nib
import numpy as np
import os
import pandas as pd
import random
from collections import Counter

from .config import *
import dataset.dataset_utils as utils

SPLICE_DISTANCE = 10
NO_LESION_KEEP_RATE = 0.05

# The goal is to make this work for other image types in the future
def open_image(image_path, convert: bool = True) -> np.ndarray:
    if os.path.isfile(image_path):
        data = nib.load(image_path).get_fdata()
        return data
    else:
        raise Exception("File Not Found")


class ExtractMetaData:

    def get_slice(self, volume, index, orientation):
        """Extract a 2D slice from a 3D volume."""
        if orientation == "sagittal":
            return volume[index, :, :]
        elif orientation == "coronal":
            return volume[:, index, :]
        else:  # axial
            return volume[:, :, index]

    def get_intersection(self, lesion_path, seg_path) -> np.ndarray:
        lesion_data = open_image(lesion_path, False)
        seg_data = open_image(seg_path, False)

        if lesion_data.shape != seg_data.shape:
            raise ValueError("The two NIfTI files do not have the same dimensions.")

        intersect = (lesion_data > 0.5) & (seg_data > 0)
        seg_data[~intersect] = 0

        return seg_data

    def get_slice_level(self, index, slice_range):
        slice_idx = index - slice_range[0]
        total_slices = slice_range[1] - slice_range[0]

        ratio = slice_idx / total_slices
        if ratio < 0.33:
            return "inferior"
        elif ratio < 0.66:
            return "mid"
        else:
            return "superior"

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
        Create a default metadata dictionary for a brain slice.
        """
        meta = {
            "Ventricular": False,
            "Ventricle": [],
            "NoLesion": True,
            "Orientation": None,
            "Level": None,
            "TotalSize": None,
            "LesionSide": None,
            "Territory": None,
            "TissueGroup": [],
            "Caption": None,
            "Modality": None,
            "FilePath": None,
            "LesionSlicePath": None,
            "DataSplit": None
        }

        # Dynamically create lesion location/ratio slots
        for i in range(MAX_LENGTH):
            meta[f"LesionLocation_{i}"] = None
            meta[f"LesionRatio_{i}"] = 0.0

        return meta

    def extract_modality(self, mri_path):
        mri_path = Path(mri_path)
        if mri_path.name.split("_")[0] in ["T1", "T2", "FLAIR"]:
            return mri_path.name.split("_")[0]
        else:
            return "T1-weighted"

    def label_2d(self, splice_generator, lesion_path, seg_path, output_path, tag: str, modality="T1-weighted") -> pd.DataFrame:
        intersection = self.get_intersection(lesion_path, seg_path)
        lesion_volume = open_image(lesion_path, False)

        meta_df = pd.DataFrame()

        for brain_slice, brain_orientation, brain_index, brain_range in splice_generator:

            if random.random() > 0.5:
                MAPPING_DICT = BRAIN_MAPPING
            else:
                MAPPING_DICT = SIMPLE_BRAIN_MAPPING

            label_meta = self.init_label_meta()
            total_brain_count = np.sum(brain_slice > 0.09)

            sliced_intersection = self.get_slice(intersection, brain_index, brain_orientation)
            lesion_slice = self.get_slice(lesion_volume, brain_index, brain_orientation)

            labels, counts = np.unique(sliced_intersection[sliced_intersection > 0], return_counts=True)

            label_meta["Level"] = self.get_slice_level(brain_index, brain_range)
            label_meta["Orientation"] = brain_orientation
            label_meta["Modality"] = modality

            total_ratio = np.sum(counts) / total_brain_count

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
                            label_meta["Ventricle"].append(MAPPING_DICT[label].replace("left", "").replace("right", "").strip())
                    else:
                        ratio_dict[f"LesionRatio_{recorded_regions}"] += count
                        label_meta[f"LesionLocation_{recorded_regions}"] = MAPPING_DICT[label].replace("left", "").replace(
                            "right", "").strip()

                        if "TissueGroup" in label_meta.keys():
                            label_meta["TissueGroup"].append(TISSUE_GROUPS.get(label, ""))
                        else:
                            label_meta["TissueGroup"] = [TISSUE_GROUPS.get(label, "")]

                        recorded_regions += 1

            for key in ratio_dict:
                label_meta[key] = (ratio_dict[key] / ratio_dict.total()) * 100

            label_meta["LesionSide"] = self.get_lesion_side(left_count, right_count)
            label_meta["FilePath"] = utils.save_image(brain_slice, output_path / f"{tag}_Images")
            label_meta["LesionSlicePath"] = utils.save_image(lesion_slice, output_path / f"{tag}_LesionSlices")

            meta_row = pd.DataFrame([label_meta])
            meta_df = pd.concat([meta_df, meta_row], ignore_index=True)

        return meta_df

    def splice_2d(self, mri_path):

        mri_image = open_image(mri_path, False)

        for index, brain_region in enumerate(["sagittal", "coronal", "axial"]):
            brain_range = []
            for curr_slice_index in range(mri_image.shape[index]):
                brain_slice = self.get_slice(mri_image, curr_slice_index, brain_region)

                brain_fraction = np.sum((brain_slice > 0.09).astype(np.uint8)) / (brain_slice.shape[0] * brain_slice.shape[1])
                if brain_fraction >= THRESHOLD:
                    if not brain_range:
                        brain_range = [curr_slice_index, curr_slice_index]
                    else:
                        brain_range[1] = curr_slice_index

            if not brain_range:
                brain_range = [0, mri_image.shape[index]]

            for curr_slice_index in range(brain_range[0], brain_range[1] + 1, SPLICE_DISTANCE):
                brain_slice = self.get_slice(mri_image, curr_slice_index, brain_region)
                yield brain_slice, brain_region, curr_slice_index, brain_range

    def _get_lesion_mask_slice(self, lesion_volume: np.ndarray, index: int, orientation: str) -> np.ndarray:
        """Return a binary 2D lesion mask for the given slice."""
        lesion_slice = self.get_slice(lesion_volume, index, orientation)
        return lesion_slice > 0.1

    def _has_lesion_on_slice(self, lesion_volume: np.ndarray, index: int, orientation: str) -> bool:
        """Return True if the given slice contains any lesion signal."""
        return np.any(self._get_lesion_mask_slice(lesion_volume, index, orientation))

    def _get_slice_delta_and_scenario(
            self,
            mask1: np.ndarray,
            mask2: np.ndarray,
    ) -> tuple[float | None, str]:
        """
        Compute the per-slice DeltaFraction and LesionScenario from two binary masks.

        DeltaFraction
        -------------
        sum(mask2) / sum(mask1) as a raw ratio.
        None when mask1 is empty (no T0 lesion on this slice), because the
        denominator is undefined — this always coincides with new_lesion or no_lesion.

        LesionScenario
        --------------
        no_lesion                 : neither slice has a lesion
        new_lesion                : only T1 (mask2) has a lesion, or both have lesions
                                    but masks are fully disjoint
        new_lesion_with_regression: both have lesions, masks partially overlap but
                                    neither is contained within the other
        resolved                  : only T0 (mask1) has a lesion
        stable                    : masks are contained (>=95%) and delta within [0.95, 1.05]
        growth                    : masks are contained (>=95%) and delta > 1.05
        shrinkage                 : masks are contained (>=95%) and delta < 0.95

        Containment threshold: 95% to tolerate registration noise.
        """
        CONTAINMENT_THRESHOLD = 0.95

        has1 = np.any(mask1)
        has2 = np.any(mask2)

        # ── no lesion on either slice ──────────────────────────────────────────
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

    def _extract_single_slice_meta(
            self,
            brain_volume: np.ndarray,
            intersection: np.ndarray,
            lesion_volume: np.ndarray,
            brain_index: int,
            brain_orientation: str,
            brain_range: list,
            modality: str,
            output_path: Path,
            tag: str,
    ) -> dict:
        """
        Extract all metadata for one 2D slice of a single brain.
        Returns a plain dict (one row's worth of label_meta).
        This is intentionally kept close to the original label_2d loop body
        so that the core extraction logic is not altered.
        """
        if random.random() > 0.5:
            MAPPING_DICT = BRAIN_MAPPING
        else:
            MAPPING_DICT = SIMPLE_BRAIN_MAPPING

        label_meta = self.init_label_meta()

        brain_slice = self.get_slice(brain_volume, brain_index, brain_orientation)
        total_brain_count = np.sum(brain_slice > 0.09)

        sliced_intersection = self.get_slice(intersection, brain_index, brain_orientation)
        lesion_slice = self.get_slice(lesion_volume, brain_index, brain_orientation)

        labels, counts = np.unique(sliced_intersection[sliced_intersection > 0], return_counts=True)

        label_meta["Level"] = self.get_slice_level(brain_index, brain_range)
        label_meta["Orientation"] = brain_orientation
        label_meta["Modality"] = modality

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
        label_meta["FilePath"] = utils.save_image(brain_slice, output_path / f"{tag}_Images")
        label_meta["LesionSlicePath"] = utils.save_image(lesion_slice, output_path / f"{tag}_LesionSlices")

        return label_meta

    def process_single_brain(self, mri_path, lesion_path, seg_path, output_path, tag: str = "Real"):
        generator = self.splice_2d(mri_path)
        modality = self.extract_modality(mri_path)
        meta_df = self.label_2d(generator, lesion_path, seg_path, output_path, tag, modality)
        return meta_df

    def process_brain_pair(
            self,
            brain1_path,
            brain1_lesion,
            brain1_synth,
            brain2_path,
            brain2_lesion,
            brain2_synth,
            output_path,
            tag: str = "Real",
    ) -> pd.DataFrame:
        """
        Process a matched pair of brain scans (e.g. baseline T0 and follow-up T1)
        and return a single DataFrame where every row represents one paired 2D slice.

        Both brains are assumed to be in the same space (MNI), so slicing is driven
        entirely by brain1's geometry and the same absolute index is applied to brain2.

        Sampling strategy
        -----------------
        - A slice is kept if *either* brain has a lesion on that slice.
        - Slices where *neither* brain has a lesion are kept with probability
          NO_LESION_KEEP_RATE (5 %) to preserve some negative-pair examples.

        Pair-level columns
        ------------------
        Brain1_*/Brain2_*   : All original per-brain metadata columns, prefixed so
                              they can coexist in the same row without collision.
        Brain1_MRIPath      : Absolute path to brain1's 3D MRI volume.
        Brain2_MRIPath      : Absolute path to brain2's 3D MRI volume.
        Brain1_LesionPath   : Absolute path to brain1's 3D lesion mask.
        Brain2_LesionPath   : Absolute path to brain2's 3D lesion mask.
        Brain1_HasLesion    : Boolean — whether brain1 has a lesion on this slice.
        Brain2_HasLesion    : Boolean — whether brain2 has a lesion on this slice.
        HasLesionPair       : Boolean — True when at least one brain carries a lesion.
                              Cheap downstream filter.
        DeltaFraction       : Per-slice lesion area ratio: sum(mask2) / sum(mask1),
                              rounded to nearest 0.05. None when mask1 is empty
                              (new_lesion / no_lesion cases where T0 has no lesion).
        LesionScenario      : Categorical descriptor of how the lesion changed:
                                "no_lesion"                — neither slice has a lesion
                                "new_lesion"               — T1 lesion with no T0 lesion,
                                                             or fully disjoint masks
                                "new_lesion_with_regression" — partial overlap; T1 has new
                                                             territory AND T0 area regressed
                                "resolved"                 — T0 had lesion, T1 does not
                                "growth"                   — T0 mask contained within T1
                                "shrinkage"                — T1 mask contained within T0
        PairTag             : The tag string shared by both brains in this pair.
                              Useful for grouping rows that belong to the same subject.
        """
        output_path = Path(output_path)

        # Load all volumes once — reused across every slice
        brain1_vol = open_image(brain1_path, False)
        brain2_vol = open_image(brain2_path, False)
        lesion1_vol = open_image(brain1_lesion, False)
        lesion2_vol = open_image(brain2_lesion, False)

        intersection1 = self.get_intersection(brain1_lesion, brain1_synth)
        intersection2 = self.get_intersection(brain2_lesion, brain2_synth)

        modality1 = self.extract_modality(brain1_path)
        modality2 = self.extract_modality(brain2_path)

        meta_df = pd.DataFrame()

        # Drive slicing from brain1 (both are in the same MNI space)
        for index, brain_region in enumerate(["sagittal", "coronal", "axial"]):
            brain_range = []
            for curr_slice_index in range(brain1_vol.shape[index]):
                brain_slice = self.get_slice(brain1_vol, curr_slice_index, brain_region)
                brain_fraction = (
                        np.sum((brain_slice > 0.09).astype(np.uint8))
                        / (brain_slice.shape[0] * brain_slice.shape[1])
                )
                if brain_fraction >= THRESHOLD:
                    if not brain_range:
                        brain_range = [curr_slice_index, curr_slice_index]
                    else:
                        brain_range[1] = curr_slice_index

            if not brain_range:
                brain_range = [0, brain1_vol.shape[index]]

            for curr_slice_index in range(brain_range[0], brain_range[1] + 1, SPLICE_DISTANCE):

                mask1 = self._get_lesion_mask_slice(lesion1_vol, curr_slice_index, brain_region)
                mask2 = self._get_lesion_mask_slice(lesion2_vol, curr_slice_index, brain_region)

                has_lesion1 = np.any(mask1)
                has_lesion2 = np.any(mask2)
                has_lesion_pair = has_lesion1 or has_lesion2

                # Sampling: always keep slices with a lesion; stochastically keep the rest
                if not has_lesion_pair and random.random() > NO_LESION_KEEP_RATE:
                    continue

                delta_fraction, lesion_scenario = self._get_slice_delta_and_scenario(mask1, mask2)

                meta1 = self._extract_single_slice_meta(
                    brain_volume=brain1_vol,
                    intersection=intersection1,
                    lesion_volume=lesion1_vol,
                    brain_index=curr_slice_index,
                    brain_orientation=brain_region,
                    brain_range=brain_range,
                    modality=modality1,
                    output_path=output_path,
                    tag=f"{tag}_Brain1",
                )

                meta2 = self._extract_single_slice_meta(
                    brain_volume=brain2_vol,
                    intersection=intersection2,
                    lesion_volume=lesion2_vol,
                    brain_index=curr_slice_index,
                    brain_orientation=brain_region,
                    brain_range=brain_range,
                    modality=modality2,
                    output_path=output_path,
                    tag=f"{tag}_Brain2",
                )

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
                paired_row["Brain1_HasLesion"] = bool(has_lesion1)
                paired_row["Brain2_HasLesion"] = bool(has_lesion2)
                paired_row["HasLesionPair"] = has_lesion_pair
                paired_row["DeltaFraction"] = delta_fraction
                paired_row["LesionScenario"] = lesion_scenario
                paired_row["PairTag"] = tag

                meta_df = pd.concat([meta_df, pd.DataFrame([paired_row])], ignore_index=True)

        return meta_df

if __name__ == "__main__":
    meta_instance = ExtractMetaData()
    df = meta_instance.process_single_brain(
        "data/y0_0.nii.gz",
        "data/x0_0.nii.gz",
        "data/y0_0_synthseg.nii.gz",
        "data/output",
    )
    df.to_csv("output.csv")
