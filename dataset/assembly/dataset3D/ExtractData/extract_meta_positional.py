import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

# Reuse the shared label maps / constants and the whole-volume extraction logic.
from dataset.ExtractData.config import *
from dataset3D.ExtractData.extract_meta import ExtractMetaData3D, open_image
from dataset3D.CaptionGeneration.template_generator import TemplateCaptionGenerator

MERGE_DILATION_ITERS = 3    # bridge gaps up to ~2*iters voxels so close lesions merge into one box
MIN_LESION_VOXELS = 5       # drop lesion components smaller than this (noise); matches SBLE min_volume_voxels
MIN_REGION_COVERAGE = 0.05  # a region counts as lesion-involved (excluded from normals) at >= 5% coverage
NUM_BOXES = 10              # fixed number of boxes per volume (lesion rows first, normals fill the rest)


class ExtractMetaDataPositional(ExtractMetaData3D):
    """
    Positional / box-level extractor: one row per bounding box, not per volume.

    Each 3D brain yields a multi-row DataFrame with columns
    Caption, BoundingBox, AnatomicalRegion, T1w, T1ce, T2w, FLAIR, Pathology. Two kinds of row
    are emitted, each a self-contained, grammatically-correct box↔caption pair
    (no LLM polishing downstream):

      lesion rows : one per distinct lesion. Separate lesions get separate boxes;
                    small lesions close to a larger one are merged into a single
                    box (raw mask dilated by MERGE_DILATION_ITERS before
                    connected-component labelling). Caption names the dominant
                    FreeSurfer region, e.g. "There is a stroke infarct in the
                    left thalamus."
      normal rows : NUM_NORMAL_BOXES randomly-chosen lesion-free regions, e.g.
                    "The right hippocampus is unremarkable."

    BoundingBox is (x_min, x_max, y_min, y_max, z_min, z_max). Region names keep
    their laterality from BRAIN_MAPPING.
    """

    def __init__(self):
        # ExtractMetaData3D defines no __init__; template helpers are reused for
        # pathology-aware wording (lesion_noun, get_side_phrase, ...).
        self.template = TemplateCaptionGenerator()

    def get_bounding_box(self, mask: np.ndarray):
        """Tight union box around all nonzero voxels; None when the mask is empty."""
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        return (int(mins[0]), int(maxs[0]),
                int(mins[1]), int(maxs[1]),
                int(mins[2]), int(maxs[2]))

    def _lesion_descriptor(self, pathology) -> str:
        """
        Grammatical lesion phrase, e.g. "stroke infarct" / "glioma mass". The
        pathology word is prepended to the noun only when it adds information
        (avoids "lesion lesion" / "metastasis metastasis").
        """
        noun = self.template.lesion_noun(pathology)  # stroke->infarct, glioma/tumor->mass, ...
        p = (str(pathology) if pathology is not None else "").strip().lower()
        if not p or p in ("none", "nan", "unspecified") or p == noun:
            return noun
        return f"{p} {noun}"

    def get_lesion_boxes(self, lesion_mask: np.ndarray, full_seg: np.ndarray, pathology) -> list:
        """
        One row per distinct lesion. The mask is dilated before connected-
        component labelling so small satellites within ~2*MERGE_DILATION_ITERS
        voxels of a larger lesion merge into a single box, while genuinely
        separate lesions stay distinct. Each box is named by its dominant
        FreeSurfer region.
        """
        if not lesion_mask.any():
            return []

        dilated = ndimage.binary_dilation(lesion_mask, iterations=MERGE_DILATION_ITERS)
        labeled, n = ndimage.label(dilated)

        comps = []
        for comp in range(1, n + 1):
            # Restrict back to real lesion voxels so the box hugs the lesion, not the dilation.
            comp_mask = (labeled == comp) & lesion_mask
            size = int(comp_mask.sum())
            if size < MIN_LESION_VOXELS:
                continue
            seg_vals = full_seg[comp_mask]
            seg_vals = seg_vals[seg_vals > 0]
            if seg_vals.size == 0:  # lesion entirely outside the parcellation -> unnameable
                continue
            labels, counts = np.unique(seg_vals, return_counts=True)
            region = BRAIN_MAPPING.get(int(labels[np.argmax(counts)]), "unknown")
            comps.append((size, self.get_bounding_box(comp_mask), region))

        comps.sort(key=lambda t: t[0], reverse=True)  # largest lesion first

        desc = self._lesion_descriptor(pathology)
        article = "an" if desc[:1].lower() in "aeiou" else "a"
        return [{"Caption": f"There is {article} {desc} in the {region}.",
                 "BoundingBox": box, "AnatomicalRegion": region, "with_lesion": 1}
                for _, box, region in comps]

    def get_normal_boxes(self, lesion_mask: np.ndarray, full_seg: np.ndarray, count: int) -> list:
        """
        `count` randomly-chosen lesion-free regions (background excluded). A
        region is "lesion-free" when the lesion covers less than
        MIN_REGION_COVERAGE of its volume. Regions sharing a name are collapsed
        so no two normal captions are identical.
        """
        labels, counts = np.unique(full_seg[full_seg > 0], return_counts=True)
        region_total = dict(zip(labels.tolist(), counts.tolist()))

        les_labels, les_counts = np.unique(full_seg[lesion_mask & (full_seg > 0)], return_counts=True)
        lesion_cov = {int(l): c / region_total[int(l)] for l, c in zip(les_labels, les_counts)}

        candidates = [int(l) for l in labels
                      if int(l) != 0 and lesion_cov.get(int(l), 0) < MIN_REGION_COVERAGE]
        random.shuffle(candidates)

        rows, used_names = [], set()
        for label in candidates:
            if len(rows) >= count:
                break
            name = BRAIN_MAPPING.get(label, "unknown")
            if name in used_names:
                continue
            used_names.add(name)
            rows.append({"Caption": f"The {name} is unremarkable.",
                         "BoundingBox": self.get_bounding_box(full_seg == label),
                         "AnatomicalRegion": name, "with_lesion": 0})
        return rows

    def process_single_brain(self, modality_paths: dict, lesion_path, seg_path, output_path,
                             pathology, tag: str = "Real") -> pd.DataFrame:
        """
        One row per bounding box (lesion findings first, then healthy regions)
        for a single 3D brain. `output_path` and `tag` are accepted for pipeline
        signature compatibility but unused.

        `modality_paths` maps whichever of T1w/T1ce/T2w/FLAIR are available to their
        file paths; missing modalities are left as None. `lesion_path` may be None
        when a study has no lesion mask, in which case no lesion rows are emitted.
        """
        full_seg = open_image(seg_path, False)
        if lesion_path is not None:
            lesion_raw = open_image(lesion_path, False)
            lesion_mask = (lesion_raw > 0.5) & (lesion_raw != 2) # Exclude FLAIR EXPANSION, temporary fix for now.
            if lesion_mask.shape != full_seg.shape:
                raise ValueError("The two NIfTI files do not have the same dimensions.")
        else:
            lesion_mask = np.zeros_like(full_seg, dtype=bool)

        rows = self.get_lesion_boxes(lesion_mask, full_seg, pathology)[:NUM_BOXES]
        rows += self.get_normal_boxes(lesion_mask, full_seg, NUM_BOXES - len(rows))

        for r in rows:
            for col in ("T1w", "T1ce", "T2w", "FLAIR"):
                path = modality_paths.get(col)
                r[col] = str(Path(path).resolve()) if path is not None else None
            r["Pathology"] = pathology

        return pd.DataFrame(rows, columns=["Caption", "BoundingBox", "AnatomicalRegion",
                                           "with_lesion", "T1w", "T1ce", "T2w", "FLAIR",
                                           "Pathology"])


if __name__ == "__main__":
    meta_instance = ExtractMetaDataPositional()
    df = meta_instance.process_single_brain(
        {"T1w": "data/y0_0.nii.gz"},
        "data/x0_0.nii.gz",
        "data/y0_0_synthseg.nii.gz",
        "data/output",
        pathology="stroke",
    )
    print(df[["Caption", "AnatomicalRegion", "BoundingBox"]].to_string())
    df.to_csv("output.csv")
