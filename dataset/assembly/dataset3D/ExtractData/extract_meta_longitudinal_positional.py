import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from dataset.ExtractData.config import BRAIN_MAPPING
from dataset3D.ExtractData.extract_meta import open_image
from dataset3D.ExtractData.extract_meta_positional import (
    ExtractMetaDataPositional,
    MERGE_DILATION_ITERS,
    MIN_LESION_VOXELS,
    MIN_REGION_COVERAGE,
    NUM_BOXES,
)


class ExtractMetaDataLongitudinalPositional(ExtractMetaDataPositional):
    """
    Longitudinal (two-timepoint) positional extractor: one row per matched
    lesion (or lesion-free region) pair across a ref/main timepoint. Reuses
    ExtractMetaDataPositional's bounding-box/region logic and
    ExtractMetaData3D._get_delta_and_scenario's containment/delta-ratio
    scenario classification, applied per lesion component instead of
    whole-volume.

    Each row is a self-contained comparative box <-> caption pair with columns
    Caption, BoundingBox_ref, BoundingBox_main, AnatomicalRegion_ref,
    AnatomicalRegion_main, LesionScenario, DeltaFraction, with_lesion, plus
    per-modality paths for both timepoints and Pathology.

      matched lesion rows : a lesion present at both timepoints, matched by
                            spatial overlap of its connected component across
                            ref/main. Classified stable/growth/shrinkage/
                            new_lesion_with_regression via _get_delta_and_scenario.
      resolved rows       : a lesion present only at ref (gone by main).
      new_lesion rows     : a lesion present only at main (absent at ref).
      normal rows         : NUM_NORMAL_BOXES randomly-chosen regions lesion-free
                            at BOTH timepoints, e.g. "The right hippocampus is
                            unremarkable at both timepoints."
    """

    def _lesion_components(self, lesion_mask: np.ndarray, full_seg: np.ndarray) -> list:
        """Per-component {mask, size, box, region} dicts, largest first.

        Same dilate -> connected-component -> filter -> majority-region logic
        as ExtractMetaDataPositional.get_lesion_boxes, but keeps the boolean
        component mask (needed for cross-timepoint overlap matching) instead
        of building a caption directly.
        """
        if not lesion_mask.any():
            return []

        dilated = ndimage.binary_dilation(lesion_mask, iterations=MERGE_DILATION_ITERS)
        labeled, n = ndimage.label(dilated)

        comps = []
        for comp in range(1, n + 1):
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
            comps.append({"mask": comp_mask, "size": size,
                          "box": self.get_bounding_box(comp_mask), "region": region})

        comps.sort(key=lambda c: c["size"], reverse=True)
        return comps

    def _article(self, desc: str) -> str:
        return "an" if desc[:1].lower() in "aeiou" else "a"

    def _pair_caption(self, scenario: str, region_ref, region_main, pathology) -> str:
        desc = self._lesion_descriptor(pathology)
        article = self._article(desc)

        if scenario == "stable":
            return f"There is {article} {desc} in the {region_ref}, stable compared to the prior timepoint, with no interval change."
        elif scenario == "growth":
            suffix = f", now centered in the {region_main}" if region_main != region_ref else ""
            return f"There is {article} {desc} in the {region_ref}, increased in size compared to the prior timepoint{suffix}."
        elif scenario == "shrinkage":
            suffix = f", now centered in the {region_main}" if region_main != region_ref else ""
            return f"There is {article} {desc} in the {region_ref}, decreased in size compared to the prior timepoint{suffix}."
        elif scenario == "new_lesion_with_regression":
            return (f"Compared to the prior {desc} in the {region_ref}, there is partial regression "
                    f"with a new {desc} in the {region_main}.")
        elif scenario == "resolved":
            return f"The previously noted {desc} in the {region_ref} has resolved."
        else:  # new_lesion
            return f"There is a new {desc} in the {region_main} not present at the prior timepoint."

    def get_lesion_box_pairs(self, lesion_mask_ref: np.ndarray, full_seg_ref: np.ndarray,
                             lesion_mask_main: np.ndarray, full_seg_main: np.ndarray, pathology) -> list:
        """
        One row per lesion matched (or unmatched) across ref/main. Both masks
        are assumed already co-registered to the same (MNI) voxel space, so
        components are directly comparable by voxel overlap with no extra
        registration step.
        """
        comps_ref = self._lesion_components(lesion_mask_ref, full_seg_ref)
        comps_main = self._lesion_components(lesion_mask_main, full_seg_main)

        # Greedy best-overlap bipartite matching: score every (ref, main) pair
        # by voxel overlap, then accept highest-overlap pairs first, each
        # component used at most once. Sufficient given the small (~1-10)
        # component counts typical of lesion masks -- no need for an optimal
        # (Hungarian) assignment.
        triples = []
        for i, cr in enumerate(comps_ref):
            for j, cm in enumerate(comps_main):
                overlap = int(np.sum(cr["mask"] & cm["mask"]))
                if overlap > 0:
                    triples.append((overlap, i, j))
        triples.sort(key=lambda t: t[0], reverse=True)

        used_ref, used_main = set(), set()
        matches = []
        for overlap, i, j in triples:
            if i in used_ref or j in used_main:
                continue
            used_ref.add(i)
            used_main.add(j)
            matches.append((i, j))

        rows = []

        for i, j in matches:
            cr, cm = comps_ref[i], comps_main[j]
            _, scenario = self._get_delta_and_scenario(cr["mask"], cm["mask"])
            delta = cm["size"] / cr["size"]
            rows.append({
                "Caption": self._pair_caption(scenario, cr["region"], cm["region"], pathology),
                "BoundingBox_ref": cr["box"], "BoundingBox_main": cm["box"],
                "AnatomicalRegion_ref": cr["region"], "AnatomicalRegion_main": cm["region"],
                "LesionScenario": scenario, "DeltaFraction": delta,
                "with_lesion": 1, "_size": max(cr["size"], cm["size"]),
            })

        for i, cr in enumerate(comps_ref):
            if i in used_ref:
                continue
            rows.append({
                "Caption": self._pair_caption("resolved", cr["region"], None, pathology),
                "BoundingBox_ref": cr["box"], "BoundingBox_main": None,
                "AnatomicalRegion_ref": cr["region"], "AnatomicalRegion_main": None,
                "LesionScenario": "resolved", "DeltaFraction": 0.0,
                "with_lesion": 1, "_size": cr["size"],
            })

        for j, cm in enumerate(comps_main):
            if j in used_main:
                continue
            rows.append({
                "Caption": self._pair_caption("new_lesion", None, cm["region"], pathology),
                "BoundingBox_ref": None, "BoundingBox_main": cm["box"],
                "AnatomicalRegion_ref": None, "AnatomicalRegion_main": cm["region"],
                "LesionScenario": "new_lesion", "DeltaFraction": None,
                "with_lesion": 1, "_size": cm["size"],
            })

        rows.sort(key=lambda r: r["_size"], reverse=True)
        for r in rows:
            del r["_size"]
        return rows

    def _lesion_coverage(self, lesion_mask: np.ndarray, full_seg: np.ndarray) -> dict:
        """Fraction of each region's voxels covered by the lesion mask."""
        labels, counts = np.unique(full_seg[full_seg > 0], return_counts=True)
        region_total = dict(zip(labels.tolist(), counts.tolist()))

        les_labels, les_counts = np.unique(full_seg[lesion_mask & (full_seg > 0)], return_counts=True)
        return {int(l): c / region_total[int(l)] for l, c in zip(les_labels, les_counts)}

    def get_normal_box_pairs(self, lesion_mask_ref: np.ndarray, lesion_mask_main: np.ndarray,
                             full_seg_ref: np.ndarray, full_seg_main: np.ndarray, count: int) -> list:
        """
        `count` randomly-chosen regions that are lesion-free (< MIN_REGION_COVERAGE)
        at BOTH timepoints. Regions sharing a name are collapsed so no two
        normal captions are identical.
        """
        cov_ref = self._lesion_coverage(lesion_mask_ref, full_seg_ref)
        cov_main = self._lesion_coverage(lesion_mask_main, full_seg_main)

        labels_ref = set(int(l) for l in np.unique(full_seg_ref[full_seg_ref > 0]))
        labels_main = set(int(l) for l in np.unique(full_seg_main[full_seg_main > 0]))
        candidates = [
            l for l in (labels_ref & labels_main)
            if cov_ref.get(l, 0) < MIN_REGION_COVERAGE and cov_main.get(l, 0) < MIN_REGION_COVERAGE
        ]
        random.shuffle(candidates)

        rows, used_names = [], set()
        for label in candidates:
            if len(rows) >= count:
                break
            name = BRAIN_MAPPING.get(label, "unknown")
            if name in used_names:
                continue
            used_names.add(name)
            rows.append({
                "Caption": f"The {name} is unremarkable at both timepoints.",
                "BoundingBox_ref": self.get_bounding_box(full_seg_ref == label),
                "BoundingBox_main": self.get_bounding_box(full_seg_main == label),
                "AnatomicalRegion_ref": name, "AnatomicalRegion_main": name,
                "LesionScenario": "no_lesion", "DeltaFraction": None, "with_lesion": 0,
            })
        return rows

    def process_brain_pair_positional(self, modality_paths_ref: dict, modality_paths_main: dict,
                                      lesion_path_ref, lesion_path_main,
                                      seg_path_ref, seg_path_main, output_path,
                                      pathology, tag: str = "Real") -> pd.DataFrame:
        """
        One row per matched lesion (or lesion-free region) pair for a single
        subject's ref/main timepoints. `output_path` and `tag` are accepted
        for pipeline signature compatibility but unused.
        """
        full_seg_ref = open_image(seg_path_ref, False)
        full_seg_main = open_image(seg_path_main, False)

        def load_lesion_mask(lesion_path, full_seg):
            if lesion_path is None:
                return np.zeros_like(full_seg, dtype=bool)
            lesion_raw = open_image(lesion_path, False)
            mask = (lesion_raw > 0.5) & (lesion_raw != 2)  # Exclude FLAIR EXPANSION, temporary fix for now.
            if mask.shape != full_seg.shape:
                raise ValueError("The two NIfTI files do not have the same dimensions.")
            return mask

        lesion_mask_ref = load_lesion_mask(lesion_path_ref, full_seg_ref)
        lesion_mask_main = load_lesion_mask(lesion_path_main, full_seg_main)

        rows = self.get_lesion_box_pairs(lesion_mask_ref, full_seg_ref,
                                         lesion_mask_main, full_seg_main, pathology)[:NUM_BOXES]
        rows += self.get_normal_box_pairs(lesion_mask_ref, lesion_mask_main,
                                          full_seg_ref, full_seg_main, NUM_BOXES - len(rows))

        for r in rows:
            for col in ("T1w", "T1ce", "T2w", "FLAIR"):
                path_ref = modality_paths_ref.get(col)
                path_main = modality_paths_main.get(col)
                r[f"{col}_ref"] = str(Path(path_ref).resolve()) if path_ref is not None else None
                r[f"{col}_main"] = str(Path(path_main).resolve()) if path_main is not None else None
            r["Pathology"] = pathology

        columns = ["Caption", "BoundingBox_ref", "BoundingBox_main",
                   "AnatomicalRegion_ref", "AnatomicalRegion_main",
                   "LesionScenario", "DeltaFraction", "with_lesion",
                   "T1w_ref", "T1ce_ref", "T2w_ref", "FLAIR_ref",
                   "T1w_main", "T1ce_main", "T2w_main", "FLAIR_main", "Pathology"]
        return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    # Synthetic smoke test covering every scenario branch with hand-built
    # blobs, no real MRI data required (matching/classification logic is
    # dimension/content agnostic).
    meta_instance = ExtractMetaDataLongitudinalPositional()
    shape = (60, 60, 60)

    full_seg_ref = np.zeros(shape, dtype=np.int32)
    full_seg_main = np.zeros(shape, dtype=np.int32)
    # Distinct regions so each scenario lands somewhere nameable.
    full_seg_ref[0:20, 0:20, 0:20] = 10   # stable region
    full_seg_main[0:20, 0:20, 0:20] = 10
    full_seg_ref[20:40, 0:20, 0:20] = 11  # growth region
    full_seg_main[20:40, 0:20, 0:20] = 11
    full_seg_ref[0:20, 20:40, 0:20] = 12  # shrinkage region
    full_seg_main[0:20, 20:40, 0:20] = 12
    full_seg_ref[20:40, 20:40, 0:20] = 17  # resolved region (ref only)
    full_seg_ref[40:60, 0:20, 0:20] = 49   # new_lesion region (main only)
    full_seg_main[40:60, 0:20, 0:20] = 49
    full_seg_ref[40:60, 20:40, 0:20] = 4   # new_lesion_with_regression region
    full_seg_main[40:60, 20:40, 0:20] = 4
    # A large lesion-free region at both timepoints -> normal row candidate.
    full_seg_ref[0:60, 40:60, 0:60] = 8
    full_seg_main[0:60, 40:60, 0:60] = 8

    lesion_mask_ref = np.zeros(shape, dtype=bool)
    lesion_mask_main = np.zeros(shape, dtype=bool)

    # stable
    lesion_mask_ref[5:10, 5:10, 5:10] = True
    lesion_mask_main[5:10, 5:10, 5:10] = True
    # growth
    lesion_mask_ref[25:28, 5:8, 5:8] = True
    lesion_mask_main[25:32, 5:12, 5:12] = True
    # shrinkage
    lesion_mask_ref[5:12, 25:32, 5:12] = True
    lesion_mask_main[5:8, 25:28, 5:8] = True
    # resolved
    lesion_mask_ref[25:30, 25:30, 5:10] = True
    # new_lesion
    lesion_mask_main[45:50, 5:10, 5:10] = True
    # new_lesion_with_regression (partial, non-containing overlap)
    lesion_mask_ref[45:50, 25:30, 5:10] = True
    lesion_mask_main[48:53, 28:33, 5:10] = True

    # process_brain_pair_positional needs real files on disk (it opens NIfTIs
    # by path); exercise the box/matching/caption logic directly instead.
    rows = meta_instance.get_lesion_box_pairs(lesion_mask_ref, full_seg_ref,
                                              lesion_mask_main, full_seg_main, pathology="stroke")
    rows += meta_instance.get_normal_box_pairs(lesion_mask_ref, lesion_mask_main,
                                               full_seg_ref, full_seg_main, count=3)
    print(pd.DataFrame(rows)[["Caption", "LesionScenario", "AnatomicalRegion_ref",
                              "AnatomicalRegion_main", "BoundingBox_ref", "BoundingBox_main"]].to_string())
