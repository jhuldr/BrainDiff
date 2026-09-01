import math

import pandas as pd


class TemplateCaptionGenerator:
    """
    Deterministic, LLM-free caption fallback for whole-volume 3D metadata.

    Differences vs the 2D template generator:
      - No imaging plane / slice level (3D volumes have neither).
      - Lesion terminology is chosen from the `Pathology` field rather than
        being hard-coded to "infarct".
      - Supports paired (T0/T1) rows via `generate_pair_caption`, producing a
        simple comparative sentence keyed off `LesionScenario`.
    """

    IRREGULAR_PLURALS = {
        "cerebral cortex": "cerebral cortices",
        "hippocampus": "hippocampi",
    }

    # ---- helpers -----------------------------------------------------------
    def clean_string(self, *parts):
        """Join parts, removing empty strings and extra spaces."""
        cleaned = " ".join(str(p).strip() for p in parts if p and str(p).strip())
        return " ".join(cleaned.split())  # Remove double spaces

    def lesion_noun(self, pathology) -> str:
        """Pick the lesion noun from the pathology type."""
        p = (str(pathology) if pathology is not None else "").strip().lower()
        if p == "stroke":
            return "infarct"
        elif p in ("glioma", "tumor", "tumour", "mass", "neoplasm"):
            return "mass"
        elif not p or p in ("none", "nan", "unspecified"):
            return "lesion"
        return p  # use the pathology name itself (e.g. "metastasis")

    def get_side_phrase(self, side, structure):
        """Generate grammatically correct side + structure phrase."""
        if not structure:
            return ""

        side = (side or "").strip().lower()
        structure = structure.strip()

        if "bilateral" in side:
            return f"bilateral {structure}s" if not structure.endswith("s") else f"bilateral {structure}"
        elif side in ["left", "right"]:
            return f"{side} {structure}"
        else:
            return structure

    def get_territory_phrase(self, territory, lesion_noun):
        """Generate vascular territory phrase."""
        if not territory or str(territory) in ["None", "Indeterminate", "unknown", "nan"]:
            return ""

        if str(territory).lower() == "lacunar":
            return f", consistent with a lacunar {lesion_noun}"
        else:
            return f", in the {territory} territory"

    def get_size_descriptor(self, size):
        """Normalize size descriptor."""
        size = (str(size) if size is not None else "").lower().strip()
        return size if size in ["small", "moderate", "large"] else "moderate"

    def correct_grammer(self, captions: dict):
        for key in captions:
            caption = captions[key]
            for original in self.IRREGULAR_PLURALS:
                caption = caption.replace(original, self.IRREGULAR_PLURALS[original])
            captions[key] = caption
        return captions

    def _is_present(self, val) -> bool:
        if val is None:
            return False
        if isinstance(val, float) and math.isnan(val):
            return False
        return bool(str(val).strip()) and str(val).strip().lower() not in ("none", "nan")

    # ---- single-volume captions -------------------------------------------
    def generate_caption(self, meta: pd.Series) -> dict:
        """
        Generate caption variants from single-volume metadata
        (as produced by ExtractMetaData3D.process_single_brain).
        """
        meta = meta.to_dict() if isinstance(meta, pd.Series) else dict(meta)
        modality = meta.get("Modality") or "T1-weighted"
        noun = self.lesion_noun(meta.get("Pathology"))

        # Handle no lesion case
        if meta.get("NoLesion", False):
            return {
                "basic": f"{modality} brain MRI with no acute {noun}.",
                "detailed": f"{modality} brain MRI with no acute {noun}.",
                "quantitative": f"{modality} brain MRI with no acute {noun}.",
                "tissue": f"{modality} brain MRI with no acute {noun}.",
                "path": str(meta.get("MRIPath", "None")),
            }

        size = self.get_size_descriptor(meta.get("TotalSize"))
        side = meta.get("LesionSide", "")
        territory = meta.get("Territory")

        val = meta.get("LesionLocation_0")
        primary = val.strip() if self._is_present(val) else ""
        primary_ratio = meta.get("LesionRatio_0", 0) or 0
        secondary = (
            str(meta.get("LesionLocation_1")).strip()
            if self._is_present(meta.get("LesionLocation_1"))
            else None
        )

        tissue_groups = meta.get("TissueGroup", []) or []
        if isinstance(tissue_groups, str):
            import ast
            try:
                tissue_groups = ast.literal_eval(tissue_groups)
            except (ValueError, SyntaxError):
                tissue_groups = []
        unique_tissues = list(dict.fromkeys(t for t in tissue_groups if t and t != "unknown"))

        territory_phrase = self.get_territory_phrase(territory, noun)
        location_phrase = self.get_side_phrase(side, primary) or ""

        captions = {}

        # === BASIC ===
        captions["basic"] = self.clean_string(
            f"{modality} MRI demonstrating a {size}",
            f"{noun} in the {location_phrase}{territory_phrase}."
        )

        # === DETAILED ===
        if secondary:
            captions["detailed"] = self.clean_string(
                f"{modality} brain MRI demonstrating a {size}",
                f"{noun} involving the {location_phrase}",
                f"and {secondary}{territory_phrase}."
            )
        else:
            captions["detailed"] = captions["basic"]

        # === QUANTITATIVE ===
        if primary_ratio and primary_ratio >= 5:
            percentage = int(round(primary_ratio))
            captions["quantitative"] = self.clean_string(
                f"{modality} MRI with a {size} {noun}",
                f"affecting approximately {percentage}% of the {location_phrase}{territory_phrase}."
            )
        else:
            captions["quantitative"] = captions["basic"]

        # === TISSUE-BASED ===
        if unique_tissues:
            tissue_str = " and ".join(unique_tissues[:2])
            captions["tissue"] = self.clean_string(
                f"{modality} MRI showing a {noun}",
                f"involving the {tissue_str}{territory_phrase}."
            )
        else:
            captions["tissue"] = captions["basic"]

        # === VENTRICULAR EXTENSION ===
        if meta.get("Ventricular", False):
            ventricles = meta.get("Ventricle", [])
            if isinstance(ventricles, str):
                import ast
                try:
                    ventricles = ast.literal_eval(ventricles)
                except (ValueError, SyntaxError):
                    ventricles = []
            if ventricles:
                vent_str = ", ".join(str(v) for v in ventricles)
                extension = f" There is intraventricular extension into the {vent_str}."
                for key in captions:
                    captions[key] = captions[key].rstrip(".") + "." + extension

        captions = self.correct_grammer(captions)
        captions["path"] = str(meta.get("MRIPath", "None"))
        return captions

    # ---- paired (T0/T1) comparative caption -------------------------------
    def generate_pair_caption(self, meta: pd.Series) -> dict:
        """
        Build a simple comparative caption from a paired (Brain1_/Brain2_) row.
        Keyed off LesionScenario; intended only as an offline fallback.
        """
        meta = meta.to_dict() if isinstance(meta, pd.Series) else dict(meta)
        scenario = str(meta.get("LesionScenario", "no_lesion"))
        noun = self.lesion_noun(meta.get("Pathology"))
        modality = meta.get("Brain1_Modality") or "T1-weighted"

        t0_side = meta.get("Brain1_LesionSide", "")
        t0_loc = meta.get("Brain1_LesionLocation_0")
        t1_loc = meta.get("Brain2_LesionLocation_0")
        t0_phrase = self.get_side_phrase(t0_side, t0_loc.strip()) if self._is_present(t0_loc) else ""
        t1_side = meta.get("Brain2_LesionSide", "")
        t1_phrase = self.get_side_phrase(t1_side, t1_loc.strip()) if self._is_present(t1_loc) else ""

        if scenario == "no_lesion":
            caption = f"{modality} MRI with no acute {noun} at either timepoint."
        elif scenario == "new_lesion":
            caption = self.clean_string(
                f"{modality} MRI showing interval development of a {noun}",
                f"in the {t1_phrase}" if t1_phrase else "",
                "on follow-up, with no lesion at baseline."
            )
        elif scenario == "resolved":
            caption = self.clean_string(
                f"Previously identified {noun}",
                f"in the {t0_phrase}" if t0_phrase else "",
                "on baseline imaging has resolved on follow-up."
            )
        elif scenario == "stable":
            caption = self.clean_string(
                f"{modality} MRI demonstrating a {noun}",
                f"in the {t0_phrase}" if t0_phrase else "",
                "stable compared to baseline, with no interval change."
            )
        elif scenario == "shrinkage":
            caption = self.clean_string(
                f"Compared to baseline {noun}",
                f"in the {t0_phrase}," if t0_phrase else ",",
                "follow-up demonstrates interval reduction in lesion volume."
            )
        elif scenario == "growth":
            caption = self.clean_string(
                f"Compared to baseline {noun}",
                f"in the {t0_phrase}," if t0_phrase else ",",
                "follow-up demonstrates interval enlargement in lesion volume."
            )
        else:  # new_lesion_with_regression
            caption = self.clean_string(
                f"Compared to baseline {noun}",
                f"in the {t0_phrase}," if t0_phrase else ",",
                "follow-up demonstrates partial regression with a new lesion",
                f"in the {t1_phrase}." if t1_phrase else "elsewhere."
            )

        caption = self.correct_grammer({"caption": caption})["caption"]
        return {
            "caption": caption,
            "path": str(meta.get("Brain1_MRIPath", "None")),
        }

    def generate_caption_from_df(self, df: pd.DataFrame):
        is_paired = "LesionScenario" in df.columns
        caption_df = pd.DataFrame()
        for _, row in df.iterrows():
            if is_paired:
                out = self.generate_pair_caption(row)
            else:
                out = self.generate_caption(row)
            caption_df = pd.concat([caption_df, pd.DataFrame([out])], ignore_index=True)
        return caption_df
