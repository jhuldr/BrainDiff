# EXTERNAL IMPORTS
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime
#from monai.transforms import Compose, Resize, Resized, SaveImage, LoadImage
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.affines import rescale_affine
    

# INTERNAL IMPORTS — 3D-specific stages
from dataset3D.CaptionGeneration.generate_caption import CaptionGenerator3D
from dataset3D.ExtractData.extract_meta import ExtractMetaData3D
from dataset3D.ExtractData.extract_meta_positional import ExtractMetaDataPositional
from dataset3D.CaptionGeneration.generate_caption_positional import CaptionGeneratorPositional
from dataset3D.ExtractData.extract_meta_longitudinal_positional import ExtractMetaDataLongitudinalPositional
from dataset3D.dataset_adapter import *

# INTERNAL IMPORTS — shared stages reused unchanged from the 2D `dataset` package
from dataset.LesionModification.main import PairGen
from dataset.freesurfer_interface import freesurfer_segment
from dataset.dataset_utils import *
from dataset.LesionMapping.map_lesion import generate_usb_aligned_lesions
from dataset.ants_interface import align_image_to_mni
from dataset.DatasetIntake.ucfs import intake_ucfs


MODALITY_COLUMN = {"T1W": "T1w", "T2W": "T2w", "T1CE": "T1ce", "FLAIR": "FLAIR"}


def resize_nifti_in_place(path, spatial_size=(96, 96, 96), mode="trilinear"):
    """Resize a NIfTI to spatial_size with MONAI and overwrite it in place.

    Masks should pass mode="nearest" to keep labels intact.
    """
    path = Path(path)
    if not path.exists():
        return

    # Load image
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)

    original_shape = data.shape[:3]

    # Convert to tensor: (N, C, D, H, W)
    tensor = (
        torch.from_numpy(data)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    if mode == "nearest":
        resized = F.interpolate(
            tensor,
            size=spatial_size,
            mode="nearest",
        )
    else:
        resized = F.interpolate(
            tensor,
            size=spatial_size,
            mode="trilinear",
            align_corners=False,
        )

    resized = resized.squeeze().cpu().numpy()

    # Compute the new affine
    new_affine = rescale_affine(
        nii.affine,
        original_shape,
        nii.header.get_zooms()[:3],
        spatial_size,
    )

    # Preserve header
    header = nii.header.copy()
    header.set_data_shape(spatial_size)

    out = nib.Nifti1Image(resized, new_affine, header)
    nib.save(out, str(path))

class Pipeline3D:
    """
    Whole-volume (3D) variant of dataset.main.Pipeline.

    The registration, synthetic-pair generation, and segmentation stages are
    reused verbatim from the 2D `dataset` package. Only metadata extraction and
    caption generation are swapped for their 3D counterparts, and every run is
    tagged with a single `pathology` type (e.g. "stroke", "glioma").
    """

    def __init__(self, use_llm=False):
        self.meta_instance = ExtractMetaData3D()
        self.gen_instance = PairGen()
        self.use_llm = use_llm
        self.gen_caption = CaptionGenerator3D(use_llm=use_llm)

        # Positional (single-volume) collaborators — see generate_positional_dataset.
        self.pos_meta_instance = ExtractMetaDataPositional()
        self.pos_gen_caption = CaptionGeneratorPositional(use_llm=use_llm)

        # Longitudinal positional (two-timepoint) collaborator — see
        # generate_longitudinal_positional_dataset.
        self.pos_meta_instance_longitudinal = ExtractMetaDataLongitudinalPositional()

    def generate_dataset(self, data_path, aligned_data_path, seg_save_path, 
                         synth_save_path, output_path, pathology,
                         lesion_save_path=None, output_csv_name: str = None):

        data_path = Path(data_path)
        seg_save_path = Path(seg_save_path)
        synth_save_path = Path(synth_save_path)
        output_path = Path(output_path)
        # this will become the data_path shortly
        aligned_data_path = Path(aligned_data_path)

        # Create directories if they don't exist
        seg_save_path.mkdir(parents=True, exist_ok=True)
        synth_save_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)
        aligned_data_path.mkdir(parents=True, exist_ok=True)

        mri_files = [
            p for p in data_path.glob("*.nii*")
            if not (p.name.startswith("x0") or "LESION" in p.name.upper() or "SYNTHSEG" in p.name.upper())
        ]

        # align everything (image + lesion) to nihpd space
        for file in tqdm(mri_files, desc="Aligning MRI files", unit="file"):
            lesion_name = modify_name(file.name, "lesion")
            align_image_to_mni(
                file,
                aligned_data_path / file.name,
                lesion_path=data_path / lesion_name,
                lesion_output_path=aligned_data_path / lesion_name,
            )
        data_path = aligned_data_path
        # end alignment

        brain_pairs = generate_usb_aligned_lesions(data_path, mri_files, data_path, lesion_save_path)
        pairs_to_process = []
        for brain, lesion in tqdm(brain_pairs, desc="Generating pairs", unit="pair"):
            pairs_to_process.extend(
                self.gen_instance.genPairs(
                    str(synth_save_path),
                    brain,
                    lesion,
                    3,
                    (0.1, 0.3),
                )
            )

        # segment everything at the same time, save at the same location
        freesurfer_segment(data_path, seg_save_path)
        freesurfer_segment(synth_save_path, seg_save_path)

        meta_holder = pd.DataFrame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_csv_name is None:
            csv_path = output_path / f"meta_df_{timestamp}.csv"
        else:
            csv_path = output_path / output_csv_name

    
        for i, pair_to_process in enumerate(tqdm(pairs_to_process, desc="Processing brains")):
            brain1_path = Path(data_path) / pair_to_process[0]
            brain1_lesion = Path(lesion_save_path) / pair_to_process[1]
            brain1_seg = Path(seg_save_path) / modify_name(Path(pair_to_process[0]).name)

            brain2_path = Path(data_path) / pair_to_process[2]
            brain2_lesion = Path(lesion_save_path) / pair_to_process[3]
            brain2_seg = Path(seg_save_path) / modify_name(Path(pair_to_process[2]).name)

            # One row per pair (whole-volume). Pathology is recorded for every row.
            meta_df = self.meta_instance.process_brain_pair(
                brain1_path, brain1_lesion, brain1_seg,
                brain2_path, brain2_lesion, brain2_seg,
                output_path, pathology=pathology
            )
            meta_holder = pd.concat([meta_holder, meta_df])

            # checkpoint the meta dataframe periodically
            if (i + 1) % 10 == 0:
                meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                meta_holder = pd.DataFrame()

        meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)

        self.gen_caption.generate_caption(csv_path, output_path)
        return csv_path

    def generate_positional_dataset(self, data_path, aligned_data_path, seg_save_path,
                                    output_path, pathology, modality: str = "T1w",
                                    output_csv_name: str = "boxes.csv",
                                    extract_modality_fn=None, find_lesion_fn=None):
        """
        Single-volume positional dataset: align -> segment -> extract -> caption.

        Mirrors generate_dataset but is NOT paired — no synthetic pair generation
        (PairGen). `data_path` may hold either flat single-modality files (each
        tagged as `modality`) or per-study subfolders containing several modality
        files (T1w/T1ce/T2w/FLAIR) plus a lesion mask. Every image is aligned to
        MNI; only one representative MRI per study (T1w preferred) is segmented.
        Aligned output is always written flat into aligned_data_path — study-folder
        outputs are prefixed with the folder name to avoid collisions. Each study
        yields one row of positional metadata (bounding_box, anatomical_region,
        per-modality paths) via ExtractMetaDataPositional. `pathology` is recorded
        per row.

        `extract_modality_fn`, if given, takes a file Path from a study folder and
        returns its modality column name (one of "T1w"/"T1ce"/"T2w"/"FLAIR").
        Defaults to `extract_modality` + MODALITY_COLUMN.

        `find_lesion_fn`, if given, takes a study folder Path and returns the
        lesion file Path, or None if the study has no lesion. Defaults to
        `folder / "lesion.nii.gz"` if present, else None.
        """
        def default_extract_modality(f):
            return MODALITY_COLUMN.get(self.pos_meta_instance.extract_modality(f), "T1w")

        def default_find_lesion(folder):
            p = folder / "lesion.nii.gz"
            return p if p.is_file() else None

        extract_modality_fn = extract_modality_fn or default_extract_modality
        find_lesion_fn = find_lesion_fn or default_find_lesion

        data_path = Path(data_path)
        aligned_data_path = Path(aligned_data_path)
        seg_save_path = Path(seg_save_path)
        output_path = Path(output_path)

        # Create directories if they don't exist
        seg_save_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)
        aligned_data_path.mkdir(parents=True, exist_ok=True)

        def is_wanted(p):
            return not (p.name.startswith("x0") or "LESION" in p.name.upper() or "SYNTHSEG" in p.name.upper())

        entries = list(data_path.iterdir())
        flat_files = sorted(
            p for p in entries
            if p.is_file() and is_wanted(p) and p.name.endswith((".nii", ".nii.gz"))
        )
        study_folders = sorted(p for p in entries if p.is_dir())

        studies = []  # each: {"seg_source": Path, "modality_paths": dict, "lesion_path": Path|None}

        # --- flat single-modality files: unchanged alignment, tagged with `modality` ---
        for file in tqdm(flat_files, desc="Aligning flat MRI files", unit="file"):
            lesion_name = modify_name(file.name, "lesion")
            lesion_src = data_path / lesion_name
            aligned_img = aligned_data_path / file.name
            aligned_lesion = aligned_data_path / lesion_name if lesion_src.is_file() else None
            align_image_to_mni(
                file, aligned_img,
                lesion_path=lesion_src if lesion_src.is_file() else None,
                lesion_output_path=aligned_lesion,
            )
            # No more resize nifti, do it in the dataloader and precalulate modified coordinates using a different script.
            #resize_nifti_in_place(aligned_img)
            #if aligned_lesion is not None:
            #    resize_nifti_in_place(aligned_lesion, mode="nearest")
            studies.append({
                "seg_source": aligned_img,
                "modality_paths": {modality: aligned_img},
                "lesion_path": aligned_lesion,
            })

        # --- study folders: multiple modalities + optional lesion.nii.gz, flattened output ---
        for folder in tqdm(study_folders, desc="Aligning study folders", unit="folder"):
            lesion_src = find_lesion_fn(folder)
            candidate_files = [
                p for p in folder.iterdir()
                if p.is_file() and is_wanted(p) and p != lesion_src
            ]
            col_to_src = {}
            for f in candidate_files:
                col = extract_modality_fn(f)
                col_to_src[col] = f
            if not col_to_src:
                continue

            preferred_col = "T1w" if "T1w" in col_to_src else next(iter(col_to_src))
            aligned_lesion = aligned_data_path / f"{folder.name}_lesion.nii.gz" if lesion_src is not None else None

            modality_paths = {}
            for col, src in col_to_src.items():
                aligned_img = aligned_data_path / f"{folder.name}_{col}.nii.gz"
                if col == preferred_col:
                    align_image_to_mni(
                        src, aligned_img,
                        lesion_path=lesion_src,
                        lesion_output_path=aligned_lesion,
                    )
                else:
                    align_image_to_mni(src, aligned_img)
                #resize_nifti_in_place(aligned_img)
                modality_paths[col] = aligned_img
            #if aligned_lesion is not None:
            #    resize_nifti_in_place(aligned_lesion, mode="nearest")

            studies.append({
                "seg_source": modality_paths[preferred_col],
                "modality_paths": modality_paths,
                "lesion_path": aligned_lesion,
            })

        # --- segment only one representative MRI per study ---
        seg_staging = aligned_data_path / "_seg_staging"
        seg_staging.mkdir(parents=True, exist_ok=True)
        for s in studies:
            shutil.copy2(s["seg_source"], seg_staging / s["seg_source"].name)
        freesurfer_segment(seg_staging, seg_save_path)

        meta_holder = pd.DataFrame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_csv_name is None:
            csv_path = output_path / f"meta_df_{timestamp}.csv"
        else:
            csv_path = output_path / output_csv_name

        for i, s in enumerate(tqdm(studies, desc="Processing brains")):
            seg_path = seg_save_path / modify_name(s["seg_source"].name)

            # One row per study (whole-volume positional metadata).
            meta_df = self.pos_meta_instance.process_single_brain(
                s["modality_paths"], s["lesion_path"], seg_path, output_path, pathology
            )
            meta_holder = pd.concat([meta_holder, meta_df])

            # checkpoint the meta dataframe periodically
            if (i + 1) % 10 == 0:
                meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                meta_holder = pd.DataFrame()

        meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        return csv_path

    def generate_longitudinal_positional_dataset(self, csv_file, image_csv, aligned_data_path,
                                                  seg_save_path, output_path, pathology,
                                                  output_csv_name: str = "longitudinal_boxes.csv"):
        """
        Longitudinal (two-timepoint) positional dataset: align -> segment ->
        extract matched-lesion boxes -> caption.

        Driven by two CSVs (mirrors dataloaders/MultiModal/multi_dual_dataloader.py):
        `csv_file` has one row per study pair with `study_uid1` (ref) / `study_uid2`
        (main) columns; `image_csv` is indexed by `study_uid` with per-modality
        (T1w/T1ce/T2w/FLAIR) path columns. Lesion mask and segmentation paths are
        derived per study as sibling files of its preferred modality via
        `modify_name` (e.g. t1w.nii.gz -> t1w_lesion.nii.gz / t1w_synthseg.nii.gz),
        same convention as generate_positional_dataset. Every study referenced by
        `csv_file` is aligned to MNI and segmented once (even if it appears as
        `main` in one pair and `ref` in another); each pair then yields one row
        per matched lesion (or lesion-free region) via
        ExtractMetaDataLongitudinalPositional.process_brain_pair_positional.
        `pathology` is recorded per row.
        """
        aligned_data_path = Path(aligned_data_path)
        seg_save_path = Path(seg_save_path)
        output_path = Path(output_path)

        aligned_data_path.mkdir(parents=True, exist_ok=True)
        seg_save_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        pairs_df = pd.read_csv(csv_file)
        image_df = pd.read_csv(image_csv).set_index("study_uid")

        study_uids = pd.unique(pairs_df[["study_uid1", "study_uid2"]].values.ravel())

        studies = {}  # study_uid -> {"seg_source", "modality_paths", "lesion_path"}
        for study_uid in tqdm(study_uids, desc="Aligning studies", unit="study"):
            if study_uid not in image_df.index:
                continue
            image_row = image_df.loc[study_uid]

            col_to_src = {}
            for col in ("T1w", "T1ce", "T2w", "FLAIR"):
                path = image_row.get(col)
                if pd.notna(path) and os.path.exists(path):
                    col_to_src[col] = Path(path)
            if not col_to_src:
                continue

            preferred_col = "T1w" if "T1w" in col_to_src else next(iter(col_to_src))
            preferred_src = col_to_src[preferred_col]
            lesion_name = modify_name(preferred_src.name, "lesion")
            lesion_src = preferred_src.parent / lesion_name
            lesion_src = lesion_src if lesion_src.is_file() else None
            aligned_lesion = aligned_data_path / f"{study_uid}_lesion.nii.gz" if lesion_src is not None else None

            modality_paths = {}
            for col, src in col_to_src.items():
                aligned_img = aligned_data_path / f"{study_uid}_{col}.nii.gz"
                if col == preferred_col:
                    align_image_to_mni(
                        src, aligned_img,
                        lesion_path=lesion_src,
                        lesion_output_path=aligned_lesion,
                    )
                else:
                    align_image_to_mni(src, aligned_img)
                modality_paths[col] = aligned_img

            studies[study_uid] = {
                "seg_source": modality_paths[preferred_col],
                "modality_paths": modality_paths,
                "lesion_path": aligned_lesion,
            }

        # --- segment every referenced study once, in one batched call ---
        seg_staging = aligned_data_path / "_seg_staging"
        seg_staging.mkdir(parents=True, exist_ok=True)
        for s in studies.values():
            shutil.copy2(s["seg_source"], seg_staging / s["seg_source"].name)
        freesurfer_segment(seg_staging, seg_save_path)

        seg_paths = {
            study_uid: seg_save_path / modify_name(s["seg_source"].name)
            for study_uid, s in studies.items()
        }

        meta_holder = pd.DataFrame()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_csv_name is None:
            csv_path = output_path / f"meta_df_{timestamp}.csv"
        else:
            csv_path = output_path / output_csv_name

        for i, (_, pair_row) in enumerate(tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Processing pairs")):
            uid_ref, uid_main = pair_row["study_uid1"], pair_row["study_uid2"]
            if uid_ref not in studies or uid_main not in studies:
                continue

            meta_df = self.pos_meta_instance_longitudinal.process_brain_pair_positional(
                studies[uid_ref]["modality_paths"], studies[uid_main]["modality_paths"],
                studies[uid_ref]["lesion_path"], studies[uid_main]["lesion_path"],
                seg_paths[uid_ref], seg_paths[uid_main], output_path, pathology,
            )
            meta_df["StudyUID_ref"] = uid_ref
            meta_df["StudyUID_main"] = uid_main
            meta_holder = pd.concat([meta_holder, meta_df])

            # checkpoint the meta dataframe periodically
            if (i + 1) % 10 == 0:
                meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                meta_holder = pd.DataFrame()

        meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        return csv_path

    def debug(self):
        print("Debugging...")
        self.gen_caption.generate_caption("/path/to/code/BrainDiff/TRIAL_IMAGES/OUTPUT/meta_df_20260611_230939.csv", "/path/to/code/BrainDiff/TRIAL_IMAGES/OUTPUT")


    def test_prompts(self, meta_path):
        meta_df = pd.read_csv(meta_path)
        self.gen_caption.review_generation_prompt(meta_df)
        self.gen_caption.review_reversion_prompt(30)
       

if __name__ == "__main__":
    
    pipeline_configs = {
        "atlas_intake": {
            "method": "generate_dataset",
            "use_llm": True,
            "kwargs": {
                "data_path": "/home/data/CLIPDATA/TRIAL_DATASET_ONEDRIVE",
                "aligned_data_path": "/home/data/CLIPDATA/BRAIN_DIFF/ALIGNED",
                "seg_save_path": "/home/data/CLIPDATA/BRAIN_DIFF/SEG",
                "synth_save_path": "/home/data/CLIPDATA/BRAIN_DIFF/SYNTH_3D",
                "output_path": "/home/data/CLIPDATA/BRAIN_DIFF/OUTPUT",
                "pathology": "stroke",
                "lesion_save_path": "/home/data/CLIPDATA/BRAIN_DIFF/SYNTH_3D/ALIGNED_LESIONS",
            },
        },

        "atlas_positional": {
            "method": "generate_positional_dataset",
            "use_llm": False,
            "kwargs": {
                "data_path": "/home/data/CLIPDATA/TRIAL_DATASET_ONEDRIVE",
                "aligned_data_path": "/home/data/BRAIN_DIFF/ATLAS/ALIGNED",
                "seg_save_path": "/home/data/BRAIN_DIFF/ATLAS/SEG",
                "output_path": "/home/data/BRAIN_DIFF/ATLAS/OUTPUT",
                "pathology": "stroke",
                "modality": "T1w",
            },
        },

        "brats_men": {
            "method": "generate_positional_dataset",
            "use_llm": False,
            "kwargs": {
                "data_path": "/home/data/BraTS2023/BraTS-MEN-Train",
                "aligned_data_path": "/home/data/BRAIN_DIFF/BRATS-MEN/ALIGNED",
                "seg_save_path": "/home/data/BRAIN_DIFF/BRATS-MEN/SEG",
                "output_path": "/home/data/BRAIN_DIFF/BRATS-MEN/OUTPUT",
                "pathology": "meningioma",
                "extract_modality_fn": extract_modality_brats_men,
                "find_lesion_fn": find_seg_brats_men,
            },
        },

        "isles_22": {
            "method": "generate_positional_dataset",
            "use_llm": False,
            "kwargs": {
                "data_path": "/home/data/ISLES-2022-2/processed_data",
                "aligned_data_path": "/home/data/BRAIN_DIFF/ISLES-22/ALIGNED",
                "seg_save_path": "/home/data/BRAIN_DIFF/ISLES-22/SEG",
                "output_path": "/home/data/BRAIN_DIFF/ISLES-22/OUTPUT",
                "pathology": "stroke",
                "modality": "FLAIR",
            },
        },

        "brats_met": {
            "method": "generate_positional_dataset",
            "use_llm": False,
            "kwargs": {
                "data_path": "/home/data/BraTS-MET/single-timepoint",
                "aligned_data_path": "/home/data/BRAIN_DIFF/BRATS-MET/ALIGNED",
                "seg_save_path": "/home/data/BRAIN_DIFF/BRATS-MET/SEG",
                "output_path": "/home/data/BRAIN_DIFF/BRATS-MET/OUTPUT",
                "pathology": "metastases",
                "extract_modality_fn": extract_modality_brats_men,
                "find_lesion_fn": find_seg_brats_men,
            },
        },

        "mrrate_longitudinal": {
            "method": "generate_longitudinal_positional_dataset",
            "use_llm": False,
            "kwargs": {
                "csv_file": "/path/to/code/BrainDiff/process_mrrate/data/longitudional_meta_reports.csv",
                "image_csv": "/home/data/BRAIN_DIFF_S4/image.csv",
                "aligned_data_path": "/home/data/BRAIN_DIFF/MR-RATE-LONGITUDINAL/ALIGNED",
                "seg_save_path": "/home/data/BRAIN_DIFF/MR-RATE-LONGITUDINAL/SEG",
                "output_path": "/home/data/BRAIN_DIFF/MR-RATE-LONGITUDINAL/OUTPUT",
                "pathology": "stroke",
            },
        },
    }

    selected_config = "brats_met"

    config = pipeline_configs[selected_config]

    pipeline = Pipeline3D(use_llm=config["use_llm"])
    method = getattr(pipeline, config["method"])
    method(**config["kwargs"])
