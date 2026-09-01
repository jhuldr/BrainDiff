# EXTERNAL IMPORTS
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf

# INTERNAL IMPORTS
from dataset.CaptionGeneration.generate_caption import CaptionGenerator
from dataset.ExtractData.extract_meta import ExtractMetaData
from dataset.LesionModification.main import PairGen
from dataset.freesurfer_interface import freesurfer_segment
from dataset.dataset_utils import *
from dataset.LesionMapping.map_lesion import generate_usb_aligned_lesions

class Pipeline:

    def __init__(self, use_llm=False):
        self.meta_instance = ExtractMetaData()
        self.gen_instance = PairGen()
        self.use_llm = use_llm
        self.gen_caption = CaptionGenerator(use_llm=use_llm)


    def generate_dataset(self, data_path, seg_save_path, synth_save_path, output_path, lesion_save_path = None, output_csv_name: str = None):

        data_path = Path(data_path)
        seg_save_path = Path(seg_save_path)
        synth_save_path = Path(synth_save_path)
        output_path = Path(output_path)

        mri_files = [
            p for p in data_path.rglob("*.nii*")
            if not (p.name.startswith("x0") or "LESION" in p.name.upper() or "SYNTHSEG" in p.name.upper())
        ]

        brain_pairs = generate_usb_aligned_lesions(data_path, mri_files, seg_save_path, data_path, lesion_save_path)
        pairs_to_process = []
        for brain,lesion in brain_pairs:
            pairs_to_process.extend(self.gen_instance.genPairs(str(synth_save_path), brain, lesion, 5))

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

            brain1_path = data_path / pair_to_process[0]
            brain1_lesion = lesion_save_path / pair_to_process[1]
            brain1_synth = synth_save_path / modify_name(pair_to_process[0])

            brain2_path = data_path / pair_to_process[2]
            brain2_lesion = lesion_save_path / pair_to_process[3]
            brain2_synth = synth_save_path / modify_name(pair_to_process[2])

            meta_df = self.meta_instance.process_brain_pair(brain1_path, brain1_lesion, brain1_synth, brain2_path,
                                                            brain2_lesion, brain2_synth, pair_to_process[-1],
                                                            output_path)
            meta_holder = pd.concat([meta_holder, meta_df])

            #saving the meta dataframe
            if (i + 1) % 10 == 0:
                meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
                meta_holder = pd.DataFrame()

        meta_holder.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)

        self.gen_caption.generate_caption(csv_path, output_path)
        return csv_path


    def test_prompts(self, meta_path):
        self.gen_caption.review_generation_prompt(meta_path)
        self.gen_caption.review_reversion_prompt(30)

if __name__ == "__main__":
    Pipeline(use_llm=False).generate_captions("/home/data/CLIPDATA/TRIAL_IMAGES",
                                            "/home/data/CLIPDATA/TRIAL_IMAGES/SEG",
                                             "trial_run", align_lesion=True,
                                            lesion_save_path="/home/data/CLIPDATA/TRIAL_IMAGES/ALIGNED_LESIONS"
                                            )
    #Pipeline(use_llm=True).generate_captions("/home/data/CLIPDATA/TRIAL_DATASET_ONEDRIVE", "/home/data/CLIPDATA/TRIAL_DATASET_ONEDRIVE_SEG", "/home/data/CLIPDATA/ATLAS_DATASET_PROCESSED_FULL_REV2", align_lesion=False, lesion_save_path="/home/data/CLIPDATA/ATLAS_DATASET_ALIGNED_LESIONS")
    #Pipeline(use_llm=True).generate_captions("/home/data/CLIPDATA/TRIAL_DATASET_ONEDRIVE_TRIMMED", "/home/data/CLIPDATA/TRIAL_DATASET_ONEDRIVE_SEG", "/home/data/CLIPDATA/ATLAS_DATASET_PROCESSED_SUB_10_REV2", align_lesion=True, lesion_save_path="/home/data/CLIPDATA/ATLAS_DATASET_ALIGNED_LESIONS")
    #Pipeline(use_llm=False).test_prompts("/home/data/CLIPDATA/ATLAS_DATASET_PROCESSED/meta_df_20260329_212037.csv")
