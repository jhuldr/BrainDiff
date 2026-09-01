from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .generate_caption import CaptionGenerator3D
from .positional_generation_prompt import PositionalGenerationPrompt


class CaptionGeneratorPositional(CaptionGenerator3D):
    """
    Single-volume caption generator for the positional dataset.

    Reuses the LLM call/retry, batching, response parsing, validation and CSV I/O
    of CaptionGenerator3D. The single-volume specifics differ:
      - the prompt polishes a per-volume draft rather than comparing T0/T1,
      - resume/de-duplication keys off the single-volume `MRIPath`,
      - add_meta joins single-volume columns plus the positional fields,
      - the LLM-free / fallback path returns the deterministic `simple_caption`,
      - there is NO revision stage — the polish pass is the only LLM step.
    """

    def __init__(self, use_llm=True):
        super().__init__(use_llm=use_llm)
        self.generation_prompt = PositionalGenerationPrompt()

    def _generate_fallback(self, meta: pd.DataFrame) -> dict:
        # Offline / LLM-failure path: use the deterministic template caption.
        return {idx: cap for idx, cap in
                zip(meta.index, meta["simple_caption"].astype(str).tolist())}

    def resume_captioning(self, meta_df, output_csv):
        if not output_csv.exists():
            return meta_df
        output_df = pd.read_csv(output_csv, index_col=False)
        before = len(meta_df)

        meta_df = meta_df[~meta_df["MRIPath"].isin(output_df["mri_path"])]

        dropped = before - len(meta_df)
        print(f"[INFO] Dropped {dropped} rows based on MRIPath overlap")

        return meta_df

    def add_meta(self, target_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
        # Single-volume rows: carry the positional fields so captions.csv is
        # a self-contained record.
        meta_cols = meta_df[[
            'MRIPath',
            'LesionPath',
            'Pathology',
            'Modality',
            'bounding_box',
            'anatomical_region',
            'simple_caption',
        ]]
        result = target_df.join(meta_cols, how='inner')
        result = result.rename(columns={
            'MRIPath': 'mri_path',
            'LesionPath': 'lesion_path',
            'Pathology': 'pathology',
            'Modality': 'modality',
        })

        dropped = len(target_df) - len(result)
        if dropped:
            print(f"Warning: {dropped} caption rows dropped due to missing metadata match. This is likely caused by the LLM. If the number is small. It can be ignored.")

        return result

    def generate_caption(self, meta_df: str, output_path,
                         remove_no_lesion_ratio: float = 0.95,
                         generation_batching: int = 45, revision_batching: int = 75) -> pd.DataFrame:
        """
        Generate polished single-volume captions. Single LLM stage only (no
        revision): the generation pass polishes the template draft straight into
        `captions.csv`.
        """
        meta_df = pd.read_csv(meta_df)
        meta_df = self.resume_captioning(meta_df, Path(output_path) / "captions.csv")

        if not self.use_llm:
            captions = self._generate_fallback(meta_df)
            captions_df = pd.DataFrame.from_dict(captions, orient='index', columns=['caption'])
            captions_df = self.add_meta(captions_df, meta_df)
            self.save_df(captions_df, output_path, "captions.csv")
            return captions_df

        total_chunks_gen = (len(meta_df) + generation_batching - 1) // generation_batching

        caption_chunks = [self.generate_original_caption(chunk)
                          for chunk in tqdm(self.iter_chunks(meta_df, chunk_size=generation_batching),
                                            total=total_chunks_gen, desc="Generating captions")]
        captions_df = pd.concat(caption_chunks)
        captions_df = self.add_meta(captions_df, meta_df)

        self.save_df(captions_df, output_path, "captions.csv")
        return captions_df
