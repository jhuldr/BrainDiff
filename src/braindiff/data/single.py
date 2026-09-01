# THIS FILE IS DONE
import os
import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd,
    Rand3DElasticd, RandAffined, RandBiasFieldd, RandGaussianNoised, ToTensord
)
from braindiff.data.neurovfm_transforms import (
    NeuroVFMGridd, NeuroVFMTokenize, N_TOKENS, TOKEN_DIM)
from braindiff.data.report_text import (
    split_report_sentences, reorder_report_sections,
)
from braindiff.training.val_mask import build_token_weights
from transformers import AutoTokenizer
from braindiff.models.paths import decoder_dir
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Decoder resolved through the HF cache (models/paths.py): downloaded once, never again.

# Fixed modality order (matches the model's ["T1w", "T2w", "FLAIR"]).
# Files are named <modality>_time0.nii.gz with lowercase tokens.
MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]

# Per-sentence tokenization budget for the sentence-level contrastive loss.
MAX_SENTENCES = 32
MAX_SENTENCE_LENGTH = 64

# The "Pathologies: <labels>." caption prefix (and the matching request in the
# model's single_timepoint prompt) was removed 2026-08-07. The target is the report
# and nothing else. `strip_pathology_prefix` is deliberately kept in the eval path:
# it is a no-op on prefix-free text, so checkpoints trained with the prefix stay
# comparable to ones trained without it.



class TokenizeCaption:
    """Tokenize the caption as the Qwen3 assistant turn.

The caption ends with the single ChatML terminator <|im_end|>; Qwen3's eos_token_id IS
<|im_end|> (151645), so emitting both would duplicate it. Upstream's trailing newline after
<|im_end|> is dropped -- it trains a token past EOS that generation can never reach.

Also emits a per-sentence tokenization (sentence_input_ids/sentence_attn/sentence_mask) for
the sentence-level contrastive loss.
    """
    def __init__(self, tokenizer, max_length=128,
                 max_sentences=MAX_SENTENCES, max_sentence_length=MAX_SENTENCE_LENGTH,
                 content_weight=1.0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # >1 downweights normality/absence sentences in the captioning CE.
        # 1.0 emits all-ones and is an exact no-op.
        self.content_weight = content_weight
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tokenizer.pad_token_id

    def _tokenize_sentences(self, caption):
        # Segment with syntok, cap at max_sentences.
        sentences = split_report_sentences(caption)[:self.max_sentences]

        ids = torch.full((self.max_sentences, self.max_sentence_length), self.pad_id, dtype=torch.long)
        attn = torch.zeros((self.max_sentences, self.max_sentence_length), dtype=torch.long)
        mask = torch.zeros(self.max_sentences, dtype=torch.bool)

        for i, sent in enumerate(sentences):
            core = self.tokenizer(
                sent, add_special_tokens=False, truncation=True,
                max_length=self.max_sentence_length,
            )['input_ids']
            if not core:
                continue
            ids[i, :len(core)] = torch.tensor(core, dtype=torch.long)
            attn[i, :len(core)] = 1
            mask[i] = True

        return ids, attn, mask

    def __call__(self, data):
        # Reserve 1 slot so <|im_end|> always survives truncation.
        core = self.tokenizer(
            data['caption'],
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - 1,
        )['input_ids']
        ids = core + [self.eot_id]
        attn = [1] * len(ids)
        pad = self.max_length - len(ids)
        if pad > 0:
            ids += [self.pad_id] * pad
            attn += [0] * pad

        data['input_ids'] = torch.tensor(ids, dtype=torch.long)
        data['attention_mask'] = torch.tensor(attn, dtype=torch.long)

        # Computed HERE, per item, not precomputed by dataset index the way the
        # S4 val metric does it (train_dual.py:425). That shortcut is only sound
        # because val captions are unaugmented; train rows go through
        # reorder_report_sections, which moves every token position -- an
        # index-keyed weight vector would be silently wrong on exactly the rows
        # being trained on. Emitted unconditionally so the batch schema is the
        # same on every rank and in every branch.
        if self.content_weight == 1.0:
            data['token_weights'] = torch.ones(self.max_length, dtype=torch.float)
        else:
            data['token_weights'] = build_token_weights(
                data['caption'], self.tokenizer, self.max_length, self.content_weight)

        # Per-sentence tokenization for the sentence-level contrastive loss.
        s_ids, s_attn, s_mask = self._tokenize_sentences(data['caption'])
        data['sentence_input_ids'] = s_ids
        data['sentence_attn'] = s_attn
        data['sentence_mask'] = s_mask
        return data


class MultiModalSingleDataset(Dataset):
    """Multi-modal single-timepoint captioning dataset.

Each study supplies up to 4 modalities at one timepoint, at
input_dir/<batch>/<study_uid>/<modality>_time0.nii.gz.

An absent modality is a zero volume flagged False in the presence mask; studies with no
modality present are dropped. The timepoint is emitted as tokens_main/present_main.
Augmentation is shared across the co-registered modalities.
    """

    def __init__(self, csv_file, img_size, tokenizer,
                 max_caption_length=128, is_train=True, content_weight=1.0):
        self.img_size = img_size
        self.is_train = is_train

        # Modality keys, e.g. "main_t1w"
        self.keys = [f"main_{mod}" for mod in MODALITIES]

        # === BUILD ITEMS (drop incomplete studies) ===
        df = pd.read_csv(csv_file)
        self.items = []
        dropped = 0
        for _, row in df.iterrows():
            caption = row.get("report")
            if pd.isna(caption) or str(caption).strip() == "":
                dropped += 1
                continue

            paths, present = {}, []
            for mod in MODALITIES:
                path = row.get(mod)
                if pd.isna(path) or not os.path.exists(path):
                    present.append(False)
                else:                    
                    present.append(True)
                    paths[f"main_{mod}"] = path

            # Need >=1 modality present
            if not any(present):
                dropped += 1
                continue

            self.items.append({
                "paths": paths,
                "present_main": present,
                "caption": str(caption),
                # Swap-partner group for the counterfactual loss. Stage 2 has no
                # `classification`; pathology is the analogue -- substituting a
                # scan with a different pathology should make the report less
                # likely, whereas swapping two normals teaches nothing.
                "group": str(row.get("pathology", "unknown")),
            })
        print(f"Curriculum Step 2: kept {len(self.items)}, dropped {dropped}")

        # Sorted so the mapping is identical on every rank without communication.
        self.group_to_idx = {g: i for i, g in
                             enumerate(sorted({it["group"] for it in self.items}))}

        # === TRANSFORMS (allow_missing_keys -> tolerate absent modalities) ===
        # Grid first, augment, tokenize last: augmentation needs a spatial volume.
        # Augmentation is kept here, unlike S1 -- S2 is report generation with no
        # bounding boxes to invalidate, and it is the stage the S2 gate is judged on.
        self.load = Compose([NeuroVFMGridd(keys=self.keys, allow_missing_keys=True)])
        self.to_tokens = NeuroVFMTokenize(keys=self.keys, allow_missing_keys=True)
        # One augment over the modality group -> same random params across them
        self.aug = self._make_augment(self.keys) if is_train else None

        self.tokenize = TokenizeCaption(tokenizer, max_caption_length,
                                       content_weight=content_weight)

    def _make_augment(self, keys):
        """One timepoint, so spatial and intensity can share a Compose -- the
        modalities are co-registered and get identical geometry. Deformation is
        mild on purpose: the reports describe lesions with a median size of 8 mm."""
        return Compose([
            RandAffined(keys=keys, prob=0.3, rotate_range=(0.05, 0.05, 0.05),
                        scale_range=(0.05, 0.05, 0.05), mode="bilinear",
                        allow_missing_keys=True),
            Rand3DElasticd(keys=keys, sigma_range=(5, 8), magnitude_range=(50, 100),
                           prob=0.15, mode="bilinear", allow_missing_keys=True),
            RandBiasFieldd(keys=keys, prob=0.3, allow_missing_keys=True),
            RandGaussianNoised(keys=keys, prob=0.2, allow_missing_keys=True),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load only the present modalities, then augment the group
        data = self.load({k: v for k, v in item["paths"].items()})
        if self.is_train:
            data = self.aug(data)

        data = self.to_tokens(data)
        tokens = torch.zeros(len(self.keys), N_TOKENS, TOKEN_DIM, dtype=torch.float16)
        coords = torch.zeros(len(self.keys), N_TOKENS, 3, dtype=torch.int16)
        present = torch.zeros(len(self.keys), dtype=torch.bool)
        for i, key in enumerate(self.keys):
            entry = data.get(key)
            if entry is None or entry["tokens"].shape[0] != N_TOKENS:
                continue
            tokens[i], coords[i], present[i] = entry["tokens"], entry["coords"], True

        # Sentences reshuffled within their sections on training rows only.
        report = (reorder_report_sections(item["caption"])
                  if self.is_train else item["caption"])

        sample = {
            "tokens_main": tokens,
            "coords_main": coords,
            "present_main": present,
            "caption": report,
            "group": self.group_to_idx[item["group"]],
        }
        sample = self.tokenize(sample)
        return sample


def get_diff_caption_dataset(
    csv_file,
    img_size,
    num_workers,
    tokenizer=None,
    max_caption_length=128,
    is_train=True,
    content_weight=1.0,
):
    """
    Multi-modal single-timepoint captioning dataset.

    Expected CSV columns: study_uid, batch, edited_report.
    Images: input_dir/<batch>/<study_uid>/<modality>_time0.nii.gz
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(decoder_dir())

    return MultiModalSingleDataset(
        csv_file=csv_file,
        img_size=img_size,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        is_train=is_train,
        content_weight=content_weight,
    )


def get_diff_caption_dataloader(
    csv_file,
    img_size,
    batch_size,
    num_workers,
    tokenizer=None,
    max_caption_length=128,
    is_train=True,
    distributed: bool = False,
    content_weight=1.0,
):
    """Returns the DataLoader. Under DDP a DistributedSampler shards the data;
    access it via loader.sampler for set_epoch()."""

    ds = get_diff_caption_dataset(
        csv_file=csv_file,
        img_size=img_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        is_train=is_train,
        content_weight=content_weight,
    )

    # Under DDP, shard across ranks instead of shuffling locally.
    if distributed:
        sampler = DistributedSampler(ds, shuffle=is_train)
    else:
        sampler = None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
