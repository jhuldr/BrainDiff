# TODO FIX THIS DATALOADER
import os
import random
from collections import Counter, defaultdict
import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd,
    Rand3DElasticd, RandAffined, RandBiasFieldd, RandGaussianNoised, ToTensord
)
from braindiff.data.neurovfm_transforms import (
    NeuroVFMGridd, NeuroVFMTokenize, N_TOKENS, TOKEN_DIM)
from transformers import AutoTokenizer
from braindiff.models.paths import decoder_dir
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

# Decoder resolved through the HF cache (models/paths.py): downloaded once, never again.

# Fixed modality order (matches the model's ["T1w", "T2w", "FLAIR"]).
# Files are named <modality>_time<t>.nii.gz with lowercase tokens.
MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]
TIMEPOINTS = {"ref": 1, "main": 2}

# Per-sentence tokenization budget for the sentence-level contrastive loss.
MAX_SENTENCES = 32
MAX_SENTENCE_LENGTH = 64

# Fixed order for the auxiliary change head and the counterfactual swap. Index
# order is the S4 frequency order; the head's class weights are derived from
# counts at build time, so this only has to stay stable within a run.
CHANGE_CLASSES = (
    "Stable", "New lesion", "Indeterminate", "Progressed",
    "Improved", "Mixed interval change", "Resolved",
)
CHANGE_CLASS_TO_IDX = {name: i for i, name in enumerate(CHANGE_CLASSES)}


class BalancedBatchSampler(Sampler):
    """Yields random batches stratified evenly across `classification` values.

Each batch draws an equal share from every bucket, remainder to random buckets, so the
marginal stays near-even regardless of batch_size. Buckets smaller than their share are
reshuffled and reused within the epoch.

Under DDP every rank builds `groups` from the same full dataset, and each bucket is
truncated to `len(idxs) // num_replicas` before striding, so every rank gets the same batch
count with no communication. This drops at most (num_replicas - 1) samples per bucket.
    """

    def __init__(self, classifications, batch_size, num_replicas=1, rank=0, seed=0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        groups = defaultdict(list)
        for idx, cls in enumerate(classifications):
            groups[cls].append(idx)

        # Truncate each bucket to an equal, rank-independent size *before*
        # striding, so every rank computes the same shard length with zero
        # inter-process communication.
        self.groups = {}
        for key, idxs in groups.items():
            shard_size = len(idxs) // num_replicas
            if shard_size == 0:
                continue
            self.groups[key] = idxs[rank::num_replicas][:shard_size]

        self.keys = list(self.groups.keys())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return total // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {k: v[:] for k, v in self.groups.items()}
        for pool in pools.values():
            rng.shuffle(pool)
        cursors = {k: 0 for k in self.keys}

        n_keys = len(self.keys)
        base = self.batch_size // n_keys
        remainder = self.batch_size % n_keys

        for _ in range(len(self)):
            counts = {k: base for k in self.keys}
            for k in rng.sample(self.keys, remainder):
                counts[k] += 1

            batch = []
            for k in self.keys:
                pool = pools[k]
                for _ in range(counts[k]):
                    if cursors[k] >= len(pool):
                        rng.shuffle(pool)
                        cursors[k] = 0
                    batch.append(pool[cursors[k]])
                    cursors[k] += 1
            rng.shuffle(batch)
            yield batch


from braindiff.data.report_text import (
    split_report_sentences, reorder_report_sections,
)
from braindiff.training.val_mask import build_token_weights


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
        self.max_sentences = max_sentences
        self.content_weight = content_weight
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

        # Computed HERE, per item, exactly as in single.py -- and it MUST be here
        # rather than precomputed by dataset index, because training rows have
        # already been through reorder_report_sections in __getitem__ (it runs
        # before this call), which moves every token position. An index-keyed
        # weight vector would be silently wrong on exactly the rows being trained
        # on. Emitted unconditionally so the batch schema matches on every rank.
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


class MultiModalDiffDataset(Dataset):
    """Multi-modal longitudinal difference-captioning dataset.

Each study supplies up to 3 modalities (T1w/T2w/FLAIR) at two timepoints (time0 = ref,
time1 = main), at input_dir/<batch>/<study_uid>/<modality>_time<t>.nii.gz.

An absent modality is a zero volume flagged False in the presence mask; studies without at
least one modality present at BOTH timepoints are dropped. One random transform is shared
across the whole pair, keeping ref and main co-registered.
    """

    def __init__(self, csv_file, image_csv, img_size, tokenizer,
                 max_caption_length=128, is_train=True, content_weight=1.0):

        self.img_size = img_size
        self.is_train = is_train

        # Build the per-timepoint key lists, e.g. "ref_t1w", "main_flair"
        self.keys = {
            tp: [f"{tp}_{mod}" for mod in MODALITIES] for tp in TIMEPOINTS
        }
        self.all_keys = self.keys["ref"] + self.keys["main"]
        all_keys = self.all_keys

        # === BUILD ITEMS (drop incomplete studies) ===
        df = pd.read_csv(csv_file)
        image_df = pd.read_csv(image_csv)
        image_df = image_df.set_index("study_uid")
        self.items = []
        dropped = 0

        for _, row in df.iterrows():
            caption = row.get("generated_report")
            if pd.isna(caption) or str(caption).strip() == "":
                dropped += 1
                continue

            classification = row.get("classification")
            if pd.isna(classification) or str(classification).strip() == "":
                dropped += 1
                continue

            paths, present = {}, {tp: [] for tp in TIMEPOINTS}
            for tp, t in TIMEPOINTS.items():
                if row[f"study_uid{t}"] in image_df.index:
                    image_row = image_df.loc[row[f"study_uid{t}"]]
                else:
                    break
                for mod in MODALITIES:
                    path = image_row.get(mod)
                    if pd.isna(path) or not os.path.exists(path):
                        present[tp].append(False)
                    else:                    
                        present[tp].append(True)
                        paths[f"{tp}_{mod}"] = path

            # Need >=1 modality present at both timepoints to form a pair
            if not (any(present["ref"]) and any(present["main"])):
                dropped += 1
                continue

            self.items.append({
                "paths": paths,
                "present_ref": present["ref"],
                "present_main": present["main"],
                "caption": str(caption),
                "classification": str(classification),
            })
        print(f"Step 4 Curriculum: kept {len(self.items)}, dropped {dropped}")

        # Validate here rather than in __getitem__: an unrecognised class there is a
        # KeyError raised inside a dataloader worker, with no indication of which
        # value or how many rows are affected. Mapping the stray value to a fallback
        # class would be worse -- it silently corrupts both the change head's targets
        # and the counterfactual partner selection.
        unknown = Counter(it["classification"] for it in self.items
                          if it["classification"] not in CHANGE_CLASS_TO_IDX)
        if unknown:
            raise ValueError(
                f"{sum(unknown.values())} rows carry a `classification` outside "
                f"CHANGE_CLASSES: {dict(unknown)}. Add them to CHANGE_CLASSES in "
                f"{__name__} (order is stable within a run) or clean the CSV."
            )

        # === TRANSFORMS (allow_missing_keys -> tolerate absent modalities) ===
        # Grid first, augment, tokenize last: augmentation needs a spatial volume,
        # and warping a flat token list is not the same operation.
        self.load = Compose([NeuroVFMGridd(keys=all_keys, allow_missing_keys=True)])
        self.to_tokens = NeuroVFMTokenize(keys=all_keys, allow_missing_keys=True)
        # Spatial augmentation is shared across all 8 keys, so both timepoints
        # move together and stay in correspondence -- the delta is a token-wise
        # difference and desynchronising them would make it meaningless.
        # Intensity augmentation is applied per timepoint instead, so the two
        # scans get independent bias fields and noise. Sharing it would leave a
        # matching intensity signature on both, which is a shortcut the model can
        # use to pair them without looking at anatomy.
        self.spatial_aug = self._make_spatial_augment(self.all_keys) if is_train else None
        self.intensity_aug = {
            tp: self._make_intensity_augment(self.keys[tp]) for tp in TIMEPOINTS
        } if is_train else None

        self.tokenize = TokenizeCaption(tokenizer, max_caption_length,
                                        content_weight=content_weight)

    def _make_spatial_augment(self, keys):
        """Geometry, applied identically to every key in one call so ref and main
        stay aligned. Deformation is kept mild: the targets describe lesions with
        a median size of 8 mm, and a large warp would move a finding further than
        the finding itself measures."""
        return Compose([
            RandAffined(keys=keys, prob=0.3, rotate_range=(0.05, 0.05, 0.05),
                        scale_range=(0.05, 0.05, 0.05), mode="bilinear",
                        allow_missing_keys=True),
            Rand3DElasticd(keys=keys, sigma_range=(5, 8), magnitude_range=(50, 100),
                           prob=0.15, mode="bilinear", allow_missing_keys=True),
        ])

    def _make_intensity_augment(self, keys):
        """Appearance, randomized separately per timepoint."""
        return Compose([
            RandBiasFieldd(keys=keys, prob=0.3, allow_missing_keys=True),
            RandGaussianNoised(keys=keys, prob=0.2, allow_missing_keys=True),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # Load only the present modalities, then warp the pair together and vary
        # each timepoint's appearance independently.
        data = self.load({k: v for k, v in item["paths"].items()})
        if self.is_train:
            data = self.spatial_aug(data)
            for tp in TIMEPOINTS:
                data = self.intensity_aug[tp](data)

        data = self.to_tokens(data)

        # Per timepoint: [M, N, 1024] tokens + [M, N, 3] coords + [M] presence.
        # Absent modalities stay all-zero and are masked out of the connector's
        # attention rather than contributing a block of zeros to a fusion MLP.
        sample = {}
        for tp in TIMEPOINTS:
            tokens = torch.zeros(len(MODALITIES), N_TOKENS, TOKEN_DIM, dtype=torch.float16)
            coords = torch.zeros(len(MODALITIES), N_TOKENS, 3, dtype=torch.int16)
            present = torch.zeros(len(MODALITIES), dtype=torch.bool)
            for i, key in enumerate(self.keys[tp]):
                entry = data.get(key)
                if entry is None or entry["tokens"].shape[0] != N_TOKENS:
                    continue
                tokens[i], coords[i], present[i] = entry["tokens"], entry["coords"], True
            sample[f"tokens_{tp}"] = tokens
            sample[f"coords_{tp}"] = coords
            sample[f"present_{tp}"] = present

        # Change class: supervises the auxiliary head, and lets the trainer pick a
        # counterfactual partner whose progression differs from this sample's.
        sample["change_label"] = CHANGE_CLASS_TO_IDX[item["classification"]]
        # Dataset position, so the trainer can look up this sample's precomputed
        # content-token mask after sharding/shuffling.
        sample["sample_idx"] = idx

        # Tokenize the report. Sentences are reshuffled within their sections on
        # training rows only -- validation must score a fixed target.
        caption = item["caption"]
        if self.is_train:
            caption = reorder_report_sections(caption)
        sample["caption"] = caption
        sample = self.tokenize(sample)
        return sample


def get_diff_caption_dataset(
    csv_file,
    image_csv,
    img_size,
    num_workers,
    tokenizer=None,
    max_caption_length=128,
    is_train=True,
    content_weight=1.0,
):
    """
    Multi-modal longitudinal difference-captioning dataset.

    Expected CSV columns: study_uid, batch, edited_report.
    Images: input_dir/<batch>/<study_uid>/<modality>_time<t>.nii.gz
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(decoder_dir())

    return MultiModalDiffDataset(
        csv_file=csv_file,
        image_csv=image_csv,
        img_size=img_size,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        is_train=is_train,
        content_weight=content_weight,
    )


def get_diff_caption_dataloader(
    csv_file,
    image_csv,
    img_size,
    batch_size,
    num_workers,
    tokenizer=None,
    max_caption_length=128,
    is_train=True,
    distributed: bool = False,
    content_weight=1.0,
):
    """Returns the DataLoader. Training batches are stratified near-evenly on
    `classification` via BalancedBatchSampler; access it via
    loader.batch_sampler for set_epoch(). Non-train loaders use plain
    shuffling, sharded across ranks under DDP via DistributedSampler."""

    ds = get_diff_caption_dataset(
        csv_file=csv_file,
        image_csv=image_csv,
        img_size=img_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        is_train=is_train,
        content_weight=content_weight,
    )

    if is_train:
        num_replicas = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        batch_sampler = BalancedBatchSampler(
            classifications=[item["classification"] for item in ds.items],
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
        )
        loader = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        sampler = DistributedSampler(ds, shuffle=False) if distributed else None
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loader
