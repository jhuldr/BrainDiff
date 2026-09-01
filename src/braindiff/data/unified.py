import os
import random
from collections import Counter, defaultdict
import pandas as pd
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    NormalizeIntensityd, Rand3DElasticd, RandBiasFieldd, RandGaussianNoised, ToTensord
)
from braindiff.data.neurovfm_transforms import (
    NeuroVFMGridd, NeuroVFMTokenize, N_TOKENS, TOKEN_DIM)
from transformers import AutoTokenizer
from braindiff.models.paths import decoder_dir
from braindiff.models.prompts import build_chatml, build_user_turn
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

# Decoder resolved through the HF cache (models/paths.py): downloaded once, never again.

# Fixed modality order (matches the model's ["T1w", "T1CE", "T2w", "FLAIR"]).
MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]

# "pcls" is the pathology-classification task, merged in from what used to be a
# separate stage-1.5 dataset/trainer. Its rows share the bounding CSV's schema:
# `caption` holds the comma-separated multi-label pathology target and
# `Pathology` holds the same string again; `pathology_bucket` (the rarest label
# present) is the stratification key.
TASKS = ["aref", "gcap", "caref", "pcls"]
PATHOLOGY_TASK = "pcls"

# Fallback prompt for `pcls` rows whose `prompt` cell is blank
# Studies carry a MEAN of 1.88 intracranial pathologies (45.5% have >1, max 10),
# so the target is every one present, alphabetically ordered and comma-separated.
# The list below is alphabetical to match the required output order, and every
# name is spelled EXACTLY as it appears in the labels -- the old prompt said
# "Rathkes pouch cyst" while the label is "Rathke's pouch cyst", which the model
# now has to reproduce verbatim inside a joined string.
PATHOLOGY_PROMPT = (
    "Your job is to state which pathologies out of the following list are present in the image. "
    "Pathology List: Arachnoid cyst, Cavernous hemangioma, Cerebellar degeneration, "
    "Cerebral atrophy, Cerebral edema, Cerebral hemorrhage, Cerebral infarction, "
    "Chiari malformation, Choroid plexus cyst, Cyst of pineal gland, "
    "Demyelinating disease of central nervous system, Empty sella syndrome, Encephalomalacia, "
    "Glioma, Gliosis, Intracranial aneurysm, Intracranial meningioma, Lacunar infarct, "
    "Lipoma of brain, Mega cisterna magna, Metastatic malignant neoplasm to brain, "
    "Pituitary adenoma, Rathke's pouch cyst, Schwannoma, Silent micro-hemorrhage of brain, "
    "Structure of cave of septum pellucidum, Subdural intracranial hemorrhage, "
    "Ventriculomegaly, Watershed infarct. "
    "Output only pathologies from this list, in alphabetical order, separated by a comma and a "
    "space. If more than one is present, list them all. No additional words or statements. "
    "Which pathologies from the list do you observe in the scans?"
)

MIN_CLASS_COUNT = 20  # drop pathology classes with fewer real examples than this


LABEL_SEP = ", "


def split_labels(s):
    """Comma-joined pathology string -> list of labels. Empty/NaN -> []."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [p for p in (x.strip() for x in str(s).split(",")) if p]


def strat_key(task, with_lesion, bucket):
    """Two-level stratification key: (group, sub).

Bounding rows bucket on (with_lesion, task). Pathology rows bucket on `pathology_bucket`,
the rarest label present, rather than the caption: the caption is a multi-label string with
1,835 distinct values, which would shatter the sampler into buckets too small to shard.
    """
    if task == PATHOLOGY_TASK:
        return ("pathology", str(bucket).strip())
    return ("bounding", (bool(with_lesion), task))


def filter_rare_classes(df, min_class_count=MIN_CLASS_COUNT):
    """Strip individual pathology labels occurring fewer than `min_class_count` times from `pcls`
targets; drop a row only if its label set becomes empty.

Counting whole comma-joined strings instead would delete 4,251 rows (11.2% of pcls) for
having an uncommon combination of common findings. Rarity is a per-label property.
A no-op on the current data (rarest label is Schwannoma at 134). Bounding rows pass through.
    """
    is_pcls = df["task"] == PATHOLOGY_TASK
    if not is_pcls.any():
        return df.reset_index(drop=True)

    counts = Counter(l for s in df.loc[is_pcls, "Pathology"] for l in split_labels(s))
    rare = {l for l, n in counts.items() if n < min_class_count}
    if not rare:
        return df.reset_index(drop=True)

    kept = df.loc[is_pcls, "Pathology"].map(
        lambda s: LABEL_SEP.join(sorted(set(split_labels(s)) - rare)))
    df = df.copy()
    df.loc[is_pcls, "Pathology"] = kept
    df.loc[is_pcls, "caption"] = kept
    n_empty = int((kept == "").sum())
    print(f"filter_rare_classes: stripped {len(rare)} labels below {min_class_count} "
          f"({sorted(rare)}), dropped {n_empty} now-empty rows")
    return df[~is_pcls | (df["Pathology"] != "")].reset_index(drop=True)


class UnifiedBalancedBatchSampler(Sampler):
    """Yields random batches split 50/50 between the bounding and pathology objectives, each half
stratified across its own sub-buckets.

Half the batch goes to bounding rows (up to 6 buckets), half to pathology rows (~20 buckets
after filter_rare_classes). Within a half every bucket draws an equal share and the
remainder goes to randomly chosen buckets. A flat key over all ~26 buckets would give
base = batch_size // 26 = 0 and skew the batch ~3/4 pathology.

Buckets smaller than their per-batch share are reshuffled and reused within the epoch.
Epoch length is anchored on the larger group, so no bounding row goes unseen.

Under DDP every rank builds `groups` from the same full dataset, and each bucket is
truncated to `len(idxs) // num_replicas` before striding, so every rank gets the same batch
count with no communication. This drops at most (num_replicas - 1) samples per bucket.
    """

    GROUPS = ("bounding", "pathology")

    def __init__(self, strat_keys, batch_size, num_replicas=1, rank=0, seed=0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        buckets = defaultdict(list)
        for idx, (group, sub) in enumerate(strat_keys):
            buckets[(group, sub)].append(idx)

        # Truncate each bucket to an equal, rank-independent size *before*
        # striding, so every rank computes the same shard length with zero
        # inter-process communication.
        self.groups = {g: {} for g in self.GROUPS}
        for (group, sub), idxs in buckets.items():
            shard_size = len(idxs) // num_replicas
            if shard_size == 0:
                continue
            self.groups[group][sub] = idxs[rank::num_replicas][:shard_size]

        # Drop empty groups so a bounding-only or pathology-only CSV still runs.
        self.groups = {g: b for g, b in self.groups.items() if b}
        self.slots = self._allocate_slots()

    def _allocate_slots(self):
        """Split batch_size across the present groups, 50/50 when both exist."""
        present = list(self.groups)
        if len(present) == 1:
            return {present[0]: self.batch_size}
        half = self.batch_size // 2
        return {"bounding": half, "pathology": self.batch_size - half}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return max(
            sum(len(v) for v in self.groups[g].values()) // self.slots[g]
            for g in self.groups
        )

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {g: {k: v[:] for k, v in b.items()} for g, b in self.groups.items()}
        for group_pools in pools.values():
            for pool in group_pools.values():
                rng.shuffle(pool)
        cursors = {g: {k: 0 for k in b} for g, b in self.groups.items()}

        for _ in range(len(self)):
            batch = []
            for group, group_pools in pools.items():
                keys = list(group_pools)
                slots = self.slots[group]
                base, remainder = divmod(slots, len(keys))

                counts = {k: base for k in keys}
                for k in rng.sample(keys, remainder):
                    counts[k] += 1

                for k in keys:
                    pool = group_pools[k]
                    for _ in range(counts[k]):
                        if cursors[group][k] >= len(pool):
                            rng.shuffle(pool)
                            cursors[group][k] = 0
                        batch.append(pool[cursors[group][k]])
                        cursors[group][k] += 1
            rng.shuffle(batch)
            yield batch


class TokenizeCaption:
    """Tokenize the caption as the Qwen3 assistant turn.

The caption ends with the single ChatML terminator <|im_end|>; Qwen3's eos_token_id IS
<|im_end|> (151645), so emitting both would duplicate it. Upstream's trailing newline after
<|im_end|> is dropped -- it trains a token past EOS that generation can never reach.
    """
    def __init__(self, tokenizer, max_length=128, max_prompt_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # MEASURED over all 109,253 rows with the prompt the loader ACTUALLY uses:
        # pcls 339 | caref 271 | gcap 247 | aref 200. 384 = 339 + 45 margin.
        #
        # pcls dominates because every one of its 38,086 `prompt` cells is NaN and
        # PATHOLOGY_PROMPT (270 tokens of 29-label vocabulary) is substituted here.
        # Measuring the CSV column instead gives 70 and a dangerously low cap.
        #
        # Prompts are LEFT-PADDED to this width, so it is dead sequence on every
        # sample rather than free headroom -- but only 45 positions of it, so
        # trimming buys <5% and costs the margin. Leave it.
        # for template edits without silently eating the tail.
        self.max_prompt_length = max_prompt_length
        self.eot_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tokenizer.pad_token_id

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

        # S1 is the one stage with a PER-ROW instruction (the `prompt` column carries
        # box coordinates / region names), so it builds its prompt here rather than
        # via PromptTable -- the table keys on modality presence alone and would
        # collapse 88 distinct `caref` prompts into one. No counterfactual at S1, so
        # nothing needs the table's swap lookup.
        #
        # Hand-rolled ChatML, byte-identical to neurovfm/data/text.py::process_text
        # (asserted in braindiff/eval/verify_prompt_format.py), with one labelled
        # <|image_pad|> per present modality.
        text = build_chatml(build_user_turn(data["present_main"].tolist(),
                                            instruction=data["prompt"]))
        core = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        # 384, not 200. At 200 this silently right-truncated every `caref` and
        # `pcls` prompt (max 224 / 225 tokens): caref lost "Region: <name>", which
        # IS its conditioning signal, collapsing 88 distinct prompts to 1 across
        # 23,448 rows with 22,677 distinct captions -- an unlearnable sixth of every
        # batch. `gcap` survived at 199/200, one token from the same cliff, and its
        # visible degradation was collateral through the output surface it shares
        # with caref. `aref` had 43 tokens of slack, which is why it looked fine.
        # Raised 320 -> 384 because the modality labels add ~30 tokens.
        # Fail loudly rather than silently dropping the tail again. Truncation here
        # is not a graceful degradation -- it removes the task.
        if len(core) > self.max_prompt_length:
            raise ValueError(
                f"prompt is {len(core)} tokens, over the {self.max_prompt_length} cap "
                f"(task={data.get('task')!r}). Raise max_prompt_length; do not let this "
                f"pass silently.")

        # LEFT pad. Load-bearing twice over: _splice_prefix and _caption_nll's
        # logits_to_keep both assume the supervised span is the contiguous tail. And
        # the mask must be the REAL one -- regenerating it as ones downstream makes
        # every pad token attended, and since pad count is a deterministic function
        # of prompt length that hands the model a task identifier it can read
        # without reading the instruction.
        n_pad = self.max_prompt_length - len(core)
        data["prompt_ids"] = torch.tensor([self.pad_id] * n_pad + core, dtype=torch.long)
        data["prompt_attn"] = torch.tensor([0] * n_pad + [1] * len(core), dtype=torch.long)
        data["block_present"] = data["present_main"].clone()


        data['input_ids'] = torch.tensor(ids, dtype=torch.long)
        data['attention_mask'] = torch.tensor(attn, dtype=torch.long)
        return data


class MultiModalUnifiedDataset(Dataset):
    """Multi-modal single-timepoint dataset covering both stage-1 objectives.

Each study supplies up to 4 modalities (T1w/T1ce/T2w/FLAIR) at one timepoint. Rows carry a
`task`: aref/gcap/caref are bounding-box grounded captioning, pcls is pathology
classification. One schema so the two objectives interleave within a batch.

An absent modality is a zero volume flagged False in the presence mask; studies with no
modality present are dropped. The single timepoint is emitted as tokens_main/present_main.
Augmentation is shared across the 4 co-registered modalities.
    """

    def __init__(self, csv_file, img_size, tokenizer,
                 max_caption_length=128, max_prompt_length=384, is_train=True):

        self.img_size = img_size
        self.is_train = is_train

        # Modality keys, e.g. "main_t1w"
        self.keys = [f"main_{mod}" for mod in MODALITIES]

        # === BUILD ITEMS (drop incomplete studies) ===
        df = pd.read_csv(csv_file)
        self.items = []
        dropped = 0
        for _, row in df.iterrows():
            caption = row.get("caption")
            if pd.isna(caption) or str(caption).strip() == "":
                dropped += 1
                continue

            task = row.get("task")
            if pd.isna(task) or str(task).strip() not in TASKS:
                dropped += 1
                continue
            task = str(task).strip()

            prompt = row.get("prompt")
            if pd.isna(prompt) or str(prompt).strip() == "":
                # pcls shares one dataset-wide prompt, so a blank cell is
                # expected there; every other task needs its own.
                if task != PATHOLOGY_TASK:
                    dropped += 1
                    continue
                prompt = PATHOLOGY_PROMPT

            with_lesion = row.get("with_lesion")
            if pd.isna(with_lesion):
                # with_lesion is half the *bounding* stratification key and has
                # no meaning for pcls rows (they bucket on pathology_bucket), so the
                # merged CSV leaves it blank there. Every pcls scan has a
                # pathology by construction -> True.
                if task != PATHOLOGY_TASK:
                    dropped += 1
                    continue
                with_lesion = True

            pathology = row.get("Pathology")
            if task == PATHOLOGY_TASK and (pd.isna(pathology) or str(pathology).strip() == ""):
                dropped += 1
                continue

            # Sampler/split bucket is the rarest label present, carried as its own
            # column. Falling back to the caption would silently restore the
            # 1,835-way bucketing this column exists to avoid, so fail loudly.
            bucket = row.get("pathology_bucket")
            if task == PATHOLOGY_TASK and (pd.isna(bucket) or str(bucket).strip() == ""):
                raise RuntimeError(
                    "pcls row has no `pathology_bucket`. Regenerate the unified CSV:\n"
                    "  python -m dataset.reports.braindiff.build_multilabel_pathology --patch-s2\n"
                    "  python -m dataset.assembly.braindiff.build_unified_dataframe"
                )

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
                "prompt": str(prompt),
                "caption": str(caption),
                "task": task,
                "with_lesion": bool(with_lesion),
                "strat": strat_key(task, with_lesion, bucket),
            })
        n_pcls = sum(1 for it in self.items if it["task"] == PATHOLOGY_TASK)
        print(f"Stage 1 Curriculum: kept {len(self.items)} "
              f"(bounding {len(self.items) - n_pcls}, pathology {n_pcls}), dropped {dropped}")

        # === TRANSFORMS (allow_missing_keys -> tolerate absent modalities) ===
        self.load = Compose([NeuroVFMGridd(keys=self.keys, allow_missing_keys=True)])
        self.to_tokens = NeuroVFMTokenize(keys=self.keys, allow_missing_keys=True)

        # NO AUGMENTATION AT S1, deliberately. Half of every batch is the `aref`/
        # `caref` bounding-box task, and a spatial warp moves the anatomy without
        # moving the box coordinates -- it would teach the model wrong targets.
        # Intensity augmentation is dropped too: the encoder is frozen apart from
        # attention-LoRA, so perturbing its input mostly spends capacity undoing
        # the perturbation. It also cost ~135 ms/volume, which by itself put S1 at
        # ~2 days against a 1.5-day budget.

        self.tokenize = TokenizeCaption(tokenizer, max_caption_length,
                                        max_prompt_length=max_prompt_length)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        data = self.to_tokens(self.load({k: v for k, v in item["paths"].items()}))

        tokens = torch.zeros(len(self.keys), N_TOKENS, TOKEN_DIM, dtype=torch.float16)
        coords = torch.zeros(len(self.keys), N_TOKENS, 3, dtype=torch.int16)
        present = torch.zeros(len(self.keys), dtype=torch.bool)
        for i, key in enumerate(self.keys):
            entry = data.get(key)
            if entry is None or entry["tokens"].shape[0] != N_TOKENS:
                continue
            tokens[i], coords[i], present[i] = entry["tokens"], entry["coords"], True

        sample = {
            "tokens_main": tokens,
            "coords_main": coords,
            "present_main": present,
            "prompt": item["prompt"],
            "caption": item["caption"],
            "task": item["task"],
        }
        sample = self.tokenize(sample)
        return sample


def get_diff_caption_dataset(
    csv_file,
    img_size,
    num_workers,
    tokenizer=None,
    max_caption_length=128,
    max_prompt_length=384,
    is_train=True,
):
    """Multi-modal single-timepoint dataset for the unified stage-1 objective.

Expected CSV columns: task, with_lesion, Pathology, pathology_bucket, prompt, caption, and
one path column per entry in MODALITIES.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(decoder_dir())

    return MultiModalUnifiedDataset(
        csv_file=csv_file,
        img_size=img_size,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        max_prompt_length=max_prompt_length,
        is_train=is_train,
    )


def get_diff_caption_dataloader(
    csv_file,
    img_size,
    batch_size,
    num_workers,
    tokenizer=None,
    max_caption_length=128,
    max_prompt_length=384,
    is_train=True,
    distributed: bool = False,
):
    """Returns the DataLoader. Training batches are split 50/50 bounding/pathology via
UnifiedBalancedBatchSampler; access it through loader.batch_sampler for set_epoch().
Non-train loaders use plain shuffling, sharded across ranks by DistributedSampler.
    """

    ds = get_diff_caption_dataset(
        csv_file=csv_file,
        img_size=img_size,
        num_workers=num_workers,
        tokenizer=tokenizer,
        max_caption_length=max_caption_length,
        max_prompt_length=max_prompt_length,
        is_train=is_train,
    )

    if is_train:
        num_replicas = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        batch_sampler = UnifiedBalancedBatchSampler(
            strat_keys=[item["strat"] for item in ds.items],
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
