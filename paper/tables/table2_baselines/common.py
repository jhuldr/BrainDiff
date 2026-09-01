"""Shared per-row payload construction for the proprietary-model benchmark.
Builds the labelled image set (current + prior timepoint slices, priority-ordered, capped)
and the prior report / reference, from subset_64.csv + image_extended.csv."""
import csv, os
csv.field_size_limit(10**7)
from slice_convert import study_sequences

HERE = os.path.dirname(os.path.abspath(__file__))
SUBSET = os.path.join(HERE, "subset_64.csv")
IMAGE_CSV = "/home/data/BRAIN_DIFF_S4/image_extended.csv"

def load_subset():
    with open(SUBSET, newline="") as f:
        return list(csv.DictReader(f))

def load_image_map():
    m = {}
    with open(IMAGE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            m[r["study_uid"]] = r
    return m

def build_row_images(row, image_map, slices_per_seq=12, both_timepoints=True, max_images=96):
    """Return (labelled_images, n_current, n_prior) where labelled_images is a list of
    (label, b64png). Current timepoint first (priority T1ce>FLAIR>T2w>T1w), then prior if
    both_timepoints and budget remains; total capped at max_images."""
    cur_uid, pri_uid = row["study_uid2"], row["study_uid1"]
    labelled = []
    def add(uid, tag):
        seqs = study_sequences(image_map.get(uid, {}), slices_per_seq=slices_per_seq)
        for mod, slices in seqs:
            n = len(slices)
            for k, b in enumerate(slices):
                if len(labelled) >= max_images:
                    return
                labelled.append((f"{tag} study - {mod}, slice {k+1}/{n}", b))
    add(cur_uid, "Current")
    n_current = len(labelled)
    if both_timepoints:
        add(pri_uid, "Prior")
    n_prior = len(labelled) - n_current
    return labelled, n_current, n_prior

def rows_with_payload(slices_per_seq=12, both_timepoints=True, max_images=96, limit=None):
    subset = load_subset(); imap = load_image_map()
    for i, row in enumerate(subset):
        if limit and i >= limit:
            break
        imgs, nc, npri = build_row_images(row, imap, slices_per_seq, both_timepoints, max_images)
        yield {
            "study_uid2": row["study_uid2"], "study_uid1": row["study_uid1"],
            "prior_report": row.get("report1", ""), "ref": row.get("generated_report", ""),
            "images": imgs, "n_current": nc, "n_prior": npri,
        }
