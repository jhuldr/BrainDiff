#!/usr/bin/env python3
"""
BIND image steps, run through bind_preprocess.py (skullstrip / register / retry). Env bind_proc.

skullstrip  HD-BET (accurate, TTA, postprocess) on the NATIVE-space centre image; one mask per session
            ($BIND_OUT/_masks/{site}/{session_uid}.nii.gz), atomic writes, a done-marker per shard.
register    per session: (1) ANTs Rigid of every modality onto the centre; (2) the centre's mask applied to all
            (only valid after (1): masking first amputates brain); (3) ANTs Affine of the masked centre onto
            MNI_REFERENCE, applied to all -> 193x229x193 @ 1 mm, shape asserted. ANTs is not deterministic (random
            metric sampling). A session is "cached" when every output exists (--overwrite to redo). --follow submits
            sessions as their masks appear.
retry       stalled sessions: drop series the anatomy filter excludes plus RGB secondary captures, skip a session
            whose centre is one of them, rerun the rest with a per-session timeout, overwriting.
"""
import os

# ITK threads must be capped before ants is imported, or every worker process spawns a full thread pool.
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")

import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import ants  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from bind_preprocess import (EXPECTED_SHAPE, MAX_IN_PLANE_MM, MNI_REFERENCE, MODALITIES, mask_path,  # noqa: E402
                             masks_done, session_dir)


# ------------------------------------------------------------------ skullstrip (GPU)
def skullstrip(a):
    # BrainSegmentor keeps the ensemble resident; run_hd_bet reloads it per volume.
    sys.path.insert(0, os.environ.get("MRRATE_PREPROCESSING", "MR-RATE/data-preprocessing/src"))
    from mr_rate_preprocessing.mri_preprocessing.hdbet import BrainSegmentor
    sessions = pd.read_csv(a.data_csv, dtype={"session_uid": str, "patient_uid": str})
    if a.num_shards > 1:
        sessions = sessions.iloc[a.shard:: a.num_shards].copy()
    if a.limit:
        sessions = sessions.head(a.limit).copy()
    todo = [row for _, row in sessions.iterrows() if not mask_path(a.output_root, row["site"], row["session_uid"]).is_file()]
    print(f"shard {a.shard}/{a.num_shards}: {len(sessions):,} sessions, {len(todo):,} still need a mask")
    done_marker = Path(a.output_root) / "_masks" / f".done_shard{a.shard}of{a.num_shards}"
    if not todo:
        done_marker.parent.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        return
    device = "cpu" if str(a.device).lower() == "cpu" else int(a.device)
    segmentor = BrainSegmentor(mode=a.mode, device=device, do_tta=True, postprocess=True,   # largest component
                               compile=False, mixed_prec=device != "cpu")
    failures, elapsed = 0, 0.0
    for row in tqdm(todo, desc=f"HD-BET dev{a.device}", unit="vol"):
        out_path = mask_path(a.output_root, row["site"], row["session_uid"])
        try:
            start = time.perf_counter()
            # temporary name + rename, so a concurrent `register --follow` never reads a half-written mask
            partial = out_path.with_name(out_path.name.replace(".nii.gz", ".partial.nii.gz"))
            segmentor(row["centered_image"], partial)
            os.replace(partial, out_path)
            elapsed += time.perf_counter() - start
        except Exception as error:
            failures += 1
            print(f"Warning: HD-BET failed on {row['site']}/{row['session_uid']}: {error}")
    done_marker.touch()
    done = len(todo) - failures
    print(f"masked {done:,} volumes, {failures:,} failures" + (f"; {elapsed / done:.2f} s/volume" if done else ""))


# ------------------------------------------------------------------ register (CPU, ANTs)
def apply_mask(image, mask):
    return image.new_image_like(image.numpy() * (mask != 0))


def coregister_to_center(center_image, moving_path):
    """Rigid (head motion only within a session), MR-RATE's upstream metric settings."""
    moving = ants.image_read(str(moving_path))
    return ants.registration(fixed=center_image, moving=moving, type_of_transform="Rigid", aff_metric="Mattes",
                             aff_sampling=32, aff_random_sampling_rate=0.2)["warpedmovout"]


def align_to_mni(center_image, images, out_dir):
    """One Affine of the centre onto MNI_REFERENCE, applied (linear) to every modality."""
    fixed = ants.image_read(MNI_REFERENCE)
    with tempfile.TemporaryDirectory() as tmp_dir:
        registration = ants.registration(fixed=fixed, moving=center_image, type_of_transform="Affine",
                                         outprefix=str(Path(tmp_dir) / "tx_"))
        for name, image in images.items():
            warped = ants.apply_transforms(fixed=fixed, moving=image, transformlist=registration["fwdtransforms"],
                                           interpolator="linear")
            ants.image_write(warped, str(out_dir / f"{name}.nii.gz"))


def process_session(row, output_root, overwrite=False):
    """One session end to end -> (session_key, ok, message); intermediates stay in memory."""
    site, session_uid = row["site"], row["session_uid"]
    session_key = f"{site}/{session_uid}"
    out_dir = session_dir(output_root, site, session_uid)
    present = [m for m in MODALITIES if row.get(m) and not pd.isna(row.get(m))]
    if not overwrite and all((out_dir / f"{m}.nii.gz").is_file() for m in present):
        return session_key, True, "cached"
    mask_file = mask_path(output_root, site, session_uid)
    if not mask_file.is_file():
        return session_key, False, "missing brain mask (run skullstrip first)"
    try:
        center_image = ants.image_read(str(row["centered_image"]))
        in_center_space = {}
        for modality in present:                                           # 1. into the centre's space
            source = str(row[modality])
            if os.path.abspath(source) == os.path.abspath(str(row["centered_image"])):
                in_center_space[modality] = center_image
            else:
                in_center_space[modality] = coregister_to_center(center_image, source)
        mask = np.asarray(nib.load(str(mask_file)).dataobj)               # 2. one mask for all
        if mask.shape != tuple(center_image.shape[:3]):
            return session_key, False, f"mask shape {mask.shape} != centre {tuple(center_image.shape[:3])}"
        masked = {m: apply_mask(image, mask) for m, image in in_center_space.items()}
        out_dir.mkdir(parents=True, exist_ok=True)
        try:                                                               # 3. one affine to MNI for all
            align_to_mni(apply_mask(center_image, mask), masked, out_dir)
        except Exception as error:
            return session_key, False, f"affine registration to MNI failed: {error}"
        for modality in present:
            written = out_dir / f"{modality}.nii.gz"
            if not written.is_file():
                return session_key, False, f"{modality}.nii.gz not written"
            shape = nib.load(str(written)).shape[:3]
            if tuple(shape) != EXPECTED_SHAPE:
                return session_key, False, f"{modality} shape {shape} != {EXPECTED_SHAPE}"
        return session_key, True, "ok"
    except Exception as error:
        return session_key, False, f"{type(error).__name__}: {error}"


def follow(rows, a):
    """Submit each session once its mask exists; once every mask shard is done, submit the rest (they fail fast)."""
    pending = {f"{row['site']}/{row['session_uid']}": row for row in rows}
    results, futures = [], set()
    bar = tqdm(total=len(rows), unit="sess")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        while pending or futures:
            all_done = masks_done(a.output_root, a.mask_shards)
            ready = [k for k, row in pending.items()
                     if all_done or mask_path(a.output_root, row["site"], row["session_uid"]).is_file()]
            for k in ready:
                futures.add(pool.submit(process_session, pending.pop(k), a.output_root, a.overwrite))
            finished, futures = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
            results += [f.result() for f in finished]
            bar.update(len(finished))
    bar.close()
    return results


def register(a):
    sessions = pd.read_csv(a.data_csv, dtype={"session_uid": str, "patient_uid": str})
    if a.num_shards > 1:
        sessions = sessions.iloc[a.shard:: a.num_shards].copy()
    if a.limit:
        sessions = sessions.head(a.limit).copy()
    print(f"shard {a.shard}/{a.num_shards}: {len(sessions):,} sessions, {a.workers} workers")
    rows = [row for _, row in sessions.iterrows()]
    start = time.perf_counter()
    if a.follow:
        results = follow(rows, a)
    else:
        results = []
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(process_session, row, a.output_root, a.overwrite) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), unit="sess"):
                results.append(future.result())
    duration = time.perf_counter() - start
    report = pd.DataFrame(results, columns=["session_key", "ok", "message"])
    processed = (report["message"] != "cached").sum()
    print(f"{report['ok'].sum():,}/{len(report):,} sessions ok in {duration / 60:.1f} min"
          + (f"; {duration / processed:.1f} s/session over {processed:,} newly processed" if processed else ""))
    failures = report[~report["ok"]]
    if len(failures):
        print(f"{len(failures):,} failures:\n{failures['message'].str.slice(0, 70).value_counts().head(10)}")
    log_path = a.data_csv.parent / f"process_log_shard{a.shard}.csv"
    report.to_csv(log_path, index=False)
    print(f"Log: {log_path}")


# ------------------------------------------------------------------ retry (stalled sessions)
def junk(path):
    """Why the anatomy filter (plus RGB) excludes this series, or '' if it doesn't."""
    try:
        image = nib.load(path)
    except Exception as error:
        return f"unreadable {type(error).__name__}"
    header, shape = image.header, image.shape
    if header.get_data_dtype().names:
        return "rgb"
    if len(shape) > 3 and np.prod(shape[3:]) > 1:
        return "multi-volume"
    if len(shape) < 3 or min(shape[:3]) < 2:
        return "not 3D"
    if sorted(header.get_zooms()[:3])[1] > MAX_IN_PLANE_MM:
        return f"in-plane > {MAX_IN_PLANE_MM} mm"
    return ""


def clean_row(row):
    """-> ((row without excluded series, dropped), None) or (None, reason to skip the session)."""
    why = junk(row["centered_image"])
    if why:
        return None, f"centre image excluded: {why}"     # its HD-BET mask belongs to that image
    row = row.copy()
    dropped = []
    for modality in MODALITIES:
        if isinstance(row.get(modality), str) and junk(row[modality]):
            dropped.append(f"{modality}:{junk(row[modality])}")
            row[modality] = np.nan
    return (row, ";".join(dropped)), None


def run_one(row, output_root, queue):
    queue.put(process_session(row, output_root, overwrite=True))   # the killed run can leave a half-written file


def retry(a):
    sessions = pd.read_csv(a.data_csv, dtype={"session_uid": str, "patient_uid": str})
    done = set(pd.read_csv(a.log).session_key) if a.log.is_file() else set()
    log_new = not a.log.is_file()
    log = open(a.log, "a", newline="")
    writer = csv.writer(log)
    if log_new:
        writer.writerow(["session_key", "ok", "message", "seconds", "dropped"])
    todo = []
    for _, row in sessions.iterrows():
        key = f"{row['site']}/{row['session_uid']}"
        if key in done:
            continue
        cleaned, skip = clean_row(row)
        if skip:
            writer.writerow([key, False, skip, 0, ""])
        else:
            todo.append((key, *cleaned))
    log.flush()
    print(f"{len(sessions):,} sessions, {len(done):,} already logged, {len(todo):,} to run, "
          f"{a.workers} workers, {a.timeout:.0f} s limit", flush=True)
    ctx = mp.get_context("fork")
    running, n_ok, n_fail, n_timeout, start = {}, 0, 0, 0, time.time()
    while todo or running:
        while todo and len(running) < a.workers:
            key, row, dropped = todo.pop(0)
            queue = ctx.Queue()
            proc = ctx.Process(target=run_one, args=(row, a.output_root, queue), daemon=True)
            proc.start()
            running[key] = (proc, queue, time.time(), dropped)
        time.sleep(2)
        for key in list(running):
            proc, queue, t0, dropped = running[key]
            elapsed = time.time() - t0
            if not queue.empty():
                _, ok, message = queue.get()
                proc.join(5)
            elif not proc.is_alive():
                ok, message = False, f"worker exited {proc.exitcode} without a result"
            elif elapsed > a.timeout:
                proc.kill()
                proc.join(5)
                ok, message = False, f"timeout {a.timeout:.0f} s"
                n_timeout += 1
            else:
                continue
            del running[key]
            n_ok += bool(ok)
            n_fail += not ok
            writer.writerow([key, ok, message, round(elapsed), dropped])
            log.flush()
            n = n_ok + n_fail
            if n % 50 == 0:
                print(f"[{time.strftime('%H:%M')}] {n}/{n + len(todo) + len(running)} ok {n_ok} fail {n_fail} "
                      f"(timeout {n_timeout}) {(time.time() - start) / 3600:.2f} h", flush=True)
    print(f"done: ok {n_ok}, fail {n_fail} (timeout {n_timeout}) in {(time.time() - start) / 3600:.2f} h", flush=True)
