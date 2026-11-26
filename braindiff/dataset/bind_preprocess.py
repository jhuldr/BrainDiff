#!/usr/bin/env python3
"""
BIND preprocessing: raw BIDS tree + clinical CSVs -> series manifest, report / image-only pairs, series selection,
then HD-BET + ANTs into the 193x229x193 @ 1 mm template space. CPU except `skullstrip`. Image steps live in
bind_register.py so the table steps never import ants / torch / HD-BET.

    python bind_preprocess.py manifest                  -> $BIND_TABLES/bind_series_manifest.csv
    python bind_preprocess.py pairs                     -> bind_longitudinal_meta.csv (report pairs)
    python bind_preprocess.py select                    -> bind_longitudinal_data.csv, bind_longitudinal_meta_paired.csv
    python bind_preprocess.py select3d                  -> bind_longitudinal_data_3d.csv, bind_longitudinal_meta_paired_3d.csv
    python bind_preprocess.py imageonly                 -> bind_imageonly_data.csv, bind_imageonly_pairs.csv
    python bind_preprocess.py skullstrip --data-csv X --device 0 --shard 0 --num-shards 9      (GPU, HD-BET)
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 python bind_preprocess.py register --data-csv X --follow --mask-shards 9 --workers 60
    python bind_preprocess.py retry --data-csv X --timeout 1800 --log X.csv                    (stalled sessions)

Env: BIND_ROOT = raw BIND (Imaging/{site}/BIDS, Imaging/{site}/Clinical, PatientMergeHistory);
BIND_OUT = registered volumes; BIND_TABLES = the CSVs above; MNI_REFERENCE = mni_reference.nii.gz (193x229x193 @ 1 mm);
MRRATE_PREPROCESSING = MR-RATE's data-preprocessing/src (HD-BET wrapper).

The paper's report-pair tables predate the anatomy filter: reproduce them with `select --no-anatomy-filter`.
"""
import argparse
import ast
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ paths
#   $BIND_OUT/_masks/{site}/{session_uid}.nii.gz       native-space brain mask (of the centre image)
#   $BIND_OUT/{site}/{session_uid}/{modality}.nii.gz   193x229x193 @ 1 mm template space
BIND_ROOT = os.environ.get("BIND_ROOT", "BIND")
OUTPUT_ROOT = Path(os.environ.get("BIND_OUT", "BIND-BRAINDIFF"))
TABLES = Path(os.environ.get("BIND_TABLES", "bind_tables"))
MNI_REFERENCE = os.environ.get("MNI_REFERENCE", "mni_reference.nii.gz")
MODALITIES = ["t1w", "t1ce", "t2w", "flair"]
EXPECTED_SHAPE = (193, 229, 193)     # geometry of mni_reference.nii.gz; asserted on every output

MANIFEST_CSV = TABLES / "bind_series_manifest.csv"
META_CSV = TABLES / "bind_longitudinal_meta.csv"
DATA_CSV = TABLES / "bind_longitudinal_data.csv"
PAIRED_CSV = TABLES / "bind_longitudinal_meta_paired.csv"


def mask_path(output_root, site, session_uid):
    return Path(output_root) / "_masks" / str(site) / f"{session_uid}.nii.gz"


def session_dir(output_root, site, session_uid):
    return Path(output_root) / str(site) / str(session_uid)


def masks_done(output_root, num_shards):
    """True once every skullstrip shard of a --num-shards run has written its done-marker."""
    return all((Path(output_root) / "_masks" / f".done_shard{i}of{num_shards}").is_file() for i in range(num_shards))


# ------------------------------------------------------------------ session / patient keys
# BIND spells one session id three ways (padded, unpadded, BIDS label); raw-string joins silently match nothing.
# Canonical = padded I0001 id (= BIDS label) and verbatim I0004 id (alphanumeric; padding corrupts it).
SITES = ("I0001", "I0004")
PATIENT_MERGE_CSV = {"I0001": f"{BIND_ROOT}/PatientMergeHistory/I0001_patient_history_18thAugust2025.csv"}   # only I0001 ships one
_I0001_WIDTH = 10


def canonical_session(value, site):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.zfill(_I0001_WIDTH) if site == "I0001" else value


def canonical_session_series(values, site):
    values = values.astype("string").str.strip()
    if site == "I0001":
        values = values.str.zfill(_I0001_WIDTH)
    return values.replace("", pd.NA)


def session_from_bids_dir(name):
    name = str(name)
    return name[4:] if name.startswith("ses-") else name


def subject_from_bids_dir(name):
    name = str(name)
    return name[4:] if name.startswith("sub-") else name


def load_patient_merge_map(site):
    """{BDSPPatientID -> MergedBDSPPatientID}, keeps merged duplicate records on one side of the split; {} if none."""
    path = PATIENT_MERGE_CSV.get(site)
    if path is None:
        return {}
    m = pd.read_csv(path, dtype=str).dropna(subset=["BDSPPatientID", "MergedBDSPPatientID"])
    return dict(zip(m["BDSPPatientID"].str.strip(), m["MergedBDSPPatientID"].str.strip()))


def canonical_patient_series(values, merge_map):
    """One hop through the merge map; unmapped ids unchanged."""
    values = values.astype("string").str.strip()
    if not merge_map:
        return values
    return values.map(lambda v: merge_map.get(v, v) if pd.notna(v) else v)


# ------------------------------------------------------------------ report sections
# BIND prose does not use the S2/S4 headers. The COMPARISON section is stripped rather than dropping reports that
# mention a prior. Type == "not available" rows have no report text and are dropped.
SECTION_HEADERS = ("NARRATIVE", "TECHNIQUE", "COMPARISON STUDIES", "COMPARISON", "FINDINGS OF CLINICAL SIGNIFICANCE",
                   "FINDINGS", "IMPRESSION", "END OF IMPRESSION", "ATTESTATION", "CLINICAL INDICATION",
                   "CLINICAL HISTORY", "ADDITIONAL HISTORY", "HISTORY", "INDICATIONS", "INDICATION",
                   "PROCEDURE COMMENTS", "RECOMMENDATION", "ACCESSION NUMBER", "SUMMARY",
                   "SYNOPSIS FOR CLINICAL MANAGEMENT", "EXAM")
_HEADER_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:", re.IGNORECASE)
_COMPARISON_HEADERS = {"COMPARISON", "COMPARISON STUDIES"}
_EXAM_TYPE_RE = re.compile(r"\b((?:MRI?|CT|NM|PET|MRA)\s+[A-Z][A-Z /&-]{2,60}?)\s*(?:[:.]|$)")
_MR_BRAIN_RE = re.compile(r"\b(?:MRI?|MR)\s+(?:NS\s+)?(?:FAST\s+)?BRAIN\b", re.IGNORECASE)
_EXCLUDE_RE = re.compile(r"\b(ANGIOGRAPH|ANGIO|MRA|FUNCTIONAL|SPECTROSCOP|PERFUSION ONLY|DATSCAN"
                         r"|OUTSIDE \(NO INTERPRETATION\)|PET|SPECT)\b", re.IGNORECASE)   # "brain" but not structural


def split_sections(text):
    """Report -> [(header_or_None, body)] in document order; text before the first header has header None."""
    if not isinstance(text, str) or not text.strip():
        return []
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [(None, text)]
    sections = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1).upper(), text[match.end(): end].strip()))
    return sections


def strip_comparison(text):
    """-> (text without COMPARISON sections, had_comparison). Header-less reports are returned unchanged."""
    sections = split_sections(text)
    if not sections:
        return text, False
    if not any(header in _COMPARISON_HEADERS for header, _ in sections):
        return text, False
    kept = [f"{h}: {b}" if h else b for h, b in sections if h not in _COMPARISON_HEADERS]
    return "\n\n".join(part for part in kept if part.strip()), True


# FINDINGS onward, signatures dropped: the BIND text that is labelled and rewritten. Not valid for I0007 (its
# IMPRESSION repeats the whole report).
DROP_SECTIONS = {"ATTESTATION", "ACCESSION NUMBER"}
SIGNATURE = re.compile(r"\b(Dictated by|Interpreted by|Electronically Signed|Authored By)\b.*", re.S)   # I0004


def findings_onward(text):
    """Report text from the first FINDINGS header to the end, signatures dropped. None if no FINDINGS."""
    secs = split_sections(text)
    start = next((i for i, (h, _) in enumerate(secs) if h and "FINDINGS" in h), None)
    if start is None:
        return None
    body = " ".join(f"{h}: {b}" for h, b in secs[start:] if h not in DROP_SECTIONS)
    return " ".join(SIGNATURE.sub("", body).split())


def exam_type_from_report(text):
    """Exam name from the TECHNIQUE / NARRATIVE / EXAM body, else the first modality-prefixed phrase."""
    if not isinstance(text, str) or not text.strip():
        return ""
    for header, body in split_sections(text):
        if header in {"TECHNIQUE", "NARRATIVE", "EXAM"} and body:
            match = _EXAM_TYPE_RE.search(body[:200])
            if match:
                return match.group(1).strip().upper()
    match = _EXAM_TYPE_RE.search(text[:400])
    return match.group(1).strip().upper() if match else ""


def is_mr_brain(exam_type):
    if not isinstance(exam_type, str) or not exam_type.strip():
        return False
    if _EXCLUDE_RE.search(exam_type):
        return False
    return bool(_MR_BRAIN_RE.search(exam_type))


def has_contrast(exam_type, text=""):
    """With-and-without-contrast exam; gates the run-order T1ce fallback in select_t1_pair."""
    blob = f"{exam_type or ''} {(text or '')[:400]}".upper()
    if re.search(r"\bW\s*/?\s*O\s+(?:IV\s+)?CONTRAST\b", blob) and "AND" not in blob:
        return False
    return bool(re.search(r"\bW(?:ITH)?\s+AND\s+W(?:ITH)?\s*/?O(?:UT)?\b|\bWWO\b"
                          r"|\bWITH\s+AND\s+WITHOUT\b|\bW\s*&\s*WO\b", blob))


def selftest():
    # synthetic ids in the real formats
    assert canonical_session("123456", "I0001") == "0000123456" == canonical_session(" 123456 ", "I0001")
    assert canonical_session(123456, "I0001") == "0000123456" == canonical_session("0000123456", "I0001")
    assert canonical_session("GG0a1b2c3", "I0004") == "GG0a1b2c3"
    assert canonical_session(None, "I0001") is None and canonical_session("", "I0001") is None
    assert session_from_bids_dir("ses-0000654321") == "0000654321" and subject_from_bids_dir("sub-100000001") == "100000001"
    assert list(canonical_session_series(pd.Series(["123456", "654321"]), "I0001")) == ["0000123456", "0000654321"]
    assert list(canonical_session_series(pd.Series(["GG0a1b2c3"]), "I0004")) == ["GG0a1b2c3"]
    assert list(canonical_patient_series(pd.Series(["1", "2"]), {"1": "9"})) == ["9", "2"]
    text = ("TECHNIQUE: MRI BRAIN WITH AND WITHOUT CONTRAST\n\nCOMPARISON: MRI brain from last year.\n\n"   # synthetic
            "FINDINGS:\n\nSmall left frontal meningioma.\n\nIMPRESSION: Stable.")
    cleaned, had = strip_comparison(text)
    assert had and "COMPARISON" not in cleaned and "meningioma" in cleaned and "IMPRESSION" in cleaned
    assert exam_type_from_report(text) == "MRI BRAIN WITH AND WITHOUT CONTRAST"
    assert all(map(is_mr_brain, ["MRI BRAIN WITH AND WITHOUT CONTRAST", "MR BRAIN WO IV CONTRAST",
                                 "MRI NS FAST BRAIN WO CONTRAST LIMITED"]))
    assert not any(map(is_mr_brain, ["MRI ANGIO BRAIN", "MRA HEAD WITHOUT CONTRAST", "CT HEAD",
                                     "NM PET BRAIN ALZHEIMER DEMENTIA"]))
    assert has_contrast("MR BRAIN W AND WO CONTRAST") and has_contrast("MRI BRAIN WWO CONTRAST")
    assert not has_contrast("MR BRAIN WO IV CONTRAST") and not has_contrast("MRI BRAIN")
    plain = "No headers here, just prose about a prior study."
    assert strip_comparison(plain) == (plain, False)
    print("selftest passed")


# ------------------------------------------------------------------ manifest
# anat/ only. I0004 SeriesDescription is scrubbed ("Series 5"), so numeric sidecar fields are kept verbatim.
SIDECAR_FIELDS = ["SeriesDescription", "ProtocolName", "StudyDescription", "SeriesNumber", "RepetitionTime",
                  "EchoTime", "InversionTime", "FlipAngle", "ScanningSequence", "SequenceVariant", "ScanOptions",
                  "MRAcquisitionType", "SliceThickness", "SpacingBetweenSlices", "Manufacturer", "MagneticFieldStrength"]


def parse_bids_entities(filename):
    """'sub-1_ses-2_acq-SE_run-1_T1w.nii.gz' -> ('T1w', {'acq': 'SE', 'run': '1'}). No ce- entity in BIND, so
    contrast cannot be read off the filename."""
    stem = filename[: -len(".nii.gz")] if filename.endswith(".nii.gz") else Path(filename).stem
    parts = stem.split("_")
    entities = {}
    for part in parts[:-1]:
        if "-" in part:
            key, _, value = part.partition("-")
            entities[key] = value
    return parts[-1], entities


def acquisition_plane(orientation):
    """Plane from ImageOrientationPatient: the slice normal's dominant patient axis; '' if missing."""
    if orientation is None:
        return ""
    try:
        cosines = np.asarray(orientation, dtype=float).ravel()
    except (TypeError, ValueError):
        return ""
    if cosines.size != 6 or not np.all(np.isfinite(cosines)):
        return ""
    normal = np.cross(cosines[:3], cosines[3:])
    norm = np.linalg.norm(normal)
    if norm == 0:
        return ""
    return ["SAGITTAL", "CORONAL", "AXIAL"][int(np.argmax(np.abs(normal / norm)))]   # DICOM x=L/R, y=A/P, z=S/I


def is_derived_series(image_type, suffix, entities):
    """MR-RATE's is_derived/is_localizer: DERIVED/SECONDARY ImageType, MIP/mnIP, SWI phase, angio/pwi/swi."""
    image_type = [str(v).upper() for v in (image_type or [])]
    if "DERIVED" in image_type or "SECONDARY" in image_type or "PROJECTION IMAGE" in image_type:
        return True
    if entities.get("acq", "").lower() in {"mip", "mnip"}:
        return True
    if entities.get("part", "").lower() == "phase":
        return True
    return suffix in {"angio", "pwi", "swi"}


def read_series(nii_path):
    """One manifest row; the NIfTI header is read, never the voxels."""
    import nibabel as nib
    nii_path = Path(nii_path)
    filename = nii_path.name
    suffix, entities = parse_bids_entities(filename)
    json_path = nii_path.with_name(filename[: -len(".nii.gz")] + ".json")
    sidecar = {}
    if json_path.is_file():
        try:
            with open(json_path) as handle:
                sidecar = json.load(handle)
        except (json.JSONDecodeError, OSError):
            sidecar = {}
    try:
        header = nib.load(str(nii_path)).header
        shape = tuple(int(v) for v in header.get_data_shape()[:3])
        spacing = tuple(round(float(v), 4) for v in header.get_zooms()[:3])
        n_volumes = int(np.prod(header.get_data_shape()[3:]))   # >1: multi-echo / multi-TI / 4D
    except Exception:
        shape, spacing, n_volumes = None, None, None
    row = {"site": nii_path.parents[4].name,          # .../Imaging/{site}/BIDS/sub-X/ses-Y/anat/file.nii.gz
           "patient_id": subject_from_bids_dir(nii_path.parents[2].name),
           "session_id": session_from_bids_dir(nii_path.parents[1].name),
           "datatype": nii_path.parent.name, "nii_path": str(nii_path),
           "json_path": str(json_path) if json_path.is_file() else "", "bids_suffix": suffix,
           "acq": entities.get("acq", ""), "run": entities.get("run", ""), "part": entities.get("part", ""),
           "shape": str(shape) if shape else "", "spacing_mm": str(spacing) if spacing else "", "n_volumes": n_volumes,
           "acquisition_plane": acquisition_plane(sidecar.get("ImageOrientationPatientDICOM")),
           "is_derived": is_derived_series(sidecar.get("ImageType"), suffix, entities)}
    for field in SIDECAR_FIELDS:
        row[field] = sidecar.get(field)
    return row


def scan_subject(subject_dir):
    rows = []
    for ses in sorted(Path(subject_dir).glob("ses-*")):
        anat_dir = ses / "anat"
        if not anat_dir.is_dir():
            continue
        for nii_path in sorted(anat_dir.glob("*.nii.gz")):
            try:
                rows.append(read_series(nii_path))
            except Exception as error:
                print(f"Warning: failed on {nii_path}: {error}")
    return rows


def scan_site(site, workers, limit=None):
    from tqdm import tqdm
    bids_root = Path(BIND_ROOT) / "Imaging" / site / "BIDS"
    subjects = sorted(d for d in bids_root.glob("sub-*") if d.is_dir())
    if limit:
        subjects = subjects[:limit]
    print(f"{site}: {len(subjects):,} subjects")
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(scan_subject, str(d)) for d in subjects]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Scanning {site}", unit="subj"):
            rows.extend(future.result())
    return rows


def manifest(a):
    rows = [r for site in a.sites for r in scan_site(site, a.workers, a.limit)]
    m = pd.DataFrame(rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.output, index=False)
    print(f"Saved {len(m):,} series to {a.output}; sessions {m.groupby(['site', 'session_id']).ngroups:,}, "
          f"patients {m.groupby(['site', 'patient_id']).ngroups:,}")
    print(m["bids_suffix"].value_counts().head(15).to_string())


# ------------------------------------------------------------------ report pairs
_MISSING = {"not available", "n/a", "none", ""}   # every such Type also has Report_txt "not available"


def load_reports(site):
    """One site's reports with usable text, keyed by canonical session id."""
    reports = pd.read_csv(f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_reports.csv", dtype=str)
    reports["session_uid"] = canonical_session_series(reports["session_id"], site)
    reports["site"] = site
    before = len(reports)
    missing = (reports["Report_txt"].fillna("").str.strip().str.lower().isin(_MISSING)
               | reports["Type"].fillna("").str.strip().str.lower().isin(_MISSING))
    reports = reports[~missing].copy()
    print(f"  {site}: {before:,} rows -> {len(reports):,} with usable text")
    return reports


def load_dates(site):
    """ShiftedStudyDate (demographics only); the shift is constant per patient, so intervals are preserved."""
    d = pd.read_csv(f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_demographics.csv", dtype=str)
    d["session_uid"] = canonical_session_series(d["Session_id"], site)
    dates = d[["session_uid", "ShiftedStudyDate"]].dropna().drop_duplicates(subset=["session_uid"])
    dates["study_date"] = pd.to_datetime(dates["ShiftedStudyDate"], errors="coerce")
    return dates[["session_uid", "study_date"]].dropna(subset=["study_date"])


def build_site_studies(site):
    """One row per dated MR-brain session with usable text; `Type` decides brain MR, the body only the contrast flag."""
    studies = load_reports(site).merge(load_dates(site), on="session_uid", how="left")
    with_date = studies["study_date"].notna()
    print(f"  {site}: {with_date.sum():,}/{len(studies):,} sessions resolved a study date")
    studies = studies[with_date].copy()
    studies["exam_type"] = studies["Type"].fillna("").str.upper().str.strip()
    is_brain = studies["exam_type"].map(is_mr_brain)
    print(f"  {site}: {is_brain.sum():,} MR-brain sessions of {len(studies):,}")
    studies = studies[is_brain].copy()
    stripped = studies["Report_txt"].map(strip_comparison)
    studies["report"] = stripped.map(lambda pair: pair[0])
    studies["had_comparison"] = stripped.map(lambda pair: pair[1])
    studies["study_contains_contrast"] = [has_contrast(e, t) for e, t in zip(studies["exam_type"], studies["Report_txt"])]
    merge_map = load_patient_merge_map(site)
    studies["patient_uid"] = canonical_patient_series(studies["bdsp_patient_id"], merge_map)
    return studies[["site", "patient_uid", "session_uid", "study_date", "report", "exam_type", "had_comparison",
                    "study_contains_contrast"]]


def pair_consecutive(studies):
    """MR-RATE's rule: one study per patient per date (ties by session id), adjacent timepoints only."""
    studies = studies.sort_values(["patient_uid", "study_date", "session_uid"])
    studies = studies.drop_duplicates(subset=["patient_uid", "study_date"], keep="first")
    pairs = []
    for patient_uid, group in studies.groupby("patient_uid", sort=False):
        if len(group) < 2:
            continue
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            p, c = group.iloc[i], group.iloc[i + 1]
            pairs.append({"patient_uid": patient_uid, "site": p["site"], "study_uid1": p["session_uid"],
                          "study_uid2": c["session_uid"], "duration": (c["study_date"] - p["study_date"]).days,
                          "report1": p["report"], "report2": c["report"], "exam_type1": p["exam_type"],
                          "exam_type2": c["exam_type"], "had_comparison1": p["had_comparison"],
                          "had_comparison2": c["had_comparison"]})
    return pd.DataFrame(pairs)


def report_pairs(a):
    studies = pd.concat([build_site_studies(site) for site in a.sites], ignore_index=True)
    pairs = pair_consecutive(studies)
    before = len(pairs)
    pairs = pairs[pairs["duration"] >= a.min_duration].copy()
    print(f"Dropped {before - len(pairs):,} pairs under {a.min_duration} day(s) apart")
    pairs = pairs.dropna(subset=["report1", "report2"])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(a.output, index=False)
    print(f"Saved {len(pairs):,} pairs / {pairs['patient_uid'].nunique():,} patients to {a.output}; "
          f"per site {pairs['site'].value_counts().to_dict()}")


# ------------------------------------------------------------------ series selection
# Classification = BIDS suffix, not TR/TE/TI (a parameter rule mislabels T1 FLAIR PROPELLER, MPRAGE and STIR).
MODALITY_SUFFIXES = ("T1w", "T2w", "FLAIR")
EXCLUDED_ACQ = {"STIR", "MAGIC"}      # short-TI fat-suppressed T2 / synthetic-contrast reconstruction
MAX_IN_PLANE_MM = 1.5
# Contrast markers; intact at I0001, scrubbed at I0004 (the run-order fallback carries that site).
_POST_RE = re.compile(r"(?<![A-Za-z])(POST|GAD|GD)(?![A-Za-z])|\+\s*C(?![A-Za-z])", re.IGNORECASE)
_PRE_RE = re.compile(r"(?<![A-Za-z])(PRE|NON[- ]?CON)(?![A-Za-z])", re.IGNORECASE)
_PLANE_RANK = {"AXIAL": 1, "SAGITTAL": 2, "CORONAL": 3}


def plane_priority(plane):
    return _PLANE_RANK.get(str(plane).upper(), 4)


def load_series(manifest_csv, anatomy_filter=True):
    """Usable structural series. anatomy_filter: in-plane <= MAX_IN_PLANE_MM and one volume (drops coil maps, scouts,
    localizers that carry a T1w/T2w suffix; HD-BET rejects > 1 volume). Needs the n_volumes column."""
    m = pd.read_csv(manifest_csv, dtype={"session_id": str, "patient_id": str, "run": str, "acq": str}, low_memory=False)
    series = m[(~m["is_derived"]) & m["bids_suffix"].isin(MODALITY_SUFFIXES)
               & (~m["acq"].fillna("").str.upper().isin(EXCLUDED_ACQ))].copy()
    print(f"Manifest: {len(m):,} series -> {len(series):,} structural non-derived")
    if anatomy_filter:
        in_plane = series["spacing_mm"].map(
            lambda v: max(sorted(ast.literal_eval(v))[:2]) if isinstance(v, str) and v else np.inf)
        series = series[(in_plane <= MAX_IN_PLANE_MM) & (series["n_volumes"] == 1)].copy()
        print(f"  -> {len(series):,} anatomical (in-plane <= {MAX_IN_PLANE_MM} mm, one volume)")
    series["SeriesNumber"] = pd.to_numeric(series["SeriesNumber"], errors="coerce")
    series["SliceThickness"] = pd.to_numeric(series["SliceThickness"], errors="coerce")
    series["is_3d"] = series["MRAcquisitionType"].astype(str).str.upper().eq("3D")
    series["plane_rank"] = series["acquisition_plane"].map(plane_priority)
    description = series["SeriesDescription"].fillna("")
    series["is_post"] = description.map(lambda v: bool(_POST_RE.search(v)))
    series["is_pre"] = description.map(lambda v: bool(_PRE_RE.search(v)))
    series["session_key"] = series["site"] + "/" + series["session_id"]
    return series


def pick_best(candidates):
    """Best plane, then earliest SeriesNumber (MR-RATE pick_best_plane)."""
    if candidates.empty:
        return None
    return candidates.sort_values(["plane_rank", "SeriesNumber"], na_position="last").iloc[0]


def pick_center(t1w_series):
    """Registration target = highest-resolution T1w: 3D over 2D, thinnest slice, plane, earliest series. Contrast is
    ignored."""
    if t1w_series.empty:
        return None
    return t1w_series.sort_values(["is_3d", "SliceThickness", "plane_rank", "SeriesNumber"],
                                  ascending=[False, True, True, True], na_position="last").iloc[0]


def select_t1_pair(t1w_series, study_has_contrast):
    """-> (pre, post, source). text: SeriesDescription POST/GAD/+C (I0001 only); run_order: with-and-without-contrast
    exams only, lowest SeriesNumber = pre, highest = post (heuristic); none: one T1w, no t1ce."""
    if t1w_series.empty:
        return None, None, "none"
    post = t1w_series[t1w_series["is_post"]]
    pre = t1w_series[t1w_series["is_pre"]]
    untagged = t1w_series[~t1w_series["is_post"] & ~t1w_series["is_pre"]]
    if not post.empty:
        return pick_best(pre if not pre.empty else untagged), pick_best(post), "text"
    if study_has_contrast and len(t1w_series) > 1:
        numbered = t1w_series.dropna(subset=["SeriesNumber"])
        if len(numbered) > 1:
            first, last = numbered["SeriesNumber"].min(), numbered["SeriesNumber"].max()
            if first != last:
                return (pick_best(numbered[numbered["SeriesNumber"] == first]),
                        pick_best(numbered[numbered["SeriesNumber"] == last]), "run_order")
    return pick_best(t1w_series), None, "none"


def select_session(session_series, study_has_contrast):
    """One selection row, or None without a usable T1w (centre)."""
    by_suffix = {suffix: g for suffix, g in session_series.groupby("bids_suffix")}
    empty = session_series.iloc[0:0]
    t1w_series = by_suffix.get("T1w", empty)
    center = pick_center(t1w_series)
    if center is None:
        return None
    t1w, t1ce, source = select_t1_pair(t1w_series, study_has_contrast)
    t2w, flair = pick_best(by_suffix.get("T2w", empty)), pick_best(by_suffix.get("FLAIR", empty))
    path = lambda row: None if row is None else row["nii_path"]
    return {"site": center["site"], "session_uid": center["session_id"], "patient_uid": center["patient_id"],
            "t1w": path(t1w), "t1ce": path(t1ce), "t2w": path(t2w), "flair": path(flair),
            "centered_image": center["nii_path"], "center_is_3d": bool(center["is_3d"]),
            "center_slice_mm": center["SliceThickness"], "t1ce_source": source}


def select(a):
    series = load_series(a.manifest, anatomy_filter=not a.no_anatomy_filter)
    pairs = pd.read_csv(a.pairs, dtype={"study_uid1": str, "study_uid2": str})
    wanted = pd.concat([pairs[["site", "study_uid1"]].rename(columns={"study_uid1": "session_uid"}),
                        pairs[["site", "study_uid2"]].rename(columns={"study_uid2": "session_uid"})]).drop_duplicates()
    series = series[series["session_key"].isin(set(wanted["site"] + "/" + wanted["session_uid"]))].copy()
    print(f"Pairs {len(pairs):,}; sessions with imaging on disk {series['session_key'].nunique():,}")
    # contrast flag from the session's own report (reachable via either pair column)
    exam = pd.concat([pairs[["site", "study_uid1", "exam_type1"]].rename(columns={"study_uid1": "session_uid", "exam_type1": "exam_type"}),
                      pairs[["site", "study_uid2", "exam_type2"]].rename(columns={"study_uid2": "session_uid", "exam_type2": "exam_type"})]
                     ).drop_duplicates(subset=["site", "session_uid"])
    flag = dict(zip(exam["site"] + "/" + exam["session_uid"], exam["exam_type"].map(has_contrast)))
    rows = [r for k, g in series.groupby("session_key", sort=False) if (r := select_session(g, flag.get(k, False))) is not None]
    selected = pd.DataFrame(rows)
    have = set(selected["site"] + "/" + selected["session_uid"])
    keep = pairs.apply(lambda r: f"{r['site']}/{r['study_uid1']}" in have and f"{r['site']}/{r['study_uid2']}" in have, axis=1)
    final_pairs = pairs[keep].copy()
    used = set(final_pairs["site"] + "/" + final_pairs["study_uid1"]) | set(final_pairs["site"] + "/" + final_pairs["study_uid2"])
    selected = selected[(selected["site"] + "/" + selected["session_uid"]).isin(used)]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(a.output, index=False)
    final_pairs.to_csv(a.output_pairs, index=False)
    print(f"pairs {len(pairs):,} -> both timepoints selected {len(final_pairs):,}; sessions {len(selected):,}; "
          f"patients {final_pairs['patient_uid'].nunique():,}")
    print(f"t1ce_source by site:\n{pd.crosstab(selected['site'], selected['t1ce_source'])}")


def select3d(a):
    """Report pairs whose two centre images are 3D with every voxel side <= 1.5 mm, and their sessions' selection
    rows."""
    sel = pd.read_csv(a.data, dtype=str)
    pairs = pd.read_csv(a.pairs, dtype=str)
    m = pd.read_csv(a.manifest, usecols=["nii_path", "spacing_mm"], dtype=str)
    sp = dict(zip(m.nii_path, m.spacing_mm))
    vox = sel.centered_image.map(lambda x: max(ast.literal_eval(sp[x])) if isinstance(sp.get(x), str) else np.inf)
    is3d = (sel.center_is_3d == "True") & (vox <= MAX_IN_PLANE_MM)
    ok = set(zip(sel.site[is3d], sel.session_uid[is3d]))
    p3 = pairs[[(s, x) in ok and (s, y) in ok for s, x, y in zip(pairs.site, pairs.study_uid1, pairs.study_uid2)]]
    used = set(zip(p3.site, p3.study_uid1)) | set(zip(p3.site, p3.study_uid2))
    s3 = sel[[k in used for k in zip(sel.site, sel.session_uid)]]
    s3.to_csv(a.output, index=False)
    p3.to_csv(a.output_pairs, index=False)
    print(f"3D-centre subset: {len(p3):,} pairs, {len(s3):,} sessions")


# ------------------------------------------------------------------ image-only pairs
# Report-pipeline rules except "both reports present": brain MR from the report Type, else DICOM StudyDescription;
# series selected BEFORE pairing, so neighbours of an unusable session still pair; adjacent dates >= 1 day apart.
def imageonly_site_sessions(site, series):
    s = series[series["site"] == site]
    sess = s.groupby("session_id").agg(patient_id=("patient_id", "first"),
                                       study_desc=("StudyDescription", "first")).reset_index()
    sess = sess.rename(columns={"session_id": "session_uid"})
    reports = pd.read_csv(f"{BIND_ROOT}/Imaging/{site}/Clinical/{site}_reports.csv", dtype=str, usecols=["session_id", "Type"])
    reports["session_uid"] = canonical_session_series(reports["session_id"], site)
    exam = reports.dropna(subset=["Type"]).drop_duplicates("session_uid").set_index("session_uid")["Type"]
    exam = exam.str.upper().str.strip()
    exam = exam[~exam.str.lower().isin(_MISSING)]
    sess = sess.merge(load_dates(site), on="session_uid", how="left")
    sess["exam_type"] = sess["session_uid"].map(exam)
    from_report = sess["exam_type"].notna() & (sess["exam_type"] != "")
    sess["brain_source"] = from_report.map({True: "report_type", False: "study_description"})
    label = sess["exam_type"].where(from_report, sess["study_desc"].fillna("").str.upper())
    sess["is_brain"] = label.map(is_mr_brain)
    sess["has_contrast"] = label.map(has_contrast)
    sess["patient_uid"] = canonical_patient_series(sess["patient_id"], load_patient_merge_map(site))
    sess["site"] = site
    print(f"  {site}: {len(sess):,} imaged sessions | dated {sess.study_date.notna().sum():,} | brain {sess.is_brain.sum():,} "
          f"(by report Type {(sess.is_brain & from_report).sum():,}, by StudyDescription {(sess.is_brain & ~from_report).sum():,})")
    return sess[sess["is_brain"] & sess["study_date"].notna()]


def imageonly(a):
    series = load_series(a.manifest)
    sessions = pd.concat([imageonly_site_sessions(site, series) for site in SITES], ignore_index=True)
    if a.report_type_only:
        sessions = sessions[sessions["brain_source"] == "report_type"]
    skey = sessions["site"] + "/" + sessions["session_uid"]
    flag = dict(zip(skey, sessions["has_contrast"]))
    rows = [r for k, g in series[series["session_key"].isin(set(skey))].groupby("session_key", sort=False)
            if (r := select_session(g, flag[k])) is not None]
    selected = pd.DataFrame(rows)
    key = selected["site"] + "/" + selected["session_uid"]
    selected["patient_uid"] = key.map(dict(zip(skey, sessions["patient_uid"])))
    print(f"brain MR sessions with a date: {len(sessions):,}; with a usable T1w: {len(selected):,}")
    meta = sessions.set_index(skey)
    selected["study_date"] = key.map(meta["study_date"])
    selected["brain_source"] = key.map(meta["brain_source"])
    s = selected.sort_values(["patient_uid", "study_date", "session_uid"]).drop_duplicates(subset=["patient_uid", "study_date"], keep="first")
    pairs = []
    for patient_uid, g in s.groupby("patient_uid", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            p, c = g.iloc[i], g.iloc[i + 1]
            pairs.append(dict(patient_uid=patient_uid, site=p["site"], study_uid1=p["session_uid"],
                              study_uid2=c["session_uid"], duration=(c["study_date"] - p["study_date"]).days,
                              brain_source1=p["brain_source"], brain_source2=c["brain_source"]))
    pairs = pd.DataFrame(pairs)
    pairs = pairs[pairs["duration"] >= a.min_duration]
    used = set(pairs["site"] + "/" + pairs["study_uid1"]) | set(pairs["site"] + "/" + pairs["study_uid2"])
    out = selected[key.isin(used)].drop(columns=["study_date", "brain_source"])
    tag = "_reporttype" if a.report_type_only else ""
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out_dir / f"bind_imageonly_data{tag}.csv", index=False)
    pairs.to_csv(a.out_dir / f"bind_imageonly_pairs{tag}.csv", index=False)
    print(f"image-only: {len(pairs):,} pairs / {pairs['patient_uid'].nunique():,} patients, {len(used):,} sessions; "
          f"per site {pairs['site'].value_counts().to_dict()}; StudyDescription-only brain flag "
          f"{((pairs.brain_source1 == 'study_description') | (pairs.brain_source2 == 'study_description')).sum():,}")


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    p = sub.add_parser("manifest")
    p.add_argument("--sites", nargs="+", default=list(SITES))
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--limit", type=int, default=None, help="first N subjects per site (smoke test)")
    p.add_argument("--output", type=Path, default=MANIFEST_CSV)
    p = sub.add_parser("pairs")
    p.add_argument("--sites", nargs="+", default=list(SITES))
    p.add_argument("--output", type=Path, default=META_CSV)
    p.add_argument("--min-duration", type=int, default=1, help="drop pairs closer than this many days")
    p = sub.add_parser("select")
    p.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    p.add_argument("--pairs", type=Path, default=META_CSV)
    p.add_argument("--output", type=Path, default=DATA_CSV)
    p.add_argument("--output-pairs", type=Path, default=PAIRED_CSV)
    p.add_argument("--no-anatomy-filter", action="store_true",
                   help="the 2026-08-04 rule (no in-plane / n_volumes filter): reproduces the report-pair tables the paper used")
    p = sub.add_parser("select3d")
    p.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    p.add_argument("--data", type=Path, default=DATA_CSV)
    p.add_argument("--pairs", type=Path, default=PAIRED_CSV)
    p.add_argument("--output", type=Path, default=TABLES / "bind_longitudinal_data_3d.csv")
    p.add_argument("--output-pairs", type=Path, default=TABLES / "bind_longitudinal_meta_paired_3d.csv")
    p = sub.add_parser("imageonly")
    p.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    p.add_argument("--out-dir", type=Path, default=TABLES)
    p.add_argument("--min-duration", type=int, default=1)
    p.add_argument("--report-type-only", action="store_true", help="ablation: no StudyDescription fallback")
    p = sub.add_parser("skullstrip")
    p.add_argument("--data-csv", type=Path, default=DATA_CSV)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--device", default="0", help="GPU index, or 'cpu'")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mode", default="accurate", choices=["accurate", "fast"], help="'accurate' = MR-RATE's 5-model ensemble")
    p = sub.add_parser("register")
    p.add_argument("--data-csv", type=Path, default=DATA_CSV)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 8))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--follow", action="store_true", help="run alongside skullstrip; needs --mask-shards")
    p.add_argument("--mask-shards", type=int, default=None, help="--num-shards the concurrent skullstrip run was given")
    p = sub.add_parser("retry")
    p.add_argument("--data-csv", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--log", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "register" and a.follow and not a.mask_shards:
        ap.error("--follow needs --mask-shards")
    if a.cmd in ("skullstrip", "register", "retry"):
        import bind_register
        return getattr(bind_register, a.cmd)(a)
    {"selftest": lambda a: selftest(), "manifest": manifest, "pairs": report_pairs, "select": select,
     "select3d": select3d, "imageonly": imageonly}[a.cmd](a)


if __name__ == "__main__":
    main()
