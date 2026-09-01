"""The probes are evaluation instruments, not BrainDiff-specific scripts.

They must (a) reproduce the paper's published values from the shipped caches, and (b) accept a
plain JSON contract so any system's generations can be scored the same way.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / "paper" / "cache" / "perreport"


def _run(script, *args):
    r = subprocess.run([sys.executable, str(ROOT / "paper" / "probes" / script), *map(str, args)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_factorial_reproduces_table6():
    out = _run("factorial_probe.py",
               "--present-own", CACHE / "C_braindiff_real.json",
               "--present-other", CACHE / "C_braindiff_visroll.json",
               "--withheld-own", CACHE / "C_braindiff_noreport.json",
               "--withheld-other", CACHE / "C_braindiff_nr_visroll.json")
    for value in ["0.3837", "0.3450", "0.2600", "0.2057",      # the four cells
                  "+0.0387", "+0.0544",                         # image effects
                  "+0.1236", "+0.1393",                         # prior-report effects
                  "-0.0157"]:                                   # interaction
        assert value in out, f"Table 6 value {value} not reproduced"


def test_reversal_accepts_the_generic_contract(tmp_path):
    """A synthetic three-case file: any system can emit this shape."""
    prog = "Impression: There is interval progression of the lesion."
    impr = "Impression: Interval decrease in lesion size."
    rows = [
        {"gt": 3, "fwd": prog, "rev": impr},   # asserted direction reverses -> flip
        {"gt": 4, "fwd": impr, "rev": impr},   # unchanged under reversal    -> hold
        {"gt": 0, "fwd": prog, "rev": impr},   # Stable: not a directional case, excluded
    ]
    p = tmp_path / "sys.json"
    p.write_text(json.dumps(rows))
    out = _run("reversal_probe.py", "--reports", p, "--labels", "mysystem")
    assert "mysystem" in out
    # gt=0 is not a directional change case, so two cases remain; one flips.
    assert "0.5000" in out, out


def test_reversal_flip_definition_is_index_aligned(tmp_path):
    """Uninformative rows stay in the array as NaN so two systems compare pair-by-pair."""
    sys.path.insert(0, str(ROOT / "paper" / "probes"))
    from reversal_probe import flip_array
    from braindiff.eval.temporal_score import change_class_v2
    rows = [{"gt": 3, "fwd": "Impression: There is interval progression of the lesion.",
             "rev": "Impression: Interval decrease in lesion size."},
            {"gt": 3, "fwd": "Impression: Findings are indeterminate.",
             "rev": "Impression: Interval decrease in lesion size."}]
    f = flip_array(rows, change_class_v2)
    assert len(f) == 2 and f[0] == 1.0 and np.isnan(f[1])


@pytest.mark.parametrize("arm,cached", [("nocf", "reversal_nocf_v2"),
                                        ("cfdropout", "reversal_cfdropout_v2")])
def test_cached_flip_arrays_are_well_formed(arm, cached):
    f = np.array(json.load(open(CACHE / f"{cached}.json"))["flips"], float)
    assert len(f) == 503, "one entry per ground-truth directional change case"
    assert set(np.unique(f[np.isfinite(f)])) <= {0.0, 1.0}


# ---------------------------------------------------------------- change classifier

def _truth_and_pred(tmp_path):
    import csv as _csv
    _csv.field_size_limit(10 ** 7)
    pred_all = json.load(open(ROOT / "paper" / "cache" / "confusion" / "pred_v2.json"))
    rows = list(_csv.DictReader(open(ROOT / "paper" / "cache" / "s4_test.csv")))
    truth = {r["study_uid2"]: r["classification"] for r in rows if r.get("classification")}
    p, t = tmp_path / "pred.json", tmp_path / "true.json"
    p.write_text(json.dumps(pred_all["bd"]))
    t.write_text(json.dumps(truth))
    return p, t


def _run_cls(*args):
    r = subprocess.run([sys.executable,
                        str(ROOT / "paper" / "probes" / "change_classifier.py"), *map(str, args)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"change_classifier failed:\n{r.stdout}\n{r.stderr}"
    return r.stdout


def test_classifier_reproduces_figure3a(tmp_path):
    pred, truth = _truth_and_pred(tmp_path)
    out = _run_cls("--pred", pred, "--labels", truth)
    assert "0.4166" in out, f"Figure 3(a) balanced accuracy not reproduced:\n{out}"
    for recall in ["0.653", "0.657", "0.270", "0.085"]:
        assert recall in out, f"per-class recall {recall} missing:\n{out}"
    assert "n = 1444" in out


def test_classifier_classifies_raw_reports(tmp_path):
    reports = [
        {"id": "a", "report": "Impression: There is interval progression of the lesion."},
        {"id": "b", "report": "Impression: Interval decrease in lesion size."},
    ]
    p = tmp_path / "r.json"
    p.write_text(json.dumps(reports))
    out_file = tmp_path / "pred.json"
    _run_cls("--reports", p, "--out", out_file)
    got = json.loads(out_file.read_text())
    assert got == {"a": "Progressed", "b": "Improved"}, got


def test_classifier_accepts_a_bare_list_of_strings(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(["Impression: Interval decrease in lesion size."]))
    out_file = tmp_path / "pred.json"
    _run_cls("--reports", p, "--out", out_file)
    assert json.loads(out_file.read_text()) == {"0": "Improved"}


@pytest.mark.parametrize("version", ["v1", "v2", "v5"])
def test_all_three_rule_sets_run(tmp_path, version):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(["Impression: There is interval progression of the lesion."]))
    out = _run_cls("--reports", p, "--classifier", version)
    assert f"change_class_{version}" in out
