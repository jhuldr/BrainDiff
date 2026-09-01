"""The repository's central claim: the paper's numbers re-derive from the shipped caches.

Runs the two single-source scripts and diffs their output against committed goldens. CPU only,
no GPU, no corpora, a few seconds. If this fails, either a cache changed or a refactor moved a
path -- both are things a release must not do silently.

    pytest tests/test_reproduce.py
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATS = ROOT / "paper" / "stats"
GOLDEN = STATS / "golden"


def _values(path):
    """Numeric rows only -- prose and provenance lines are allowed to change."""
    out = []
    for line in pathlib.Path(path).read_text().splitlines():
        if re.match(r"^\| .+ \| [-+0-9.]", line):
            out.append(line)
    return out


def _run(script):
    r = subprocess.run([sys.executable, str(STATS / script)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"{script} failed:\n{r.stdout}\n{r.stderr}"


def test_paper_cis_reproduces():
    _run("all_cis.py")
    got, want = _values(STATS / "PAPER_CIS.md"), _values(GOLDEN / "PAPER_CIS.md")
    assert got, "no interval rows produced"
    assert got == want, "PAPER_CIS.md values drifted from the golden"


def test_paper_metrics_reproduces():
    _run("all_metrics.py")
    got, want = _values(STATS / "PAPER_METRICS.md"), _values(GOLDEN / "PAPER_METRICS.md")
    assert got, "no metric rows produced"
    assert got == want, "PAPER_METRICS.md values drifted from the golden"


def test_vocabularies_resolve():
    """Guards the failure mode that is silent: the module imports either way."""
    from braindiff.eval import temporal_score as t
    assert t.ANATOMY_PATH.is_file(), "anatomy vocabulary did not resolve as package data"
    assert t.PATHOLOGY_PATH.is_file(), "pathology vocabulary did not resolve as package data"
    assert len(t.CHANGE_LEXICON) == 41
