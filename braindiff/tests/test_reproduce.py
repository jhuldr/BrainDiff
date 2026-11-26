"""Paper Table 2 (brain block) re-derives from the shipped per-pair cache.

The cache check recomputes the rg_er and F1temp cells and diffs them against paper/table2_golden.json. The full check
recomputes every cell and needs the private data (`-m data`, BRAINDIFF_CKPT and READER_DIR set). CPU only.

    pytest tests/test_reproduce.py                  cache check + privacy check
    pytest -m data tests/test_reproduce.py          full check (needs the data)
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
GOLD = json.loads((PAPER / "table2_golden.json").read_text())


def _run(tmp_path, *args):
    out = tmp_path / "table2.json"
    r = subprocess.run([sys.executable, str(PAPER / "table2.py"), "--out", str(out), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"table2.py failed:\n{r.stdout}\n{r.stderr}"
    return json.loads(out.read_text())


def _diff(got, metrics=None):
    return [(part, sp, k, m) for part in ("cells", "sig") for sp in GOLD[part] for k, v in GOLD[part][sp].items() for m, x in v.items()
            if (metrics is None or m in metrics) and got[part][sp][k].get(m) != x]


def test_cache_has_no_labels():
    """Only patient groups and per-pair rg_er / F1temp, all numeric."""
    for f in (PAPER / "cache").glob("*.npz"):
        z = np.load(f, allow_pickle=False)
        assert set(z.files) == ({"group"} if f.name.startswith("groups_") else {"rger", "f1temp"}), f.name
        assert all(z[k].dtype.kind in "iuf" for k in z.files), f.name
    assert {p.suffix for p in (PAPER / "cache").iterdir()} <= {".npz", ".md"}


def test_table2_cache_cells(tmp_path):
    got = _run(tmp_path)
    assert len(GOLD["cells"]["test"]) == 9 and len(GOLD["sig"]["test"]) == 8
    n = sum(1 for part in ("cells", "sig") for sp in GOLD[part] for v in GOLD[part][sp].values() for m in v if m in ("rger", "f1temp"))
    assert n == 68 and not any(x.endswith(("/rger", "/f1temp")) for x in got["from_golden"])
    d = _diff(got, ("rger", "f1temp"))
    assert not d, f"{len(d)} recomputed cells drifted from the golden, e.g. {d[:5]}"


@pytest.mark.data
@pytest.mark.skipif(not (os.environ.get("BRAINDIFF_CKPT") and os.environ.get("READER_DIR")), reason="needs the private scorer outputs")
def test_table2_full(tmp_path):
    d = _diff(_run(tmp_path, "--full"))
    assert not d, f"{len(d)} Table 2 cells drifted from the golden, e.g. {d[:5]}"
