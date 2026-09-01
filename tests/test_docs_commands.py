"""Every command in the docs must point at something that exists.

The refactor broke six commands in docs/REPRODUCTION.md and nothing caught it: the paths lived
inside fenced code blocks, so a check for backticked file references never saw them. This walks
the shipped markdown, extracts each `python`/`pytest`/`torchrun` invocation, and resolves its
target -- a script path against the repo, a `-m` module against the import system.
"""
import importlib.util
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", ROOT / "paper" / "README.md",
        ROOT / "docs" / "REPRODUCTION.md", ROOT / "docs" / "PROBES.md",
        ROOT / "MODEL_LICENSE"]

# Placeholder inputs a reader supplies themselves -- the probes' JSON contracts.
USER_SUPPLIED = {"mysystem.json", "armA.json", "armB.json", "real.json", "vis_roll.json",
                 "noreport.json", "nr_vis_roll.json", "sys.json"}

# Historical documents: they describe the pre-release layout on purpose.
SKIP_FILES = {"CODE_MANIFEST.md", "RELEASE_PLAN.md", "DATA_TERMS.md"}


def _commands():
    for doc in DOCS:
        if not doc.exists() or doc.name in SKIP_FILES:
            continue
        for block in re.findall(r"```bash\n(.*?)```", doc.read_text(), re.S):
            joined = block.replace("\\\n", " ")
            for line in joined.splitlines():
                line = line.split("#")[0].strip()
                if line.startswith(("python", "torchrun", "pytest")):
                    yield doc.name, line


def _target(cmd):
    """-> ('module', name) or ('path', name) or None if there is nothing to resolve."""
    toks = cmd.split()
    if "-m" in toks:
        return "module", toks[toks.index("-m") + 1]
    for t in toks[1:]:
        if t.endswith(".py"):
            return "path", t
        if t.endswith(".json") and t not in USER_SUPPLIED and "/" in t:
            return "path", t
    return None


CASES = [(d, c) for d, c in _commands() if _target(c)]
assert CASES, "no commands were extracted -- the parser is broken, not the docs"


@pytest.mark.parametrize("doc,cmd", CASES, ids=[f"{d}:{c[:48]}" for d, c in CASES])
def test_command_target_resolves(doc, cmd):
    kind, name = _target(cmd)
    if kind == "path":
        assert (ROOT / name).is_file(), f"{doc}: '{cmd}' -> missing file {name}"
    else:
        assert importlib.util.find_spec(name) is not None, \
            f"{doc}: '{cmd}' -> unimportable module {name}"
