# Data terms

## What this repository contains

Code, prompts, configs, one model checkpoint (downloaded separately), aggregate results and a small per-pair numeric cache.
It contains **no radiology report text, no labels derived from reports, no patient, study or pair identifiers and no images.**

| Data | Source | Redistributed here? |
|---|---|---|
| BIND brain MRI and reports (training, internal and external test) | health-system data | No |
| MR-RATE brain MRI and reports | health-system data | No |
| MIMIC-CXR (chest X-ray rows of Table 2) | PhysioNet, credentialed access | No |
| LLM-rewritten report targets, Rad-Δ change labels, reader packets | derived from the above | No |
| `braindiff/paper/cache/` | per-pair rg_er and F1temp for each Table 2 system, a patient-group index for the bootstrap, opaque row order | Yes |
| `braindiff/paper/table2_golden.json` | aggregate Table 2 cells | Yes |

The cache holds no reference-report labels (no Rad-Δ status flags, no change / stable split), so macro-MCC, the agreement columns and
the reader cells cannot be recomputed from it; `paper/table2.py --full` recomputes them for anyone with the data.

## Prompts

The LLM prompts are shipped verbatim except one, the prior-report edit prompt, whose worked examples were replaced with synthetic text
because the original contained text from a real report ([`PROMPT_CHANGES.md`](PROMPT_CHANGES.md)). Every prompt was checked sentence by
sentence against the report corpora used in the project.
