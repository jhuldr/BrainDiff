#!/usr/bin/env python3
"""
Run the S4 longitudinal report generation over the BIND external-validation set.

Same generator, prompt and model that produced S4's `generated_report` /
`classification`, pointed at the BIND pairs. Two documented differences from how
the S4 references were made:

  * `pathology1` is "unavailable" -- BIND has no pathology label source, where
    S4's came from MR-RATE's own label file.
  * `report1`/`report2` are BIND's own text, truncated to begin at the FINDINGS
    header (97.5% / 97.9% of rows), not MR-RATE's normalized prose.

Needs OPENAI_API_KEY in the environment. Submits an OpenAI Batch job (24h
window) and blocks polling until it finishes, then writes
data/bind_eval_800_reports.csv.

    conda activate CLIP
    OPENAI_API_KEY=... python -m process_bind.run_bind_batch

To pick a submitted batch back up after an interruption:

    OPENAI_API_KEY=... python -m process_bind.run_bind_batch --resume batch_xxx
"""
import argparse
import os
import sys

sys.path.insert(0, "/path/to/code/BrainDiff")

from process_mrrate.ReportRevision.generate_longitudional_report_batch import (
    LongitudinalReportBatchGenerator,
)

META = "/path/to/code/BrainDiff/process_bind/data/bind_eval_800.csv"
OUT = "/path/to/code/BrainDiff/process_bind/data"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default=META)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--batching", type=int, default=8,
                    help="cases per request; 8 is what S4 used")
    ap.add_argument("--resume", default=None, help="existing batch id to poll instead")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    generator = LongitudinalReportBatchGenerator()

    if args.resume:
        df = generator.resume_run(args.resume, args.meta, args.out)
    else:
        df = generator.run(args.meta, args.out, batching=args.batching)

    print(f"\nrows returned: {len(df)}")
    if "classification" in df:
        print(df["classification"].value_counts().to_dict())


if __name__ == "__main__":
    main()
