"""Patient-aware partner permutation, shared by training and evaluation.

Lives at repo root rather than in s4_oracle_diag so the trainer can import it too;
the eval harness keeps its own copy in ablate.py for the dumps it already produced.

Why patient-aware: 33% of S4 val rows share an 8-row batch with another visit of the
SAME patient (patient-grouped split, study order). Pairing a scan with its own series
is not a control and not a counterfactual -- the "wrong" prior is nearly right.
"""
import torch


def batch_partner_perm(patients):
    """Index permutation pairing each row with a DIFFERENT patient's row.

    Deterministic: the candidate furthest away in the batch, ties to the lower index.
    Rows with no valid partner map to themselves; the caller is told how many so a
    silently-inert control cannot pass as a real one.
    """
    n = len(patients)
    perm, skipped = list(range(n)), 0
    for i in range(n):
        cand = [j for j in range(n) if patients[j] != patients[i]]
        if not cand:
            skipped += 1
            continue
        perm[i] = max(cand, key=lambda j: (abs(j - i), -j))
    return torch.tensor(perm, dtype=torch.long), skipped
