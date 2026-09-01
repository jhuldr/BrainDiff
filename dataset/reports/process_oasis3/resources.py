#!/usr/bin/env python3
"""
How much of the machine this pipeline is allowed to take.

Kept in one place so the CPU phases, the QC pass and the retro-strip audit all
agree, and so the ITK thread count stays consistent with the worker count -- the
two interact, and getting them wrong makes the machine slower, not faster.

CPU: default to 75% of cores. On this box (128 cores) that is 96 workers.

ITK threads: ANTs spawns a full ITK thread pool per process. 96 workers x the
usual 2 threads is 192 threads on 128 cores -- oversubscribed, and the context
switching costs more than the second thread buys. So the thread count is derived
from the worker count rather than pinned: keep workers x threads at or under the
core count. process_bind hardcoded 2 because it ran 24 workers, where 48 threads
on 128 cores was comfortable. That constant does not survive going wide.

    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS must be set BEFORE `import ants`,
    which is why callers import this module first.
"""

import os

CPU_FRACTION = 0.75


def total_cores():
    """Cores this process may actually use, honouring CPU affinity if set."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 8


def default_workers(fraction=CPU_FRACTION):
    """75% of cores by default, at least 1."""
    return max(1, int(total_cores() * fraction))


def itk_threads_for(workers):
    """Threads per ANTs worker so that workers x threads <= cores.

    Returns at least 1. At 96 workers on 128 cores this is 1; at the 24 workers
    process_bind used it reproduces that script's 2.
    """
    return max(1, total_cores() // max(1, workers))


def configure_itk(workers=None):
    """Set the ITK thread cap for `workers`. Call BEFORE importing ants.

    Respects an explicit ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS in the environment
    so a caller can still override this from the shell.
    """
    workers = default_workers() if workers is None else workers
    threads = itk_threads_for(workers)

    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(threads))

    return threads


def describe(workers=None):
    workers = default_workers() if workers is None else workers
    threads = int(os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
                                 itk_threads_for(workers)))

    return (f"{total_cores()} cores -> {workers} workers x {threads} ITK thread(s) "
            f"= {workers * threads} threads")
