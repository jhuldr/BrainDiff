"""Run notifications. The release ships a no-op; the development version posted to Slack.

Kept as a module rather than deleted so the trainers' call sites stay unchanged.
"""


def send(*_args, **_kwargs):
    """Accept and ignore. Override this module to wire up your own notifier."""
    return None
