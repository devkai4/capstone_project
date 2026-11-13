"""Shared eval helpers: per-label accuracy and latency percentiles."""

from __future__ import annotations

import math
from collections.abc import Sequence

LABELS = ("queue", "priority", "type")


def accuracy(truth: Sequence, preds: Sequence) -> dict[str, float]:
    """Exact-match accuracy per label + their mean. `preds` items may be None (= wrong).

    Both sides are objects with .queue/.priority/.type (Ticket, TicketLabels, ...).
    """
    if len(truth) != len(preds):
        raise ValueError("truth and preds must have the same length")
    n = len(truth)
    out = {}
    for label in LABELS:
        hits = sum(p is not None and getattr(t, label) == getattr(p, label)
                   for t, p in zip(truth, preds))
        out[label] = hits / n if n else 0.0
    out["mean"] = sum(out[label] for label in LABELS) / len(LABELS)
    return out


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100): the smallest value with >= q% of values at or below it."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]
