"""Entropy-trajectory construction and the shape signals.

The core of the Zhao (2026) protocol, operating on already-collected per-step answer
samples. Everything here is pure: no model, no I/O. Given, for each reasoning step, a
list of extracted answers (one per sampled completion), we compute the Shannon-entropy
trajectory and the shape statistics derived from it.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

Answer = str


def shannon_entropy(labels: Sequence[Answer], base: float | None = None) -> float:
    """Shannon entropy of the empirical distribution over ``labels``.

    Natural log by default (base=None); pass base=2 for bits. The monotonicity test and
    all shape statistics are invariant to the base, so it only affects reported units.
    Empty input has entropy 0.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    if base is not None:
        h /= math.log(base)
    return h


def entropy_trajectory(
    step_answers: Sequence[Sequence[Answer]], base: float | None = None
) -> list[float]:
    """Per-step entropy trajectory (H_0, ..., H_N) from per-step answer samples."""
    return [shannon_entropy(ans, base) for ans in step_answers]


def is_monotone(trajectory: Sequence[float], eps: float = 0.01) -> bool:
    """A trajectory is eps-monotone if entropy never rises by more than eps step-to-step.

    H_{k+1} <= H_k + eps for all k. A single violation makes the whole chain non-monotone.
    """
    return all(
        trajectory[k + 1] <= trajectory[k] + eps for k in range(len(trajectory) - 1)
    )


def violation_count(trajectory: Sequence[float], eps: float = 0.01) -> int:
    """Number of step transitions where entropy rises by more than eps (graded signal)."""
    return sum(
        1 for k in range(len(trajectory) - 1) if trajectory[k + 1] > trajectory[k] + eps
    )


def coherence(trajectory: Sequence[float]) -> float:
    """Total entropy reduction C = H_0 - H_N. The 'magnitude' summary (not predictive)."""
    if len(trajectory) < 2:
        return 0.0
    return trajectory[0] - trajectory[-1]


def max_positive_jump(trajectory: Sequence[float]) -> float:
    """Largest single step-to-step entropy increase (0 if none). Magnitude of the worst spike."""
    jumps = [
        trajectory[k + 1] - trajectory[k]
        for k in range(len(trajectory) - 1)
        if trajectory[k + 1] > trajectory[k]
    ]
    return max(jumps) if jumps else 0.0
