"""Bootstrap confidence intervals over problems.

The pre-registration names 95% percentile bootstrap CIs (resampling problems with
replacement) as the primary inferential object, with p-values secondary. This module
provides one generic helper used by both metrics.py and fpreduce.py.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def bootstrap_ci(
    records: Sequence,
    stat_fn: Callable[[Sequence], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for a scalar statistic of ``records``.

    Resamples ``records`` with replacement ``n_boot`` times, recomputes ``stat_fn`` on each
    resample, and returns the point estimate (on the full sample) plus the [alpha/2,
    1-alpha/2] quantiles. Resamples on which ``stat_fn`` is undefined (returns None/NaN or
    raises) are dropped; ``n_effective`` reports how many survived.
    """
    point = stat_fn(records)
    n = len(records)
    if n == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_effective": 0}
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [records[i] for i in idx]
        try:
            v = stat_fn(sample)
        except Exception:
            continue
        if _finite(v):
            boots.append(float(v))
    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_effective": 0}
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(point) if _finite(point) else float("nan"),
        "lo": float(lo),
        "hi": float(hi),
        "n_effective": len(boots),
    }


def bootstrap_ci_indexed(
    stat_fn: Callable[[np.ndarray], float],
    n: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Bootstrap CI where ``stat_fn`` consumes an array of resampled row indices.

    Useful when the statistic is computed from several parallel arrays (e.g. predicted
    probabilities and labels) rather than a single record list.
    """
    point = stat_fn(np.arange(n))
    if n == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_effective": 0}
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = stat_fn(idx)
        except Exception:
            continue
        if _finite(v):
            boots.append(float(v))
    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_effective": 0}
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(point) if _finite(point) else float("nan"),
        "lo": float(lo),
        "hi": float(hi),
        "n_effective": len(boots),
    }
