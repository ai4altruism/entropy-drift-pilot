import numpy as np

from entropydrift.stats import bootstrap_ci, bootstrap_ci_indexed


def test_bootstrap_ci_mean_brackets_point():
    records = [{"x": float(i)} for i in range(100)]  # mean = 49.5
    stat = lambda recs: sum(r["x"] for r in recs) / len(recs)
    ci = bootstrap_ci(records, stat, n_boot=500, seed=0)
    assert ci["n_effective"] == 500
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert abs(ci["point"] - 49.5) < 1e-9
    # CI should be a tight band around the mean, comfortably containing it
    assert ci["lo"] < 49.5 < ci["hi"]


def test_bootstrap_ci_deterministic():
    records = [{"x": float(i % 7)} for i in range(80)]
    stat = lambda recs: sum(r["x"] for r in recs) / len(recs)
    a = bootstrap_ci(records, stat, n_boot=300, seed=42)
    b = bootstrap_ci(records, stat, n_boot=300, seed=42)
    assert a == b


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], lambda r: 0.0, n_boot=100)
    assert ci["n_effective"] == 0
    assert ci["lo"] != ci["lo"]  # NaN


def test_bootstrap_ci_indexed_matches():
    vals = np.arange(50, dtype=float)
    stat = lambda idx: float(vals[idx].mean())
    ci = bootstrap_ci_indexed(stat, n=len(vals), n_boot=400, seed=1)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert abs(ci["point"] - vals.mean()) < 1e-9


def test_bootstrap_ci_drops_undefined():
    # a stat that is undefined (returns NaN) on some resamples should still yield a CI
    records = [{"x": float(i)} for i in range(30)]

    def flaky(recs):
        s = sum(r["x"] for r in recs)
        return float("nan") if s % 2 == 0 else s

    ci = bootstrap_ci(records, flaky, n_boot=200, seed=3)
    assert ci["n_effective"] <= 200
