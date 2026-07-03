"""False-positive reduction: a learned triage filter over cheap trajectory signals.

The bare rule "trust the answer if the entropy trajectory is monotone" carries a high
false-positive rate (monotone but wrong). Contribution 3 of the pilot asks whether a
small learned combination of signals (violation-count + final-answer confidence, by
default) does better: at the same coverage, a lower false-positive rate.

This module is deliberately dependency-light: a self-contained L2-regularized logistic
regression (numpy only), a stratified split for honest out-of-sample estimates, and a
coverage-vs-selective-accuracy curve compared against the monotonicity operating point.
Swap in scikit-learn later if desired; the interface is small.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .stats import bootstrap_ci_indexed

DEFAULT_FEATURES = ("violations", "final_confidence")


class LogisticRegression:
    """L2-regularized logistic regression via full-batch gradient descent (numpy only)."""

    def __init__(self, l2: float = 1.0, lr: float = 0.5, epochs: int = 2000):
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y) -> "LogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = (X - self.mean_) / self.std_
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = _sigmoid(Xs @ self.w + self.b)
            err = p - y
            grad_w = Xs.T @ err / n + self.l2 * self.w / n
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        return _sigmoid(Xs @ self.w + self.b)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# --------------------------------------------------------------- data helpers


def features_matrix(records: Sequence[dict], feature_names: Sequence[str]):
    """Build (X, y) from run records; bool features (e.g. 'monotone') become 0/1."""
    X = [[float(r[f]) for f in feature_names] for r in records]
    y = [1 if r["correct"] else 0 for r in records]
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def stratified_split(y, test_frac: float = 0.4, seed: int = 0):
    """Indices for a class-stratified train/test split."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train: list[int] = []
    test: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * test_frac))
        test.extend(idx[:k].tolist())
        train.extend(idx[k:].tolist())
    return sorted(train), sorted(test)


# --------------------------------------------------------------- triage analysis


def triage_curve(p_correct, y, thresholds=None) -> list[dict]:
    """Coverage vs selective accuracy as the accept-threshold on P(correct) sweeps up."""
    p = np.asarray(p_correct, dtype=float)
    y = np.asarray(y, dtype=int)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    curve = []
    for t in thresholds:
        accepted = p >= t
        cov = float(accepted.mean())
        sel = float(y[accepted].mean()) if accepted.any() else float("nan")
        curve.append(
            {
                "threshold": float(t),
                "coverage": cov,
                "selective_acc": sel,
                "false_pos_rate": (1.0 - sel) if accepted.any() else float("nan"),
                "n_accepted": int(accepted.sum()),
            }
        )
    return curve


def monotone_baseline(records: Sequence[dict]) -> dict:
    """The 'trust if monotone' operating point: its coverage, selective accuracy, FP rate."""
    mono = [r for r in records if r["monotone"]]
    n = len(records)
    if not mono:
        return {"coverage": 0.0, "selective_acc": float("nan"), "false_pos_rate": float("nan")}
    acc = sum(1 for r in mono if r["correct"]) / len(mono)
    return {
        "coverage": len(mono) / n if n else 0.0,
        "selective_acc": acc,
        "false_pos_rate": 1.0 - acc,
    }


def at_coverage(p_correct, y, target_coverage: float) -> dict:
    """Learned filter's selective accuracy when its coverage is matched to a target.

    Note: with small m, predicted probabilities are heavily tied (final_confidence takes
    only m+1 values), so the quantile threshold can over- or under-shoot the target
    coverage. Report the full ``triage_curve`` as the primary artifact; this matched-point
    comparison is a convenience summary, not a precise operating point.
    """
    p = np.asarray(p_correct, dtype=float)
    y = np.asarray(y, dtype=int)
    if target_coverage <= 0:
        return {"coverage": 0.0, "selective_acc": float("nan"), "false_pos_rate": float("nan")}
    thr = float(np.quantile(p, 1.0 - target_coverage))
    accepted = p >= thr
    cov = float(accepted.mean())
    sel = float(y[accepted].mean()) if accepted.any() else float("nan")
    return {
        "threshold": thr,
        "coverage": cov,
        "selective_acc": sel,
        "false_pos_rate": (1.0 - sel) if accepted.any() else float("nan"),
    }


def _fp_reduction(test_records: Sequence[dict], p, y) -> float:
    """FP-rate reduction of the learned filter over the monotonicity baseline, matched
    to the baseline's coverage. Model is treated as fixed; only the evaluation set varies."""
    baseline = monotone_baseline(test_records)
    matched = at_coverage(p, y, baseline["coverage"])
    if _isnan(baseline["false_pos_rate"]) or _isnan(matched["false_pos_rate"]):
        return float("nan")
    return baseline["false_pos_rate"] - matched["false_pos_rate"]


def evaluate(
    records: Sequence[dict],
    feature_names: Sequence[str] = DEFAULT_FEATURES,
    test_frac: float = 0.4,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict:
    """Fit the filter on a train split, evaluate out-of-sample, compare to the baseline.

    Headline number: false-positive-rate reduction of the learned filter over the
    monotonicity rule, holding coverage fixed to the baseline's coverage (all on the
    held-out test split), with a 95% bootstrap CI that resamples the test set (the model
    is held fixed, so the CI reflects evaluation-set sampling uncertainty).
    """
    X, y = features_matrix(records, feature_names)
    train_idx, test_idx = stratified_split(y, test_frac, seed)
    model = LogisticRegression().fit(X[train_idx], y[train_idx])
    p_test = model.predict_proba(X[test_idx])
    y_test = y[test_idx]
    test_records = [records[i] for i in test_idx]

    baseline = monotone_baseline(test_records)
    matched = at_coverage(p_test, y_test, baseline["coverage"])
    fp_reduction = _fp_reduction(test_records, p_test, y_test)

    def _stat(idx: np.ndarray) -> float:
        return _fp_reduction([test_records[i] for i in idx], p_test[idx], y_test[idx])

    ci = bootstrap_ci_indexed(_stat, n=len(test_idx), n_boot=n_boot, seed=seed + 7)
    ci_pp = {
        k: (v * 100.0 if isinstance(v, float) and not _isnan(v) else v)
        for k, v in ci.items()
        if k in ("point", "lo", "hi")
    }
    ci_pp["n_effective"] = ci["n_effective"]

    return {
        "features": list(feature_names),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "weights": dict(zip(feature_names, model.w.tolist())),
        "baseline_monotone": baseline,
        "learned_at_baseline_coverage": matched,
        "fp_rate_reduction_pp": fp_reduction * 100.0 if not _isnan(fp_reduction) else float("nan"),
        "fp_rate_reduction_ci_pp": ci_pp,
        "curve": triage_curve(p_test, y_test),
    }


def _isnan(x) -> bool:
    return isinstance(x, float) and x != x


def format_report(result: dict) -> str:
    b = result["baseline_monotone"]
    m = result["learned_at_baseline_coverage"]
    lines = [
        f"features={result['features']}  n_train={result['n_train']} n_test={result['n_test']}",
        f"weights={ {k: round(v, 3) for k, v in result['weights'].items()} }",
        "",
        "BASELINE (trust if monotone)",
        f"  coverage={b['coverage']:.3f}  selective_acc={b['selective_acc']:.3f}"
        f"  false_pos_rate={b['false_pos_rate']:.3f}",
        "",
        "LEARNED FILTER (matched to baseline coverage)",
        f"  coverage={m['coverage']:.3f}  selective_acc={m['selective_acc']:.3f}"
        f"  false_pos_rate={m['false_pos_rate']:.3f}",
        "",
        f"FALSE-POSITIVE REDUCTION: {result['fp_rate_reduction_pp']:+.1f} pp",
    ]
    ci = result.get("fp_rate_reduction_ci_pp")
    if ci and not _isnan(ci.get("lo", float("nan"))):
        lines.append(f"  95% bootstrap CI: [{ci['lo']:+.1f}, {ci['hi']:+.1f}] pp")
    return "\n".join(lines)
