import numpy as np

from entropydrift.fpreduce import (
    LogisticRegression,
    at_coverage,
    evaluate,
    features_matrix,
    monotone_baseline,
    stratified_split,
    triage_curve,
)


def _make_records(n=400, seed=0):
    """Synthetic run records where confidence + violations genuinely predict correctness,
    and the monotone flag is a noisy proxy (so a learned filter can beat it)."""
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        conf = rng.random()  # final_confidence in [0,1)
        viol = rng.integers(0, 4)  # 0..3 violations
        # true correctness probability rises with confidence, falls with violations
        p = 1 / (1 + np.exp(-(3 * (conf - 0.5) - 0.8 * viol)))
        correct = rng.random() < p
        # monotone flag: a noisy thresholded proxy of the same signal
        monotone = (viol == 0) and (rng.random() < 0.85)
        records.append(
            {
                "monotone": bool(monotone),
                "violations": int(viol),
                "final_confidence": float(conf),
                "coherence": float(rng.random()),
                "correct": bool(correct),
            }
        )
    return records


def test_logreg_learns_separable():
    X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(epochs=3000).fit(X, y)
    proba = model.predict_proba(X)
    assert proba[0] < 0.5 < proba[-1]
    assert proba[-1] > proba[0]


def test_features_matrix_and_split():
    recs = _make_records(100)
    X, y = features_matrix(recs, ("violations", "final_confidence"))
    assert X.shape == (100, 2)
    train, test = stratified_split(y, test_frac=0.4, seed=1)
    assert len(train) + len(test) == 100
    assert set(train).isdisjoint(test)


def test_triage_curve_monotonic_in_threshold():
    p = np.linspace(0, 1, 100)
    y = (p > 0.5).astype(int)
    curve = triage_curve(p, y)
    covs = [pt["coverage"] for pt in curve]
    # coverage is non-increasing as threshold rises
    assert all(covs[i] >= covs[i + 1] for i in range(len(covs) - 1))


def test_at_coverage_matches_target():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    pt = at_coverage(p, y, target_coverage=0.3)
    assert abs(pt["coverage"] - 0.3) < 0.05


def test_learned_filter_reduces_false_positives():
    recs = _make_records(600, seed=2)
    result = evaluate(recs, feature_names=("violations", "final_confidence"), seed=2)
    base = result["baseline_monotone"]
    learned = result["learned_at_baseline_coverage"]
    # at matched coverage, the learned filter should not be worse and should typically
    # reduce the false-positive rate
    assert learned["coverage"] > 0
    assert result["fp_rate_reduction_pp"] >= 0.0


def test_monotone_baseline_fields():
    recs = _make_records(50)
    b = monotone_baseline(recs)
    assert 0.0 <= b["coverage"] <= 1.0
    assert set(b) == {"coverage", "selective_acc", "false_pos_rate"}
