from entropydrift.metrics import summarize


def _rec(monotone, violations, coherence, correct):
    return {
        "monotone": monotone,
        "violations": violations,
        "coherence": coherence,
        "correct": correct,
    }


def test_summary_shape_dissociation():
    # monotone chains mostly correct; non-monotone mostly wrong; coherence (magnitude)
    # varies the same way in both groups, so it must NOT track correctness.
    records = []
    for i in range(40):
        records.append(_rec(True, 0, 0.1 * (i % 7), correct=(i % 5 != 0)))  # 80% correct
    for i in range(40):
        records.append(_rec(False, 2, 0.1 * (i % 7), correct=(i % 5 == 0)))  # 20% correct
    s = summarize(records)
    assert s["n"] == 80
    assert s["shape"]["monotone_acc"] > s["shape"]["non_monotone_acc"]
    assert s["shape"]["gap_pp"] > 0
    assert s["shape"]["odds_ratio"] > 1
    # coherence (magnitude) should not track correctness here
    assert abs(s["magnitude"]["spearman_rho"]) < 0.5
    # violation buckets present
    assert "0" in s["violations"]["accuracy_by_violations"]


def test_summary_with_bootstrap_ci():
    records = []
    for i in range(60):
        records.append(_rec(True, 0, 0.1 * (i % 7), correct=(i % 5 != 0)))
    for i in range(60):
        records.append(_rec(False, 2, 0.1 * (i % 7), correct=(i % 5 == 0)))
    s = summarize(records, n_boot=200, seed=0)
    assert "ci" in s
    for key in ("shape_gap_pp", "magnitude_rho", "violation_rho"):
        ci = s["ci"][key]
        assert ci["lo"] <= ci["hi"]
        assert ci["n_effective"] > 0
    # the shape gap CI should sit on the positive side here
    assert s["ci"]["shape_gap_pp"]["lo"] > 0
