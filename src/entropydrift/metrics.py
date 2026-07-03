"""Shape-vs-magnitude metrics over a set of per-problem trajectory records.

Reproduces the two headline analyses from Zhao (2026):
  - the shape (binary monotonicity) vs magnitude (scalar coherence) dissociation, and
  - the graded violation-count signal.

Consumes a list of records, each a dict with at least:
  monotone: bool, violations: int, coherence: float, correct: bool
"""

from __future__ import annotations

from typing import Sequence

from scipy import stats

from .stats import bootstrap_ci


def _acc(records: Sequence[dict]) -> float:
    n = len(records)
    return sum(1 for r in records if r["correct"]) / n if n else 0.0


def shape_signal(records: Sequence[dict]) -> dict:
    """Monotone vs non-monotone accuracy, the gap, and a Fisher-exact odds ratio + p."""
    mono = [r for r in records if r["monotone"]]
    non = [r for r in records if not r["monotone"]]
    mono_correct = sum(1 for r in mono if r["correct"])
    non_correct = sum(1 for r in non if r["correct"])
    # 2x2: rows = monotone/non-monotone, cols = correct/incorrect
    table = [
        [mono_correct, len(mono) - mono_correct],
        [non_correct, len(non) - non_correct],
    ]
    odds_ratio, p = stats.fisher_exact(table, alternative="greater")
    mono_acc = _acc(mono)
    non_acc = _acc(non)
    return {
        "n_monotone": len(mono),
        "n_non_monotone": len(non),
        "monotone_acc": mono_acc,
        "non_monotone_acc": non_acc,
        "gap_pp": (mono_acc - non_acc) * 100.0,
        "odds_ratio": float(odds_ratio),
        "p_value": float(p),
        "monotone_coverage": len(mono) / len(records) if records else 0.0,
        "false_positive_rate": 1.0 - mono_acc if mono else 0.0,
    }


def magnitude_signal(records: Sequence[dict]) -> dict:
    """Correlation of coherence magnitude with correctness (expected: near zero)."""
    coh = [r["coherence"] for r in records]
    correct = [1 if r["correct"] else 0 for r in records]
    if len(set(coh)) < 2 or len(set(correct)) < 2:
        return {"spearman_rho": 0.0, "p_value": 1.0}
    rho, p = stats.spearmanr(coh, correct)
    return {"spearman_rho": float(rho), "p_value": float(p)}


def violation_signal(records: Sequence[dict], max_bucket: int = 3) -> dict:
    """Accuracy by violation count and the Spearman rho of violations vs correctness."""
    buckets: dict[str, list[dict]] = {}
    for r in records:
        v = r["violations"]
        key = f">={max_bucket}" if v >= max_bucket else str(v)
        buckets.setdefault(key, []).append(r)

    def _order(k: str) -> int:
        return max_bucket if k.startswith(">=") else int(k)

    table = {k: _acc(buckets[k]) for k in sorted(buckets, key=_order)}
    counts = {k: len(buckets[k]) for k in sorted(buckets, key=_order)}

    v = [r["violations"] for r in records]
    correct = [1 if r["correct"] else 0 for r in records]
    if len(set(v)) < 2 or len(set(correct)) < 2:
        rho, p = 0.0, 1.0
    else:
        rho, p = stats.spearmanr(v, correct)
    return {
        "accuracy_by_violations": table,
        "counts_by_violations": counts,
        "spearman_rho": float(rho),
        "p_value": float(p),
    }


def _gap_pp(records: Sequence[dict]) -> float:
    return shape_signal(records)["gap_pp"]


def _magnitude_rho(records: Sequence[dict]) -> float:
    return magnitude_signal(records)["spearman_rho"]


def _violation_rho(records: Sequence[dict]) -> float:
    return violation_signal(records)["spearman_rho"]


def bootstrap_summary(
    records: Sequence[dict], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """95% bootstrap CIs (pre-registered primary inference) for the headline statistics."""
    return {
        "shape_gap_pp": bootstrap_ci(records, _gap_pp, n_boot, alpha, seed),
        "magnitude_rho": bootstrap_ci(records, _magnitude_rho, n_boot, alpha, seed + 1),
        "violation_rho": bootstrap_ci(records, _violation_rho, n_boot, alpha, seed + 2),
    }


def summarize(
    records: Sequence[dict], n_boot: int = 0, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Full metrics bundle for a run. If ``n_boot`` > 0, include bootstrap CIs."""
    out = {
        "n": len(records),
        "overall_acc": _acc(records),
        "shape": shape_signal(records),
        "magnitude": magnitude_signal(records),
        "violations": violation_signal(records),
    }
    if n_boot:
        out["ci"] = bootstrap_summary(records, n_boot=n_boot, alpha=alpha, seed=seed)
    return out


def format_report(summary: dict) -> str:
    """Human-readable one-screen report."""
    s, m, v = summary["shape"], summary["magnitude"], summary["violations"]
    lines = [
        f"n={summary['n']}  overall_acc={summary['overall_acc']:.3f}",
        "",
        "SHAPE (binary monotonicity)",
        f"  monotone     acc={s['monotone_acc']:.3f}  (n={s['n_monotone']})",
        f"  non-monotone acc={s['non_monotone_acc']:.3f}  (n={s['n_non_monotone']})",
        f"  gap={s['gap_pp']:+.1f} pp   OR={s['odds_ratio']:.2f}  p={s['p_value']:.2g}",
        f"  coverage={s['monotone_coverage']:.3f}  false_pos_rate={s['false_positive_rate']:.3f}",
        "",
        "MAGNITUDE (coherence = H_0 - H_N)",
        f"  spearman_rho={m['spearman_rho']:+.3f}  p={m['p_value']:.2g}   (expected: ~0)",
        "",
        "VIOLATIONS (graded)",
    ]
    for k in v["accuracy_by_violations"]:
        acc = v["accuracy_by_violations"][k]
        cnt = v["counts_by_violations"][k]
        lines.append(f"  {k:>3}: acc={acc:.3f}  (n={cnt})")
    lines.append(
        f"  spearman_rho={v['spearman_rho']:+.3f}  p={v['p_value']:.2g}"
    )
    if "ci" in summary:
        ci = summary["ci"]
        lines += [
            "",
            "95% BOOTSTRAP CI (pre-registered primary inference)",
            f"  shape gap (pp): {ci['shape_gap_pp']['point']:+.1f}"
            f"  [{ci['shape_gap_pp']['lo']:+.1f}, {ci['shape_gap_pp']['hi']:+.1f}]",
            f"  magnitude rho:  {ci['magnitude_rho']['point']:+.3f}"
            f"  [{ci['magnitude_rho']['lo']:+.3f}, {ci['magnitude_rho']['hi']:+.3f}]",
            f"  violation rho:  {ci['violation_rho']['point']:+.3f}"
            f"  [{ci['violation_rho']['lo']:+.3f}, {ci['violation_rho']['hi']:+.3f}]",
        ]
    return "\n".join(lines)
