"""Analyses added for the write-up, all off-pod and CPU-only.

Every number here is recomputed from the stored per-step entropy trajectories, so no
GPU and no regeneration is involved. Four groups:

  tolerance  the pre-specified eps sweep over {0, 0.01, 0.05, 0.10}, plus the entropy
             grid that explains why the sweep is close to inert at m=5
  prefix     the pre-specified early-exit analysis over the first k transitions
  baselines  the original's own confounder control and scalar-baseline comparison,
             exploratory and outside the registration
  holm       Holm-Bonferroni over the sixteen-test panel-wide family of H1 and H3,
             which the paper keeps as an exploratory robustness check; the registered
             four-test family is in scripts/paper/registered_family.py

    python scripts/paper/paper_analyses.py [results-dir] [group ...]

``results-dir`` defaults to ``results``; with no group named, all four run.

It imports the scoring functions from the harness in ``src/`` so that the definitions
of monotonicity, violation count and the bootstrap are the registered ones rather than
reimplementations.

NOTE: ``len(trajectory)`` counts prefixes that yielded an extractable answer, not
segmentation units, so the length variable used in the confounder control is an
approximation to the original's chain length.
"""
import json
import os
import sys

import numpy as np
from scipy import optimize, stats

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, SRC)
from entropydrift.metrics import shape_signal, summarize, violation_signal  # noqa: E402
from entropydrift.stats import bootstrap_ci  # noqa: E402
from entropydrift.trajectory import is_monotone, violation_count  # noqa: E402

PANEL = [
    "qwen7b-gsm8k", "qwen7b-math500", "mistral7b-gsm8k", "mistral7b-math500",
    "llama31-8b-gsm8k", "llama31-8b-math500", "r1-distill-7b-gsm8k", "r1-distill-7b-math500",
]


def load(results, cell):
    """Usable records for a cell: deduplicated by index, status ok only."""
    seen, out = set(), []
    with open(os.path.join(results, cell, "records.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["index"] in seen or r.get("status") != "ok":
                continue
            seen.add(r["index"])
            out.append(r)
    return out


def rescore(records, eps, k=None):
    """Recompute the shape fields from the stored trajectory at a given eps and prefix."""
    out = []
    for r in records:
        t = r["trajectory"][: k + 1] if k is not None else r["trajectory"]
        if len(t) < 2:
            continue
        out.append({
            "monotone": is_monotone(t, eps),
            "violations": violation_count(t, eps),
            "coherence": t[0] - t[-1],
            "correct": r["correct"],
        })
    return out


def tolerance(results):
    print("=== eps sweep (primary eps = 0.01) ===")
    print(f"{'cell':<24} {'eps':>5} {'cov':>6} {'gap_pp':>8} {'gapCI':>18} {'OR':>6} {'violrho':>8}")
    for cell in PANEL:
        recs = load(results, cell)
        for eps in (0.0, 0.01, 0.05, 0.10):
            rs = rescore(recs, eps)
            sh = shape_signal(rs)
            ci = bootstrap_ci(rs, lambda x: shape_signal(x)["gap_pp"], 1000, 0.05, 0)
            print(f"{cell:<24} {eps:>5.2f} {sh['monotone_coverage']:>6.3f} {sh['gap_pp']:>+8.2f} "
                  f"[{ci['lo']:>+7.2f},{ci['hi']:>+7.2f}] {sh['odds_ratio']:>6.2f} "
                  f"{violation_signal(rs)['spearman_rho']:>+8.3f}")

    # Why the sweep is close to inert: the entropy grid an m=5 estimator can produce.
    vals, rises = {}, []
    for cell in PANEL:
        for r in load(results, cell):
            t = r["trajectory"]
            for h in t:
                vals[round(h, 10)] = vals.get(round(h, 10), 0) + 1
            rises += [b - a for a, b in zip(t, t[1:]) if b > a]
    grid = sorted(vals)
    print(f"\ndistinct entropy values panel-wide: {len(grid)}")
    print(f"minimum spacing: {min(b - a for a, b in zip(grid, grid[1:])):.6f} nats")
    print(f"step-to-step rises: {len(rises)}; at or below 0.10 nats: "
          f"{sum(1 for d in rises if d <= 0.10)} "
          f"({100.0 * sum(1 for d in rises if d <= 0.10) / len(rises):.1f}%)")


def prefix(results):
    print("=== early exit: first k transitions only, eps = 0.01 ===")
    print(f"{'cell':<24} {'k':>3} {'n':>5} {'cov':>6} {'gap_pp':>8} {'gapCI':>18} {'%full':>7}")
    for cell in PANEL:
        recs = load(results, cell)
        full = shape_signal(rescore(recs, 0.01))["gap_pp"]
        for k in (1, 2, 3, None):
            rs = rescore(recs, 0.01, k=k)
            sh = shape_signal(rs)
            ci = bootstrap_ci(rs, lambda x: shape_signal(x)["gap_pp"], 1000, 0.05, 0)
            frac = 100.0 * sh["gap_pp"] / full if full else float("nan")
            print(f"{cell:<24} {'all' if k is None else k:>3} {len(rs):>5} "
                  f"{sh['monotone_coverage']:>6.3f} {sh['gap_pp']:>+8.2f} "
                  f"[{ci['lo']:>+7.2f},{ci['hi']:>+7.2f}] {frac:>6.0f}%")


def _residual(y, x):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _logistic(X, y):
    X = np.column_stack([np.ones(len(y)), X])

    def nll(b):
        z = X @ b
        return np.sum(np.logaddexp(0, z) - y * z)

    b = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    return b, 1.0 / (1.0 + np.exp(-(X @ b)))


def _auroc(y, score):
    u = stats.mannwhitneyu(score[y == 1], score[y == 0], alternative="two-sided").statistic
    return u / (np.sum(y == 1) * np.sum(y == 0))


def baselines(results):
    """Zhao's A.7 confounder control and his Table 3 scalar-baseline comparison.

    Exploratory and outside the registration. Reported because both bear on how the
    confirmatory results should be read: the baseline ordering does not reproduce.
    """
    print("=== confounder control and scalar baselines (exploratory) ===")
    print(f"{'cell':<24} {'partial r':>10} {'p':>9} {'b_mono':>7} {'b_len':>7} {'b_Hn':>7} "
          f"{'AUROC mono':>11} {'viol':>7} {'-H_N':>7} {'C':>7}")
    for cell in PANEL:
        rs = load(results, cell)
        y = np.array([1.0 if r["correct"] else 0.0 for r in rs])
        mono = np.array([1.0 if r["monotone"] else 0.0 for r in rs])
        ln = np.array([float(len(r["trajectory"])) for r in rs])
        hn = np.array([float(r["final_entropy"]) for r in rs])
        viol = np.array([-float(r["violations"]) for r in rs])
        coh = np.array([float(r["coherence"]) for r in rs])
        r_p, p_p = stats.pearsonr(_residual(mono, ln), _residual(y, ln))

        def z(v):
            return (v - v.mean()) / (v.std() or 1.0)

        b, _ = _logistic(np.column_stack([z(mono), z(ln), z(hn)]), y)
        print(f"{cell:<24} {r_p:>+10.3f} {p_p:>9.2g} {b[1]:>+7.2f} {b[2]:>+7.2f} {b[3]:>+7.2f} "
              f"{_auroc(y, mono):>11.3f} {_auroc(y, viol):>7.3f} {_auroc(y, -hn):>7.3f} "
              f"{_auroc(y, coh):>7.3f}")


def holm(results):
    """Holm-Bonferroni over the sixteen-test panel-wide family (exploratory, not the registered one)."""
    tests = []
    for cell in PANEL:
        s = summarize(load(results, cell))
        tests.append((f"H1 {cell}", s["shape"]["p_value"]))
        tests.append((f"H3 {cell}", s["violations"]["p_value"]))
    tests.sort(key=lambda t: t[1])
    m = len(tests)
    print(f"=== Holm-Bonferroni, family of {m}, alpha = 0.05 ===")
    still = True
    for i, (name, p) in enumerate(tests):
        thr = 0.05 / (m - i)
        ok = still and p <= thr
        still = ok
        print(f"{i + 1:>3}. {name:<28} p={p:<10.3g} threshold={thr:<9.3g} "
              f"{'reject' if ok else 'RETAIN'}")


GROUPS = {"tolerance": tolerance, "prefix": prefix, "baselines": baselines, "holm": holm}

if __name__ == "__main__":
    args = sys.argv[1:]
    default_results = "results"
    results = args[0] if args and args[0] not in GROUPS else default_results
    wanted = [a for a in args if a in GROUPS] or list(GROUPS)
    for name in wanted:
        GROUPS[name](results)
        print()
