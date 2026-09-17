"""H5 as registered: exact-k acceptance at the baseline's coverage.

The as-run `at_coverage` thresholds on a quantile of predicted P(correct). Because
`final_confidence` takes only m+1 distinct values the probabilities are heavily tied,
so `p >= thr` sweeps in whole tie blocks and coverage overshoots its target. This
accepts exactly k = round(coverage * n) items by descending rank, breaking ties with a
seeded uniform that never inspects the outcome, and averages over R draws so no single
tie-break decides the verdict. Model, features, split and bootstrap match the as-run path.
The as-run column is what scripts/fp_reduce.py reported for each cell.

    python scripts/paper/h5_exact_k.py [results-dir]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from entropydrift.fpreduce import (DEFAULT_FEATURES, LogisticRegression,
                                   features_matrix, monotone_baseline, stratified_split)

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
CELLS = ["qwen7b-gsm8k", "qwen7b-math500", "mistral7b-gsm8k", "mistral7b-math500",
         "llama31-8b-gsm8k", "llama31-8b-math500", "r1-distill-7b-gsm8k",
         "r1-distill-7b-math500"]
R_TIEBREAK, N_BOOT = 100, 1000


def load(cell):
    seen, out = set(), []
    for line in open(os.path.join(BASE, cell, "records.jsonl")):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("status") != "ok" or r["index"] in seen: continue
        seen.add(r["index"]); out.append(r)
    return out


def exact_k_fp(p, y, k, rng):
    """FP rate of accepting exactly k items by descending p, ties broken at random."""
    if k <= 0: return float("nan")
    order = np.lexsort((rng.random(len(p)), -p))[:k]
    return 1.0 - y[order].mean()


def reduction(recs, p, y, rng, r_tie=R_TIEBREAK):
    base = monotone_baseline(recs)
    if not np.isfinite(base["false_pos_rate"]): return float("nan"), float("nan"), float("nan")
    k = int(round(base["coverage"] * len(recs)))
    fps = [exact_k_fp(p, y, k, rng) for _ in range(r_tie)]
    return base["false_pos_rate"] - np.mean(fps), base["coverage"], k / len(recs)


print(f"{'cell':22}{'cov base':>9}{'cov filt':>9}{'FP red':>9}{'95% CI':>20}{'as-run':>9}")
print("-" * 80)
AS_RUN = {"qwen7b-gsm8k": +1.1, "qwen7b-math500": +11.9, "mistral7b-gsm8k": +0.1,
          "mistral7b-math500": +1.6, "llama31-8b-gsm8k": +0.7, "llama31-8b-math500": +12.2,
          "r1-distill-7b-gsm8k": -6.4, "r1-distill-7b-math500": +50.0}
for cell in CELLS:
    recs = load(cell)
    X, y = features_matrix(recs, DEFAULT_FEATURES)
    tr, te = stratified_split(y, 0.4, 0)
    model = LogisticRegression().fit(X[tr], y[tr])
    te_recs = [recs[i] for i in te]
    p, yte = model.predict_proba(X[te]), y[te]
    rng = np.random.default_rng(0)
    pt, cov_b, cov_f = reduction(te_recs, p, yte, rng)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(te_recs), len(te_recs))
        v, _, _ = reduction([te_recs[j] for j in i], p[i], yte[i], rng, r_tie=5)
        if np.isfinite(v): boot.append(v)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) > 50 else (np.nan, np.nan))
    print(f"{cell:22}{cov_b:>9.3f}{cov_f:>9.3f}{pt*100:>+9.1f}"
          f"{f'[{lo*100:+.1f}, {hi*100:+.1f}]':>20}{AS_RUN[cell]:>+9.1f}")
