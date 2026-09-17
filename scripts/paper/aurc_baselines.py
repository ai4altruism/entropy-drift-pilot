"""Scalar baselines under Zhao's own statistic: area under the risk-coverage curve.

Section 6.1 and Appendix F of the paper rank signals by AUROC, which is not the statistic the original
uses and which structurally disadvantages a binary flag. This recomputes the same four
signals as AURC (lower is better), the statistic the original reports, sweeping coverage
over every prefix of the confidence ranking and averaging risk. Ties are broken by a
seeded uniform, averaged over R draws, so the massively tied binary flag is not decided
by sort order.

The original does not state the range its AURC integrates over, and its Table 3 fixes
coverage at 73.7% for the accuracy columns beside it. So the area is printed twice: over
the full curve, coverage 1/n to 1, and truncated at 73.7% coverage, averaging risk over
coverage 1/n to 0.737 only. The same tie-break draws serve both.

    python scripts/paper/aurc_baselines.py [results-dir]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
CELLS = ["qwen7b-gsm8k", "qwen7b-math500", "mistral7b-gsm8k", "mistral7b-math500",
         "llama31-8b-gsm8k", "llama31-8b-math500", "r1-distill-7b-gsm8k",
         "r1-distill-7b-math500"]
R = 50

def load(cell):
    seen, out = set(), []
    for line in open(os.path.join(BASE, cell, "records.jsonl")):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("status") != "ok" or r["index"] in seen: continue
        seen.add(r["index"]); out.append(r)
    return out

def aurc(score, y, rng, r=R, upto=1.0):
    """Mean risk over coverage levels 1/n..upto, ties broken at random, averaged."""
    score = np.asarray(score, float); y = np.asarray(y, int); n = len(y)
    vals = []
    for _ in range(r):
        order = np.lexsort((rng.random(n), -score))
        err = 1 - y[order]
        risk = np.cumsum(err) / np.arange(1, n + 1)
        vals.append(risk[:round(upto * n)].mean())
    return float(np.mean(vals))

SIGNALS = [("monotone",10),("violations",12),("final H_N",11),("coherence",11)]
for title, upto in [("full curve, coverage 1/n to 1", 1.0),
                    ("truncated at 73.7% coverage", 0.737)]:
    print(f"\nAURC, {title}")
    print(f"{'cell':22}{'acc':>7}{'monotone':>10}{'violations':>12}{'final H_N':>11}{'coherence':>11}{'best':>13}")
    print("-" * 88)
    for cell in CELLS:
        recs = load(cell); rng = np.random.default_rng(0)
        y = [1 if r["correct"] else 0 for r in recs]
        sig = {"monotone":  [1.0 if r["monotone"] else 0.0 for r in recs],
               "violations": [-r["violations"] for r in recs],
               "final H_N":  [-r["final_entropy"] for r in recs],
               "coherence":  [r["coherence"] for r in recs]}
        a = {k: aurc(v, y, rng, upto=upto) for k, v in sig.items()}
        best = min(a, key=a.get)
        print(f"{cell:22}{np.mean(y):>7.3f}" + "".join(f"{a[k]:>{w}.4f}" for k, w in SIGNALS) + f"{best:>13}")
    print("-" * 88)
print("Lower is better. Zhao, pilot GSM8K n=300: monotonicity AURC 0.311, scalar coherence 0.408,")
print("full-coverage accuracy 0.630. AURC is bounded by a cell's base error rate, so compare")
print("within a row, not across rows.")
