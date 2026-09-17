"""Holm-Bonferroni over the family the registration actually names.

prereg-v1.0 sec.8: "the primary confirmatory family (H1 both datasets, H3, H5) is
corrected with Holm-Bonferroni. H4 is estimation and is not part of the null-testing
family." H1 is registered on the anchor only (sec.4); the other models are secondary
confirmatory replication (sec.7). So the registered family is four tests on the anchor,
not sixteen across the panel. The H5 lines restate what scripts/paper/h5_exact_k.py prints.

    python scripts/paper/registered_family.py [results-dir]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from scipy import stats

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"

def load(cell):
    seen, out = set(), []
    for line in open(os.path.join(BASE, cell, "records.jsonl")):
        try: r = json.loads(line)
        except Exception: continue
        if r.get("status") != "ok" or r["index"] in seen: continue
        seen.add(r["index"]); out.append(r)
    return out

def h1(recs):
    mono = [r for r in recs if r["monotone"]]; non = [r for r in recs if not r["monotone"]]
    a, b = sum(r["correct"] for r in mono), len(mono) - sum(r["correct"] for r in mono)
    c, d = sum(r["correct"] for r in non), len(non) - sum(r["correct"] for r in non)
    _, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    gap = (a/len(mono) - c/len(non)) * 100
    return gap, p

def h3(recs):
    v = [r["violations"] for r in recs]; y = [1 if r["correct"] else 0 for r in recs]
    r_ = stats.spearmanr(v, y)
    return r_.statistic, r_.pvalue

tests = []
for ds in ("gsm8k", "math500"):
    recs = load(f"qwen7b-{ds}")
    g, p = h1(recs); tests.append((f"H1 anchor {ds}", g, p))
for ds in ("gsm8k", "math500"):
    recs = load(f"qwen7b-{ds}")
    r_, p = h3(recs); tests.append((f"H3 anchor {ds}", r_, p))

print("The registered primary confirmatory family, on the anchor model")
print(f"{'test':26}{'statistic':>12}{'p':>12}")
for name, s_, p in tests: print(f"{name:26}{s_:>+12.3f}{p:>12.3g}")
print("\nH5 anchor: the registered criterion is CI-based, not a p-value.")
print("  exact-k reduction  gsm8k  +1.4 pp  CI [+0.3, +2.8]  -> excludes 0 favorably")
print("  exact-k reduction  math500 +15.5 pp CI [+2.7, +31.3] -> excludes 0 favorably")

ps = sorted([(p, n) for n, _, p in tests])
m = len(ps) + 1  # + H5, whose CI criterion is met
print(f"\nHolm-Bonferroni over the registered family (m={m}: H1 x2, H3, H5):")
rejected = True
for i, (p, n) in enumerate(ps):
    alpha = 0.05 / (m - i)
    ok = p < alpha and rejected
    rejected = ok
    print(f"  {n:26} p={p:<11.3g} alpha={alpha:.4f}  {'reject' if ok else 'retain'}")
print("  H5                         criterion is a CI, not a p-value; met on both anchor cells")
