"""The original's violation-magnitude check, recomputed on every panel cell.

Among non-monotone chains, rank-correlate the largest upward entropy step with
correctness, and print the violation-count correlation beside it for comparison.

    python scripts/paper/violation_magnitude.py [results-dir]
"""
import json, glob, os, sys, numpy as np
from scipy import stats
BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
EPS=0.01
cells=["qwen7b-gsm8k","qwen7b-math500","mistral7b-gsm8k","mistral7b-math500",
       "llama31-8b-gsm8k","llama31-8b-math500","r1-distill-7b-gsm8k","r1-distill-7b-math500"]
print(f"{'cell':24}{'non-mono n':>11}{'rho(max dH, Y)':>16}{'p':>9}{'rho(violations, Y)':>20}")
print("-"*82)
allrows=[]
for c in cells:
    f=os.path.join(BASE,c,"records.jsonl")
    if not os.path.exists(f): print(f"{c:24}  MISSING"); continue
    seen=set(); recs=[]
    for line in open(f):
        try: r=json.loads(line)
        except Exception: continue
        if r.get("status")!="ok": continue
        if r["index"] in seen: continue
        seen.add(r["index"]); recs.append(r)
    nm=[]
    for r in recs:
        t=r["trajectory"]
        rises=[t[k+1]-t[k] for k in range(len(t)-1) if t[k+1]-t[k] > EPS]
        if rises: nm.append((max(rises), 1 if r["correct"] else 0))
    if len(nm)>=10 and len({y for _,y in nm})>1:
        x=np.array([a for a,_ in nm]); y=np.array([b for _,b in nm])
        rho=stats.spearmanr(x,y)
        vr=stats.spearmanr([r["violations"] for r in recs],[1 if r["correct"] else 0 for r in recs])
        print(f"{c:24}{len(nm):>11}{rho.statistic:>+16.3f}{rho.pvalue:>9.3f}{vr.statistic:>+20.3f}")
        allrows.append((c,len(nm),rho.statistic,rho.pvalue))
    else:
        print(f"{c:24}{len(nm):>11}{'  (too few)':>16}")
print("-"*82)
print("Zhao, pilot GSM8K n=300, among non-monotone chains: rho(max positive dH, Y) = -0.017, p=0.88")
sig=[r for r in allrows if r[3]<0.05]
print(f"cells with p<0.05: {len(sig)} of {len(allrows)}" + (f" -> {[r[0] for r in sig]}" if sig else ""))
