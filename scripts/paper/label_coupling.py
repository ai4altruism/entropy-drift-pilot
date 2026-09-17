"""Decouple the correctness label from the final entropy on the traced run.

The panel scores the majority answer of the last prefix's five probes, and the final
entropy is computed from those same five. The traced run keeps the reference chain, so
the label can be recomputed from the chain's own answer and each signal scored against
both labels.

    python scripts/paper/label_coupling.py [trace-dir]

`trace-dir` holds one directory per traced arm and defaults to `trace`.
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from entropydrift.answers import extract_final_number, extract_math_answer
from scipy import stats

def auroc(score, y):
    score=np.asarray(score,float); y=np.asarray(y,int)
    pos=score[y==1]; neg=score[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    allv=np.concatenate([pos,neg]); r=stats.rankdata(allv)
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

ARMS=[("trace-qwen7b-math500","math500",extract_math_answer),
      ("trace-mistral7b-gsm8k","gsm8k",extract_final_number)]
BASE = sys.argv[1] if len(sys.argv) > 1 else "trace"

for name,ds,extract in ARMS:
    recs=[json.loads(l) for l in open(f"{BASE}/{name}/records.jsonl")]
    ok=[r for r in recs if r.get("status")=="ok" and r.get("trace")]
    print("="*100); print(f"{name}  ok={len(ok)} of {len(recs)}")
    ref_ans=[extract(r["trace"]["reference_text"]) for r in ok]
    y_probe=np.array([1 if r["correct"] else 0 for r in ok])
    y_ref  =np.array([1 if a==r["gold"] else 0 for a,r in zip(ref_ans,ok)])
    noans  =sum(1 for a in ref_ans if a=="")
    coh=np.array([r["coherence"] for r in ok]); hn=np.array([r["final_entropy"] for r in ok])
    mono=np.array([1 if r["monotone"] else 0 for r in ok]); vio=np.array([r["violations"] for r in ok])
    agree=(y_probe==y_ref).mean()
    print(f"  reference chain yields no extractable answer: {noans}")
    print(f"  accuracy, probe-majority (as scored) : {y_probe.mean():.3f}")
    print(f"  accuracy, reference-chain answer     : {y_ref.mean():.3f}")
    print(f"  the two labels agree on              : {agree:.3f} of items")
    print(f"  pred == reference answer             : {np.mean([r['pred']==a for r,a in zip(ok,ref_ans)]):.3f}")
    print(f"  {'':34}{'PROBE-MAJORITY':>18}{'REFERENCE-CHAIN':>18}")
    for lab,sc,flip in [("Spearman rho(coherence, Y)",coh,None),
                        ("Spearman rho(final entropy, Y)",hn,None)]:
        a=stats.spearmanr(sc,y_probe); b=stats.spearmanr(sc,y_ref)
        print(f"  {lab:34}{a.statistic:>+18.3f}{b.statistic:>+18.3f}")
    for lab,sc in [("AUROC(-final entropy)",-hn),("AUROC(monotone flag)",mono),
                   ("AUROC(-violation count)",-vio),("AUROC(coherence)",coh)]:
        print(f"  {lab:34}{auroc(sc,y_probe):>18.3f}{auroc(sc,y_ref):>18.3f}")
