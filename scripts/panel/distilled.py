"""Per-cell report for the reasoning-distilled cells (contribution 2).

Produced the numbers in the panel's distilled write-up: exclusions, trajectory
lengths, monotone coverage, H1/H2/H3 with bootstrap CIs, and the violation buckets.

    python scripts/panel/distilled.py [results-dir]
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from entropydrift.metrics import summarize

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
for cell in ("r1-distill-7b-gsm8k", "r1-distill-7b-math500"):
    rs = [json.loads(l) for l in open(f"{RESULTS}/{cell}/records.jsonl") if l.strip()]
    ok = [r for r in rs if r.get("status") == "ok"]
    # over ALL records, so skipped ones contribute 0 to the min; medians are unaffected
    st = [len(r.get("trajectory") or []) for r in rs]
    s = summarize(ok, n_boot=1000)
    sh, mg, vi = s["shape"], s["magnitude"], s["violations"]
    print(f"=== {cell} ===")
    print(f"  records={len(rs)} usable={len(ok)} excluded={len(rs)-len(ok)} ({100*(len(rs)-len(ok))/len(rs):.1f}%)")
    print(f"  steps min/med/max={min(st)}/{sorted(st)[len(st)//2]}/{max(st)}  acc={s['overall_acc']:.3f}")
    print(f"  coverage={sh['monotone_coverage']:.3f}  fp_rate={sh['false_positive_rate']:.3f}")
    print(f"  H1 gap={sh['gap_pp']:+.2f}pp OR={sh['odds_ratio']:.2f} p={sh['p_value']:.2g}  n_mono={sh['n_monotone']} n_non={sh['n_non_monotone']}")
    print(f"  H2 magnitude rho={mg['spearman_rho']:+.4f} p={mg['p_value']:.2g}")
    print(f"  H3 violation rho={vi['spearman_rho']:+.4f}")
    c = s.get("ci", {})
    if c:
        print(f"  CI gap[{c['shape_gap_pp']['lo']:+.2f},{c['shape_gap_pp']['hi']:+.2f}] "
              f"mag[{c['magnitude_rho']['lo']:+.4f},{c['magnitude_rho']['hi']:+.4f}] "
              f"viol[{c['violation_rho']['lo']:+.4f},{c['violation_rho']['hi']:+.4f}]")
    print(f"  violation buckets: {[(k, round(v,3), s['violations']['counts_by_violations'][k]) for k,v in s['violations']['accuracy_by_violations'].items()]}")
