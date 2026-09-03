"""Deviation check: does deduplicating a records file change its statistics?

Run after the mistral7b-math500 duplicate-record incident, comparing the raw file
against the corrected one. Needs records.jsonl.raw-with-duplicate beside records.jsonl.

    python scripts/panel/recompute.py [results-dir]
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from entropydrift.metrics import summarize

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
for tag, path in (("WITH duplicate (as run)", f"{RESULTS}/mistral7b-math500/records.jsonl.raw-with-duplicate"),
                  ("DEDUPLICATED (corrected)", f"{RESULTS}/mistral7b-math500/records.jsonl")):
    rs = [json.loads(l) for l in open(path) if l.strip()]
    rs = [r for r in rs if r.get("status") == "ok"]
    s = summarize(rs, n_boot=1000)
    sh = s["shape"]
    print(f"=== {tag} ===")
    print(f"  n={s['n']}  gap={sh['gap_pp']:+.2f}pp  OR={sh['odds_ratio']:.3f}  p={sh['p_value']:.2g}")
    print(f"  magnitude rho={s['magnitude']['spearman_rho']:+.4f}  violation rho={s['violations']['spearman_rho']:+.4f}")
    if "ci" in s:
        c = s["ci"]["shape_gap_pp"]
        print(f"  gap CI [{c['lo']:+.2f}, {c['hi']:+.2f}]")
