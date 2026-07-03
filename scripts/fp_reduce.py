#!/usr/bin/env python
"""Post-hoc false-positive-reduction analysis over a completed run.

Reads a run's records.jsonl, fits the learned triage filter, compares it to the
monotonicity baseline, and writes fp_analysis.json alongside the run.

    python scripts/fp_reduce.py results/<run-name>
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from entropydrift.fpreduce import DEFAULT_FEATURES, evaluate, format_report


def load_records(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "records.jsonl")
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    # keep only usable ('ok') records; skipped entries lack metric fields
    return [r for r in records if r.get("status", "ok") == "ok"]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/fp_reduce.py results/<run-name> [feature ...]")
        return 2
    run_dir = sys.argv[1]
    features = tuple(sys.argv[2:]) or DEFAULT_FEATURES
    records = load_records(run_dir)
    result = evaluate(records, feature_names=features)
    print(format_report(result))
    out = os.path.join(run_dir, "fp_analysis.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
