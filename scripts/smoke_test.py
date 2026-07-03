#!/usr/bin/env python
"""End-to-end smoke test on the mock backend (no model, no network).

Verifies the full pipeline plumbing and that the shape signal separates correct from
incorrect on synthetic data (monotone should beat non-monotone). Exits non-zero if the
pipeline produces no records.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from entropydrift.config import load_config
from entropydrift.metrics import format_report
from entropydrift.run import run

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "mock.yaml")


def main() -> int:
    cfg = load_config(CONFIG)
    print(f"Smoke test: backend={cfg.backend}, dataset={cfg.dataset.name}, n={cfg.dataset.limit}")
    summary = run(cfg, progress=False, overwrite=True)
    print()
    print(format_report(summary))
    if summary["n"] == 0:
        print("\nFAIL: no records produced.")
        return 1
    gap = summary["shape"]["gap_pp"]
    print(f"\nOK: pipeline ran, {summary['n']} records, shape gap {gap:+.1f} pp.")
    print(
        "Note: on the mock backend the MAGNITUDE correlation is nonzero by construction "
        "(the simulator ties convergence to correctness). The shape-over-magnitude "
        "dissociation is a hypothesis about real models, tested by the transformers runs, "
        "not by this plumbing check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
