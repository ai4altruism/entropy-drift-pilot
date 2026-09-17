"""Stage-1 gate for the temperature arm: compare two reference-only scans.

Reads the ``ref_summary.json`` of two scans that differ only in the temperature their
reference chains were sampled at, and prints the quantities that decide whether a full
cell is worth renting for: how much shorter the chains get, how much less they truncate,
and how many fewer segmentation units they carry.

    python scripts/paper/refscan_compare.py <scan-dir-A> <scan-dir-B>

A scan computes no entropy and scores no hypothesis, so nothing here can move rho. It
bounds plausibility only.
"""
import json
import os
import sys


def load(d):
    with open(os.path.join(d, "ref_summary.json")) as f:
        return json.load(f)


def row(label, a, b, fmt="{:>10}"):
    print(f"  {label:<34}" + fmt.format(a) + fmt.format(b))


def main(dir_a, dir_b):
    a, b = load(dir_a), load(dir_b)
    na, nb = os.path.basename(dir_a.rstrip("/")), os.path.basename(dir_b.rstrip("/"))
    print(f"\n  A = {na}\n  B = {nb}")
    print(f"\n{'':36}{'A':>10}{'B':>10}")
    print("  " + "-" * 52)
    row("reference budget", a["reference_budget"], b["reference_budget"])
    row("temperature (from manifest)", _temp(dir_a), _temp(dir_b))
    row("n", a["n"], b["n"])
    print()
    ta, tb = a["truncated"], b["truncated"]
    row("truncated", f"{ta['n']}/{ta['of']}", f"{tb['n']}/{tb['of']}")
    row("truncation rate", f"{ta['rate']:.1%}", f"{tb['rate']:.1%}")
    print()
    ka, kb = a["tokens"], b["tokens"]
    for k in ("min", "p50", "p90", "p95", "p99", "max", "mean"):
        row(f"chain tokens {k}", _fmt(ka[k]), _fmt(kb[k]))
    print()
    ua, ub = a["raw_units"], b["raw_units"]
    for k in ("min", "p50", "p90", "p95", "p99", "max", "mean"):
        row(f"raw units {k}", _fmt(ua[k]), _fmt(ub[k]))
    cap = a.get("max_steps", 8)
    row(f"share with >= {cap} units", _cap_share(ua, cap), _cap_share(ub, cap))

    print("\n  Reading it")
    print("  A large fall in truncation and in raw units means the temperature difference")
    print("  materially changes the chains the protocol measures, so it is a live")
    print("  explanation for the H2 result and stage 2 is worth the money. Distributions")
    print("  that sit on top of each other mean it is not, and the paper's limitation")
    print("  can stand as written. Neither outcome is a rho measurement.")
    print("\n  Both scans are independent generations, not replays: vLLM is not")
    print("  bit-reproducible under seed 0, so a small difference is not evidence.\n")


def _temp(d):
    try:
        with open(os.path.join(d, "ref_manifest.json")) as f:
            return json.load(f)["config"]["sampling"]["temperature"]
    except Exception:
        return "?"


def _fmt(v):
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def _cap_share(u, cap):
    h = u.get("histogram")
    if not h:
        return "n/a"
    total = sum(h.values())
    at = sum(n for k, n in h.items() if int(k) >= cap)
    return f"{at / total:.1%}" if total else "n/a"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
