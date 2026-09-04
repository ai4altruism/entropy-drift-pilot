"""Reference-only diagnostic scan: measure the chain, never the trajectory.

Two open questions about the panel cannot be answered from the records it wrote, because
both are about the *reference chain* while the records describe the *trajectory* built on
top of it:

  1. **Does a 1280-token reference budget still truncate?** The registered panel ran at
     600 and no run before 2026-09-02 recorded reference-chain token counts at all.
  2. **Is the spike at 8 trajectory points structure or extraction?** Trajectory length
     counts prefixes that yielded an extractable answer, so it cannot distinguish a chain
     capped by ``max_steps`` from a longer chain that lost a prefix to extraction.

Answering either needs only the reference chain: generate it, tokenize it, segment it,
record the counts. It does not need the ``m`` continuations at every prefix, and those
continuations are what make a real cell take hours. This module is that path.

> [!warning] This is a DIAGNOSTIC, not a cell
> It computes no entropy, scores no hypothesis, and writes no quantity the registration
> mentions; the output has no ``trajectory`` field to score. It is off the registered
> execution path by construction and its numbers must never be reported as confirmatory.
>
> Generation is sampled at temperature 0.7 and vLLM is not bit-reproducible under a fixed
> seed, so a scan characterizes the chain-length **distribution** its config produces. It
> does not recover the individual chains an earlier run generated, and a scan at 600 is an
> independent generation from the registered cell that ran at 600, not a replay of it.

What it settles, stated at the right strength. Question 1 it answers outright: the token
distribution against the budget is exactly the measurement that was missing. Question 2 it
**bounds** rather than closes. The scan reports ``raw_units`` (segmentation units before
the cap) and ``prefixes`` (units after it, plus the empty prefix), so it says how much of
an 8-point spike the cap alone can explain; measuring the extraction loss on top of that
still needs the continuations.

Usage::

    PYTHONPATH=src python -m entropydrift.refscan --config configs/diagnostic/<cfg>.yaml

Writes ``ref_records.jsonl``, ``ref_manifest.json`` and ``ref_summary.json`` into
``results/<run.name>/``. The filenames are deliberately distinct from a run's
``records.jsonl`` so a scan can never be mistaken for, or overwrite, a cell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

from .backends import make_backend
from .config import Config, load_config
from .datasets import load_examples
from .provenance import build_manifest
from .run import chain_diagnostics, check_config_match, load_jsonl
from .segment import cumulative_prefixes


def reference_chains(backend, questions: list[str]) -> list[str]:
    """Generate one reference chain per question, batched when the backend can.

    vLLM serves a list of prompts through one paged-attention pass, which is the whole
    reason a scan costs minutes where a cell costs hours. Backends without a batched
    method fall back to the one-at-a-time call, so the mock and transformers paths keep
    working unchanged.
    """
    batched = getattr(backend, "reference_chains", None)
    if batched is not None:
        return batched(questions)
    return [backend.reference_chain(q) for q in questions]


def _percentile(sorted_xs: list[int], q: float) -> int:
    """Nearest-rank percentile: the smallest value at or above ``q`` through the data.

    Nearest-rank rather than interpolated because these are token and unit *counts*, and
    an interpolated 1279.5 would invite reading a boundary the data cannot support.
    """
    k = max(1, math.ceil(q * len(sorted_xs)))
    return sorted_xs[k - 1]


def _dist(xs: list[int]) -> dict:
    s = sorted(xs)
    return {
        "n": len(s),
        "min": s[0],
        "p50": _percentile(s, 0.50),
        "p90": _percentile(s, 0.90),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
        "mean": round(sum(s) / len(s), 2),
    }


def summarize_scan(records: list[dict], budget: int, max_steps: int) -> dict:
    """Reduce scan records to the two questions' answers.

    ``reference_tokens`` is null when the backend exposes no tokenizer (the mock path), so
    every token-derived field is reported as null rather than fabricated from characters.
    """
    summary: dict = {
        "n": len(records),
        "reference_budget": budget,
        "max_steps": max_steps,
        "tokens": None,
        "truncated": None,
        "raw_units": None,
        "prefixes": None,
    }
    if not records:
        return summary

    toks = [r["reference_tokens"] for r in records if r.get("reference_tokens") is not None]
    if toks:
        summary["tokens"] = _dist(toks)

    flags = [r["reference_truncated"] for r in records if r.get("reference_truncated") is not None]
    if flags:
        n_trunc = sum(1 for f in flags if f)
        summary["truncated"] = {
            "n": n_trunc,
            "of": len(flags),
            "rate": round(n_trunc / len(flags), 4),
        }

    units = [r["raw_units"] for r in records if r.get("raw_units") is not None]
    if units:
        # A chain with at least max_steps units is one the cap bites: its prefixes are
        # pinned regardless of how much longer the chain actually ran. This is the share
        # of an 8-point spike that structure alone accounts for.
        capped = sum(1 for u in units if u >= max_steps)
        summary["raw_units"] = {
            **_dist(units),
            "histogram": {str(k): v for k, v in sorted(Counter(units).items())},
            "at_or_above_max_steps": {
                "n": capped,
                "rate": round(capped / len(units), 4),
            },
        }

    pfx = [r["prefixes"] for r in records if r.get("prefixes") is not None]
    if pfx:
        summary["prefixes"] = {
            **_dist(pfx),
            "histogram": {str(k): v for k, v in sorted(Counter(pfx).items())},
        }
    return summary


def format_scan_report(summary: dict) -> str:
    lines = [
        f"reference-only scan: n={summary['n']}, "
        f"budget={summary['reference_budget']} tokens, max_steps={summary['max_steps']}",
    ]
    t = summary.get("tokens")
    if t:
        lines.append(
            f"  reference tokens  mean {t['mean']}  p50 {t['p50']}  p90 {t['p90']}  "
            f"p95 {t['p95']}  p99 {t['p99']}  max {t['max']}"
        )
    else:
        lines.append("  reference tokens  (no tokenizer on this backend)")
    tr = summary.get("truncated")
    if tr:
        lines.append(f"  truncated         {tr['n']}/{tr['of']}  ({tr['rate']:.1%})")
    u = summary.get("raw_units")
    if u:
        cap = u["at_or_above_max_steps"]
        lines.append(
            f"  raw units         p50 {u['p50']}  p90 {u['p90']}  max {u['max']}   "
            f"at/above max_steps {cap['n']} ({cap['rate']:.1%})"
        )
    p = summary.get("prefixes")
    if p:
        lines.append(f"  prefixes          {p['histogram']}")
    return "\n".join(lines)


def _scan_one(cfg, backend, i: int, chain: str) -> dict:
    prefixes = cumulative_prefixes(
        chain,
        strategy=cfg.segmentation.strategy,
        window_tokens=cfg.segmentation.window_tokens,
        max_steps=cfg.segmentation.max_steps,
    )
    # Reused from run.py rather than reimplemented: a scan is only useful if its counts
    # mean exactly what the same-named fields in a cell's records mean.
    return {"index": i, **chain_diagnostics(cfg, backend, chain, len(prefixes))}


def scan(
    cfg: Config,
    batch_size: int = 32,
    progress: bool = True,
    resume: bool = False,
    overwrite: bool = False,
) -> dict:
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    out_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "ref_records.jsonl")
    manifest_path = os.path.join(out_dir, "ref_manifest.json")

    existing: list[dict] = []
    if os.path.exists(records_path) and resume and not overwrite:
        existing = load_jsonl(records_path)
        # The budget is the variable under study here, so resuming a 600 scan under a
        # 1280 config would silently mix two distributions into one file.
        check_config_match(cfg, manifest_path)
    elif os.path.exists(records_path) and not overwrite:
        raise FileExistsError(
            f"{records_path} already exists. Pass resume=True/--resume to continue, "
            "overwrite=True/--overwrite to restart, or choose a new run.name."
        )
    else:
        if os.path.exists(records_path):
            os.remove(records_path)
        with open(manifest_path, "w") as f:
            json.dump(build_manifest(cfg), f, indent=2)

    completed = {r["index"] for r in existing}
    records: list[dict] = list(existing)

    backend = make_backend(cfg)
    examples = load_examples(cfg)
    todo = [(i, ex) for i, ex in enumerate(examples) if i not in completed]

    with open(records_path, "a") as out_f:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            chains = reference_chains(backend, [ex.question for _, ex in batch])
            for (i, _), chain in zip(batch, chains):
                rec = _scan_one(cfg, backend, i, chain)
                completed.add(i)
                records.append(rec)
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            if progress:
                print(f"  ...{min(start + batch_size, len(todo))}/{len(todo)}")

    summary = summarize_scan(
        records, cfg.sampling.reference_budget, cfg.segmentation.max_steps
    )
    with open(os.path.join(out_dir, "ref_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reference-only diagnostic scan (generates no trajectories)."
    )
    ap.add_argument("--config", required=True, help="Path to a YAML run config.")
    ap.add_argument("--resume", action="store_true", help="Continue an existing scan.")
    ap.add_argument("--overwrite", action="store_true", help="Restart an existing scan.")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Reference chains per backend call (vLLM batches these; default 32).",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(
        f"Scanning '{cfg.run.name}' (backend={cfg.backend}, dataset={cfg.dataset.name}, "
        f"reference budget={cfg.sampling.reference_budget})"
    )
    summary = scan(
        cfg, batch_size=args.batch_size, resume=args.resume, overwrite=args.overwrite
    )
    print()
    print(format_scan_report(summary))
    print(f"\nWrote scan to {os.path.join(cfg.run.out_dir, cfg.run.name)}/")


if __name__ == "__main__":
    main()
