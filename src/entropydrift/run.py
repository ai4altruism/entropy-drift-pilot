"""Orchestration: build entropy trajectories over a dataset and score the shape signals.

For each example: sample a reference chain, segment it into cumulative prefixes, sample
`m` continuations at each prefix, extract answers, build the entropy trajectory, and
record monotonicity / violation-count / coherence / correctness.

Records are written incrementally (one JSON line per example, flushed) so a long or
interrupted run is durable and resumable. Pass ``resume=True`` (``--resume``) to continue
a run that already has partial results: completed indices are skipped and the run picks up
where it left off. A provenance manifest is written at the start; the metrics summary is
(re)computed over all completed records at the end.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from .answers import extract_final_number, extract_math_answer
from .backends import make_backend
from .config import Config, load_config
from .datasets import load_examples
from .metrics import format_report, summarize
from .provenance import build_manifest
from .segment import count_units, cumulative_prefixes
from .trajectory import coherence, entropy_trajectory, is_monotone, violation_count

_METRIC_KEYS = ("monotone", "violations", "coherence", "final_confidence", "correct")


def _extractor(dataset_name: str):
    return extract_math_answer if dataset_name == "math500" else extract_final_number


def _majority(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0] if answers else ""


def _load_jsonl(path: str) -> list[dict]:
    """Read records, keeping the FIRST record per index and tolerating a torn tail.

    A crash can leave two hazards behind. The last line may be half-written, since a
    kill can land mid-write; that line is dropped rather than aborting the resume.
    And a record can be on disk but not yet visible to the next read, which on a
    network filesystem lets resume reprocess an index it already has, appending a
    second copy that then double-counts in every statistic. Deduplicating on read
    makes resume idempotent regardless of how the previous process died.

    First-write-wins is deliberate: it is a rule that never inspects the values, so
    recovery cannot become a choice between two results.
    """
    seen: set = set()
    out: list[dict] = []
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    for n, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if n == len(lines) - 1:
                continue  # torn final line from an interrupted write
            raise
        i = rec.get("index")
        if i in seen:
            continue
        seen.add(i)
        out.append(rec)
    return out


def _config_diff(prior: dict, current: dict, defaults: dict, path: str = "") -> list[str]:
    """Config keys that genuinely differ, as readable 'key: old -> new' lines.

    A key absent from ``prior`` but present in ``current`` means the schema gained a
    field after that run was written. That is not a config change unless the field is
    actually set. Comparing whole-config hashes flags it anyway, which locks you out of
    resuming every earlier run the moment an unrelated option is added.
    """
    diffs: list[str] = []
    for key in sorted(set(prior) | set(current)):
        where = f"{path}{key}"
        if key in prior and key in current:
            pv, cv = prior[key], current[key]
            if isinstance(pv, dict) and isinstance(cv, dict):
                sub = defaults.get(key)
                diffs += _config_diff(pv, cv, sub if isinstance(sub, dict) else {}, f"{where}.")
            elif pv != cv:
                diffs.append(f"{where}: {pv!r} -> {cv!r}")
        elif key in current:
            if key not in defaults or current[key] != defaults[key]:
                diffs.append(f"{where}: (not in the earlier run) -> {current[key]!r}")
        else:
            diffs.append(f"{where}: {prior[key]!r} -> (dropped from the config schema)")
    return diffs


def _check_config_match(cfg: Config, manifest_path: str) -> None:
    """On resume, refuse to append to a run created under a different config."""
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path) as f:
        prior = json.load(f)
    prior_cfg = prior.get("config")
    if prior_cfg is None:
        return
    diffs = _config_diff(prior_cfg, cfg.to_dict(), Config().to_dict())
    if diffs:
        detail = "\n  ".join(diffs)
        raise ValueError(
            f"resume config mismatch: {manifest_path} was created under a different config.\n"
            f"  {detail}\n"
            "Use a new run.name or reconcile the config."
        )


def _metric_dict(record: dict) -> dict:
    return {k: record[k] for k in _METRIC_KEYS}


def run(
    cfg: Config, progress: bool = True, resume: bool = False, overwrite: bool = False
) -> dict:
    out_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
    os.makedirs(out_dir, exist_ok=True)
    records_path = os.path.join(out_dir, "records.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")

    existing: list[dict] = []
    if os.path.exists(records_path) and resume and not overwrite:
        existing = _load_jsonl(records_path)
        _check_config_match(cfg, manifest_path)
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
    per_problem: list[dict] = list(existing)

    backend = make_backend(cfg)
    examples = load_examples(cfg)
    extract = _extractor(cfg.dataset.name)
    eps = cfg.monotonicity.eps

    with open(records_path, "a") as out_f:
        for i, ex in enumerate(examples):
            if i in completed:
                continue
            record = _process_one(cfg, backend, extract, eps, i, ex)
            completed.add(i)  # belt and braces: never write an index twice in one run
            per_problem.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            if progress and (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(examples)}")

    metric_records = [_metric_dict(r) for r in per_problem if r.get("status", "ok") == "ok"]
    summary = summarize(
        metric_records, n_boot=cfg.analysis.n_boot, alpha=cfg.analysis.alpha, seed=cfg.run.seed
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _chain_diagnostics(cfg, backend, chain: str, n_prefixes: int) -> dict:
    """Per-chain facts the trajectory cannot carry.

    Trajectory length is the number of prefixes that yielded an extractable answer, so
    on its own it distinguishes neither a capped chain from an uncapped one nor a long
    chain from a lossy extraction. These fields separate the three.
    """
    tokenizer = getattr(backend, "tokenizer", None)
    n_tokens = len(tokenizer.encode(chain)) if tokenizer is not None else None
    budget = getattr(backend, "reference_budget", None)
    return {
        "reference_chars": len(chain),
        "reference_tokens": n_tokens,
        "reference_truncated": (
            None if n_tokens is None or budget is None else n_tokens >= budget
        ),
        "raw_units": count_units(
            chain, cfg.segmentation.strategy, cfg.segmentation.window_tokens
        ),
        "prefixes": n_prefixes,
    }


def _process_one(cfg, backend, extract, eps, i, ex) -> dict:
    """Process one example into a record. status='skipped' when no usable trajectory."""
    chain = backend.reference_chain(ex.question)
    prefixes = cumulative_prefixes(
        chain,
        strategy=cfg.segmentation.strategy,
        window_tokens=cfg.segmentation.window_tokens,
        max_steps=cfg.segmentation.max_steps,
    )
    step_answers: list[list[str]] = []
    for pfx in prefixes:
        comps = backend.continue_from(ex.question, pfx, cfg.sampling.m)
        answers = [a for a in (extract(c) for c in comps) if a]
        if answers:
            step_answers.append(answers)

    diag = _chain_diagnostics(cfg, backend, chain, len(prefixes))
    diag["extracted_prefixes"] = len(step_answers)

    if len(step_answers) < 2:
        return {"index": i, "status": "skipped", **diag}

    traj = entropy_trajectory(step_answers)
    last = step_answers[-1]
    pred = _majority(last)
    final_confidence = last.count(pred) / len(last) if last else 0.0
    return {
        "index": i,
        "status": "ok",
        "gold": ex.gold,
        "pred": pred,
        "trajectory": traj,
        "monotone": is_monotone(traj, eps),
        "violations": violation_count(traj, eps),
        "coherence": coherence(traj),
        "final_confidence": final_confidence,
        "final_entropy": traj[-1],
        "correct": pred == ex.gold,
        **diag,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the entropy-drift pilot.")
    ap.add_argument("--config", required=True, help="Path to a YAML run config.")
    ap.add_argument("--resume", action="store_true", help="Continue an existing run.")
    ap.add_argument("--overwrite", action="store_true", help="Restart an existing run from scratch.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"Running '{cfg.run.name}' (backend={cfg.backend}, dataset={cfg.dataset.name})")
    summary = run(cfg, resume=args.resume, overwrite=args.overwrite)
    print()
    print(format_report(summary))
    print(f"\nWrote results to {os.path.join(cfg.run.out_dir, cfg.run.name)}/")


if __name__ == "__main__":
    main()
