"""Orchestration: build entropy trajectories over a dataset and score the shape signals.

For each example: sample a reference chain, segment it into cumulative prefixes, sample
`m` continuations at each prefix, extract answers, build the entropy trajectory, and
record monotonicity / violation-count / coherence / correctness. Writes per-problem
records, a metrics summary, and a provenance manifest.
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
from .segment import cumulative_prefixes
from .trajectory import coherence, entropy_trajectory, is_monotone, violation_count


def _extractor(dataset_name: str):
    return extract_math_answer if dataset_name == "math500" else extract_final_number


def _majority(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0] if answers else ""


def run(cfg: Config, progress: bool = True) -> dict:
    backend = make_backend(cfg)
    examples = load_examples(cfg)
    extract = _extractor(cfg.dataset.name)
    eps = cfg.monotonicity.eps

    records: list[dict] = []
    per_problem: list[dict] = []

    for i, ex in enumerate(examples):
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

        if len(step_answers) < 2:
            continue  # not enough parseable steps to form a trajectory

        traj = entropy_trajectory(step_answers)
        last = step_answers[-1]
        pred = _majority(last)
        # final-answer confidence = self-consistency agreement at the last step
        final_confidence = last.count(pred) / len(last) if last else 0.0
        correct = pred == ex.gold
        rec = {
            "monotone": is_monotone(traj, eps),
            "violations": violation_count(traj, eps),
            "coherence": coherence(traj),
            "final_confidence": final_confidence,
            "final_entropy": traj[-1],
            "correct": correct,
        }
        records.append(rec)
        per_problem.append(
            {"index": i, "gold": ex.gold, "pred": pred, "trajectory": traj, **rec}
        )
        if progress and (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(examples)}")

    summary = summarize(records)
    _write_outputs(cfg, per_problem, summary)
    return summary


def _write_outputs(cfg: Config, per_problem: list[dict], summary: dict) -> None:
    out_dir = os.path.join(cfg.run.out_dir, cfg.run.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "w") as f:
        for r in per_problem:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(build_manifest(cfg), f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the entropy-drift pilot.")
    ap.add_argument("--config", required=True, help="Path to a YAML run config.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(f"Running '{cfg.run.name}' (backend={cfg.backend}, dataset={cfg.dataset.name})")
    summary = run(cfg)
    print()
    print(format_report(summary))
    print(f"\nWrote results to {os.path.join(cfg.run.out_dir, cfg.run.name)}/")


if __name__ == "__main__":
    main()
