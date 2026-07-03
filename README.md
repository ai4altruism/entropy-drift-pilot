# entropy-drift-pilot

A4A's first ML-legible arXiv artifact: an independent reproduction and reasoning-model
stress-test of the entropy-trajectory reasoning-reliability signal from **Zhao (2026)**,
run as the Phase-1 pilot of the **Reasoning Drift** program.

Planning artifact (goals, scope, decisions, venues) lives in the `offload` wiki:
`wiki/analyses/entropy-drift-pilot-2026.md`. This repo holds the code and experiments.

## What it measures

For a chain-of-thought answer, at each reasoning-step boundary we sample `m` short
completions, extract the final answer from each, compute the Shannon entropy of the
answer distribution, and test whether that entropy trajectory decreases monotonically.
The **shape** (is it monotone?) predicts correctness; the **magnitude** of total entropy
drop does not. See the plan and `src/entropydrift/trajectory.py`.

## Contributions (locked 2026-07-03)

1. **Replication** of the shape-over-magnitude dissociation + graded violation-count signal
   on GSM8K and MATH-500 (anchor: Qwen2.5-7B-Instruct).
2. **Reasoning-model stress-test** (headline novelty): does the signal survive on a
   reasoning-distilled model (DeepSeek-R1-Distill-Qwen-7B) that Zhao never tested?
   Step-segmentation of unstructured reasoning traces is the crux.
3. **False-positive-reduction delta**: a small learned combination (violation-count +
   final-answer confidence) reported as a triage filter with a cost/accuracy curve.

## Layout

```
docs/               environment.md (compute setup), preregistration.md (OSF pre-reg)
configs/            YAML run configs (smoke, full)
src/entropydrift/   library
  trajectory.py     entropy / monotonicity / violation-count (pure, tested)
  metrics.py        shape-vs-magnitude stats, accuracy gaps (pure, tested)
  answers.py        GSM8K / MATH answer extraction + normalization (pure, tested)
  segment.py        step-segmentation strategies (pure, tested)
  metrics.py        shape-vs-magnitude stats, accuracy gaps (pure, tested)
  fpreduce.py       learned triage filter vs the monotonicity baseline (contribution 3)
  backends.py       generation backends: mock, transformers, vllm, openai-compatible
  datasets.py       GSM8K / MATH-500 loaders
  run.py            orchestration: build trajectories over a dataset, write results
  provenance.py     per-run manifest (config hash, seeds, versions)
scripts/
  smoke_test.py     end-to-end on the mock backend (no model needed)
  fp_reduce.py      post-hoc false-positive-reduction analysis over a completed run
tests/              unit tests for the pure logic
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # verify the pure logic
python scripts/smoke_test.py    # end-to-end on the mock backend (no GPU)
```

## Running against a real model

The default smoke test uses a **mock** backend so the pipeline is verifiable without a
GPU or API key. To run the actual reproduction:

- **Local** (GPU): `pip install -r requirements-local.txt`, set `backend: transformers`
  and `model.name: Qwen/Qwen2.5-7B-Instruct` in a config. Add `model.quantization: 4bit`
  to fit a 7-8B model on a small (12GB) GPU (dev only; confirmatory runs are fp16).
- **Panel throughput** (recommended for the full run): `pip install -r requirements-vllm.txt`
  in a fresh venv, set `backend: vllm` (`configs/full-vllm.yaml`). Same true token-level
  continuation, batched via vLLM.
- **Hosted** (OpenAI-compatible endpoint): set `backend: openai_compatible`,
  `model.base_url`, and `model.api_key_env`. Note: this path uses an approximate
  prefix-in-prompt continuation, not the headline method.

First real milestone: reproduce the shape signal on **Qwen2.5-7B-Instruct / GSM8K n=300**
before scaling the panel.

```bash
python -m entropydrift.run --config configs/smoke.yaml
```

## False-positive reduction (contribution 3)

The bare "trust if monotone" rule has a high false-positive rate. `fpreduce.py` fits a
small logistic filter over cheap signals (violation-count + final-answer confidence by
default) and compares it to the monotonicity baseline out-of-sample. Run it over any
completed run directory:

```bash
python scripts/fp_reduce.py results/<run-name>
# writes results/<run-name>/fp_analysis.json (weights, baseline point, curve, FP reduction)
```

The full coverage-vs-selective-accuracy `curve` is the primary artifact; the matched-point
FP-reduction number is a convenience summary (predicted probabilities are tied at small m).
