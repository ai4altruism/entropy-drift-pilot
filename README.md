# entropy-drift-pilot

A4A's first ML-legible arXiv artifact: an independent reproduction and reasoning-model
stress-test of the entropy-trajectory reasoning-reliability signal from **Zhao (2026)**,
run as the Phase-1 pilot of the **Reasoning Drift** program.

**The paper under reproduction:** Xinghao Zhao, *Entropy Trajectory Shape Predicts LLM
Reasoning Reliability*, [arXiv:2603.18940v2](https://arxiv.org/abs/2603.18940), 30 March
2026 (Huazhong University of Science and Technology).

This repo holds the code and experiments. The pre-registered protocol lives in
`docs/preregistration.md` and the compute setup in `docs/environment.md`; between them
they carry the scope, the fixed parameters, and the analysis plan.

## What it measures

For a chain-of-thought answer, at each reasoning-step boundary we sample `m` short
completions, extract the final answer from each, compute the Shannon entropy of the
answer distribution, and test whether that entropy trajectory decreases monotonically.
The **shape** (is it monotone?) predicts correctness; the **magnitude** of total entropy
drop does not. See `docs/preregistration.md` and `src/entropydrift/trajectory.py`.

## Reading a record

Each line of `results/<run>/records.jsonl` carries the trajectory and its statistics, plus
five per-chain diagnostics that the trajectory cannot express:

| field | meaning |
|---|---|
| `reference_tokens` / `reference_chars` | length of the reference chain (`None` when the backend has no tokenizer) |
| `reference_truncated` | did the chain hit `reference_max_tokens`, i.e. is it cut off |
| `raw_units` | segmentation units **before** the `max_steps` cap |
| `prefixes` | prefixes actually sampled, i.e. units after the cap, plus the empty prefix |
| `extracted_prefixes` | prefixes that yielded at least one extractable answer |

**`len(trajectory)` equals `extracted_prefixes`, not `prefixes`.** A prefix whose `m`
continuations produce no parseable answer contributes no trajectory point, so trajectory
length alone cannot tell a capped chain from a shorter one that parsed cleanly. Use
`raw_units` to ask whether the cap bound, and `prefixes - extracted_prefixes` to ask how
much extraction cost you.

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
configs/            YAML run configs (smoke, full, panel, exploratory, diagnostic)
src/entropydrift/   library
  trajectory.py     entropy / monotonicity / violation-count (pure, tested)
  metrics.py        shape-vs-magnitude stats, accuracy gaps (pure, tested)
  answers.py        GSM8K / MATH answer extraction + normalization (pure, tested)
  segment.py        step-segmentation strategies (pure, tested)
  stats.py          bootstrap CIs (the pre-registered primary inference)
  fpreduce.py       learned triage filter vs the monotonicity baseline (contribution 3)
  backends.py       generation backends: mock, transformers, vllm, openai-compatible
  datasets.py       GSM8K / MATH-500 loaders
  run.py            orchestration: build trajectories over a dataset, write results
  refscan.py        reference-only diagnostic scan: measure the chain, skip the trajectory
  config.py         YAML config schema + validation
  provenance.py     per-run manifest (config hash, seeds, versions)
scripts/
  smoke_test.py     end-to-end on the mock backend (no model needed)
  fp_reduce.py      post-hoc false-positive-reduction analysis over a completed run
  panel/            as-run analysis scripts behind the published panel tables
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

## Reference-only diagnostic scan

Some questions are about the **reference chain**, not the trajectory built on it: whether
a token budget still truncates, or how much of a spike in trajectory length the
`max_steps` cap alone accounts for. A cell cannot answer them cheaply, because almost all
of its cost is the `m` continuations at every prefix, and none of that work bears on the
question. `refscan` generates the reference chains and nothing else.

```bash
python -m entropydrift.refscan --config configs/diagnostic/qwen7b-math500-refscan-600.yaml
python -m entropydrift.refscan --config configs/diagnostic/qwen7b-math500-refscan-1280.yaml
# each writes results/<run-name>/{ref_records.jsonl, ref_manifest.json, ref_summary.json}
```

Run the pair: a single budget has nothing to compare against, and the registered cells
recorded no reference-chain token counts, so 600's own distribution was never measured
either. `--resume` continues an interrupted scan, and refuses to append to one started
under a different budget. `--batch-size` (default 32) sets how many chains go to the
backend per call; vLLM serves them in one pass, which is what makes a scan cheap.

**A scan is a diagnostic, not a cell.** It computes no entropy, scores no hypothesis, and
writes no quantity the registration mentions: its records have no `trajectory` field to
score. Two limits it does not paper over. Generation is sampled at temperature 0.7 and
vLLM is not bit-reproducible under a fixed seed, so a scan characterizes the chain-length
*distribution* a config produces rather than recovering the chains an earlier run
generated. And on the cap-versus-extraction question it gives a **bound**, not an answer:
`raw_units` and `prefixes` say how much of a spike the cap can explain, while measuring
the extraction loss on top of that still needs the continuations.

## False-positive reduction (contribution 3)

The bare "trust if monotone" rule has a high false-positive rate. `fpreduce.py` fits a
small logistic filter over cheap signals (violation-count + final-answer confidence by
default) and compares it to the monotonicity baseline out-of-sample. Run it over any
completed run directory:

```bash
python scripts/fp_reduce.py results/<run-name>
# writes results/<run-name>/fp_analysis.json (weights, baseline point, curve, FP reduction)
```

The full coverage-vs-selective-accuracy `curve` is the primary artifact. The single
FP-reduction number is a convenience summary and **not** a matched-coverage comparison:
predicted probabilities are tied at small `m`, so the threshold takes whole tie blocks and
actual coverage can land well above the baseline's. The report prints the miss when it
happens. Where the filter wins on coverage *and* selective accuracy, state that as
dominance rather than as a like-for-like false-positive rate.
