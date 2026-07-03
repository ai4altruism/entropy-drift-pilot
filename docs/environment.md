# Experimental Environment Configuration

How to provision and configure a reproducible environment to run the entropy-drift pilot.
Decisions locked 2026-07-03: **rented cloud GPU**, budget ceiling **$100-500**, headline
runs use the `transformers` backend's **true token-level prefix continuation**.

## 1. Why a GPU we control

The paper's headline method is token-level continuation from a reasoning prefix: render
the model's chat template, append the prefix as a partial assistant turn, and continue
generation. Hosted chat APIs do not expose this, so their prefix-in-prompt approximation
(`openai_compatible` backend) is reserved for sensitivity checks only. A rented single GPU
runs the clean `transformers` path and keeps cost trivial at this scale.

## 1a. Local dev box vs cloud (precision matters)

A small local GPU (e.g. 12 GB) cannot hold a 7-8B model in fp16 (weights alone are
~14-16 GB); it can only run **4-bit quantized** (~4 GB weights, fits with KV headroom).
Quantization changes the output distribution the trajectories measure, so:

- **Local 12 GB box, 4-bit:** development, plumbing, and the offline anchor smoke test
  (`configs/smoke-local.yaml`). Also the pre-registered **quantization-robustness**
  exploratory (4-bit vs fp16 on the anchor). Free and offline.
- **Cloud GPU, fp16:** all **confirmatory** runs (H1-H5) and the panel, one precision
  across all four models so the cross-model comparison (H4) is clean.

Enable 4-bit with `model.quantization: 4bit` (needs `bitsandbytes`, in
`requirements-local.txt`).

## 2. Hardware target

- **Models:** four 7-8B open-weight models, run **one at a time** (sequential across the
  panel), so a single GPU suffices. No multi-GPU.
  - `Qwen/Qwen2.5-7B-Instruct` (anchor)
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `meta-llama/Llama-3.1-8B-Instruct` (gated: accept the license, needs an HF token)
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (reasoning-distilled)
- **GPU:** a single **40-80 GB** card. bf16 weights are ~15 GB; the rest is headroom for
  batched sampling (`num_return_sequences = m`). A100 40 GB is sufficient; L40S/A6000
  48 GB or H100 80 GB give more batch throughput (useful for the longer distilled chains).
- **Disk:** ~15 GB per model plus HF cache. Provision **~150 GB**.
- **Providers:** Lambda Cloud, RunPod, or vast.ai. Indicative rates: A100 40 GB
  ~$1.1-1.5/hr, H100 80 GB ~$2-3/hr (interruptible cheaper).

## 3. Compute budget estimate

Per problem the harness generates: one reference chain (long) plus, for each of up to
`max_steps + 1` (~9) prefixes, `m = 5` short (<=150 token) continuations, so ~46
generations/problem. Full run volume:

| Dataset | n | gens/model (~46 x n) |
|---|---|---|
| GSM8K (full) | 1319 | ~61k |
| MATH-500 | 500 | ~23k |

~84k generations/model x 4 models = **~340k short generations** total. On a 40-80 GB GPU
with batched decoding of 7-8B models this is on the order of **10-30 GPU-hours** across the
panel (the distilled model is slowest, longer chains). At $1.5-3/hr that is **~$20-90**,
well inside the $100-500 ceiling with room for the pre-specified sensitivity sweeps
(segmentation strategies; `m = 5/10/20` on a dev subset).

> [!note] Throughput
> The `transformers` backend issues many `generate()` calls and is the **correctness
> reference**, not a throughput engine. For the full panel, the recommended production
> path is a **local vLLM server hit on its completions endpoint** with the rendered chat
> template + prefix (true token-level continuation, batched). That is a small backend
> addition (see Open code tasks) and cuts wall-clock by an order of magnitude. Keep a
> `transformers`-run validation subset to confirm the vLLM path matches.

## 4. Software environment

- **OS / driver:** Ubuntu 22.04, CUDA 12.x driver (provider images ship this).
- **Python:** 3.10-3.12, project `.venv`.
- **Install:**
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  pip install -r requirements-local.txt   # torch build matching the image's CUDA
  ```
- **HF auth (gated models):** `huggingface-cli login` or `HUGGING_FACE_HUB_TOKEN`; accept
  the Llama-3.1 license once on the Hub.
- **Version pinning:** after install, capture `pip freeze > requirements-lock.txt` and
  commit it for the run. Pin each model to a **specific Hub revision SHA** and record it
  (see provenance below); "latest" is not reproducible.

## 5. Reproducibility and provenance

Sampling at `temperature = 0.7` is stochastic, so reproducibility means: fixed seed +
pinned model revision + pinned deps + recorded configs, with results reported as
**distributions with bootstrap CIs**, not point values.

- Fixed `run.seed` per (model, dataset) cell; unique `run.name` per cell.
- `manifest.json` already records config hash + package versions. **Extend
  `provenance.py`** to also capture: model revision SHA, GPU name, CUDA/torch versions, and
  the `requirements-lock.txt` hash. (Open code task.)
- Tag the git commit used for the official run (e.g. `run-2026-07`), so configs + lockfile
  + code are one referenceable state.

## 6. Run procedure

1. Provision the GPU instance; clone the repo; install (Section 4).
2. `pytest -q` and `python scripts/smoke_test.py` (mock, no model) to confirm plumbing.
3. **Local real-model check (optional, 4-bit):** on the small GPU,
   `python -m entropydrift.run --config configs/smoke-local.yaml` to confirm the harness
   works against a real (quantized) model, offline.
4. **Anchor smoke (confirmatory, fp16 on cloud):**
   `python -m entropydrift.run --config configs/smoke.yaml` (Qwen2.5-7B / GSM8K n=300).
   Confirm the shape signal reproduces Zhao before scaling.
5. **Panel:** duplicate `configs/full.yaml` per (model x dataset) cell with a unique
   `run.name`; run each. Then `python scripts/fp_reduce.py results/<run-name>` per cell.
6. Pull `results/` down (or push to object storage); **tear down the instance** to stop
   billing.

## 7. Cost controls

- Spin the instance down when idle; models re-download from cache-on-disk if the volume
  persists, else budget re-download time.
- Interruptible/spot instances are safe **once** incremental result writing lands (Open
  code task) so a preemption does not lose a partial run.

## 8. Open code tasks this plan surfaces

Tracked so they are not lost:

1. **vLLM completions backend** (true continuation, batched) for panel-scale throughput.
2. **Incremental / resumable result writing** in `run.py` (spot-safe, long-run-safe).
3. **Extend `provenance.py`** to record model revision SHA, GPU name, CUDA/torch, lockfile
   hash.
4. ~~**Bootstrap-CI reporting**~~ **done** (`stats.py`; wired into `metrics.summarize`,
   `fpreduce.evaluate`, and the run output; `analysis.n_boot`/`analysis.alpha` config).
