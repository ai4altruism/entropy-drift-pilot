"""Generation backends.

A backend produces (a) a reference chain-of-thought for a question, and (b) `n` sampled
continuations from a given reasoning prefix. The caller extracts answers from the raw
completions (extraction is dataset-specific, so it does not live here).

    reference_chain(question)            -> str
    continue_from(question, prefix, n)   -> list[str]   (n raw completions)

Three backends:
  MockBackend            deterministic synthetic convergence; no model, for plumbing/tests
  TransformersBackend    local HF model, TRUE token-level prefix continuation (real path)
  OpenAICompatibleBackend  hosted chat endpoint; prefix carried in the prompt (approximate)
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Protocol


class Backend(Protocol):
    def reference_chain(self, question: str) -> str: ...
    def continue_from(self, question: str, prefix: str, n: int) -> list[str]: ...


# --------------------------------------------------------------------------- helpers

_SYSTEM = (
    "Solve the problem step by step. Put each reasoning step on its own line. "
    "End with the final answer."
)


def _seed_from(*parts: object) -> int:
    h = hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16)


def mock_gold(question: str, seed: int = 0) -> int:
    """Deterministic hidden gold for a question under the mock backend (10..99).

    Shared with the synthetic dataset so its example golds match what the mock 'solves'.
    """
    return _seed_from("gold", seed, question) % 90 + 10


# --------------------------------------------------------------------------- mock


class MockBackend:
    """Synthetic backend that fabricates a believable shape-vs-correctness dissociation.

    Each question gets a deterministic hidden gold and an easy/hard label. Easy items
    converge toward the gold as the prefix deepens (monotone, usually correct); hard items
    fluctuate mid-chain (non-monotone, usually wrong). Emits completions in the requested
    answer format so the real extractors parse them unchanged. Fully deterministic.
    """

    def __init__(
        self,
        n_steps: int = 5,
        hard_fraction: float = 0.35,
        answer_format: str = "gsm8k",
        seed: int = 0,
    ):
        self.n_steps = n_steps
        self.hard_fraction = hard_fraction
        self.answer_format = answer_format
        self.seed = seed

    def _gold(self, question: str) -> int:
        return mock_gold(question, self.seed)

    def _is_hard(self, question: str) -> bool:
        r = (_seed_from("hard", self.seed, question) % 1000) / 1000.0
        return r < self.hard_fraction

    def _format(self, value: int) -> str:
        if self.answer_format == "math":
            return f"The reasoning leads here. \\boxed{{{value}}}"
        return f"The reasoning leads here. #### {value}"

    def reference_chain(self, question: str) -> str:
        return "\n\n".join(f"Step {i + 1}: work on the problem." for i in range(self.n_steps))

    def _p_gold(self, question: str, depth: int) -> float:
        if self._is_hard(question):
            # low base, a confusion dip mid-chain, noisy: yields spikes -> non-monotone
            mid = self.n_steps / 2
            dip = 0.15 * (1.0 - abs(depth - mid) / max(mid, 1))
            jitter = ((_seed_from("j", self.seed, question, depth) % 100) / 100.0 - 0.5) * 0.3
            return max(0.05, min(0.6, 0.25 - dip + jitter))
        # easy: monotonic ramp toward certainty
        return min(1.0, 0.30 + 0.16 * depth)

    def continue_from(self, question: str, prefix: str, n: int) -> list[str]:
        depth = prefix.count("Step ")
        gold = self._gold(question)
        p = self._p_gold(question, depth)
        rng = random.Random(_seed_from("cont", self.seed, question, depth))
        # a small distractor pool; more spread -> higher entropy
        distractors = [gold + d for d in (-2, -1, 1, 2, 3) if gold + d > 0]
        out = []
        for _ in range(n):
            value = gold if rng.random() < p else rng.choice(distractors)
            out.append(self._format(value))
        return out


# --------------------------------------------------------------------------- transformers


class TransformersBackend:
    """Local HuggingFace backend with true token-level prefix continuation.

    reference_chain: one sampled full CoT. continue_from: render the chat template with the
    prefix as a partial assistant turn, then continue generating n times. This is the
    methodologically clean path used for the real experiments.
    """

    def __init__(
        self,
        name: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        reference_max_tokens: int | None = None,
        dtype: str = "auto",
        device: str = "auto",
        quantization: str = "none",
        revision: str = "",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reference_budget = reference_max_tokens or max_tokens * 4
        self.quantization = quantization
        self.revision = revision or None
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=self.revision)

        model_kwargs: dict = {"device_map": device, "revision": self.revision}
        if quantization in ("4bit", "8bit"):
            # bitsandbytes path for small local GPUs (e.g. a 12GB card). NOT for the
            # confirmatory panel: quantization changes the output distribution the
            # trajectories measure, so headline runs stay fp16 (quantization="none").
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)

    def _render(self, question: str, prefix: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + prefix

    def _generate(self, prompt: str, n: int, max_tokens: int) -> list[str]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=max_tokens,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1] :]
        return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen]

    def reference_chain(self, question: str) -> str:
        return self._generate(self._render(question, ""), n=1, max_tokens=self.reference_budget)[0]

    def continue_from(self, question: str, prefix: str, n: int) -> list[str]:
        return self._generate(self._render(question, prefix), n=n, max_tokens=self.max_tokens)


# --------------------------------------------------------------------------- openai-compatible


class OpenAICompatibleBackend:
    """Hosted chat endpoint (vLLM server, or a cheap open-weight provider).

    Prefix continuation is approximated by carrying the reasoning-so-far in the prompt
    rather than as a true assistant prefill (most chat endpoints do not expose token-level
    continuation). Documented as approximate; prefer TransformersBackend for the paper's
    headline runs.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.7,
        max_tokens: int = 150,
        reference_max_tokens: int | None = None,
        timeout: int = 120,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get(api_key_env, "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reference_budget = reference_max_tokens or max_tokens * 4
        self.timeout = timeout

    def _chat(self, messages: list[dict], n: int, max_tokens: int) -> list[str]:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.name,
                "messages": messages,
                "n": n,
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return [c["message"]["content"] for c in resp.json()["choices"]]

    def reference_chain(self, question: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ]
        return self._chat(messages, n=1, max_tokens=self.reference_budget)[0]

    def continue_from(self, question: str, prefix: str, n: int) -> list[str]:
        user = question if not prefix else (
            f"{question}\n\nReasoning so far:\n{prefix}\n\n"
            "Continue from here and end with the final answer."
        )
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ]
        return self._chat(messages, n=n, max_tokens=self.max_tokens)


# --------------------------------------------------------------------------- vllm


class VLLMBackend:
    """In-process vLLM backend: true token-level prefix continuation, high throughput.

    Renders the chat template with the tokenizer, appends the reasoning prefix, and submits
    the result as a raw prompt to vLLM (so the continuation is genuine token-level, not the
    prefix-in-prompt approximation). vLLM batches the ``n`` samples of each call via paged
    attention, so this is the recommended path for the fp16 panel runs on cloud.

    ``quantization`` is a **vLLM** method name (e.g. "awq", "gptq") or None; it is not the
    bitsandbytes "4bit/8bit" of the transformers backend. Confirmatory runs are fp16
    (quantization None).
    """

    def __init__(
        self,
        name: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        reference_max_tokens: int | None = None,
        quantization: str | None = None,
        revision: str | None = None,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "auto",
    ):
        from vllm import LLM

        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reference_budget = reference_max_tokens or max_tokens * 4
        self.llm = LLM(
            model=name,
            revision=revision or None,
            quantization=quantization or None,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _render(self, question: str, prefix: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + prefix

    def _generate(self, prompt: str, n: int, max_tokens: int) -> list[str]:
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=self.temperature, max_tokens=max_tokens)
        out = self.llm.generate([prompt], sp, use_tqdm=False)
        return [o.text for o in out[0].outputs]

    def reference_chain(self, question: str) -> str:
        return self._generate(self._render(question, ""), n=1, max_tokens=self.reference_budget)[0]

    def reference_chains(self, questions: list[str]) -> list[str]:
        """One reference chain per question, served in a single batched pass.

        Used by the reference-only diagnostic scan, never by a cell. The panel walks one
        problem and one prefix at a time on purpose: that is the registered execution
        path, and restructuring it to save GPU-hours would perturb a registered run to
        save $20. A scan is off that path entirely and generates nothing but reference
        chains, so batching there costs no registered property and is the whole reason a
        scan takes minutes where a cell takes hours.

        vLLM returns outputs in prompt order, so the caller can zip them back.
        """
        from vllm import SamplingParams

        prompts = [self._render(q, "") for q in questions]
        sp = SamplingParams(
            n=1, temperature=self.temperature, max_tokens=self.reference_budget
        )
        out = self.llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in out]

    def continue_from(self, question: str, prefix: str, n: int) -> list[str]:
        return self._generate(self._render(question, prefix), n=n, max_tokens=self.max_tokens)


# --------------------------------------------------------------------------- factory


def make_backend(cfg) -> Backend:
    """Construct a backend from a resolved config (see config.py)."""
    kind = cfg.backend
    m = cfg.model
    if kind == "mock":
        return MockBackend(
            answer_format=("math" if cfg.dataset.name == "math500" else "gsm8k"),
            seed=cfg.run.seed,
        )
    if kind == "transformers":
        return TransformersBackend(
            name=m.name,
            temperature=cfg.sampling.temperature,
            max_tokens=cfg.sampling.max_tokens,
            reference_max_tokens=cfg.sampling.reference_budget,
            quantization=m.quantization,
            revision=m.revision,
        )
    if kind == "openai_compatible":
        return OpenAICompatibleBackend(
            name=m.name,
            base_url=m.base_url,
            api_key_env=m.api_key_env,
            temperature=cfg.sampling.temperature,
            max_tokens=cfg.sampling.max_tokens,
            reference_max_tokens=cfg.sampling.reference_budget,
        )
    if kind == "vllm":
        return VLLMBackend(
            name=m.name,
            temperature=cfg.sampling.temperature,
            max_tokens=cfg.sampling.max_tokens,
            reference_max_tokens=cfg.sampling.reference_budget,
            quantization=(None if m.quantization in ("none", "") else m.quantization),
            revision=m.revision,
            gpu_memory_utilization=cfg.vllm.gpu_memory_utilization,
            max_model_len=cfg.vllm.max_model_len,
            dtype=cfg.vllm.dtype,
        )
    raise ValueError(f"unknown backend: {kind!r}")
