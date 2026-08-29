"""Dataset loaders. Returns a list of Example(question, gold), gold pre-normalized.

  gsm8k      HuggingFace 'openai/gsm8k' (main config)
  math500    HuggingFace 'HuggingFaceH4/MATH-500'
  synthetic  offline fixtures whose golds match the mock backend (for smoke/tests)
"""

from __future__ import annotations

from dataclasses import dataclass

from .answers import extract_gsm8k_gold, extract_boxed, normalize_math
from .backends import mock_gold

# Hub repo ids, pinned. Must be fully qualified "namespace/name": huggingface_hub
# rejects bare legacy ids such as "gsm8k" with HfUriError.
GSM8K_REPO = "openai/gsm8k"
MATH500_REPO = "HuggingFaceH4/MATH-500"


@dataclass
class Example:
    question: str
    gold: str


def _load_gsm8k(split: str, limit: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset(GSM8K_REPO, "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return [Example(x["question"], extract_gsm8k_gold(x["answer"])) for x in ds]


def _load_math500(split: str, limit: int) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset(MATH500_REPO, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for x in ds:
        gold = x.get("answer") or extract_boxed(x.get("solution", ""))
        out.append(Example(x["problem"], normalize_math(gold)))
    return out


def _load_synthetic(limit: int, seed: int) -> list[Example]:
    limit = limit or 200
    out = []
    for i in range(limit):
        q = f"Synthetic problem {i}"
        out.append(Example(q, str(mock_gold(q, seed))))
    return out


def load_examples(cfg) -> list[Example]:
    name = cfg.dataset.name
    if name == "gsm8k":
        return _load_gsm8k(cfg.dataset.split, cfg.dataset.limit)
    if name == "math500":
        return _load_math500(cfg.dataset.split, cfg.dataset.limit)
    if name == "synthetic":
        return _load_synthetic(cfg.dataset.limit, cfg.run.seed)
    raise ValueError(f"unknown dataset: {name!r}")
