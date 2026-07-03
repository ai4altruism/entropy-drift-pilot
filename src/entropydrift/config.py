"""Typed run configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import yaml


@dataclass
class ModelCfg:
    name: str = "mock"
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    quantization: str = "none"  # none | 4bit | 8bit (transformers backend, local dev only)
    revision: str = ""  # pin a specific Hub commit/tag for reproducibility ("" = default)


@dataclass
class SamplingCfg:
    m: int = 5
    temperature: float = 0.7
    max_tokens: int = 150


@dataclass
class DatasetCfg:
    name: str = "gsm8k"  # gsm8k | math500 | synthetic
    split: str = "test"
    limit: int = 300


@dataclass
class SegmentationCfg:
    strategy: str = "blank_line"  # blank_line | newline | sentence | token_window
    window_tokens: int = 40
    max_steps: int = 8


@dataclass
class MonotonicityCfg:
    eps: float = 0.01


@dataclass
class AnalysisCfg:
    n_boot: int = 1000  # bootstrap resamples for CIs (0 disables)
    alpha: float = 0.05  # 1 - alpha = CI level (0.05 -> 95%)


@dataclass
class VLLMCfg:
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    dtype: str = "auto"


@dataclass
class RunCfg:
    seed: int = 0
    out_dir: str = "results"
    name: str = "run"


@dataclass
class Config:
    backend: str = "mock"  # mock | transformers | openai_compatible
    model: ModelCfg = field(default_factory=ModelCfg)
    sampling: SamplingCfg = field(default_factory=SamplingCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    segmentation: SegmentationCfg = field(default_factory=SegmentationCfg)
    monotonicity: MonotonicityCfg = field(default_factory=MonotonicityCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    vllm: VLLMCfg = field(default_factory=VLLMCfg)
    run: RunCfg = field(default_factory=RunCfg)

    def to_dict(self) -> dict:
        return asdict(self)


_SECTIONS = {
    "model": ModelCfg,
    "sampling": SamplingCfg,
    "dataset": DatasetCfg,
    "segmentation": SegmentationCfg,
    "monotonicity": MonotonicityCfg,
    "analysis": AnalysisCfg,
    "vllm": VLLMCfg,
    "run": RunCfg,
}


def from_dict(d: dict) -> Config:
    kwargs = {"backend": d.get("backend", "mock")}
    for key, cls in _SECTIONS.items():
        kwargs[key] = cls(**(d.get(key) or {}))
    return Config(**kwargs)


def load_config(path: str) -> Config:
    with open(path) as f:
        return from_dict(yaml.safe_load(f) or {})
