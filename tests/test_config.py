from entropydrift.config import Config, from_dict


def test_defaults():
    c = Config()
    assert c.backend == "mock"
    assert c.model.quantization == "none"
    assert c.analysis.n_boot == 1000
    assert c.sampling.m == 5
    assert c.vllm.gpu_memory_utilization == 0.9
    assert c.vllm.max_model_len is None


def test_from_dict_parses_vllm_section():
    c = from_dict(
        {
            "backend": "vllm",
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct"},
            "vllm": {"gpu_memory_utilization": 0.85, "max_model_len": 4096},
        }
    )
    assert c.backend == "vllm"
    assert c.vllm.gpu_memory_utilization == 0.85
    assert c.vllm.max_model_len == 4096
    assert c.vllm.dtype == "auto"


def test_from_dict_parses_quantization_and_sections():
    c = from_dict(
        {
            "backend": "transformers",
            "model": {"name": "Qwen/Qwen2.5-7B-Instruct", "quantization": "4bit"},
            "dataset": {"name": "gsm8k", "limit": 100},
            "analysis": {"n_boot": 500, "alpha": 0.1},
        }
    )
    assert c.backend == "transformers"
    assert c.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert c.model.quantization == "4bit"
    assert c.dataset.limit == 100
    assert c.analysis.n_boot == 500
    assert c.analysis.alpha == 0.1


def test_from_dict_empty_uses_defaults():
    c = from_dict({})
    assert c.model.quantization == "none"
    assert c.run.seed == 0


def test_reference_budget_defaults_to_four_times_max_tokens():
    """Default must reproduce the historical 4x behavior exactly."""
    from entropydrift.config import SamplingCfg

    assert SamplingCfg().reference_budget == 600
    assert SamplingCfg(max_tokens=150).reference_budget == 600
    assert SamplingCfg(max_tokens=256).reference_budget == 1024


def test_reference_budget_is_settable_without_touching_continuations():
    """The m continuations are the measuring instrument and must not move."""
    from entropydrift.config import SamplingCfg

    s = SamplingCfg(max_tokens=150, reference_max_tokens=1280)
    assert s.reference_budget == 1280
    assert s.max_tokens == 150
