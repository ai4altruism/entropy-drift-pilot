from entropydrift.config import Config, from_dict


def test_defaults():
    c = Config()
    assert c.backend == "mock"
    assert c.model.quantization == "none"
    assert c.analysis.n_boot == 1000
    assert c.sampling.m == 5


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
