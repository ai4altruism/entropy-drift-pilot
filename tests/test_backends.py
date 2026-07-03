import pytest

from entropydrift.backends import MockBackend, make_backend
from entropydrift.config import Config


def test_make_backend_mock_default():
    b = make_backend(Config())
    assert isinstance(b, MockBackend)


def test_make_backend_unknown_raises():
    cfg = Config(backend="does-not-exist")
    with pytest.raises(ValueError):
        make_backend(cfg)


def test_mock_backend_trajectory_shape():
    # a mock continuation returns exactly n completions and converges with depth
    b = MockBackend(seed=0)
    chain = b.reference_chain("Synthetic problem 1")
    assert chain.count("Step ") == 5
    early = b.continue_from("Synthetic problem 1", "", 5)
    assert len(early) == 5
