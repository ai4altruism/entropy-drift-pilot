from entropydrift.config import Config, ModelCfg
from entropydrift.provenance import build_manifest


def test_manifest_core_fields():
    m = build_manifest(Config())
    assert "created_utc" in m
    assert "config" in m
    assert len(m["config_sha256"]) == 64
    assert set(m["packages"]) >= {"numpy", "scipy"}


def test_manifest_resolved_section():
    m = build_manifest(Config())
    r = m["resolved"]
    for key in (
        "git",
        "model_name",
        "model_revision_config",
        "model_commit_sha",
        "gpu",
        "cuda",
        "torch",
        "lockfile_sha256",
    ):
        assert key in r
    # mock model resolves to no Hub SHA
    assert r["model_name"] == "mock"
    assert r["model_commit_sha"] is None
    # git info is either a well-formed dict (in a repo) or None (out of one)
    assert r["git"] is None or set(r["git"]) == {"commit", "dirty"}


def test_manifest_records_configured_revision():
    cfg = Config(model=ModelCfg(name="Qwen/Qwen2.5-7B-Instruct", revision="abc123"))
    r = build_manifest(cfg)["resolved"]
    assert r["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert r["model_revision_config"] == "abc123"


def test_revision_changes_config_hash():
    a = build_manifest(Config(model=ModelCfg(name="m", revision="")))
    b = build_manifest(Config(model=ModelCfg(name="m", revision="v1")))
    assert a["config_sha256"] != b["config_sha256"]
