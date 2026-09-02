import json
import os

import pytest

from entropydrift.config import AnalysisCfg, Config, DatasetCfg, RunCfg
from entropydrift.run import run


def _cfg(tmp_path, name="t", limit=30):
    return Config(
        backend="mock",
        dataset=DatasetCfg(name="synthetic", limit=limit),
        analysis=AnalysisCfg(n_boot=50),
        run=RunCfg(out_dir=str(tmp_path), name=name),
    )


def _records(run_dir):
    with open(os.path.join(run_dir, "records.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_full_run_writes_records_and_summary(tmp_path):
    cfg = _cfg(tmp_path)
    summary = run(cfg, progress=False)
    run_dir = os.path.join(str(tmp_path), "t")
    assert os.path.exists(os.path.join(run_dir, "manifest.json"))
    assert os.path.exists(os.path.join(run_dir, "summary.json"))
    recs = _records(run_dir)
    # every example gets exactly one record (ok or skipped)
    assert len(recs) == 30
    assert {r["index"] for r in recs} == set(range(30))
    assert summary["n"] == sum(1 for r in recs if r.get("status") == "ok")


def test_rerun_without_resume_raises(tmp_path):
    cfg = _cfg(tmp_path)
    run(cfg, progress=False)
    with pytest.raises(FileExistsError):
        run(cfg, progress=False)


def test_resume_completes_after_truncation(tmp_path):
    cfg = _cfg(tmp_path)
    full = run(cfg, progress=False)
    run_dir = os.path.join(str(tmp_path), "t")
    recs = _records(run_dir)

    # simulate an interruption: keep only the first 12 records
    with open(os.path.join(run_dir, "records.jsonl"), "w") as f:
        for r in recs[:12]:
            f.write(json.dumps(r) + "\n")

    resumed = run(cfg, progress=False, resume=True)
    recs2 = _records(run_dir)
    # all indices present again, no duplicates
    assert {r["index"] for r in recs2} == set(range(30))
    assert len(recs2) == 30
    # identical summary to the uninterrupted run (deterministic mock + fixed seed)
    assert resumed["n"] == full["n"]
    assert resumed["shape"]["gap_pp"] == full["shape"]["gap_pp"]


def test_overwrite_restarts_fresh(tmp_path):
    cfg = _cfg(tmp_path)
    run(cfg, progress=False)
    run_dir = os.path.join(str(tmp_path), "t")
    # corrupt the file with an extra stray record; overwrite should discard it
    with open(os.path.join(run_dir, "records.jsonl"), "a") as f:
        f.write(json.dumps({"index": 999, "status": "ok"}) + "\n")
    run(cfg, progress=False, overwrite=True)
    recs = _records(run_dir)
    assert {r["index"] for r in recs} == set(range(30))
    assert 999 not in {r["index"] for r in recs}


def test_resume_config_mismatch_raises(tmp_path):
    run(_cfg(tmp_path, limit=30), progress=False)
    # same run.name, different config -> manifest hash mismatch
    changed = _cfg(tmp_path, limit=20)
    changed.sampling.m = 9
    with pytest.raises(ValueError):
        run(changed, progress=False, resume=True)


def test_resume_drops_duplicate_indices(tmp_path):
    """A crash can leave an index on disk that the next read misses, so resume
    reprocesses it and appends a second copy. Reading must keep only the first."""
    import json

    from entropydrift.run import _load_jsonl

    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps({"index": 0, "correct": True}) + "\n"
        + json.dumps({"index": 1, "correct": False}) + "\n"
        + json.dumps({"index": 1, "correct": True}) + "\n"   # later duplicate
        + json.dumps({"index": 2, "correct": True}) + "\n"
    )
    recs = _load_jsonl(str(p))
    assert [r["index"] for r in recs] == [0, 1, 2]
    assert recs[1]["correct"] is False, "must keep the FIRST copy, not the later one"


def test_resume_tolerates_a_torn_final_line(tmp_path):
    """A kill mid-write leaves a half-written last line; it must not abort resume."""
    import json

    from entropydrift.run import _load_jsonl

    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps({"index": 0, "correct": True}) + "\n"
        + '{"index": 1, "corr'  # torn
    )
    recs = _load_jsonl(str(p))
    assert [r["index"] for r in recs] == [0]


def test_resume_survives_a_config_field_added_after_the_run(tmp_path):
    """A code upgrade that adds an optional config field must not lock you out of
    resuming runs written before it existed. Comparing whole-config hashes did:
    the new key changed every prior run's hash though no registered parameter moved,
    which cost a record in the 2026-09 panel."""
    cfg = _cfg(tmp_path)
    run(cfg, progress=False)
    run_dir = os.path.join(str(tmp_path), "t")

    # rewrite the manifest as an older version of the schema would have: without the
    # field the code has since gained
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    del manifest["config"]["sampling"]["reference_max_tokens"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    recs = _records(run_dir)
    with open(os.path.join(run_dir, "records.jsonl"), "w") as f:
        for r in recs[:12]:
            f.write(json.dumps(r) + "\n")

    run(cfg, progress=False, resume=True)   # must not raise
    assert {r["index"] for r in _records(run_dir)} == set(range(30))


def test_resume_refuses_when_the_added_field_is_actually_set(tmp_path):
    """The permissive case above must not become a hole: a field absent from the
    earlier run is fine only while it holds its default."""
    cfg = _cfg(tmp_path)
    run(cfg, progress=False)
    run_dir = os.path.join(str(tmp_path), "t")

    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    del manifest["config"]["sampling"]["reference_max_tokens"]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    cfg.sampling.reference_max_tokens = 1280
    with pytest.raises(ValueError, match="reference_max_tokens"):
        run(cfg, progress=False, resume=True)


def test_resume_mismatch_names_the_offending_key(tmp_path):
    cfg = _cfg(tmp_path)
    run(cfg, progress=False)
    changed = _cfg(tmp_path)
    changed.sampling.temperature = 0.9
    with pytest.raises(ValueError, match=r"sampling\.temperature: 0\.7 -> 0\.9"):
        run(changed, progress=False, resume=True)
