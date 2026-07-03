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
