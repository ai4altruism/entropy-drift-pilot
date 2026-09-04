"""The reference-only diagnostic scan.

A scan exists to measure the reference chain without paying for the trajectory built on
it. Two properties matter and both are tested here: it must produce the same counts a
real cell records, or its numbers cannot be compared to the panel's; and it must remain
incapable of producing a confirmatory statistic, so it can never be mistaken for a cell.
"""

import json

import pytest

from entropydrift import refscan
from entropydrift.config import from_dict
from entropydrift.refscan import _percentile, scan, summarize_scan


def _chain(n_units: int) -> str:
    """A chain of ``n_units`` blank-line units, each two whitespace tokens long."""
    return "\n\n".join(f"step {i}" for i in range(n_units))


class _Backend:
    """A batched backend with a tokenizer, as VLLMBackend has. Budget 20 tokens.

    Each unit is two whitespace tokens, so a chain is truncated once it reaches 10 units.
    """

    reference_budget = 20

    class tokenizer:
        @staticmethod
        def encode(text):
            return text.split()

    def __init__(self, units_for=lambda q: 5):
        self.units_for = units_for
        self.batch_sizes = []

    def reference_chain(self, question):
        self.batch_sizes.append(1)
        return _chain(self.units_for(question))

    def reference_chains(self, questions):
        self.batch_sizes.append(len(questions))
        return [_chain(self.units_for(q)) for q in questions]


class _Unbatched:
    """A backend offering only the one-at-a-time call (mock, transformers, hosted)."""

    def __init__(self):
        self.calls = 0

    def reference_chain(self, question):
        self.calls += 1
        return _chain(3)


def _cfg(tmp_path, limit=8, budget=20, name="scan"):
    return from_dict(
        {
            "backend": "mock",  # ignored: make_backend is patched in these tests
            "dataset": {"name": "synthetic", "limit": limit},
            "sampling": {"max_tokens": 150, "reference_max_tokens": budget},
            "segmentation": {"strategy": "blank_line", "max_steps": 8},
            "run": {"out_dir": str(tmp_path), "name": name},
        }
    )


@pytest.fixture
def patched(monkeypatch):
    def install(backend):
        monkeypatch.setattr(refscan, "make_backend", lambda cfg: backend)
        return backend

    return install


# --------------------------------------------------------------------- end to end


def test_scan_writes_a_record_per_example_and_the_three_files(tmp_path, patched):
    patched(_Backend())
    cfg = _cfg(tmp_path, limit=8)
    summary = scan(cfg, batch_size=3, progress=False)

    out = tmp_path / "scan"
    assert (out / "ref_manifest.json").exists()
    assert (out / "ref_summary.json").exists()
    records = [json.loads(ln) for ln in (out / "ref_records.jsonl").read_text().splitlines()]
    assert [r["index"] for r in records] == list(range(8))
    assert summary["n"] == 8
    assert json.loads((out / "ref_summary.json").read_text()) == summary


def test_a_scan_record_cannot_carry_a_confirmatory_statistic(tmp_path, patched):
    """The guarantee that keeps a scan off the registered path: there is nothing to score.

    No trajectory, so no monotonicity, no violation count, no coherence, no correctness.
    A scan cannot be reported as a cell because it never computes what a cell computes.
    """
    patched(_Backend())
    scan(_cfg(tmp_path), batch_size=4, progress=False)
    rec = json.loads((tmp_path / "scan" / "ref_records.jsonl").read_text().splitlines()[0])

    for forbidden in ("trajectory", "monotone", "violations", "coherence", "correct", "pred"):
        assert forbidden not in rec, f"a scan must not record {forbidden!r}"
    assert set(rec) == {
        "index",
        "reference_chars",
        "reference_tokens",
        "reference_truncated",
        "raw_units",
        "prefixes",
    }


def test_scan_counts_agree_with_what_a_cell_records(tmp_path, patched):
    """raw_units is the pre-cap count and prefixes the post-cap count, as in run.py.

    Comparability with the panel is the whole point, so these must not drift: a 12-unit
    chain is 12 raw units and 9 prefixes under max_steps=8, exactly as a cell records it.
    """
    patched(_Backend(units_for=lambda q: 12))
    scan(_cfg(tmp_path, limit=1), progress=False)
    rec = json.loads((tmp_path / "scan" / "ref_records.jsonl").read_text().splitlines()[0])
    assert rec["raw_units"] == 12
    assert rec["prefixes"] == 9


# ------------------------------------------------------------------------ batching


def test_the_batched_backend_method_is_used_and_the_last_batch_is_short(tmp_path, patched):
    backend = patched(_Backend())
    scan(_cfg(tmp_path, limit=10), batch_size=4, progress=False)
    assert backend.batch_sizes == [4, 4, 2]


def test_backends_without_a_batched_method_still_work(tmp_path, patched):
    backend = patched(_Unbatched())
    summary = scan(_cfg(tmp_path, limit=5), batch_size=4, progress=False)
    assert backend.calls == 5
    assert summary["n"] == 5
    assert summary["tokens"] is None, "no tokenizer means no token figures, not guesses"


# -------------------------------------------------------------------------- resume


def test_resume_skips_completed_indices_and_does_not_regenerate_them(tmp_path, patched):
    backend = patched(_Backend())
    cfg = _cfg(tmp_path, limit=8)
    scan(cfg, batch_size=8, progress=False)

    backend.batch_sizes.clear()
    records_path = tmp_path / "scan" / "ref_records.jsonl"
    kept = records_path.read_text().splitlines()[:5]
    records_path.write_text("\n".join(kept) + "\n")

    summary = scan(cfg, batch_size=8, progress=True, resume=True)
    assert backend.batch_sizes == [3], "only the three missing indices are regenerated"
    assert summary["n"] == 8
    indices = [json.loads(ln)["index"] for ln in records_path.read_text().splitlines()]
    assert sorted(indices) == list(range(8))


def test_an_existing_scan_is_not_silently_appended_to(tmp_path, patched):
    patched(_Backend())
    cfg = _cfg(tmp_path)
    scan(cfg, progress=False)
    with pytest.raises(FileExistsError):
        scan(cfg, progress=False)


def test_resuming_under_a_different_budget_is_refused(tmp_path, patched):
    """The budget is the variable under study; mixing two into one file destroys it."""
    patched(_Backend())
    scan(_cfg(tmp_path, budget=20), batch_size=4, progress=False)
    with pytest.raises(ValueError, match="resume config mismatch"):
        scan(_cfg(tmp_path, budget=1280), batch_size=4, progress=False, resume=True)


# ------------------------------------------------------------------------ summary


def test_percentile_is_nearest_rank():
    xs = list(range(1, 101))
    assert _percentile(xs, 0.50) == 50
    assert _percentile(xs, 0.90) == 90
    assert _percentile(xs, 0.99) == 99
    assert _percentile([7], 0.99) == 7, "a single value is every percentile of itself"


def test_truncation_rate_counts_chains_at_or_above_the_budget():
    records = [
        {"reference_tokens": 600, "reference_truncated": True, "raw_units": 9, "prefixes": 9},
        {"reference_tokens": 400, "reference_truncated": False, "raw_units": 5, "prefixes": 6},
        {"reference_tokens": 200, "reference_truncated": False, "raw_units": 3, "prefixes": 4},
        {"reference_tokens": 610, "reference_truncated": True, "raw_units": 11, "prefixes": 9},
    ]
    s = summarize_scan(records, budget=600, max_steps=8)
    assert s["truncated"] == {"n": 2, "of": 4, "rate": 0.5}
    assert s["tokens"]["max"] == 610
    assert s["tokens"]["p50"] == 400
    assert s["reference_budget"] == 600


def test_the_summary_bounds_how_much_of_a_spike_the_cap_explains():
    """Question 2, stated at the strength the scan can support.

    Three of four chains reach max_steps, so the cap pins their prefixes at 9 and can
    account for at most that share of a spike in trajectory length. What extraction loses
    on top of it is not visible here, and the scan does not pretend otherwise.
    """
    records = [
        {"raw_units": u, "prefixes": min(u, 8) + 1, "reference_tokens": None,
         "reference_truncated": None}
        for u in (4, 8, 9, 12)
    ]
    s = summarize_scan(records, budget=600, max_steps=8)
    assert s["raw_units"]["at_or_above_max_steps"] == {"n": 3, "rate": 0.75}
    assert s["raw_units"]["histogram"] == {"4": 1, "8": 1, "9": 1, "12": 1}
    assert s["prefixes"]["histogram"] == {"5": 1, "9": 3}
    assert s["tokens"] is None and s["truncated"] is None


def test_a_zero_batch_size_is_rejected_before_anything_loads(tmp_path, patched):
    """A typo on a metered box should say so, not raise range()'s opaque complaint."""
    patched(_Backend())
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        scan(_cfg(tmp_path), batch_size=0, progress=False)


def test_an_empty_scan_summarizes_without_crashing():
    s = summarize_scan([], budget=600, max_steps=8)
    assert s["n"] == 0
    assert s["tokens"] is None and s["raw_units"] is None
