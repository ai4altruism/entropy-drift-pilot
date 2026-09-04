"""run.capture_text: keep the text a run would otherwise discard.

The point of the flag is that it changes nothing except what is stored, so the tests
that matter are the two boundary ones: off must serialize exactly as before, and a
skipped record must still be traced, since a chain the segmenter cannot split is the
case most worth looking at and the one that gets dropped.
"""
from types import SimpleNamespace

import pytest

from entropydrift.config import Config
from entropydrift.run import _process_one

EX = SimpleNamespace(question="q", gold="42")


class Segmented:
    """Three blank-line units, every continuation parseable."""

    def reference_chain(self, q):
        return "Step 1: a.\n\nStep 2: b.\n\nStep 3: c."

    def continue_from(self, q, pfx, n):
        return ["the answer is 42"] * n


class Unsegmentable:
    """One line, no blank lines, nothing parseable: the Mistral failure shape."""

    def reference_chain(self, q):
        return "Step 1: a. Step 2: b. Step 3: c."

    def continue_from(self, q, pfx, n):
        return ["no number here"] * n


def _cfg(capture):
    c = Config()
    c.run.capture_text = capture
    c.segmentation.strategy = "blank_line"
    c.segmentation.max_steps = 8
    c.sampling.m = 5
    return c


def test_off_by_default():
    assert Config().run.capture_text is False


def test_capture_off_stores_no_trace():
    rec = _process_one(_cfg(False), Segmented(), lambda c: "42", 0.01, 0, EX)
    assert rec["status"] == "ok"
    assert "trace" not in rec


def test_capture_on_stores_chain_units_and_continuations():
    rec = _process_one(_cfg(True), Segmented(), lambda c: "42", 0.01, 0, EX)
    t = rec["trace"]
    assert t["reference_text"] == Segmented().reference_chain("q")
    assert len(t["units"]) == 3
    # the empty prefix plus one per unit
    assert len(t["prefix_traces"]) == len(t["units"]) + 1
    assert len(t["prefix_traces"][0]["continuations"]) == 5
    assert t["prefix_traces"][0]["extracted"] == ["42"] * 5


def test_skipped_records_are_traced_and_show_parse_failures():
    rec = _process_one(_cfg(True), Unsegmentable(), lambda c: None, 0.01, 0, EX)
    assert rec["status"] == "skipped"
    t = rec["trace"]
    # one unit is why it was skipped, and the trace has to show that
    assert len(t["units"]) == 1
    assert t["prefix_traces"][0]["extracted"] == [None] * 5


def test_capped_units_matches_the_prefixes_actually_measured():
    from entropydrift.segment import capped_units, cumulative_prefixes

    text = "\n\n".join(f"u{i}" for i in range(20))
    units = capped_units(text, "blank_line", 40, max_steps=8)
    prefixes = cumulative_prefixes(text, "blank_line", 40, max_steps=8)
    assert len(units) == 8
    assert len(prefixes) == len(units) + 1
    # the tail is merged rather than dropped: nothing is lost from the chain
    assert "u19" in units[-1]
