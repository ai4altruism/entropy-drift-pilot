"""Per-chain diagnostics recorded alongside the trajectory.

Trajectory length alone is ambiguous: it counts prefixes that yielded an extractable
answer, so a capped chain that lost one prefix to extraction is indistinguishable from
an uncapped chain one unit shorter. The September 2026 panel read a shift in that
distribution as evidence the segmentation cap had relaxed, which it could not show.
"""

from entropydrift.answers import extract_final_number
from entropydrift.config import Config
from entropydrift.run import _process_one

CHAIN_12 = "\n\n".join(f"step {i}" for i in range(12))
CHAIN_5 = "\n\n".join(f"step {i}" for i in range(5))


class _Example:
    question = "q"
    gold = "42"
    answer = "42"


class _Backend:
    """Answers every prefix except the ones listed in ``duds`` (1-based call order)."""

    def __init__(self, chain=CHAIN_12, duds=()):
        self.chain, self.duds, self.calls = chain, set(duds), 0

    def reference_chain(self, question):
        return self.chain

    def continue_from(self, question, prefix, n):
        self.calls += 1
        return ["no answer"] * n if self.calls in self.duds else ["The answer is 42"] * n


class _Tokenized(_Backend):
    """A backend with a tokenizer and a budget, as the real generation backends have."""

    reference_budget = 20

    class tokenizer:
        @staticmethod
        def encode(text):
            return text.split()


def _run(backend):
    return _process_one(Config(), backend, extract_final_number, 0.01, 0, _Example())


def test_raw_units_is_the_pre_cap_count_and_prefixes_the_post_cap_count():
    rec = _run(_Backend(CHAIN_12))
    assert rec["raw_units"] == 12, "must report the chain's real unit count"
    assert rec["prefixes"] == 9, "max_steps=8 caps units at 8, so 9 prefixes"


def test_trajectory_length_hides_the_cap_but_the_fields_do_not():
    """The exact confusion the panel hit: two very different chains, same trajectory."""
    capped_lossy = _run(_Backend(CHAIN_12, duds=[4]))   # 12 units, one prefix unextractable
    short_clean = _run(_Backend(CHAIN_5))               # 5 units, everything extracts

    assert len(capped_lossy["trajectory"]) == 8
    assert len(short_clean["trajectory"]) == 6
    # trajectory length alone would say the first chain has 7 units; it has 12
    assert capped_lossy["raw_units"] == 12
    assert capped_lossy["prefixes"] == 9
    assert capped_lossy["extracted_prefixes"] == 8
    assert short_clean["raw_units"] == 5
    assert short_clean["prefixes"] == short_clean["extracted_prefixes"] == 6


def test_extracted_prefixes_matches_trajectory_length():
    rec = _run(_Backend(CHAIN_12, duds=[2, 5]))
    assert rec["extracted_prefixes"] == len(rec["trajectory"]) == 7


def test_reference_tokens_and_truncation_flag():
    rec = _run(_Tokenized(CHAIN_12))         # 24 whitespace tokens, budget 20
    assert rec["reference_tokens"] == 24
    assert rec["reference_truncated"] is True
    rec = _run(_Tokenized(CHAIN_5))          # 10 tokens, under budget
    assert rec["reference_tokens"] == 10
    assert rec["reference_truncated"] is False


def test_backends_without_a_tokenizer_report_none_rather_than_guessing():
    rec = _run(_Backend(CHAIN_5))
    assert rec["reference_tokens"] is None
    assert rec["reference_truncated"] is None
    assert rec["reference_chars"] == len(CHAIN_5)


def test_skipped_records_carry_the_diagnostics_too():
    """A skipped record is the extreme extraction failure, so it is where these
    fields matter most: without them nothing says why the record was dropped."""
    rec = _run(_Backend(CHAIN_12, duds=range(1, 10)))
    assert rec["status"] == "skipped"
    assert rec["raw_units"] == 12
    assert rec["prefixes"] == 9
    assert rec["extracted_prefixes"] == 0
