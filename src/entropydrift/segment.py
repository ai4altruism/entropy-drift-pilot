"""Step-segmentation of a chain-of-thought trace into cumulative prefixes.

This is the crux methodological choice for contribution 2: standard instruct models
produce enumerable steps, but reasoning-distilled models emit long free-form chains with
no clean markers. We expose several strategies and return *cumulative prefixes*
(s_0, s_1, ..., s_N) where s_k is the chain truncated to the end of step k. The trajectory
sampler continues generation from each prefix.

Strategies:
  blank_line   split on blank lines (paragraph-style CoT)
  newline      split on single newlines
  sentence     split on sentence-terminal punctuation
  token_window fixed-size windows of whitespace tokens (robust to unstructured traces)
"""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_units(text: str, strategy: str, window_tokens: int) -> list[str]:
    if strategy == "blank_line":
        return [u for u in re.split(r"\n\s*\n", text) if u.strip()]
    if strategy == "newline":
        return [u for u in text.split("\n") if u.strip()]
    if strategy == "sentence":
        return [u for u in _SENTENCE_RE.split(text) if u.strip()]
    if strategy == "token_window":
        toks = text.split()
        if window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        return [
            " ".join(toks[i : i + window_tokens])
            for i in range(0, len(toks), window_tokens)
        ]
    raise ValueError(f"unknown segmentation strategy: {strategy!r}")


def cumulative_prefixes(
    text: str,
    strategy: str = "blank_line",
    window_tokens: int = 40,
    max_steps: int | None = None,
) -> list[str]:
    """Return cumulative prefixes of ``text`` under a segmentation strategy.

    The empty prefix (before any reasoning) is included as the first element, so a chain
    with N units yields N+1 prefixes: the trajectory has one entropy value per prefix.
    ``max_steps`` optionally caps the number of *units* (useful to bound compute); when
    set, later units are merged into the final prefix.
    """
    units = _split_units(text, strategy, window_tokens)
    if max_steps is not None and len(units) > max_steps:
        head = units[: max_steps - 1]
        tail = " ".join(units[max_steps - 1 :])
        units = head + [tail]

    prefixes = [""]
    acc: list[str] = []
    for u in units:
        acc.append(u)
        prefixes.append("\n\n".join(acc))
    return prefixes
