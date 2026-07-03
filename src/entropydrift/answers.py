"""Answer extraction and normalization for GSM8K and MATH.

Pure functions. GSM8K answers are integers/decimals delimited by '#### ' in the gold
field; generated answers are taken as the last number in the completion. MATH answers
are LaTeX inside \\boxed{...} and need light normalization before comparison.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- GSM8K

_GSM8K_GOLD_RE = re.compile(r"####\s*(.+)")
_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def normalize_number(s: str) -> str:
    """Canonicalize a numeric string: strip $ and thousands commas, drop a trailing '.0'."""
    s = s.strip().replace("$", "").replace(",", "")
    if s.endswith("."):
        s = s[:-1]
    # normalize '12.0' -> '12', but keep '12.5'
    if re.fullmatch(r"-?\d+\.\d+", s):
        s = s.rstrip("0").rstrip(".")
    return s


def extract_gsm8k_gold(answer_field: str) -> str:
    """Pull the gold answer from a GSM8K 'answer' field ('... #### 42')."""
    m = _GSM8K_GOLD_RE.search(answer_field)
    return normalize_number(m.group(1)) if m else ""


def extract_final_number(text: str) -> str:
    """Extract a generated numeric answer as the last number appearing in the text."""
    nums = _NUMBER_RE.findall(text)
    return normalize_number(nums[-1]) if nums else ""


# ---------------------------------------------------------------- MATH


def extract_boxed(text: str) -> str:
    r"""Return the content of the last \boxed{...}, with balanced-brace matching."""
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return ""
    i = idx + len(r"\boxed")
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return ""
    depth = 0
    start = i + 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j]
    return ""


_MATH_STRIP = [
    (re.compile(r"\\left"), ""),
    (re.compile(r"\\right"), ""),
    (re.compile(r"\\!"), ""),
    (re.compile(r"\\,"), ""),
    (re.compile(r"\\ "), ""),
    (re.compile(r"\s+"), ""),
    (re.compile(r"\\text\{[^}]*\}"), ""),
    (re.compile(r"\\dfrac"), r"\\frac"),
    (re.compile(r"\\tfrac"), r"\\frac"),
]


def normalize_math(s: str) -> str:
    """Light MATH-answer normalization: strip spacing/formatting macros, unify frac, trim $."""
    s = s.strip()
    s = s.strip("$")
    for pat, repl in _MATH_STRIP:
        s = pat.sub(repl, s)
    # a trailing period is not part of the answer
    if s.endswith("."):
        s = s[:-1]
    return s


def extract_math_answer(text: str) -> str:
    r"""Generated MATH answer: prefer the last \boxed{...}, else fall back to a final number."""
    boxed = extract_boxed(text)
    if boxed:
        return normalize_math(boxed)
    return normalize_number(extract_final_number(text))
