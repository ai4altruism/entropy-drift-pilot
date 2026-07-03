from entropydrift.answers import (
    extract_boxed,
    extract_final_number,
    extract_gsm8k_gold,
    extract_math_answer,
    normalize_math,
    normalize_number,
)


def test_normalize_number():
    assert normalize_number("1,234") == "1234"
    assert normalize_number("$42") == "42"
    assert normalize_number("12.0") == "12"
    assert normalize_number("12.50") == "12.5"
    assert normalize_number(" 7. ") == "7"


def test_gsm8k_gold():
    assert extract_gsm8k_gold("Reasoning...\n#### 18") == "18"
    assert extract_gsm8k_gold("no marker") == ""


def test_final_number():
    assert extract_final_number("first 3 then the answer is 27.") == "27"
    assert extract_final_number("cost was $1,000 total") == "1000"
    assert extract_final_number("no digits here") == ""


def test_boxed_balanced():
    assert extract_boxed(r"the answer is \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{x} then \boxed{y}") == "y"  # last one
    assert extract_boxed("nothing") == ""


def test_normalize_math():
    assert normalize_math(r"\left(3\right)") == "(3)"
    assert normalize_math(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert normalize_math(r"$5$") == "5"


def test_extract_math_answer_prefers_boxed():
    assert extract_math_answer(r"work \boxed{42} done") == "42"
    assert extract_math_answer("plain answer 42") == "42"
