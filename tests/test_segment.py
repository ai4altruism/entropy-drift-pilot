from entropydrift.segment import cumulative_prefixes


def test_blank_line_prefixes():
    text = "step one\n\nstep two\n\nstep three"
    pfx = cumulative_prefixes(text, strategy="blank_line")
    # empty prefix + one per unit
    assert pfx[0] == ""
    assert len(pfx) == 4
    assert pfx[1] == "step one"
    assert pfx[-1] == "step one\n\nstep two\n\nstep three"


def test_token_window():
    text = "a b c d e f g"
    pfx = cumulative_prefixes(text, strategy="token_window", window_tokens=3)
    # units: "a b c", "d e f", "g" -> 3 units -> 4 prefixes
    assert len(pfx) == 4
    assert pfx[1] == "a b c"


def test_sentence():
    text = "First idea. Second idea. Third!"
    pfx = cumulative_prefixes(text, strategy="sentence")
    assert len(pfx) == 4


def test_max_steps_caps_units():
    text = "\n\n".join(f"s{i}" for i in range(10))
    pfx = cumulative_prefixes(text, strategy="blank_line", max_steps=4)
    # 4 units max -> 5 prefixes; the last unit absorbs the overflow
    assert len(pfx) == 5
    assert "s9" in pfx[-1]


def test_empty_text():
    assert cumulative_prefixes("", strategy="blank_line") == [""]
