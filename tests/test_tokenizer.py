import sympy as sp

from tokenizer.prefix_converter import infix_to_prefix, prefix_to_sympy


def test_tokenizer_roundtrip():
    expr = "sin(x) + x**2/2"
    tokens = infix_to_prefix(expr)
    out = prefix_to_sympy(tokens)
    assert sp.simplify(sp.sympify(expr) - out) == 0


def test_exact_rational_tokens_present():
    expr = "x**3/6"
    tokens = infix_to_prefix(expr)
    assert "INT_1" in tokens
    assert "INT_6" in tokens
