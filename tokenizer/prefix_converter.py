from fractions import Fraction
from typing import Any, List, Tuple

import sympy as sp


FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
}

OPERATORS = {"add", "sub", "mul", "div", "pow", "neg"}


def _int_token(value: int) -> str:
    return f"INT_{int(value)}"


def _flatten_binary(op: str, args: List[Any]) -> List[str]:
    if len(args) == 1:
        return sympy_to_prefix(args[0])
    left = sympy_to_prefix(args[0])
    right = _flatten_binary(op, args[1:])
    return [op] + left + right


def _number_to_tokens(n: Any) -> List[str]:
    if n.is_Integer:
        return [_int_token(int(n))]
    if n.is_Rational:
        n = sp.Rational(n)
        if n.q == 1:
            return [_int_token(int(n.p))]
        return ["div", _int_token(int(n.p)), _int_token(int(n.q))]
    frac = Fraction(str(n)).limit_denominator()
    if frac.denominator == 1:
        return [_int_token(frac.numerator)]
    return ["div", _int_token(frac.numerator), _int_token(frac.denominator)]


def sympy_to_prefix(expr: Any) -> List[str]:
    expr = sp.sympify(expr)
    if expr.is_Symbol:
        return [str(expr)]
    if expr.is_Number:
        return _number_to_tokens(expr)
    if expr.func == sp.Add:
        args = [sp.sympify(a) for a in expr.as_ordered_terms()]
        return _flatten_binary("add", args)
    if expr.func == sp.Mul:
        coeff, terms = expr.as_coeff_mul()
        if coeff == -1 and len(terms) == 1:
            return ["neg"] + sympy_to_prefix(terms[0])
        symbolic_terms = [sp.sympify(t) for t in terms]
        factors = list(symbolic_terms)
        if coeff != 1:
            # Keep symbolic factors first for more readable, stable token sequences.
            factors.append(sp.sympify(coeff))
        return _flatten_binary("mul", factors)
    if expr.func == sp.Pow:
        return ["pow"] + sympy_to_prefix(expr.args[0]) + sympy_to_prefix(expr.args[1])
    if expr.func in FUNCTIONS.values():
        fn_name = expr.func.__name__
        return [fn_name] + sympy_to_prefix(expr.args[0])
    # Robust fallback for unsupported constructs to avoid recursive expansion loops.
    text = str(expr)
    if text in {"zoo", "nan", "oo", "-oo"}:
        return ["<UNK>"]
    if expr.args:
        # Attempt generic function-style flattening.
        fn = expr.func.__name__.lower()
        tokens = [fn]
        for a in expr.args:
            tokens.extend(sympy_to_prefix(sp.sympify(a)))
        return tokens
    return ["<UNK>"]


def infix_to_prefix(infix: str) -> List[str]:
    expr = sp.sympify(infix)
    return sympy_to_prefix(expr)


def _parse_number(token: str) -> sp.Integer:
    if not token.startswith("INT_"):
        raise ValueError(f"Unexpected numeric token: {token}")
    return sp.Integer(int(token.split("INT_", 1)[1]))


def _parse(tokens: List[str], idx: int = 0) -> Tuple[Any, int]:
    # Bounds check
    if idx >= len(tokens):
        return sp.Symbol("INVALID"), idx
    
    token = tokens[idx]
    
    # Handle special tokens
    if token in {"<PAD>", "<SOS>", "<EOS>", "<UNK>"}:
        return sp.Symbol("INVALID"), idx + 1
    
    if token == "add":
        a, idx = _parse(tokens, idx + 1)
        if idx >= len(tokens):
            return a, idx
        b, idx = _parse(tokens, idx)
        return a + b, idx
    if token == "sub":
        a, idx = _parse(tokens, idx + 1)
        if idx >= len(tokens):
            return a, idx
        b, idx = _parse(tokens, idx)
        return a - b, idx
    if token == "mul":
        a, idx = _parse(tokens, idx + 1)
        if idx >= len(tokens):
            return a, idx
        b, idx = _parse(tokens, idx)
        return a * b, idx
    if token == "div":
        a, idx = _parse(tokens, idx + 1)
        if idx >= len(tokens):
            return a, idx
        b, idx = _parse(tokens, idx)
        return a / b, idx
    if token == "pow":
        a, idx = _parse(tokens, idx + 1)
        if idx >= len(tokens):
            return a, idx
        b, idx = _parse(tokens, idx)
        return a ** b, idx
    if token == "neg":
        a, idx = _parse(tokens, idx + 1)
        return -a, idx
    if token in FUNCTIONS:
        a, idx = _parse(tokens, idx + 1)
        return FUNCTIONS[token](a), idx
    if token.startswith("INT_"):
        return _parse_number(token), idx + 1
    return sp.Symbol(token), idx + 1


def prefix_to_sympy(tokens: List[str]) -> sp.Expr:
    if not tokens:
        return sp.Integer(0)
    try:
        expr, idx = _parse(tokens, 0)
        # Don't require full consumption - model may generate extra tokens
        return sp.simplify(expr)
    except Exception:
        # Fallback for unparseable sequences
        return sp.Symbol("INVALID")


def prefix_to_infix(tokens: List[str]) -> str:
    return str(prefix_to_sympy(tokens))
