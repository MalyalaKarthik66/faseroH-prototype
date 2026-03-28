import random
from dataclasses import dataclass
from typing import Any, List, cast

import sympy as sp


x: Any = cast(Any, sp.Symbol("x"))


@dataclass
class ExpressionGeneratorConfig:
    max_depth: int = 3
    polynomial_degree: int = 3
    seed: int = 42


def _random_coeff(rng: random.Random, low: int = -5, high: int = 5) -> int:
    value = 0
    while value == 0:
        value = rng.randint(low, high)
    return value


def _make_polynomial(rng: random.Random, degree: int) -> Any:
    coeffs = [_random_coeff(rng) for _ in range(degree + 1)]
    poly = sum(c * x ** i for i, c in enumerate(coeffs))
    return sp.expand(poly)


def _simple_term(rng: random.Random) -> Any:
    """Small building-block expression safe to combine."""
    c = rng.choice([1, -1, 2, -2, 3])
    kind = rng.choice(["cx", "cx2", "sin", "cos"])
    if kind == "cx":
        return c * x
    if kind == "cx2":
        return c * x ** 2
    if kind == "sin":
        return sp.sin(x)
    return sp.cos(x)


def _base_expr(rng: random.Random, cfg: ExpressionGeneratorConfig) -> Any:
    family = rng.choice([
        "poly", "poly", "poly",   # extra weight for poly diversity
        "sin", "sin_nx",
        "cos", "cos_nx",
        "exp", "exp_cx",
        "log",
        "rat", "rat2",
        "tanh",
        "combo_add", "combo_mul",
    ])
    deg = rng.randint(1, cfg.polynomial_degree + 1)
    if family == "poly":
        return _make_polynomial(rng, deg)
    if family == "sin":
        return sp.sin(x)
    if family == "sin_nx":
        n = rng.randint(2, 3)
        return sp.sin(n * x)
    if family == "cos":
        return sp.cos(x)
    if family == "cos_nx":
        n = rng.randint(2, 3)
        return sp.cos(n * x)
    if family == "exp":
        return sp.exp(x)
    if family == "exp_cx":
        c = rng.choice([-2, -1, 2])
        return sp.exp(c * x)
    if family == "log":
        c = rng.randint(1, 4)
        return sp.log(c + x)
    if family == "rat":
        a = rng.choice([1, 2, 3])
        return 1 / (1 + a * x ** 2)
    if family == "rat2":
        a = rng.choice([1, 2])
        b = rng.choice([1, 2, 3])
        return b / (a + x)
    if family == "tanh":
        return sp.tanh(x)
    if family == "combo_add":
        return _simple_term(rng) + _simple_term(rng)
    if family == "combo_mul":
        return _simple_term(rng) * _simple_term(rng)
    return sp.exp(sp.sin(x))


def _small_poly(rng: random.Random) -> Any:
    c0 = rng.randint(-2, 2)
    c1 = rng.randint(-2, 2)
    c2 = rng.choice([0, 1])
    return c0 + c1 * x + c2 * x**2


def _compose(rng: random.Random, expr: Any, depth: int) -> Any:
    # Keep composition shallow and affine to keep symbolic series expansion fast.
    for _ in range(depth):
        op = rng.choice(["add", "mul", "sin", "cos", "exp"])
        if op == "add":
            expr = expr + _small_poly(rng)
        elif op == "mul":
            expr = expr * (1 + rng.choice([1, 2]) * x)
        elif op == "sin":
            expr = sp.sin(expr)
        elif op == "cos":
            expr = sp.cos(expr)
        elif op == "exp":
            expr = sp.exp(expr)
    return expr


def generate_expression(rng: random.Random, cfg: ExpressionGeneratorConfig) -> Any:
    expr = _base_expr(rng, cfg)
    if rng.random() < 0.35:
        extra_depth = rng.randint(1, min(cfg.max_depth, 1))
    else:
        extra_depth = 0
    if extra_depth > 0:
        expr = _compose(rng, expr, extra_depth)
    return expr


def generate_expressions(count: int, cfg: ExpressionGeneratorConfig) -> List[Any]:
    rng = random.Random(cfg.seed)
    expressions: List[Any] = []
    seen = set()
    attempts = 0
    max_attempts = max(count * 80, 50000)
    while len(expressions) < count and attempts < max_attempts:
        attempts += 1
        expr = generate_expression(rng, cfg)
        # Skip expressions that are likely to cause very slow symbolic ops.
        if sp.count_ops(expr, visual=False) > 40:
            continue
        key = sp.srepr(expr)
        if key in seen:
            continue
        seen.add(key)
        expressions.append(expr)
    return expressions


if __name__ == "__main__":
    cfg = ExpressionGeneratorConfig()
    exprs = generate_expressions(10, cfg)
    for e in exprs:
        print(e)
