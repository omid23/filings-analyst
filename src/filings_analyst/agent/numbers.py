"""Safe arithmetic and number matching.

Models make arithmetic slips and transcription errors on financial tables, so
the agent must compute with ``calculate`` and every number in the final
answer is checked against the cited pages or the calculator log.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"abs": abs, "round": lambda x, n=0: round(x, int(n)), "min": min, "max": max, "sum": lambda *a: sum(a), "sqrt": math.sqrt,
          "avg": lambda *a: sum(a) / len(a), "log": math.log, "exp": math.exp}


class CalcError(ValueError):
    pass


def _clean_expression(expr: str) -> str:
    expr = expr.replace("$", "").replace("−", "-").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expr)  # 1,577 -> 1577
    expr = re.sub(r"\((\d[\d.]*)\)", r"(-\1)", expr) if re.fullmatch(r"\s*\(\d[\d.]*\)\s*", expr) else expr
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", expr)
    return expr


def calculate(expr: str) -> float:
    """Evaluate arithmetic without eval(): numbers, + - * / ** % //, parentheses and a few functions."""
    if len(expr) > 500:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(_clean_expression(expr), mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"cannot parse expression: {exc.msg}") from exc

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise CalcError("exponent too large")
            try:
                return float(_BIN[type(node.op)](left, right))
            except ZeroDivisionError as exc:
                raise CalcError("division by zero") from exc
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](ev(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return float(_FUNCS[node.func.id](*[ev(a) for a in node.args]))
        raise CalcError(f"unsupported syntax: {type(node).__name__}")

    value = ev(tree)
    if math.isnan(value) or math.isinf(value):
        raise CalcError("result is not finite")
    return value


# -- extracting numbers from text --------------------------------------------------------------

_SCALE_WORDS = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6, "billion": 1e9,
                "bn": 1e9, "b": 1e9, "trillion": 1e12}
_NUM = re.compile(
    r"(?<![\w.])(?P<neg>\(|-|−)?\s*\$?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*\)?"
    r"\s*(?P<pct>%|percent\b|pp\b|bps\b|basis points\b)?"
    r"\s*(?P<scale>thousand|million|billion|trillion|bn\b|mn\b|mm\b|[kmb]\b)?",
    re.I,
)


@dataclass(frozen=True)
class Number:
    text: str
    value: float  # as written (mantissa)
    scale: float  # multiplier from a scale word, 1 if none
    decimals: int
    percent: bool

    @property
    def absolute(self) -> float:
        return self.value * self.scale


def extract_numbers(text: str) -> list[Number]:
    out: list[Number] = []
    for m in _NUM.finditer(text):
        raw = m.group("num")
        value = float(raw.replace(",", ""))
        if m.group("neg") and m.group("neg") in "(-−" and m.group(0).strip().startswith(("(", "-", "−")):
            value = -value
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        scale_word = (m.group("scale") or "").lower()
        if scale_word in ("k", "m", "b") and "$" not in m.group(0):
            scale_word = ""  # "3M" is a company, "$3M" is money
        out.append(Number(m.group(0).strip(), value, _SCALE_WORDS.get(scale_word, 1.0), decimals,
                          bool(m.group("pct"))))
    return out


def is_trivial(n: Number) -> bool:
    """Years, small counts and list markers do not need a source."""
    v = abs(n.value)
    if n.decimals == 0 and not n.percent and n.scale == 1 and (1990 <= v <= 2040 or v <= 12):
        return True
    return False


def matches(answer: Number, candidates: list[float], rel_tol: float = 0.005) -> bool:
    """Does ``answer`` equal some candidate, allowing for rounding and unit scale (thousands/millions)?

    A source table "in millions" shows 1,577 while an answer may say "$1.58 billion"; the comparison
    therefore tries the candidate at several scales and accepts it if rounding the candidate to the
    answer's precision reproduces the answer.
    """
    target = answer.absolute
    half_step = 0.5 * (10 ** -answer.decimals) * answer.scale
    for c in candidates:
        for k in (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100.0, 0.01):
            v = c * k
            if v == 0 and target == 0:
                return True
            if abs(abs(v) - abs(target)) <= max(half_step * 1.0001, abs(target) * 1e-9):
                return True
            if v and abs(abs(v) - abs(target)) / abs(v) <= rel_tol and answer.decimals == 0 and abs(target) >= 1000:
                return True
    return False
