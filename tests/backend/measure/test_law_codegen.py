"""The exact law compiler's contract with the law language."""
from __future__ import annotations

from fractions import Fraction

import pytest

from backend.arch.law_text import read_walk_law
from backend.measure.auto_sweep import evaluate


def test_rationals_stay_exact_until_a_float_enters():
    answer = evaluate("B/3 + B/6 + floor(B/2)", {"B": 7})
    assert answer == Fraction(13, 2)
    assert type(answer) is Fraction
    assert isinstance(evaluate("B + PEAK", {"B": 1, "PEAK": 0.1}), float)


def test_extrema_compare_exact_values_as_written():
    assert evaluate("Max(1/3, B/4)", {"B": 1}) == Fraction(1, 3)
    assert type(evaluate("Max(1/3, B/4)", {"B": 1})) is Fraction
    assert evaluate("Min(1/3, B/4)", {"B": 1}) == Fraction(1, 4)


def test_the_unselected_case_is_not_evaluated():
    """A false arm may divide by zero at precisely that point."""
    assert evaluate("Cases(B > 0, 1/B, 0)", {"B": 0}) == 0
    assert evaluate("Cases(B > 0, 1/B, 1/3)", {"B": 0}) == Fraction(1, 3)


def test_a_quotient_is_taken_only_where_its_whole_is_not_zero():
    assert evaluate("Quotient(Eq(N, 0), 0, B, N)", {"B": 7, "N": 0}) == 0
    assert evaluate("Quotient(Eq(N, 0), 0, B, N)",
                    {"B": 7, "N": 2}) == Fraction(7, 2)


def test_the_compiled_law_agrees_with_the_read_expression():
    text = ("floor(B/3) + ceiling(B/4) + Max(B/2, 1/3) "
            "+ Cases(B > 3, B/6, Quotient(Eq(Mod(B, 2), 0), B, B + 1, 2))")
    expression = read_walk_law(text)
    for b in range(13):
        expected = expression.subs("B", b)
        # `Cases` and `Quotient` are authors' functions: SymPy keeps them
        # until their conditions are numbers, exactly as the compiler does.
        expected = expected.doit() if hasattr(expected, "doit") else expected
        assert evaluate(text, {"B": b}) == expected


def test_an_unsupported_symbol_is_refused_rather_than_invented():
    with pytest.raises(ValueError, match="does not bind"):
        evaluate("B + NOT_BOUND", {"B": 1})
