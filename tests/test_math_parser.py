"""
Tests for safe math parser and evaluator.
"""

import pytest
from src.math_parser import evaluate_expression, normalize_expression


def test_normalize_expression():
    assert normalize_expression("12 + 7 =") == "12 + 7"
    assert normalize_expression("8 × 5") == "8 * 5"
    assert normalize_expression("10 ÷ 2") == "10 / 2"
    assert normalize_expression("7 x 3") == "7 * 3"
    assert normalize_expression("  4 + 6 = ") == "4 + 6"


def test_basic_arithmetic():
    res1 = evaluate_expression("12 + 7")
    assert res1["success"] is True
    assert res1["result"] == 19

    res2 = evaluate_expression("25 - 9")
    assert res2["success"] is True
    assert res2["result"] == 16

    res3 = evaluate_expression("6 * 7")
    assert res3["success"] is True
    assert res3["result"] == 42

    res4 = evaluate_expression("100 / 4")
    assert res4["success"] is True
    assert res4["result"] == 25


def test_unicode_symbols():
    res_mul = evaluate_expression("9 × 9")
    assert res_mul["success"] is True
    assert res_mul["result"] == 81

    res_div = evaluate_expression("45 ÷ 5")
    assert res_div["success"] is True
    assert res_div["result"] == 9

    res_eq = evaluate_expression("15 + 15 =")
    assert res_eq["success"] is True
    assert res_eq["result"] == 30


def test_operator_precedence_and_parentheses():
    res = evaluate_expression("2 + 3 * 4")
    assert res["success"] is True
    assert res["result"] == 14

    res_paren = evaluate_expression("(2 + 3) * 4")
    assert res_paren["success"] is True
    assert res_paren["result"] == 20


def test_floating_and_negative_numbers():
    res_float = evaluate_expression("5.5 + 4.5")
    assert res_float["success"] is True
    assert res_float["result"] == 10

    res_neg = evaluate_expression("-5 + 12")
    assert res_neg["success"] is True
    assert res_neg["result"] == 7


def test_division_by_zero():
    res = evaluate_expression("10 / 0")
    assert res["success"] is False
    assert "nol" in res["error"].lower() or "zero" in res["error"].lower()


def test_empty_and_invalid():
    res_empty = evaluate_expression("")
    assert res_empty["success"] is False

    res_syntax = evaluate_expression("3 + * 2")
    assert res_syntax["success"] is False

    # Security check: arbitrary code injection must fail
    res_inject = evaluate_expression("__import__('os').system('ls')")
    assert res_inject["success"] is False
    assert res_inject["result"] is None
