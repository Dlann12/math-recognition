"""
Safe Mathematical Expression Parser and Evaluator.

Implements AST-based safe evaluation without raw Python eval() execution,
meeting FR-11 and FR-12 requirements.
"""

import ast
import operator
import re
from typing import Any, Dict, Optional, Union

# Allowed operators mapping
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def normalize_expression(expr: str) -> str:
    """
    Clean and normalize mathematical expression string.
    - Replaces visual operators (×, ÷, x) with standard arithmetic (*, /).
    - Removes trailing '=' signs.
    - Normalizes multiple spaces and common symbol variations.
    """
    if not expr:
        return ""

    cleaned = str(expr).strip()

    # Remove trailing '=' or ':='
    cleaned = re.sub(r"[=:]+\s*$", "", cleaned).strip()

    # Replace visual symbols
    replacements = {
        "×": "*",
        "÷": "/",
        ":": "/",
        "−": "-",  # unicode minus
        "—": "-",  # em-dash
        "–": "-",  # en-dash
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    # Replace standalone 'x' or 'X' between numbers as multiplication (maintaining spacing)
    cleaned = re.sub(r"(?<=\d|\))\s*[xX]\s*(?=\d|\()", " * ", cleaned)

    return cleaned.strip()


def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    """
    Recursively evaluate an AST node strictly within allowed math operations.
    Raises ValueError on forbidden nodes or operations.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Tipe konstanta tidak diizinkan: {type(node.value).__name__}")

    if isinstance(node, ast.BinOp):
        left_val = _eval_ast_node(node.left)
        right_val = _eval_ast_node(node.right)
        op_type = type(node.op)

        if op_type not in OPERATORS:
            raise ValueError(f"Operator tidak didukung: {op_type.__name__}")

        if op_type in (ast.Div, ast.FloorDiv) and right_val == 0:
            raise ZeroDivisionError("Pembagian dengan nol (division by zero)")

        op_func = OPERATORS[op_type]
        return op_func(left_val, right_val)

    if isinstance(node, ast.UnaryOp):
        operand_val = _eval_ast_node(node.operand)
        op_type = type(node.op)

        if op_type not in OPERATORS:
            raise ValueError(f"Operator unary tidak didukung: {op_type.__name__}")

        op_func = OPERATORS[op_type]
        return op_func(operand_val)

    raise ValueError(f"Sintaks tidak diizinkan: {type(node).__name__}")


def evaluate_expression(expr: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression.

    Returns:
        dict with:
            - 'success': bool
            - 'expression': normalized string
            - 'result': numeric answer (int or float) or None
            - 'formatted_result': readable string output
            - 'error': error message if failed, else None
    """
    normalized = normalize_expression(expr)

    if not normalized:
        return {
            "success": False,
            "expression": "",
            "result": None,
            "formatted_result": None,
            "error": "Ekspresi matematika kosong.",
        }

    # Whitelist characters: digits, decimal, operators, parentheses, whitespace
    if not re.fullmatch(r"^[0-9+\-*/().\s]+$", normalized):
        return {
            "success": False,
            "expression": normalized,
            "result": None,
            "formatted_result": None,
            "error": "Ekspresi mengandung karakter tidak valid.",
        }

    try:
        parsed = ast.parse(normalized, mode="eval")
        result = _eval_ast_node(parsed.body)

        # Format integer results without unnecessary decimal places (.0)
        if isinstance(result, float) and result.is_integer():
            final_val: Union[int, float] = int(result)
        elif isinstance(result, float):
            final_val = round(result, 6)
        else:
            final_val = result

        return {
            "success": True,
            "expression": normalized,
            "result": final_val,
            "formatted_result": str(final_val),
            "error": None,
        }
    except ZeroDivisionError as zde:
        return {
            "success": False,
            "expression": normalized,
            "result": None,
            "formatted_result": None,
            "error": str(zde),
        }
    except (SyntaxError, ValueError) as err:
        return {
            "success": False,
            "expression": normalized,
            "result": None,
            "formatted_result": None,
            "error": f"Sintaks ekspresi tidak valid: {str(err)}",
        }
    except Exception as exc:
        return {
            "success": False,
            "expression": normalized,
            "result": None,
            "formatted_result": None,
            "error": f"Gagal mengevaluasi ekspresi: {str(exc)}",
        }
