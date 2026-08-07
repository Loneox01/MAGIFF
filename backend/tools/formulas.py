"""Safe arithmetic expressions for model-requested statistical rankings."""

import ast
from dataclasses import dataclass
import math
from typing import Any


MAX_FORMULA_LENGTH = 160
MAX_EXPRESSION_DEPTH = 8


class FormulaError(ValueError):
    """Raised when a formula is unsafe, unsupported, or invalid."""


class ZeroDenominatorError(ArithmeticError):
    """Raised when evaluating a formula would divide by zero."""


@dataclass(frozen=True)
class ParsedFormula:
    tree: ast.Expression
    fields: tuple[str, ...]
    canonical: str


def _validate_node(
    node: ast.AST,
    allowed_fields: set[str],
    fields: set[str],
    depth: int,
) -> None:
    if depth > MAX_EXPRESSION_DEPTH:
        raise FormulaError(
            f"formula exceeds maximum expression depth {MAX_EXPRESSION_DEPTH}"
        )

    if isinstance(node, ast.Expression):
        _validate_node(node.body, allowed_fields, fields, depth + 1)
        return
    if isinstance(node, ast.Name):
        if node.id not in allowed_fields:
            raise FormulaError(f"unsupported formula field: {node.id}")
        fields.add(node.id)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError("formula constants must be numbers")
        if not math.isfinite(float(node.value)):
            raise FormulaError("formula constants must be finite")
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise FormulaError("only +, -, *, and / operators are supported")
        _validate_node(node.left, allowed_fields, fields, depth + 1)
        _validate_node(node.right, allowed_fields, fields, depth + 1)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise FormulaError("only unary + and - are supported")
        _validate_node(node.operand, allowed_fields, fields, depth + 1)
        return
    raise FormulaError(f"unsupported formula syntax: {type(node).__name__}")


def parse_formula(formula: str, allowed_fields: set[str]) -> ParsedFormula:
    """Parse and validate a small arithmetic expression without executing it."""
    expression = formula.strip()
    if not expression:
        raise FormulaError("formula cannot be empty")
    if len(expression) > MAX_FORMULA_LENGTH:
        raise FormulaError(
            f"formula cannot exceed {MAX_FORMULA_LENGTH} characters"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise FormulaError("formula is not a valid arithmetic expression") from error

    fields: set[str] = set()
    _validate_node(tree, allowed_fields, fields, depth=0)
    if not fields:
        raise FormulaError("formula must reference at least one statistic")
    return ParsedFormula(tree, tuple(sorted(fields)), ast.unparse(tree))


def _evaluate(node: ast.AST, row: dict[str, Any]) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, row)
    if isinstance(node, ast.Name):
        value = row.get(node.id)
        if value is None:
            raise ValueError("formula input is null")
        return float(value)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, row)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, row)
        right = _evaluate(node.right, row)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ZeroDenominatorError
        return left / right
    raise FormulaError(f"unsupported formula syntax: {type(node).__name__}")


def evaluate_formula(parsed: ParsedFormula, row: dict[str, Any]) -> float:
    """Evaluate a previously validated formula for one statistical row."""
    result = _evaluate(parsed.tree, row)
    if not math.isfinite(result):
        raise ValueError("formula result is not finite")
    return result
