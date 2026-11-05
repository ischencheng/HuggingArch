"""The scheme dialect — the little language quant schemes are written in.

`quant_schemes/*.yaml` declares metadata siblings with `when:` predicates,
`cardinality:` / shape formulas over `in` / `out` / scheme fields, and a
`numeric_format_resolver`. This module is that dialect's evaluator, moved out
of `materialize` whole: `eval.formats` speaks the same dialect, and while the
evaluator lived in its own consumer, `formats` had to import its caller late
to break the cycle. A language does not belong to either of its speakers.

Nothing here imports the rest of the package or the old world — the dialect
is `re` + `ast` + a whitelist, and the one type it touches (`QuantScheme`) is
structural.
"""
from __future__ import annotations

import ast
from fractions import Fraction
import re
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional


if TYPE_CHECKING:
    from backend.arch.quant_schemes import QuantScheme


_PREDICATE_TRUE_FALSE = re.compile(r"\b(true|false)\b")


def eval_predicate(expr_str: str, bindings: Dict[str, Any]) -> bool:
    """Evaluate a YAML-author `when:` predicate against `bindings`.

    Supports: `==`, `!=`, `and`, `or`, `not`, parens, name references,
    string / int / bool literals, the bare keywords `true` / `false`.
    Refuses any other AST node — no function calls, no attribute
    access, no subscripts.

    Raises ValueError on:
      - unsupported AST node
      - identifier not present in bindings
      - syntax error
    """
    if expr_str is None:
        return True
    src = expr_str.strip()
    if not src:
        return True
    # Normalize bare true/false to Python identifiers for ast.parse.
    src_py = _PREDICATE_TRUE_FALSE.sub(
        lambda m: "True" if m.group(1) == "true" else "False",
        src,
    )
    try:
        tree = ast.parse(src_py, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in predicate {expr_str!r}: {e}") from e

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in ("True", "False"):
                return node.id == "True"
            if node.id in bindings:
                return bindings[node.id]
            raise ValueError(
                f"Predicate {expr_str!r} references unbound name "
                f"{node.id!r}; available: {sorted(bindings)}"
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _walk(node.operand)
        if isinstance(node, ast.BoolOp):
            vals = [_walk(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
            raise ValueError(
                f"Predicate {expr_str!r}: unsupported BoolOp {type(node.op).__name__}"
            )
        if isinstance(node, ast.Compare):
            left = _walk(node.left)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = _walk(comparator)
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                else:
                    raise ValueError(
                        f"Predicate {expr_str!r}: unsupported "
                        f"comparison {type(op).__name__}"
                    )
                if not ok:
                    return False
                left = right
            return True
        raise ValueError(
            f"Predicate {expr_str!r}: unsupported AST node "
            f"{type(node).__name__}"
        )

    return bool(_walk(tree))


# Sibling cardinality formulas reference `in` / `out` symbols (the
# linear's input / output dim). `in` is a Python keyword and trips
# ast.parse, so we rewrite to safe aliases purely for evaluation —
# yaml authors keep writing `in` / `out`.
_FORMULA_KEYWORD_REWRITES = (
    (re.compile(r"\bin\b"),  "_in"),
    (re.compile(r"\bout\b"), "_out"),
)
_FORMULA_BINDINGS_RENAME = {"in": "_in", "out": "_out"}


def _shape_product(axes, bindings: Dict[str, Any]) -> int:
    """A declared sibling shape, as an element count — the product of its axes.

    The shape is the declaration and the count is a projection of it, which is
    the same move `storage.py` made for weights: a count cannot be checked
    against anything, a shape can be checked against the header's axis by axis.
    """
    total = 1
    for axis in axes:
        total *= eval_formula(axis, bindings)
    return total


def formula_symbols(formula: str) -> FrozenSet[str]:
    """Return the identifiers consumed by one arithmetic formula.

    This uses the dialect's own keyword rewrite and parser so projection
    callers can require provenance for every value without maintaining a
    second, regex-based interpretation of the language.
    """
    if formula is None:
        raise ValueError("formula is None")
    src = formula.strip()
    if not src:
        raise ValueError("formula is empty")
    safe_src = src
    for pattern, replacement in _FORMULA_KEYWORD_REWRITES:
        safe_src = pattern.sub(replacement, safe_src)
    try:
        tree = ast.parse(safe_src, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in formula {formula!r}: {exc}") from exc
    inverse = {value: key for key, value in _FORMULA_BINDINGS_RENAME.items()}
    return frozenset(
        inverse.get(node.id, node.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    )


def eval_formula(formula: str, bindings: Dict[str, Any]) -> int:
    """Evaluate an arithmetic formula like `(in * out) / group_size`
    against `bindings`. Supports +, -, *, /, //, %, **, parens, unary
    minus, integer literals, identifier references. AST-walked with
    only safe nodes whitelisted (no function calls, no attribute
    access, no subscripts).

    Raises ValueError on unbound names, unsupported syntax, or
    non-integer result.
    """
    if formula is None:
        raise ValueError("formula is None")
    src = formula.strip()
    if not src:
        raise ValueError("formula is empty")

    # Rewrite `in` / `out` keywords; rename bindings to match.
    safe_src = src
    for pat, repl in _FORMULA_KEYWORD_REWRITES:
        safe_src = pat.sub(repl, safe_src)
    safe_bindings = {
        _FORMULA_BINDINGS_RENAME.get(k, k): v
        for k, v in bindings.items()
    }

    try:
        tree = ast.parse(safe_src, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in formula {formula!r}: {e}") from e

    def _walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in safe_bindings:
                return safe_bindings[node.id]
            # Translate back for error message readability.
            inv = {v: k for k, v in _FORMULA_BINDINGS_RENAME.items()}
            display = inv.get(node.id, node.id)
            raise ValueError(
                f"Formula {formula!r} references unbound symbol {display!r}; "
                f"available: {sorted(bindings)}"
            )
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                try:
                    return Fraction(left) / Fraction(right)
                except (TypeError, ValueError, ZeroDivisionError) as exc:
                    raise ValueError(
                        f"Formula {formula!r}: exact division failed: {exc}"
                    ) from exc
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError(
                f"Formula {formula!r}: unsupported BinOp "
                f"{type(node.op).__name__}"
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -_walk(node.operand)
        raise ValueError(
            f"Formula {formula!r}: unsupported AST node "
            f"{type(node).__name__}"
        )

    val = _walk(tree)
    try:
        integer = int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Formula {formula!r} did not evaluate to an integer "
            f"(got {val!r}): {e}"
        ) from e
    if val != integer:
        raise ValueError(
            f"Formula {formula!r} did not evaluate to an integer "
            f"(got {val!r})"
        )
    return integer




def logical_dtype_label(numeric_format: Optional[str], logical_bits: int,
                          storage_dtype: str) -> str:
    """Translate `(numeric_format selection, logical bits)` → display label.

    For unquantized (no numeric_format selected): label = storage_dtype.
    For quantized: pick a frontend-recognized label based on family.

    A family with no label here is NOT the storage dtype: the storage dtype
    of a quantized tensor is the container the bytes ship in (MXFP4's
    E2M1 pairs ride in a `U8`), and handing that back labels the format by
    its packaging. It raises instead, because a scheme that declares a
    numeric format this does not know is a scheme whose format nothing can
    read — including the measurement corpus, which has no token for a
    container.
    """
    if not numeric_format:
        return storage_dtype
    if numeric_format in ("int_sym", "int_asym"):
        return f"INT{logical_bits}"
    if numeric_format == "fp_e2m1":
        return f"FP{logical_bits}"      # FP4 typically
    if numeric_format == "mx_e2m1":
        # OCP Microscaling: E2M1 elements under one UE8M0 exponent per
        # block. The element type is E2M1 either way, but the format is not
        # NVFP4's — the scale is constrained to a power of two, so a
        # consumer that places the weight on a grid places it on a
        # different one (`kernel_bench.quant.FORMATS`: `pow2_scale`), and
        # a kernel written for one declines the other. One label, so the
        # two cannot share a measurement.
        return "MXFP4"
    if numeric_format == "fp_e4m3":
        return "F8_E4M3"
    if numeric_format == "fp_e5m2":
        return "F8_E5M2"
    if numeric_format == "nf4":
        return "NF4"
    raise ValueError(
        f"numeric format {numeric_format!r} has no dtype label; its tensors "
        f"would be labelled by their storage container "
        f"({storage_dtype!r}), which is the packaging and not the format")


def select_numeric_format(
    scheme: QuantScheme,
    per_role_bindings: Dict[str, Any],
) -> Optional[str]:
    """Walk numeric_format_resolver in order, return first matching
    rule's format. None if no rules registered (informational only).
    """
    for when, fmt in scheme.numeric_format_resolver:
        if eval_predicate(when, per_role_bindings):
            return fmt
    return None
