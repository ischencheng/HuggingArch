"""What keeps a verifying trace sound, held over the traced trees by AST.

A trace names the code a computation executed (`backend.code_trace.
provider`). Three ways a computation can depend on code it did not execute,
each closed here at the source rather than hoped for:

  * a memo hit — the value is returned and the code that made it is not run.
    Every memo of `backend/arch` and `backend/inference` goes through
    `backend.code_trace.memo`, whose hits name the code of their miss: no
    `functools.lru_cache` / `cache` / `cached_property`, and no hand-rolled
    `global X` … `if X is None` initialiser;
  * reading a function's signature without calling it — its defaults,
    annotations or code are part of that function's item, which only a call
    puts in a trace. Class annotations are in the module skeleton, every
    trace's, so `get_type_hints` of a class stays allowed;
  * the provider itself must sit below what it names: `code_trace` imports
    nothing else from `backend`, and `arch/eval` imports only its memos.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BACKEND = PROJECT_ROOT / "backend"
TRACED = (BACKEND / "arch", BACKEND / "inference")
FUNCTOOLS_MEMOS = {"lru_cache", "cache", "cached_property"}
SIGNATURE_READS = {"__defaults__", "__kwdefaults__", "__annotations__",
                   "__code__"}
INSPECT_SIGNATURE = {"signature", "getfullargspec", "getargspec"}


def _modules() -> Iterator[Tuple[Path, ast.Module]]:
    for root in TRACED:
        for path in sorted(root.rglob("*.py")):
            yield path, ast.parse(path.read_text(encoding="utf-8"))


def _where(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}"


def test_no_functools_memo_in_the_traced_trees():
    found: List[str] = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            imported = (isinstance(node, ast.ImportFrom)
                        and node.module == "functools"
                        and {a.name for a in node.names} & FUNCTOOLS_MEMOS)
            reached = (isinstance(node, ast.Attribute)
                       and isinstance(node.value, ast.Name)
                       and node.value.id == "functools"
                       and node.attr in FUNCTOOLS_MEMOS)
            if imported or reached:
                found.append(_where(path, node))
    assert found == [], (
        "a functools memo hides the code of its miss from every trace; use "
        f"backend.code_trace.memo (memoize / memo_property / Memo): {found}")


def test_no_hand_rolled_global_memo_in_the_traced_trees():
    """`global X` and a test of `X` against None in one function is a lazy
    initialiser: a memo the trace cannot see."""
    found: List[str] = []
    for path, tree in _modules():
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {n for stmt in ast.walk(fn) if isinstance(stmt, ast.Global)
                     for n in stmt.names}
            found.extend(
                f"{_where(path, node)} ({node.left.id})"
                for node in ast.walk(fn)
                if isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id in names
                and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops)
                and any(isinstance(c, ast.Constant) and c.value is None
                        for c in node.comparators))
    assert found == [], (
        "a lazily initialised module global is a memo the trace cannot see; "
        f"use backend.code_trace.memo: {found}")


def test_no_signature_read_in_the_traced_trees():
    found: List[str] = []
    for path, tree in _modules():
        classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in SIGNATURE_READS:
                found.append(f"{_where(path, node)} ({node.attr})")
            elif (isinstance(node, ast.Attribute)
                  and isinstance(node.value, ast.Name)
                  and node.value.id == "inspect"
                  and node.attr in INSPECT_SIGNATURE):
                found.append(f"{_where(path, node)} (inspect.{node.attr})")
            elif (isinstance(node, ast.Call)
                  and getattr(node.func, "id", getattr(node.func, "attr", None))
                  == "get_type_hints"):
                arg = node.args[0] if node.args else None
                if not (isinstance(arg, ast.Name)
                        and (arg.id == "cls" or arg.id in classes)):
                    found.append(f"{_where(path, node)} (get_type_hints)")
    assert found == [], (
        "reading a function's signature without calling it depends on code "
        f"no trace names: {found}")


def _imports(path: Path) -> Iterator[Tuple[ast.AST, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node, node.module


def test_code_trace_imports_nothing_else_from_backend():
    found = [f"{_where(path, node)} ({name})"
             for path in sorted((BACKEND / "code_trace").rglob("*.py"))
             for node, name in _imports(path)
             if name.startswith("backend") and not name.startswith("backend.code_trace")]
    assert found == []


def test_eval_imports_only_the_memos_of_code_trace():
    found = [f"{_where(path, node)} ({name})"
             for path in sorted((BACKEND / "arch" / "eval").rglob("*.py"))
             for node, name in _imports(path)
             if name.startswith("backend.code_trace")
             and name != "backend.code_trace.memo"]
    assert found == []
