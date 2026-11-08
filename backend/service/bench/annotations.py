"""The annotation rules that turn a lake input into a lookup key.

This module is deliberately separate from the HTTP composition root.  Its
import closure is the source of `rules_rev`, so a route, startup, or health
edit must not look like a change to the data format.  Routes re-export the
functions for callers that already imported them from `routes`.
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
from functools import lru_cache
from typing import Any, Dict, Mapping

from backend.fingerprint import shared_data_version
from backend.inference.benchmark_identity import (
    current_shape,
    workload_coordinates,
)
from backend.inference.op_calibration import physical_identity
from backend.inference.op_signatures import all_signatures
from backend.service.bench import store as _store


def _catalog():
    # kernel_bench.providers imports torch (~2 s): loaded on first use.
    from kernel_bench import providers
    return providers


@lru_cache(maxsize=1)
def _shape_identity(op: str, shape_json: str) -> tuple[dict, str, str]:
    """One op/shape's catalog shape and physical identity.

    Startup streams inputs ordered by `(op, shape_json)`, so this one-entry
    memo computes the expensive physical half once for every duplicate input
    in that run without keeping a per-process shape table. It is the same
    annotation road `row_annotations` always takes; callers never derive it.
    """
    shape = current_shape(op, _store.canonical_json(shape_json))
    identity = physical_identity(op, shape)
    return shape, identity["signature"], identity["physical_shape"]


def row_annotations(row: Mapping[str, Any]) -> dict:
    """The annotations of one row's input (`op`, `shape_json`, `phase`,
    `kernel_impl` read)."""
    shape, signature, physical_shape = _shape_identity(
        row["op"], row["shape_json"])
    batch, context = workload_coordinates(row["phase"], shape)
    return {
        "signature": signature,
        "physical_shape": physical_shape,
        "workload_batch": batch,
        "workload_context": context,
        "catalog_kernel": _catalog().catalog_kernel(row["kernel_impl"]),
    }


def registered_kernels() -> Dict[str, Dict[str, str]]:
    """Every registered kernel, by library: `{library: {kernel: signature}}`."""
    out: Dict[str, Dict[str, str]] = {}
    for p in _catalog().all_providers():
        out.setdefault(p.library, {})[p.kernel_impl] = p.signature
    return out


_REPO = pathlib.Path(__file__).resolve().parents[3]


def rule_roots() -> set[str]:
    """Where the annotation rules start: the modules defining
    `row_annotations` and the functions it calls — read off those functions,
    so a call moved to another module moves the root with it."""
    return {f.__module__ for f in (row_annotations, current_shape, physical_identity,
                                   workload_coordinates, _store.canonical_json,
                                   _catalog().catalog_kernel)}


def _module_file(name: str) -> pathlib.Path | None:
    if not name.split(".")[0] in ("backend", "kernel_bench"):
        return None
    base = _REPO.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def _imports(path: pathlib.Path, name: str) -> set[str]:
    """Every module `path` (module `name`) imports, at any depth of its
    body — a function-level import is part of what it can run."""
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                up = parts[:len(parts) - node.level + 1]
                base = ".".join([*up, base] if base else up)
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


@lru_cache(maxsize=1)
def rules_rev() -> str:
    """The revision of the annotation rules that decide which rows a lookup
    key selects and the migration owner brings an older lake to: a hash of
    the source of every repository module the rules can run (the import
    closure of `rule_roots()`, function-level imports included), the data
    migration owner (`data_rules.py`), the engine's shared data
    (`shared_data_version`: the
    migration tables `current_shape` reads among them), and the catalogs
    as loaded — every registered measurement signature and every registered
    kernel. Any edit that can change a selection moves it; computed once,
    at the start (a catalog change takes effect at the next start)."""
    seen: dict[str, pathlib.Path] = {}
    migration_owner = _module_file("backend.service.bench.data_rules")
    if migration_owner is not None:
        seen["backend.service.bench.data_rules"] = migration_owner
    todo = list(rule_roots())
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        path = _module_file(name)
        if path is None:
            continue
        seen[name] = path
        todo.extend(_imports(path, name) - set(seen))
    h = hashlib.sha256()
    for path in sorted(set(seen.values())):
        h.update(path.relative_to(_REPO).as_posix().encode())
        h.update(path.read_bytes())
    h.update(shared_data_version().encode())
    h.update(json.dumps(sorted(repr(s) for s in all_signatures())).encode())
    h.update(json.dumps(registered_kernels(), sort_keys=True).encode())
    return h.hexdigest()[:16]
