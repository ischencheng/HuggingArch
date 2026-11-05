"""The boundary ratchet — the debts `eval` owes, counted so they can only shrink.

`eval/__init__.py` promises a package whose contract lives in protocols rather
than in defensive accesses, and whose only coupling to the machinery it replaces
is the input syntax it consumes. Both promises are quantitative, and a
quantitative promise kept in prose is kept by nobody: `formats.py` reached
twenty-three getattr calls without any single commit looking like the one that
crossed a line.

So the two counts are pinned EXACTLY, the same way the oracle pins its model
list. Going above a baseline is a regression by definition. Coming in below one
fails too — with a different message — because a stale baseline is how a ratchet
stops being one: the next regression hides in the slack the last improvement
left behind.

What is counted
---------------
`ACCESS` counts `getattr`/`hasattr` call sites per module, by AST — a mention in
a docstring is not an access. These sit almost entirely where `eval` reads
old-world objects, which is exactly where the old contract-by-defensive-access
pattern re-enters if unwatched.

`IMPORTS` counts import sites of the five old-world modules, per (module,
target) pair. `schema` and `spec_tree` are the authored input side — `Program.of`
consumes what the loader built, so those survive the swap. Remaining `loader`
imports are the `ResolvedSpec` ingestion path (same reason). `materialize` is
already gone (see `eval/__init__.py`) — its presence in OLD_WORLD pins the
count at zero so the retired module can't creep back in.

`engine` is not a baseline, it is a ban. The side-by-side design holds only
while neither side imports the other, so that one is asserted at zero with no
allowance to grow into.

Dependency direction is asserted the same way, at zero, over every import
statement in the tree (function-level ones included — a lazy import is still
a dependency): eval's core may not import `layout`, `inference` or
`rendering`, with `transforms.py` the one documented composition seam allowed
`layout` alone; nothing under `arch/` imports `inference`; nothing under
`inference/` imports `rendering`. The contract lives in CLAUDE.md §2.7;
this gate is what keeps it from being prose.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

BACKEND_DIR = PROJECT_ROOT / "backend"
ARCH_DIR = BACKEND_DIR / "arch"
EVAL_DIR = ARCH_DIR / "eval"
INFERENCE_DIR = BACKEND_DIR / "inference"

#: Old-world modules whose imports are tracked. `engine` is handled separately.
OLD_WORLD = {"schema", "spec_tree", "loader", "materialize"}

#: getattr/hasattr call sites per module. Exact: above is a regression, below
#: means this baseline must be lowered in the same commit.
ACCESS = {
    # 11 -> 12 with cached_node_uids: one kv_cached read off the component
    # graph, the boundary this counts. 12 -> 10 when `dim.substitute` took
    # over the two `.subs(CACHE_LEN, ...)` sites: the guards they carried
    # were asking whether a law had an expression in it, which is the
    # substitution's own question and now lives with it.
    # 10 → 8 when the leftover shard-on-read fallback left `_sequence_shard`.
    # 8 → 2 when the six `getattr(graph, F, None)` became `graph.F`: graph is
    # `program.component(...)` → Optional[BlockGraph] (a single type), the None
    # case is already guarded by the surrounding `if graph else`, and every F
    # (kv_cached / kv_cache_dtype / kv_label / attn_variant / outputs) is a
    # declared BlockGraph field — the getattr default was dead.
    # 2 → 1 when the cache-law walk read `inputs_obj` off the SpecIRNode
    # children it iterates (always declared).
    # 1 → 0 when `workload_dim` read the declared `Analysis.binding` directly.
    "cache.py": 0,
    # The projection package is the declared presentation layer — the one
    # place analyses serialize into consumer shapes. Its read surface is
    # pinned for the same reason as everything else here: presentation may
    # read the analysis's public fields and the shared IR substrate only.
    "projection/result.py": 1,
    "dim.py": 2,
    # 24 → 4 when 20 getattr on the override / per-role / scheme / variant
    # objects became direct field reads: each `entry` (OverrideEntry /
    # PerRoleEntry), `selector` (OverrideSelector) and `variant` (FormatVariant)
    # is a single declared type — read directly, with an explicit `is not None`
    # guard where the value is Optional — and `scheme` (QuantScheme, past
    # `_scheme`'s None-return) always carries its fields. `entry.model_extra` is
    # a pydantic attribute always present under `extra="allow"`. The 4 that
    # remain are the HF-config probe (a foreign Mapping) and three reflective
    # `getattr(x, <var>)` sites where the attribute name is a runtime variable.
    "formats.py": 3,
    # 11 → 15 with the DAG matcher and module-parent grouping: op_choices /
    # inputs / template / display_path reads on pattern specs and old-world
    # IR nodes, the boundary this counts. 15 → 16 with `optional_inputs`
    # beside the existing `inputs` read — same optional-field category on
    # the same pattern specs.
    # 15 → 13 when sibling grouping read its parent off the tree instead of
    # the `template` / `display_path` probes.
    # 13 → 1 when every pattern read became a field read: patterns are the
    # declared `NaivePattern` / `PatternNode` / `ParallelPatternForm`, and
    # program nodes (`SpecIRNode`, `FusedNode`) both declare `template`. The
    # one left is `FusedProgram.__getattr__` delegating to its base.
    "fusion.py": 1,
    "layer.py": 2,
    # 19 → 9 when infer_shape became the only shape path and `_derive`
    # left, taking its out_key / axis_locked / permute_to reads with it.
    # 9 → 6 (OpView Stage 0) when the dead custom-KV branch left — CustomOpSpec
    # declares no KV law, so those reads were always None. 6 → 2 (Stage 1) when
    # secondary_outputs + the primitive kv_*_fn / kv_block_anchor reads became
    # direct (an OpSpec read after the `if self._custom:` early-return). The 2
    # left (is_buffer) await the OpView totalisation (Stage 2).
    # 2 → 0 (Stage 2) when is_buffer became a direct read —
    # CustomOpSpec now totals it via an honest @property (OpView).
    # advances_residual left when model-level edges became authored `inputs:`.
    "op_ref.py": 0,
    # 5 → 0 when the leftover transition machine left the module.
    "transition.py": 0,
    # 11 → 10 when `_declared_shape` stopped reading `weight_shape_fn`.
    # 10 → 8 (OpView Stage 1) when weight_tensor_suffix / weight_shape became
    # direct reads — both are declared on OpSpec and CustomOpSpec. 8 → 5
    # (Stage 2) when the three `tensors` reads became direct (CustomOpSpec totals
    # `tensors -> ()` via @property). 5 → 3 (Stage 3) when the two
    # `StorageKind.of` object-None reads (is_buffer / shape_kind) took an explicit
    # `op is not None` guard and then read the field directly. The 3 left are the
    # enum bridge (`getattr(kind,"name",str(kind))`) and the two `name` op-name
    # error messages — union-hole floor (CustomOpSpec has no `name`). 3 → 2
    # when `StorageKind.of` asked `ops.is_lookup_table` (the op's declaration)
    # instead of bridging `shape_kind` to a name.
    "storage.py": 2,
    # 3 -> 1 when the span-weighted split replaced the equal one: the two
    # `is_view_op` reads it dropped were guarding against an op the registry
    # does not know, which is `view_only_components`'s question and is now
    # asked there. 1 -> 0 when the workload-dim reader became the one
    # `cache.workload_dim`.
    "traffic.py": 0,
    # 3 → 1 when rebind stopped probing `spec.components` / `decode_variant`
    # with getattr: both are declared fields (`ArchSpec.components`,
    # `BlockGraph.decode_variant`).
    "transforms.py": 1,
}

#: Import sites of old-world modules, per "<file>:<target>" pair. The `loader`
#: rows are scheduled deletions, not permanent residents; `materialize` is
#: already at zero and stays there.
IMPORTS = {
    # The composition root (`transforms`) wraps a resolved spec as a one-stage
    # model to project isolated blocks — the loader/schema reads below are its
    # declared, pinned dependencies.
    "program.py:schema": 1,
    "program.py:spec_tree": 1,
    "storage.py:spec_tree": 1,
    # 3 → 1 when the three function-body loader imports (get_default_loader +
    # two ResolvedSpec) were hoisted to a single module-top import (lazy_edges
    # L#1) — fewer old-world import statements, the improvement this ratchet
    # wants recorded.
    "transforms.py:loader": 1,
    "transforms.py:schema": 1,
}


def _scan() -> tuple[Dict[str, int], Dict[str, int]]:
    access: Dict[str, int] = {}
    imports: Dict[str, int] = {}
    for path in sorted(EVAL_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        n = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "hasattr")
        )
        if n:
            access[str(path.relative_to(EVAL_DIR))] = n
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            mod = node.module or ""
            top = None
            if node.level >= 1 and mod.split(".")[0] in OLD_WORLD:
                top = mod.split(".")[0]
            elif mod.startswith("backend.arch."):
                cand = mod.split(".")[2]
                if cand in OLD_WORLD:
                    top = cand
            if top:
                key = f"{path.relative_to(EVAL_DIR)}:{top}"
                imports[key] = imports.get(key, 0) + 1
    return access, imports


def _resolved_imports(path: Path) -> Iterator[str]:
    """Every import in `path`, as an absolute dotted module — function-level
    and relative imports included, because both are the same dependency."""
    package = path.parent.relative_to(BACKEND_DIR).parts
    if path.name == "__init__.py":
        package = (*package, "__init__")
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                mod = ".".join(("backend", *base, mod)) if mod \
                    else ".".join(("backend", *base))
            yield mod


#: Dependency directions asserted at zero, like `engine`. (tree, banned
#: import prefixes, files exempt from the FIRST prefix only, the reason).
DIRECTION_BANS: Tuple[Tuple[Path, Tuple[str, ...], Tuple[str, ...], str], ...] = (
    (EVAL_DIR, ("backend.arch.layout", "backend.inference", "backend.rendering"),
     ("transforms.py",),
     "eval's core may not import layout/inference/rendering — transforms.py "
     "is the one composition seam, and only for layout (CLAUDE.md §2.7)"),
    (ARCH_DIR, ("backend.inference", "backend.rendering", "backend.service"), (),
     "arch is the fact source; inference, rendering and the service layer "
     "read it, never the reverse — a spec the service holds elsewhere reaches "
     "the loader through the spec-miss hook, not an import"),
    (INFERENCE_DIR, ("backend.rendering",), (),
     "rendering may read inference facts; inference never imports the "
     "presentation layer"),
    (BACKEND_DIR / "measure", ("backend.inference", "backend.api", "backend.service",
                               "backend.rendering", "kernel_bench"), (),
     "backend.measure (the measurement scope, the result identity, the "
     "Kernel Bench lookup client, the result caches, the process pool and the "
     "sweep engine `auto_sweep`) is read by inference, the API, the service "
     "and the kernel_bench harness, and reads none of them: inference supplies "
     "the sweep engine its model facts through the interface auto_sweep "
     "declares, and imports it"),
)


def _direction_scan() -> List[str]:
    offenders: List[str] = []
    for root, banned, exempt, _why in DIRECTION_BANS:
        for path in sorted(root.rglob("*.py")):
            rel = str(path.relative_to(root))
            for mod in _resolved_imports(path):
                for i, prefix in enumerate(banned):
                    if i == 0 and rel in exempt:
                        continue
                    if mod == prefix or mod.startswith(prefix + "."):
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {mod}")
    return offenders


#: Consumers read a leaf's `declaration`; they do not look an op up by name.
#: (tree, the names of `backend.arch.ops` they may not import, reason.)
#: `spec_worker` / `op_workers` are author-side: they list the registry to
#: brief an agent before any node exists, so they are not in this scan.
REGISTRY_ASK_BANS: Tuple[Tuple[Path, Tuple[str, ...], str], ...] = (
    (BACKEND_DIR / "rendering", ("OP_REGISTRY", "get_op"),
     "rendering reads SpecIRNode.declaration and the projections _node_build stamps"),
    (INFERENCE_DIR, ("OP_REGISTRY", "get_op"),
     "inference reads the node's declaration (phases) or the row it already has"),
    (PROJECT_ROOT / "kernel_bench", ("OP_REGISTRY", "get_op"),
     "kernel_bench reads the row / Invocation, never the registry"),
    (BACKEND_DIR / "analyzer.py", ("OP_REGISTRY", "get_op"),
     "the analyzer reads Analysis"),
)


def _registry_ask_scan() -> List[str]:
    offenders: List[str] = []
    for root, names, _why in REGISTRY_ASK_BANS:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("arch.ops"):
                    offenders.extend(
                        f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {alias.name}"
                        for alias in node.names if alias.name in names)
                elif isinstance(node, ast.Attribute) and node.attr in names \
                        and isinstance(node.value, ast.Name) and node.value.id in ("ops", "arch_ops"):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: ops.{node.attr}")
    return offenders


class TestEvalBoundary(unittest.TestCase):
    def test_consumers_do_not_ask_the_op_registry(self) -> None:
        """A consumer that looks an op up by name has rebuilt a fact the
        tree already carries (`SpecIRNode.declaration`)."""
        self.assertEqual(_registry_ask_scan(), [])

    maxDiff = None

    def test_engine_is_banned(self) -> None:
        _, imports = _scan()
        offenders = [k for k in imports if k.endswith(":engine")]
        self.assertEqual(
            offenders, [],
            "eval imported `engine`. The side-by-side design is void the "
            f"moment either side imports the other: {offenders}",
        )

    def test_dependency_direction_holds(self) -> None:
        offenders = _direction_scan()
        reasons = "; ".join(why for _root, _b, _e, why in DIRECTION_BANS)
        self.assertEqual(
            offenders, [],
            "an import points the wrong way (file: module). A lazy import "
            "counts too — direction is structure, not timing. The rules: "
            f"{reasons}. Offenders: {offenders}",
        )

    def test_defensive_access_ratchet(self) -> None:
        access, _ = _scan()
        grew = {k: (ACCESS.get(k, 0), v) for k, v in access.items()
                if v > ACCESS.get(k, 0)}
        self.assertEqual(
            grew, {},
            "getattr/hasattr sites grew (baseline, now). A new read of an "
            "old-world object wants a field on a protocol, not a default: "
            f"{grew}",
        )
        shrank = {k: (v, access.get(k, 0)) for k, v in ACCESS.items()
                  if access.get(k, 0) < v}
        self.assertEqual(
            shrank, {},
            "getattr/hasattr sites shrank (baseline, now) — good, but lower "
            f"ACCESS in this file in the same commit, or the slack hides the "
            f"next regression: {shrank}",
        )

    def test_old_world_import_ratchet(self) -> None:
        _, imports = _scan()
        grew = {k: (IMPORTS.get(k, 0), v) for k, v in imports.items()
                if v > IMPORTS.get(k, 0)}
        self.assertEqual(
            grew, {},
            "old-world import sites grew (baseline, now). `loader` rows are "
            "scheduled deletions and `materialize` must stay at zero; adding "
            f"a site deepens a coupling the swap has to undo: {grew}",
        )
        shrank = {k: (v, imports.get(k, 0)) for k, v in IMPORTS.items()
                  if imports.get(k, 0) < v}
        self.assertEqual(
            shrank, {},
            "old-world import sites shrank (baseline, now) — lower IMPORTS "
            f"in this file in the same commit: {shrank}",
        )


if __name__ == "__main__":
    unittest.main()
