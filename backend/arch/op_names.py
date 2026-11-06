"""Sets of op names, declared so a name that does not exist cannot hide.

A module-level `frozenset` of op names is a branch condition written as data,
and it fails in the one way data cannot report: a name nobody registered
matches nothing, forever, while reading exactly like a live rule. Four of them
were in that state — `_ACT_OPS` naming three activations that do not exist,
`_STORAGE_WALKTHROUGH_OPS` two view ops, `_RANK_STRICT` a `matmul`,
`_TENSOR_MATH_OPS` an `activation` — and nothing anywhere said so.

Declare one with `op_names(...)` and it registers itself. `verify_all()` then
checks every set that has been declared against the real universe, so the
check needs no list of its own to keep in sync with the lists it checks.

Verification is deferred rather than done at construction because the universe
is not fully known at import time: primitives come from `ops.OP_REGISTRY`,
compound names come from the component YAML, and the loader that reads them
imports several of the modules declaring these sets.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
from backend.code_trace.memo import memoize
import inspect
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

#: `(module, qualified name, names)` for every set declared through
#: `op_names`. Populated at import time by the declaring module.
_DECLARED: List[Tuple[str, FrozenSet[str]]] = []


def op_names(*names: str) -> FrozenSet[str]:
    """A set of op / component names, registered for verification.

    Use for any constant whose members are matched against `node.op` — the
    caller gets a plain `frozenset` and `verify_all` gets something to check.
    """
    where = "?"
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        where = frame.f_back.f_globals.get("__name__", "?")
    declared = frozenset(names)
    _DECLARED.append((where, declared))
    return declared


def op_keyed(table: Dict[str, object]) -> Dict[str, object]:
    """A dispatch table KEYED by op name, registered for the same check.

    Same failure as a set — a key nobody registered dispatches for nothing —
    and the same fix, without asking the caller to restate its keys.
    """
    where = "?"
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        where = frame.f_back.f_globals.get("__name__", "?")
    _DECLARED.append((where, frozenset(table)))
    return table


def _component_graphs() -> Dict[str, Dict]:
    """`{component name: its graph}`, read off the library on disk.

    A component's name is its FILE STEM — `spec_parsing._RETIRED_IDENTITY_KEYS` says
    so, and retired the `name:` line that used to repeat it. Both readers below
    scan for the same thing, so they ask here rather than each carrying its own
    idea of what a component file looks like.
    """
    import yaml

    graphs: Dict[str, Dict] = {}
    for path in (Path(__file__).resolve().parent / "components").rglob("*.yaml"):
        try:
            doc = load_yaml(path.read_text()) or {}
        except yaml.YAMLError as e:
            # A malformed component would silently vanish from universe(),
            # making any spec that uses it look like an unknown op. Fail
            # loudly, naming the file. (tests/meta/test_component_yaml_parses
            # gates this at the source, so it can't fire on shipped data.)
            raise ValueError(
                f"component library YAML {path} is malformed: {e}"
            ) from e
        graph = doc.get("graph") if isinstance(doc, dict) else None
        if isinstance(graph, dict) and graph:
            graphs[path.stem] = graph
    return graphs


def universe() -> FrozenSet[str]:
    """Every name a spec node's `op` may legally hold: registered primitives
    plus every compound the component library declares."""
    from backend.arch.ops import OP_REGISTRY

    return frozenset(set(OP_REGISTRY) | set(_component_graphs()))


@memoize(maxsize=1)
def view_only_components() -> FrozenSet[str]:
    """Components whose every node is a view op, so bytes pass through them.

    Derived, and it has to be: the hand-kept set this replaced held
    `head_split` and `head_broadcast` and not `head_merge`, which is
    `transpose` + `view_merge` — two view ops, the same shape of thing as its
    two siblings, and the only one of the three that a storage walk stopped
    at. Nothing distinguished them but which names someone had typed.

    Cached, because it reads the component library off disk.
    """
    from backend.arch.ops import OP_REGISTRY

    found = set()
    for name, graph in _component_graphs().items():
        ops = [n.get("op") for n in graph.values() if isinstance(n, dict)]
        if ops and all(o in OP_REGISTRY and OP_REGISTRY[o].is_view_op
                       for o in ops):
            found.add(name)
    return frozenset(found)


def is_view_name(op: str) -> bool:
    """True if `op` re-indexes rather than rewrites.

    A primitive answers from `OpSpec.is_view_op`. A component answers if
    every node inside it is a view (`view_only_components`). One function
    so a compute-graph absorb and a traffic walk cannot disagree on the
    same name.
    """
    from backend.arch.ops import OP_REGISTRY

    spec = OP_REGISTRY.get(op)
    if spec is not None:
        return bool(spec.is_view_op)
    return op in view_only_components()


def is_join_name(op: str) -> bool:
    """True if `op` lays several tensors end to end along one axis.

    The property a caller wants from this is that the op READS EVERY BRANCH
    and makes nothing new — so whatever is true of all its inputs is true of
    its output. Which axis they are joined on does not enter into it:
    `concat` packs features, `kv_stream_concat` unions cache positions, and
    both hand a reader all of what they were handed.

    Asked of the registry rather than matched against a name, because a
    literal `op == "concat"` is a rule that silently narrows every time a
    second join is registered, and the way it narrows is invisible: an
    attention reading two cache streams through one union reads, as far as
    such a rule can tell, no cache at all.
    """
    from backend.arch.ops import OP_REGISTRY, ShapeKind

    spec = OP_REGISTRY.get(op)
    return spec is not None and spec.shape_kind is ShapeKind.AXIS_CONCAT


def verify_all() -> Dict[str, List[str]]:
    """`{module: [names that do not exist]}`, empty when every set is real."""
    known = universe()
    bad: Dict[str, List[str]] = {}
    for module, names in _DECLARED:
        missing = sorted(names - known)
        if missing:
            bad.setdefault(module, []).extend(missing)
    return bad
