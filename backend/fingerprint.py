"""Engine and spec identity hashes — the one definition.

`engine_version()` is a semantic hash of the estimation engine's source and
all its data; `spec_fingerprint(resolved)` is a content hash of one resolved
spec. A cached result for one model is keyed on `result_version(resolved)`:
the engine's code, the data every model shares, and only the data that model
can read (`ModelFootprint`), so an edit to one model's components or to a
fusion plan another model cannot match leaves the other models' results
cached. Every result cache's key folds one of these in (the caches are
`backend.measure.cache`'s; the keys their callers') and the analysis replay memo of validate (`backend.inference.analysable`)
build their keys from here; `engine_version()` stays the stamp for what
spans every model (the gallery frontier request, `code_version`).
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import functools
import hashlib
import json
import pathlib
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, ClassVar, Dict, FrozenSet, Iterator, Tuple


from backend.arch.eval.program import Program
from backend.code_trace import provider as code_trace_provider
from backend.code_trace.source import normalized_py_source
from backend.arch.eval.transforms import (rebind_decode_variants,
                                          splice_sampled_head)
from backend.arch.loader import ResolvedSpec, strip_machine_paths
from backend.arch.ops import OP_REGISTRY
from backend.inference.framework_bindings import (FRAMEWORK_BINDINGS,
                                                  _FRAMEWORK_BINDINGS_DIR)
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.materialize_cast import GRAFTED_OPS


# ---------------------------------------------------------------------------
# Versioning / keying — the single source of truth.
# ---------------------------------------------------------------------------


#: `backend/`, under which the engine trees live.
_ROOT = pathlib.Path(__file__).resolve().parent


def _key_derivation(fn: Any) -> Any:
    """Run `fn` as key derivation: no recorded computation counts the code
    it runs (`code_trace.provider.untraced`). Every version here names what
    a cached value is keyed on, never what the value is: a computation that
    asks for a key — or for a stamp — does not depend on how it is hashed,
    and whether it ran the hashing or hit a memo of it would otherwise be
    two traces of one computation."""
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        with code_trace_provider.provider().untraced():
            return fn(*args, **kwargs)
    return run


def _engine_files(suffixes: Tuple[str, ...]) -> Iterator[pathlib.Path]:
    """Every file of the engine trees with one of `suffixes`, in one order."""
    for sub in ("inference", "arch"):
        for p in sorted((_ROOT / sub).rglob("*")):
            if p.suffix in suffixes and p.is_file():
                yield p


@lru_cache(maxsize=1)
def _code_chunks() -> Tuple[bytes, ...]:
    """What `_hash_code` hashes, in order: each engine `.py`'s path and its
    normalized source. Normalizing parses every file (seconds), and both
    `engine_code_version` and `engine_version` hash it: read once per
    process, like the versions themselves."""
    chunks = []
    for p in _engine_files((".py",)):
        chunks.append(p.relative_to(_ROOT).as_posix().encode())
        raw = p.read_bytes()
        try:
            chunks.append(normalized_py_source(raw.decode("utf-8")).encode())
        except UnicodeDecodeError:
            # A non-UTF-8 engine file (encoding cookie) still hashes
            # distinctly — semantics we cannot parse we take verbatim.
            chunks.append(b"\x00raw-bytes\x00" + raw)
    return tuple(chunks)


def _hash_code(h: Any) -> None:
    for chunk in _code_chunks():
        h.update(chunk)


def _hash_data(h: Any, skipped: FrozenSet[str]) -> None:
    for p in _engine_files((".yaml", ".yml", ".csv")):
        rel = p.relative_to(_ROOT)
        if rel.parts[1] in skipped:
            continue
        h.update(rel.as_posix().encode())
        h.update(p.read_bytes())


@lru_cache(maxsize=1)
@_key_derivation
def engine_code_version() -> str:
    """Semantic hash of the engine's code alone: the normalized source of
    every `.py` under `backend/inference` and `backend/arch`. Global by
    nature — code is shared, and which functions one model's run reaches is
    not something a key can enumerate."""
    h = hashlib.sha256()
    _hash_code(h)
    return h.hexdigest()[:16]


# `GlobalHash` traces are this hash; the provider cannot import the engine
# trees it describes, so the hash is named to it here. A lambda, so a test
# that clears or patches `engine_code_version` is followed.
code_trace_provider.use_engine_code_version(lambda: engine_code_version())  # noqa: PLW0108


@lru_cache(maxsize=1)
@_key_derivation
def shared_data_version() -> str:
    """Hash of the engine data every model reads, or may read in a way its
    spec does not name: the dims / roles / cache-family / migration
    registries, the quant schemes (a request's dtype override can reach any
    of them) and the GPU table. Everything outside `ModelFootprint.DATA_DIRS`
    lands here, so a new data file is shared until shown otherwise."""
    h = hashlib.sha256()
    _hash_data(h, ModelFootprint.DATA_DIRS)
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
@_key_derivation
def engine_version() -> str:
    """Semantic hash of the estimation/rendering engine: normalized source of
    every ``.py`` under ``backend/inference`` and ``backend/arch`` (see
    ``normalized_py_source``) plus the raw bytes of the shared engine data
    files. Any result-affecting edit (a new cost term, a KV formula, a walk
    fix, a component/dims/binding data change) flips this, so every cached
    result keyed on it is transparently superseded — while comment,
    docstring and formatting edits no longer do. A *genuinely global*
    signal: engine changes affect every model. Computed once per process —
    the source can't change under a live server, and ``ha restart``
    re-imports and recomputes."""
    h = hashlib.sha256()
    _hash_code(h)
    # Engine data (component library, dims/roles, quant schemes, fusion
    # plans, GPU table). Per-model specs (models/, snapshots/) invalidate one
    # model at a time via spec_fingerprint.
    _hash_data(h, frozenset({"models", "snapshots"}))
    return h.hexdigest()[:16]


@_key_derivation
def spec_fingerprint(resolved: Any) -> str:
    """Per-model spec identity: a content hash of a *resolved* spec — its own
    params + inlined components, config-independent. Identical specs collide (as
    they should); a refined or regenerated spec for one model flips only that
    model's fingerprint, so a new spec never busts another model's cache. Live:
    it reflects the loader's current state, so after an
    ``invalidate_default_loader()`` + re-match it moves on its own."""
    spec = resolved.spec if isinstance(resolved, ResolvedSpec) else resolved
    if spec is None:
        # A model with no spec has no spec fingerprint. Returning a sentinel
        # made "no spec" a cache key — and a transient failure upstream
        # (HF timeout) that left `resolved` None got persisted as an error
        # entry under it. The caller decides what to do without a key.
        raise ValueError("spec_fingerprint: no spec to fingerprint")
    # A spec that does not serialize is a bug in the schema, not a reason to
    # hash `repr(resolved)` (which carries an object address: a permanent miss).
    # `source_path` fields are absolute CHECKOUT paths — stripping them makes
    # the fingerprint reproducible across machines, which is the whole point
    # of the shared remote cache tier (CI warms, every deployment hits).
    blob = json.dumps(
        strip_machine_paths(spec.model_dump(mode="json")), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ModelFootprint:
    """The engine data one model's results read, beyond the code and the
    shared data: its spec, the decode twin its spec names, the fusion plans
    that can match its graph, and the binding entries for those plans.

    Each is enumerated from the resolved spec alone, and each inclusion is
    sound only because what is left out cannot reach the model:

    - the spec's `ArchSpec` carries every component it binds (the loader
      copies the transitive closure in) and their claims (`axioms`); the one
      component a run reads besides is a `decode_variant` twin, fetched from
      the library at decode (a sparse attention's at prefill too — the same
      twin, `phase_form`), so the decode-rebound spec is hashed too;
    - a plan fires only where every pattern node's op is a leaf of the
      program it is matched against (`eval.fusion`), and that program is the
      walked one — the spec or its twin, lowered, with the sampled head
      spliced — plus the collectives layout inserts, the activation casts
      `materialize_casts` grafts, and the fused ops a rewrite writes
      (`ops`). A plan with a node no op of `ops` satisfies cannot fire here,
      and neither can its binding entries.
    """

    resolved: Any

    #: The engine data directories whose files reach a result only through
    #: what this class hashes: specs and snapshots, the component library
    #: and its claims, fusion plans, framework bindings.
    DATA_DIRS: ClassVar[FrozenSet[str]] = frozenset({
        "models", "snapshots", "components", "claims", "fusion_plans",
        "framework_bindings"})

    @cached_property
    def specs(self) -> Tuple[Any, ...]:
        """The spec, and its decode form when it declares one."""
        twin = rebind_decode_variants(self.resolved)
        return (self.resolved,) if twin is None else (self.resolved, twin)

    @cached_property
    def walked_ops(self) -> FrozenSet[str]:
        """The leaf ops of every program this model's runs walk."""
        out = set()
        for spec in self.specs:
            for walked in (spec, splice_sampled_head(spec)):
                if walked is None:
                    continue
                program = Program.of(walked).lowered()
                out.update(node.op for node in program.nodes()
                           if node.op and not node.children)
        return frozenset(out)

    @cached_property
    def ops(self) -> FrozenSet[str]:
        """Every op a program this model is fused on can hold: the walked
        leaves, the collectives layout inserts, the activation casts and
        stored-transform ops a deployment can graft, and the fused ops a
        rewrite writes."""
        return (self.walked_ops | frozenset(FUSION_SCHEMES)
                | frozenset(name for name, spec in OP_REGISTRY.items()
                           if spec.is_comm_op)
                | GRAFTED_OPS)

    def admits(self, scheme: Any) -> bool:
        """Whether every node of `scheme`'s pattern has an op in `ops`."""
        pattern = scheme.naive_pattern
        forms = ((pattern.parallel,) if pattern.parallel is not None
                 else tuple(pattern.nodes.values()))
        return all(({form.op} if form.op else set(form.op_choices))
                   & self.ops for form in forms)

    @cached_property
    def plans(self) -> Dict[str, Any]:
        """The fusion plans that can match this model, as loaded."""
        return {name: scheme.model_dump(mode="json")
                for name, scheme in sorted(FUSION_SCHEMES.items())
                if self.admits(scheme)}

    @cached_property
    def bindings(self) -> Dict[str, Any]:
        """Per framework, its audit tag and the entries this model can
        reach, in the order the binding tries them. An entry is reached when
        its plan can match, or when a walked leaf's op already is its fused
        op (a row is labelled fused by that name)."""
        out: Dict[str, Any] = {}
        for framework, binding in sorted(FRAMEWORK_BINDINGS.items()):
            entries = _binding_entries(framework)
            out[framework] = {
                "audited_at": [binding.audited_at.repo,
                               binding.audited_at.tag,
                               binding.audited_at.commit],
                "entries": [[op, entries[op]]
                            for op in binding.fused_ops()
                            if op in self.plans or op in self.walked_ops],
            }
        return out

    @cached_property
    def digest(self) -> str:
        blob = json.dumps({
            "specs": [spec_fingerprint(spec) for spec in self.specs],
            "plans": self.plans,
            "bindings": self.bindings,
        }, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@lru_cache(maxsize=None)
def _binding_entries(framework: str) -> Dict[str, Any]:
    """One binding file's entries as parsed: comments do not move them.

    A list-valued op is the multi-implementation spelling and stays a list in
    the fingerprint (including its order). The common single mapping remains
    a mapping, so existing footprints do not move merely because the loader
    learned the list spelling.
    """
    path = pathlib.Path(_FRAMEWORK_BINDINGS_DIR) / f"{framework}.yaml"
    with path.open(encoding="utf-8") as f:
        return {
            op: (raw if isinstance(raw, list) and len(raw) > 1
                 else (raw[0] if isinstance(raw, list) and raw else raw))
            for op, raw in ((load_yaml(f) or {}).get("op_implementations")
                            or {}).items()
        }


#: A model's footprint digest by what it is a function of: the fingerprints
#: of its spec and decode twin, and the config and dims its program is built
#: at. The registries it also reads are fixed for the process. Bounded: a
#: server resolves a fresh spec object per request.
_FOOTPRINTS: Dict[str, str] = {}
_FOOTPRINTS_MAX = 4096


def _key_code() -> str:
    """The code half of a cache key: the engine's code hash when the
    process's provider keys on the code (`GlobalHash`), else a constant — the
    code is then each entry's trace (`backend.code_trace.provider`)."""
    if code_trace_provider.provider().keys_code:
        return engine_code_version()
    return "code:traced"


@_key_derivation
def result_version(resolved: Any) -> str:
    """The engine identity one model's cached result is keyed on: the code
    (`_key_code`), the shared data, and that model's own footprint
    (`ModelFootprint`)."""
    return hashlib.sha256("\x00".join((
        _key_code(), shared_data_version(),
        _footprint_digest(resolved))).encode()).hexdigest()[:16]


@_key_derivation
def cache_engine_version() -> str:
    """`engine_version()` as a key or an artifact path names what spans every
    model: exactly `engine_version()` when the provider keys on the code;
    else the engine's data alone, the code being each entry's trace. A stamp
    that reports which engine answered stays `engine_version()`."""
    if code_trace_provider.provider().keys_code:
        return engine_version()
    h = hashlib.sha256(_key_code().encode())
    _hash_data(h, frozenset({"models", "snapshots"}))
    return h.hexdigest()[:16]


@_key_derivation
def data_version(resolved: Any) -> str:
    """`result_version` without the code: the shared data and the model's
    footprint. What a key names when the code half is answered elsewhere —
    a process-local memo, whose code cannot change under it, or a trace
    (`backend.code_trace.provider`)."""
    return hashlib.sha256("\x00".join((
        shared_data_version(), _footprint_digest(resolved))).encode()
    ).hexdigest()[:16]


def _footprint_digest(resolved: Any) -> str:
    """`ModelFootprint(resolved).digest`, memoised by what it is a function
    of."""
    footprint = ModelFootprint(resolved)
    inputs = hashlib.sha256(json.dumps([
        [spec_fingerprint(spec) for spec in footprint.specs],
        resolved.config, resolved.dim_values,
    ], sort_keys=True, default=str).encode()).hexdigest()
    digest = _FOOTPRINTS.get(inputs)
    if digest is None:
        if len(_FOOTPRINTS) >= _FOOTPRINTS_MAX:
            _FOOTPRINTS.clear()
        digest = _FOOTPRINTS[inputs] = footprint.digest
    return digest
