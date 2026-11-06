"""The dim-symbol registry — one classification and one meaning per symbol.

`dims.yaml` beside this module is the declaration; this is the reader. A
model spec's `dims:` block binds a symbol to a config field, which is a fact
about that model; what the symbol IS — what it counts, and which rank groups
may split it — is a fact about the symbol, and it is written once, here.

Two fields say that, because per-rank extent is the product of three facts
owned by three different people: which axes a tensor is shaped on (the spec
author's), which worlds are mathematically allowed to split an occurrence of
that axis (this registry's), and what layout an endpoint or propagation
actually selects (the sharding pipeline's). `kind` is the first half of the
classification and `shardable_by` the capability half. A capability is not an
actual placement: two occurrences of the same symbol may choose differently.

Three tables used to hold that between them: the divisible kinds in
`eval/transforms.py`, the `whole` set in `tests/backend/arch/test_eval_axis_kinds.py`, and
the prose in `dims_glossary.md`, read back out through a markdown regex. Two
were unenforceable and all three had to be edited together; the three DSv4
compressor axes show what that costs — `Cdim_io` was sharded by the first
table and unknown to the other two, so the symbol whose width TP divides had
no meaning anyone could look up.

`divisible_kinds()` and `eval.transforms.tp_divides()` are projections of
the registry rather than literals, which is the whole point: adding a symbol
is one edit, and forgetting one is a load error rather than a tensor that
silently never shards. They are read when asked, never copied at import,
because the registry a running process answers from is dims.yaml plus the
symbols the specs it has loaded declare for themselves (`LocalSymbols`).
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import itertools
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

import sympy as sp

from backend.arch.law_text import LAW_OPERATORS
from backend.code_trace.memo import memoize

_REGISTRY_PATH = Path(__file__).parent / "dims.yaml"

#: Process-local extras from a proposal `dims.yaml` fragment. Filled only by
#: `apply_pending_op_patch` in a spec-gen child; serving traffic must not.
_PENDING: Dict[str, dict] = {}


#: What a `kind` may say: what the number COUNTS. Architecture vocabulary,
#: and deliberately silent about deployment — a head count is a head count
#: whether or not any rank group splits it. `whole` is an extent that counts
#: nothing distributable: `d_model` is whole because the hidden size is not a
#: thing ranks divide between them, even though every tensor shaped on it
#: shards through the head-count axis beside it.
#:
#: This used to be a map to what divides each kind, which made it the answer
#: to two questions at once — and the second answer was a closed set. There
#: was exactly one degree a head axis could be split by, so an axis with its
#: own rank group had nowhere to be filed and had to be misfiled as `whole`.
#: `shardable_by:` answers that question now, and it is open.
KINDS: FrozenSet[str] = frozenset({
    "head_count",     # parallel group axes: attention / indexer / SSM /
                      # vision heads, DSv4 lora groups
    "split_width",    # column-row split widths: FFN intermediates, conv io
    "vocab",          # the vocabulary axis
    "sequence",       # token axes — POSITIONS within a request
    "batch",          # a count of independent requests
    "expert_copies",  # a count of experts, not a width
    "layer_count",    # a count of layers, and no world's to divide — see
                      # `_KINDS_WITHOUT_WORLD`
    "whole",          # an extent no rank group divides
    "constant",       # counts nothing: a real number a function takes (a
                      # clamp bound, an activation's beta). Not an extent,
                      # so it may be fractional and no rank group divides
                      # it; a config that does not publish it leaves it
                      # unbound (`dim_resolution.resolve_dimensions`).
})

#: The meshes a world can be a subgroup of. A split in every one of them
#: divides a SHAPE — the tensor gets smaller along an axis. Expert parallelism
#: is no exception: an expert bank's weights and activations carry their
#: experts as an axis (`n_expert_copies`, declared in each weight's `shape:`
#: and in `permute`'s output), so a rank holding a slice of that axis is
#: shape division like any other. What a mesh says is WHICH axis, and what
#: the rank may then read across.
#:
#: `dp` and `cp` both cut the token stream and are NOT the same cut, which is
#: the distinction this mesh exists to make: `cp` divides the positions of one
#: sequence, so an op that reads across positions (attention) is broken by it
#: and owes a ring or an all_to_all; `dp` divides whole REQUESTS, which are
#: independent by construction, so the same op is separable and owes nothing.
#: Spelling both `Sharded("s_ctx")` made every DP-attention deployment price a
#: context-parallel repair it does not perform — the two cuts have to be two
#: axes for the algebra to tell them apart. See `attn_data` in `dims.yaml`.
#:
#: `pp` is not here. A world's whole content is a DEGREE that every axis in it
#: is divided by, and pipeline parallelism has no such number: it partitions
#: the layer index space, and the partition is uneven by design
#: (`stage_layer_ranges(28, 8) == [3,3,3,4,4,4,4,3]`, vLLM's rule). Declaring
#: `n_layers` in a world at degree `pp` is exactly the retired 28/8 = 3.5 bug,
#: written down as a fact instead of computed as one. Which layers a stage
#: holds is placement's answer (`layout.stages.StagePlacement`), per stage.
MESHES: FrozenSet[str] = frozenset({"tp", "cp", "ep", "dp"})

#: The serving phases a world's `phases:` may name. Declared here — in the
#: registry, beside the worlds — because this is the layer where a phase is a
#: piece of DEPLOYMENT VOCABULARY rather than a step being priced.
#: `encode` runs an encoder stack over its media inputs, ahead of the language
#: model's `prefill` and `decode`.
#: `sharding.PREFILL` / `DECODE` re-export these, and `inference.phases` prices
#: them; all three read one spelling, so a typo in a yaml `phases:` is a load
#: error rather than a world that quietly splits nothing in either phase.
PHASES: Tuple[str, ...] = ("encode", "prefill", "decode")

#: Kinds whose meaning fixes their mesh — a token axis is context
#: parallelism's to split and nothing else's. Checked at load so the two
#: fields cannot drift into disagreeing about the same symbol.
_KIND_MESH: Mapping[str, str] = {
    "sequence": "cp",
    "expert_copies": "ep",
    "batch": "dp",
}

#: Kinds that may name no world, and must not, for two opposite reasons.
#: `whole` and `constant` because nothing splits the symbol. `layer_count`
#: because what splits it cannot be said as a degree — see `MESHES`. All are
#: checked so that "no world" stays a classification rather than an omission.
_KINDS_WITHOUT_WORLD: FrozenSet[str] = frozenset(
    {"whole", "layer_count", "constant"})


#: The nouns a COUNT of something may be displayed in (`DimSymbol.unit`):
#: `× 61 layers`, `× 256 experts`. Closed, because the alternative is
#: synonyms — a reader cannot tell `copies` from `instances` from `banks`
#: apart if three authors each pick one. Grown by adding a word here with
#: the symbol that needs it, not by writing a new one at a call site.
#:
#: Deliberately WITHOUT a generic `copies`. Every symbol the corpus repeats
#: wants one of the three words above, so a fourth generic one would have no
#: user — and an unreachable vocabulary entry is an invitation, this one
#: especially: `copies` is the wrong label the substring guess this field
#: replaces produces for a layer loop. A symbol that declares no unit
#: publishes None and the display shows the symbol's own name instead:
#: `× 61 copies` for a layer loop tells the reader something false, while
#: `× 61 n_mtp_layers` is true and incomplete.
COPY_UNITS: FrozenSet[str] = frozenset({"layers", "experts", "banks"})


class CallTokens(str, Enum):
    """The roles a symbol can have in the tokens one kernel call runs over
    (`DimSymbol.call_tokens`)."""
    NONE = ""
    FACTOR = "factor"
    TOTAL = "total"
    CONTEXT = "context"


@dataclass(frozen=True)
class DimSymbol:
    """One registered symbol."""
    name: str
    kind: str
    meaning: str
    group: str
    #: Typical HF config field a spec binds this to, for the author of the
    #: next spec. Documentation: the binding itself lives in each `dims:`.
    config: str = ""
    #: The binding a spec writes when it is not the config field's value
    #: itself: a presence switch (`present(config.x) ?? …`) or the literal the
    #: forward source fixes. Documentation like `config`, and a row says one
    #: or the other.
    binding: str = ""
    #: A name no spec writes: its value is always this expression over other
    #: symbols (`n_expert_copies` = `n_experts`, `n_route_targets` =
    #: `n_experts + n_zero_experts`). Evaluated at model resolution
    #: (`dim_resolution._apply_derived_defaults`) and AGAIN in every component
    #: or block-assignment frame whose `dims:` rebinds a name it reads
    #: (`Binding.frame`), so a call that rebinds `n_experts` carries its own
    #: expert axis and denominator. A derivation whose inputs a model never
    #: resolves is not that model's symbol and is skipped.
    derived: str = ""
    #: What an AUTHORABLE symbol is when a model's `dims:` does not bind it
    #: (`Qb` = 0, `d_ffn_shared` = `d_ffn_expert`). Filled once, at model
    #: resolution; a child frame inherits the model's value like any other,
    #: because the frame cannot tell a defaulted value from a written one.
    #: A row declares `default:` or `derived:`, never both.
    default: str = ""
    #: An upper bound, an expression over other symbols, that this one never
    #: exceeds: a top-k picks at most what it chooses among
    #: (`n_experts_active` <= `n_route_targets`, `n_groups_active` <=
    #: `n_expert_groups`). Checked once the model's dims are resolved
    #: (`dim_resolution.resolve_dimensions`); a config that breaks it is
    #: refused, naming the bound, rather than evaluated as some other model.
    at_most: str = ""
    #: Which architecture introduced it, for the per-arch one-offs.
    used_by: str = ""
    #: Rank groups mathematically allowed to split this symbol. Empty means
    #: whole; it does not mean a declaration was forgotten. This is a
    #: capability list, not the layout of every occurrence. Weight/cache/input
    #: endpoint contracts choose an actual member, and propagation carries it.
    shardable_by: Tuple[str, ...] = ()
    #: True when the engine supplies the value and no call site binds it:
    #: `B` and `s_ctx` are threaded from the analyze request, the phase
    #: binding is resolved once the phase is known. Every other symbol is a
    #: parameter — a model's `dims:` block or a component call site says how
    #: wide it is — and that difference is what
    #: `ops_base.reject_divisible_parameter_names` turns on. Declared per
    #: symbol rather than read off `group`, which is a glossary section: a
    #: rule that keyed on the section would silently stop applying the day an
    #: engine-supplied symbol is filed under a new heading.
    engine_supplied: bool = False
    #: What the symbol is of the token count a measured shape is drawn along
    #: (Kernel Bench's per-op curves): `factor` — a factor of it, multiplied
    #: with the others only where a signature's contract folds them into one
    #: axis (`OpSignature.token_runs`: a GEMM's rows `B · s_query`; attention
    #: keeps `B` and `s_query` apart); `total` — the symbol is the count
    #: itself (a grouped expert kernel's routed `n_tokens`), and the factors
    #: the shape also has move with it; `context` — the keys a call attends
    #: per sequence (`s_kv`): a signature whose contract has it runs along
    #: the batch times it, the KV tokens one call reads. Empty: not a token
    #: count.
    call_tokens: str = ""
    #: True on the axes whose product is the tokens a step runs: the batch
    #: and the query length under each name it is bound by (`B`, `s_ctx`,
    #: `s_query`, `n_verify`). Not `kind`: the batch is kind `batch`,
    #: `n_verify` is `whole`, and `s_kv` is a `sequence` counting the keys
    #: attended. A cached tensor's row is what is left once these are taken
    #: off (`step_token_axes`). Engine-supplied only: no spec binds a step.
    #: Not `call_tokens`, which names the axes a Kernel Bench curve is drawn
    #: along (`B`, `s_query` and a routed `n_tokens`): a walked tensor also
    #: carries the step's tokens as `s_ctx` and `n_verify`, which no curve is
    #: drawn along, and a cache row keeps the routed count.
    step_token_axis: bool = False
    #: True on a count of the (query, key) pairs of the matrix a node scores
    #: (`s_pairs`): a rank that holds a share of the matrix's cells — its
    #: query rows cut (PCP), its keys cut (DCP) — scores that share of the
    #: pairs, whatever the mask (`Shape.pairs`). Not derivable by
    #: substituting per-rank `s_query` / `s_kv` into the pairs' law — the
    #: causal pairs of `s_query / cp` queries are not the pairs `cp` ranks
    #: split — so the per-rank value is the global one times the share
    #: (`layout/physical.pairs_share`).
    score_pairs: bool = False
    #: The display noun for a count of this symbol's copies — the word in
    #: `× 61 layers` / `× 2 experts`, one of `COPY_UNITS`. Declared here
    #: because `kind` cannot carry it: `expert_copies` drags
    #: `shardable_by: [moe_copies]` with it, and expert parallelism does not
    #: shard `n_shared_experts`, so spelling the shared experts' kind to fix
    #: a word would break their sharding. Optional and NOT defaulted — a
    #: symbol with no unit publishes None and the display falls back to the
    #: symbol's own name, which is true where a guessed noun is not. Required
    #: exactly where it is load-bearing, by a corpus gate
    #: (`tests/backend/arch/eval/test_copy_factors.py`) rather than by this
    #: schema: a symbol that no spec repeats needs no noun.
    unit: str = ""

    @property
    def binding_hint(self) -> str:
        """How a spec binds this symbol, as the author of the next spec is
        told it: the declared binding form, else the config field, else the
        derivation that fills it when unbound."""
        if self.binding:
            return self.binding
        if self.config:
            return f"`config.{self.config}`"
        return f"= `{self.when_unbound}`" if self.when_unbound else ""

    @property
    def when_unbound(self) -> str:
        """The expression that fills this symbol at model resolution when
        the model's `dims:` does not bind it: its derivation or its default."""
        return self.derived or self.default

    @property
    def reads(self) -> FrozenSet[str]:
        """The symbols `when_unbound` names (function names excluded)."""
        return _names_in(self.when_unbound)

    @property
    def bound_reads(self) -> FrozenSet[str]:
        """The symbols `at_most` names."""
        return _names_in(self.at_most)

    def __post_init__(self) -> None:
        if self.unit and self.unit not in COPY_UNITS:
            raise ValueError(
                f"symbol {self.name!r} has unit {self.unit!r}, not one of "
                f"{sorted(COPY_UNITS)}. The vocabulary is closed so two "
                f"authors cannot name one noun twice; add the word here with "
                f"the symbol that needs it rather than at a call site.")
        if self.call_tokens not in {role.value for role in CallTokens}:
            raise ValueError(
                f"symbol {self.name!r} has call_tokens {self.call_tokens!r}, "
                f"not one of {sorted(role.value for role in CallTokens if role.value)}")
        if self.step_token_axis and not self.engine_supplied:
            raise ValueError(
                f"symbol {self.name!r} is a `step_token_axis` the engine does "
                f"not supply; a step's tokens are the engine's to bind")
        if self.derived and self.default:
            raise ValueError(
                f"symbol {self.name!r} declares both `derived:` and "
                f"`default:`. `derived:` is a name no spec writes, computed "
                f"in every frame; `default:` fills a name a spec may write. "
                f"Keep the one that is true of it")
        if self.binding and self.config:
            raise ValueError(
                f"symbol {self.name!r} declares both `config:` and "
                f"`binding:`; a row whose binding is not the config field's "
                f"value says how to bind it in `binding:` alone")
        if isinstance(self.shardable_by, list):
            object.__setattr__(self, "shardable_by", tuple(self.shardable_by))
        if not isinstance(self.shardable_by, tuple) or any(
            not isinstance(world, str) or not world
            for world in self.shardable_by
        ):
            raise ValueError(
                f"symbol {self.name!r} has invalid `shardable_by` capabilities "
                f"{self.shardable_by!r}"
            )
        if len(set(self.shardable_by)) != len(self.shardable_by):
            raise ValueError(
                f"symbol {self.name!r} repeats a `shardable_by` capability"
            )


@dataclass(frozen=True)
class World:
    """A rank group an axis can live in."""
    slug: str
    #: `tp` / `cp` / `ep` / `dp` (`MESHES`).
    mesh: str
    meaning: str
    #: The `ParallelConfig` field carrying this world's degree. Empty for a
    #: world no deployment knob has named yet: it runs at 1, so every axis in
    #: it comes out whole by arithmetic rather than by declaration.
    degree: str = ""
    #: The world whose ranks this one PARTITIONS, `""` for a world that spans
    #: ranks of its own. The distinction is a cluster-size fact: a world with
    #: ranks of its own multiplies the deployment's GPU count, and a subgroup
    #: does not — it re-reads ranks the parent world already counted.
    #:
    #: vLLM says this outright about the pair this field exists for.
    #: `config/parallel.py:833` puts prefill context parallelism in the
    #: product (`world_size = pp * tp * pcp`) and `:342` says of decode
    #: context parallelism "DCP does not expand the process world size.
    #: Without PCP, DCP reuses TP ranks". Modelling the second as another
    #: factor asks for `dcp x` the GPUs a real deployment uses, which is a
    #: capacity answer that is wrong in the direction that looks safe.
    #:
    #: A subgroup's degree must divide its parent's, since it is a partition
    #: of that world's ranks and not a request for more.
    subgroup_of: str = ""
    #: The serving phases in which this world actually splits anything. Empty
    #: — the normal case — means every phase: a weight split is a weight split
    #: whichever way the step runs.
    #:
    #: `sequence` is the one world that has to say otherwise. vLLM's PCP
    #: partitions a prefill chunk across its ranks and REPLICATES the decode
    #: tokens over the same group (`attention/pcp.py::_gather_prefill_cache_inputs`
    #: — "Keep replicated decode writes local and gather partitioned prefills"),
    #: so in decode its degree counts real ranks that are each doing the same
    #: work, and nothing on any axis is divided.
    #:
    #: Declared here rather than branched on downstream so that the gate is the
    #: declaration: `sharding.degree_in_phase` reads this field and knows about
    #: no world by name, which is what keeps a second prefill-only world from
    #: needing a second hand-written `if` that the yaml has no way to reach.
    phases: Tuple[str, ...] = ()
    #: Whether a caller binds this world's axes at their GLOBAL extent (the
    #: default, and what the shape walk assumes: `Parallelism.policy` divides
    #: by the degree to get one rank's share) or already at one rank's.
    #:
    #: `attn_data` is the one world bound per rank, and it is bound that way
    #: because its axis is an ANSWER rather than a request. `B` comes from
    #: `capacity`, which sizes a batch against ONE GPU's free KV memory —
    #: `capacity.py` multiplies back up (`max_batch_total = max_batch *
    #: unique_tokens_multiplier`) precisely because what it solved for is per
    #: rank. Every other world's axes come from the architecture or the
    #: request, which know nothing of ranks.
    #:
    #: Declared rather than left to the caller's discipline because the two
    #: readings differ by a silent factor. `ParallelConfig.eval_parallelism`
    #: has always SAID `dp_attention` divides nothing, and passed no
    #: `dp_attention=`; it fills `world_degrees` from the whole registry, which
    #: handed the walk the same number by the other door and would have
    #: rescaled every shape if the binding still divided on read.
    #:
    #: An axis declared here is a fact somebody wrote down. Its cluster
    #: extent is recoverable by the obvious multiplication — which is
    #: exactly what the MoE dispatch's `M` does with `B` and `dp_attention`.
    bound_per_rank: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.phases, list):        # yaml gives a list
            object.__setattr__(self, "phases", tuple(self.phases))

    def active_in(self, phase: str) -> bool:
        """Whether this world splits anything in `phase`."""
        return not self.phases or phase in self.phases

    @property
    def label(self) -> str:
        """The name a PLACEMENT gives this world: the degree field, or the
        slug where no field exists.

        Two names for one world are two names too many, and both halves of
        this one are load-bearing. The registry keys a world by SLUG
        (`sequence`, `kv_shard`), which is what an axis declaration points at.
        `TensorShardingState` keys a placement by the DEGREE FIELD (`cp`,
        `dcp`), because that is the name a deployment types, the name
        `Parallelism.degree_of_world` turns back into a number, and the name a
        collective is reported under. A world with no field is one no knob
        names — `ParallelConfig.world_degrees` reaches it by slug — so the slug
        is what a placement on it can be called.

        Either way the label names a world whose degree can be READ, which is
        what makes it safe to key a placement on: `Placement` carries no degree
        of its own, and a label nothing resolves prices its collective at 1.
        This property is the only place the choice between the two is made.
        """
        return self.degree or self.slug


@dataclass(frozen=True)
class DimGroup:
    """A documentation grouping — the sections the glossary reads as."""
    slug: str
    title: str
    intro: str = ""


def _parse_symbol(name: str, entry, *, slugs, worlds) -> DimSymbol:
    """One registry row. Same constructor `_load` uses; a fragment that
    would not load must fail here, not later in a child process."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"dims.yaml: symbol {name!r} is not a mapping")
    try:
        sym = DimSymbol(name=name, **dict(entry))
    except TypeError as e:
        raise ValueError(f"dims.yaml: symbol {name!r}: {e}") from e
    if sym.kind not in KINDS:
        raise ValueError(
            f"dims.yaml: symbol {name!r} has kind {sym.kind!r}, not one of "
            f"{sorted(KINDS)}. An extent no rank group divides is `whole`, "
            f"which is a classification and not a default."
        )
    unknown_worlds = [world for world in sym.shardable_by if world not in worlds]
    if unknown_worlds:
        raise ValueError(
            f"dims.yaml: symbol {name!r} is shardable by "
            f"{unknown_worlds!r}, which no `worlds:` entry declares (worlds: "
            f"{sorted(worlds)}). A capability must name a group whose "
            "topology and degree can be resolved."
        )
    if (sym.kind in _KINDS_WITHOUT_WORLD) != (not sym.shardable_by):
        reason = (
            f"`{sym.kind}` says no rank group may split the symbol, and a "
            "capability says one may"
            if sym.kind in ("whole", "constant")
            else "`layer_count` is split by a partition of the layer index "
            "space, which has no single degree to name — a capability would "
            "claim n_layers/pp layers on every stage"
            if sym.kind == "layer_count"
            else "every other kind must name at least one group capable of "
            "splitting it"
        )
        raise ValueError(
            f"dims.yaml: symbol {name!r} is kind {sym.kind!r} with "
            f"shardable_by={list(sym.shardable_by)!r}. {reason}."
        )
    if sym.kind == "expert_copies" and not sym.derived:
        raise ValueError(
            f"dims.yaml: symbol {name!r} is kind 'expert_copies' and is not "
            f"`derived:`. The expert axis is a name the walk derives "
            f"(`n_expert_copies` = `n_experts`, in every frame), never one a "
            f"spec writes: a pool with its own expert count rebinds "
            f"`n_experts` where it is called, and its axis follows.")
    want_mesh = _KIND_MESH.get(sym.kind)
    wrong_meshes = [
        world
        for world in sym.shardable_by
        if want_mesh and worlds[world].mesh != want_mesh
    ]
    if wrong_meshes:
        raise ValueError(
            f"dims.yaml: symbol {name!r} is kind {sym.kind!r}, which is "
            f"{want_mesh}'s to split, but capabilities {wrong_meshes!r} "
            "are on another mesh."
        )
    if sym.group not in slugs:
        raise ValueError(
            f"dims.yaml: symbol {name!r} is in group {sym.group!r}, which "
            f"no `groups:` entry declares (groups: {sorted(slugs)})."
        )
    if not sym.meaning:
        raise ValueError(
            f"dims.yaml: symbol {name!r} has no meaning. What the number "
            f"counts is the half of the registry a human reads."
        )
    return sym


@memoize(maxsize=1)
def _load_base() -> Tuple[Dict[str, DimSymbol], Tuple[DimGroup, ...],
                          Dict[str, World]]:
    """`dims.yaml` (+ a spec-gen child's `_PENDING` fragment), parsed once.

    A spec's own `dim_symbols:` never clears this: re-parsing the file costs
    ~60 ms, and a layer is added and withdrawn per loaded spec."""
    raw = load_yaml(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    groups = tuple(DimGroup(**g) for g in (raw.get("groups") or []))
    slugs = {g.slug for g in groups}
    worlds: Dict[str, World] = {}
    for entry in (raw.get("worlds") or []):
        world = World(**entry)
        if world.mesh not in MESHES:
            raise ValueError(
                f"dims.yaml: world {world.slug!r} is on mesh {world.mesh!r}, "
                f"not one of {sorted(MESHES)}."
            )
        bad_phases = [p for p in world.phases if p not in PHASES]
        if bad_phases:
            raise ValueError(
                f"dims.yaml: world {world.slug!r} restricts itself to phases "
                f"{bad_phases}, which are not among {list(PHASES)}. A phase "
                f"name nothing prices is a world that splits nothing and says "
                f"so nowhere."
            )
        worlds[world.slug] = world
    for world in worlds.values():
        if world.subgroup_of and world.subgroup_of not in worlds:
            raise ValueError(
                f"dims.yaml: world {world.slug!r} is a subgroup of "
                f"{world.subgroup_of!r}, which no `worlds:` entry declares. A "
                f"subgroup partitions ranks some other world already spans; "
                f"naming one that does not exist leaves it spanning ranks "
                f"nothing counted, which is the cluster-size bug it exists "
                f"to prevent."
            )
    symbols: Dict[str, DimSymbol] = {}
    symbol_items = dict(raw.get("symbols") or {})
    symbol_items.update(_PENDING)
    for name, entry in symbol_items.items():
        symbols[name] = _parse_symbol(name, entry, slugs=slugs, worlds=worlds)
    return symbols, groups, worlds


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: A name applied as a function (`ceiling(...)`): not a dim.
_APPLIED = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _names_in(expr: str) -> FrozenSet[str]:
    """The symbols an expression names, function names excluded."""
    applied = set(_APPLIED.findall(expr))
    return frozenset(i for i in _IDENT.findall(expr) if i not in applied)



class LocalSymbols:
    """The symbols specs declare for themselves (`dim_symbols:`), installed
    for as long as the loader that loaded each spec holds it.

    Gates never read this: whether a spec may bind a name is asked of
    dims.yaml and that spec's own declarations (`registered_for`). The
    runtime projections (`world_of`, `capabilities_of`, the shard policy)
    read dims.yaml plus every live layer, because they are asked by name with
    no spec in hand; the gate guarantees a spec only asks about names it
    declared. One lock covers checks, installs and the merged table."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._layers: Dict[int, Tuple[str, Dict[str, DimSymbol]]] = {}
        self._tokens = itertools.count(1)
        self._merged: Optional[Dict[str, DimSymbol]] = None

    def parse(self, owner: str,
              entries: Mapping[str, Any]) -> Dict[str, DimSymbol]:
        """Each row through the constructor dims.yaml is loaded with, plus
        the rules for a row a spec writes. Raises with the fix."""
        if not isinstance(entries, Mapping):
            raise ValueError(
                f"dim_symbols must map a symbol to its row (kind, "
                f"shardable_by, group, meaning), got {type(entries).__name__}")
        base, groups, worlds = _load_base()
        slugs = {g.slug for g in groups}
        out: Dict[str, DimSymbol] = {}
        for name, entry in entries.items():
            name = str(name)
            if name in base:
                sym = base[name]
                raise ValueError(
                    f"dim_symbols.{name}: {name!r} is registered in "
                    f"backend/arch/dims.yaml (kind {sym.kind}, meaning: "
                    f"{sym.meaning.strip()}). Delete this declaration and bind "
                    f"the registered symbol if it is the same quantity; "
                    f"otherwise declare the new quantity under another name.")
            if not _IDENT.fullmatch(name) or not _is_plain_name(name):
                raise ValueError(
                    f"dim_symbols.{name}: {name!r} is not usable as an axis "
                    f"name (an identifier that is neither a law operator nor "
                    f"a name SymPy reads as a builtin, such as `beta`, `E`, "
                    f"`I`). Choose a descriptive name like the dims.yaml "
                    f"ones: `n_` for a count, `d_` for a width, `s_` for "
                    f"positions.")
            # `used_by` is stamped at promotion; only the engine supplies.
            authored = [f for f in ("used_by", "engine_supplied")
                        if isinstance(entry, Mapping) and f in entry]
            if authored:
                raise ValueError(
                    f"dim_symbols.{name}: remove {authored}. `used_by` is "
                    f"written when the symbol is promoted into dims.yaml, and "
                    f"a spec's symbols are bound by the spec, not supplied by "
                    f"the engine.")
            try:
                sym = _parse_symbol(name, entry, slugs=slugs, worlds=worlds)
            except ValueError as e:
                raise ValueError(
                    f"dim_symbols.{name}: "
                    f"{str(e).removeprefix('dims.yaml: ')}") from None
            out[name] = sym
        for name, sym in out.items():
            unknown = sorted(
                ident for ident in sym.reads | sym.bound_reads
                if ident not in base and ident not in out)
            if unknown:
                field = ("derived" if sym.derived else "default"
                         if sym.default else "at_most")
                raise ValueError(
                    f"dim_symbols.{name}: `{field}:` names "
                    f"{unknown}, which neither dims.yaml nor this spec's "
                    f"dim_symbols declares.")
        return out

    def declare(self, owner: str, entries: Mapping[str, Any]) -> Optional[int]:
        """Install `owner`'s declarations; returns the token that withdraws
        them, or None when there are none."""
        if not entries:
            return None
        with self._lock:
            parsed = self.parse(owner, entries)
            for name, sym in parsed.items():
                # Another record's layer is neither named nor quoted.
                for _owner, layer in self._layers.values():
                    held = layer.get(name)
                    if held is None or _numeric(held) == _numeric(sym):
                        continue
                    raise ValueError(
                        f"dim_symbols.{name}: another spec loaded in this "
                        f"server classifies {name!r} differently on kind / "
                        f"shardable_by / derived / default; this spec says "
                        f"kind {sym.kind}, shardable_by "
                        f"{list(sym.shardable_by)}"
                        f"{f', derived {sym.derived}' if sym.derived else ''}"
                        f"{f', default {sym.default}' if sym.default else ''}. "
                        f"Rename the symbol.")
            token = next(self._tokens)
            self._layers[token] = (owner, parsed)
            self._merged = None
            return token

    def withdraw(self, token: Optional[int]) -> None:
        if token is None:
            return
        with self._lock:
            if self._layers.pop(token, None) is not None:
                self._merged = None

    def merged(self) -> Dict[str, DimSymbol]:
        """dims.yaml ∪ every live layer. A dict merge, not a re-parse.

        The fast path reads `_merged` without the lock. A reader racing an
        `add` / `withdraw` may get the table built before that change: a
        whole table, never a half-merged one, because a table is only ever
        published complete and never mutated after."""
        merged = self._merged
        if merged is not None:
            return merged
        with self._lock:
            if self._merged is None:
                table = dict(_load_base()[0])
                for _owner, layer in self._layers.values():
                    for name, sym in layer.items():
                        table.setdefault(name, sym)
                self._merged = table
            return self._merged

    def reset(self) -> None:
        """Forget the merged table (the base changed)."""
        with self._lock:
            self._merged = None




def _is_plain_name(name: str) -> bool:
    """Whether a law reads `name` as an axis: not one of its operators, and
    not a name SymPy's default namespace owns."""
    return name not in LAW_OPERATORS and name not in vars(sp)


def _numeric(sym: DimSymbol) -> Tuple[Any, ...]:
    """The fields of a symbol that change a number: what it counts, who may
    split it, and its value when unbound. Two live spec-local declarations of
    one name must agree on these; `meaning` / `config` / `group` are
    documentation and may differ. Promotion compares every field but
    `used_by`."""
    return (sym.kind, frozenset(sym.shardable_by), sym.derived, sym.default)


LOCAL = LocalSymbols()


def declare_local(owner: str, entries: Mapping[str, Any]) -> Optional[int]:
    """Install a spec's `dim_symbols:` (see `LocalSymbols`)."""
    return LOCAL.declare(owner, entries)


def withdraw_local(token: Optional[int]) -> None:
    LOCAL.withdraw(token)


def local_symbols(entries: Mapping[str, Any], owner: str = "") -> Dict[str, DimSymbol]:
    """A spec's `dim_symbols:` rows as symbols, checked, not installed."""
    return LOCAL.parse(owner, entries or {})


def base_symbols() -> Mapping[str, DimSymbol]:
    """dims.yaml's symbols, without any spec's own."""
    return _load_base()[0]


def registered_for(local: Optional[Mapping[str, Any]] = None) -> FrozenSet[str]:
    """The names a spec may bind: dims.yaml's and its own `dim_symbols`.
    Never another spec's — the gate a validate subprocess and the server
    answer the same way."""
    return frozenset(_load_base()[0]) | frozenset(local or ())


def _load() -> Tuple[Dict[str, DimSymbol], Tuple[DimGroup, ...],
                     Dict[str, World]]:
    """The runtime table: dims.yaml ∪ live spec-local layers, with the
    base's groups and worlds (a spec cannot declare a world)."""
    _, groups, worlds = _load_base()
    return LOCAL.merged(), groups, worlds


def parse_row(name: str, entry: Any, registry: Mapping[str, Any]) -> DimSymbol:
    """One symbol row checked against the `groups:` / `worlds:` of a parsed
    dims.yaml document (another tree's, at promotion), by the constructor
    this process's registry is loaded with."""
    slugs = {g["slug"] for g in registry.get("groups") or []}
    worlds = {w["slug"]: World(**w) for w in registry.get("worlds") or []}
    return _parse_symbol(name, entry, slugs=slugs, worlds=worlds)


def validate_fragment(
    entries: Mapping[str, Mapping], *, known: Optional[set] = None,
) -> None:
    """Raise if `entries` would not load as registry symbols.

    Same constructor as `_load`. `known` defaults to the live registry
    (including any overlay); pass the target file's names when splicing
    into a text that is not this process's registry.
    """
    if not isinstance(entries, Mapping):
        raise ValueError(
            "dims.yaml fragment must be a mapping of symbol -> entry, got "
            f"{type(entries).__name__}"
        )
    _, groups, worlds = _load_base()
    slugs = {g.slug for g in groups}
    if known is None:
        known = set(_load_base()[0])
    for name, entry in entries.items():
        if name in known:
            raise ValueError(
                f"dims.yaml artifact re-declares existing symbol {name!r}. "
                f"Reuse it, or propose a differently named symbol."
            )
        _parse_symbol(name, entry, slugs=slugs, worlds=worlds)


def overlay_pending(entries: Optional[Mapping[str, Mapping]] = None) -> None:
    """Merge (or clear) proposal symbols into this process's registry.

    Validates with `_parse_symbol` before installing, so a bad fragment
    fails here instead of poisoning later `_load` calls. Does not write
    `dims.yaml`. Replaces any previous overlay.
    """
    _PENDING.clear()
    _reload_base()
    if entries is None:
        return
    try:
        validate_fragment(entries)
        for name, entry in entries.items():
            _PENDING[str(name)] = dict(entry)
        _reload_base()
        _load_base()
    except Exception:
        _PENDING.clear()
        _reload_base()
        raise


def _reload_base() -> None:
    _load_base.cache_clear()
    world_by_label.cache_clear()
    LOCAL.reset()


def symbols() -> Mapping[str, DimSymbol]:
    """Every registered symbol, in declaration order."""
    return _load()[0]


def groups() -> Tuple[DimGroup, ...]:
    """The documentation sections, in declaration order."""
    return _load()[1]


def call_token_axes() -> Mapping[str, Tuple[str, ...]]:
    """The symbols each `call_tokens` role names, by role, in name order."""
    out: Dict[str, Tuple[str, ...]] = {role.value: () for role in CallTokens if role.value}
    table = symbols()
    for name in sorted(table):
        role = table[name].call_tokens
        if role:
            out[role] += (name,)
    return out


def worlds() -> Mapping[str, World]:
    """Every declared rank group, in declaration order."""
    return _load()[2]


def capabilities_of(symbol: str) -> Tuple[World, ...]:
    """Rank groups allowed to split `symbol`; occurrence placement is separate."""
    sym = _load()[0].get(symbol)
    if sym is None:
        return ()
    all_worlds = _load()[2]
    return tuple(all_worlds[slug] for slug in sym.shardable_by)


def world_of(symbol: str) -> Optional[World]:
    """Single-capability projection: the world that owns this symbol.

    New endpoint code must call `capabilities_of` when a symbol may sit
    in more than one world. This projection fails loudly the first time
    a symbol gains two capabilities rather than picking one by
    declaration order.
    """
    capabilities = capabilities_of(symbol)
    if len(capabilities) > 1:
        raise ValueError(
            f"symbol {symbol!r} has multiple sharding capabilities "
            f"{[world.slug for world in capabilities]!r}; the legacy "
            "`world_of` projection cannot choose an actual layout"
        )
    return capabilities[0] if capabilities else None


def in_world(world: str) -> FrozenSet[str]:
    """Every symbol the named rank group splits.

    The inverse of `world_of`, and the form a KNOB wants: `divided_by` answers
    by MESH, which cannot separate two worlds that share one — `sequence` and
    `kv_shard` are both `cp` and run at different degrees, which is the whole
    point of a subgroup. Ask by world where the degrees differ.
    """
    return frozenset(
        name for name, sym in _load()[0].items() if world in sym.shardable_by
    )


def step_token_axes() -> FrozenSet[str]:
    """The symbols whose product is the tokens a step runs
    (`DimSymbol.step_token_axis`): the batch and the query length."""
    return frozenset(name for name, sym in _load()[0].items()
                     if sym.step_token_axis)


def query_axes() -> FrozenSet[str]:
    """The query length under each name it is bound by: the step-token
    axes that are not the batch (`s_ctx`, `s_query`, `n_verify`)."""
    return frozenset(name for name, sym in _load()[0].items()
                     if sym.step_token_axis and sym.kind != "batch")


def sequence_axes() -> FrozenSet[str]:
    """Every symbol that counts rows of the step's token/context streams.

    This is the axis identity a post-walk shape question such as “how many
    fresh KV rows did this producer land?” needs. It is a projection of the
    one symbol registry, not a per-consumer spelling of `s_ctx`/`s_kv`: a
    component may name its token axis locally, and a compressed stream may
    build one axis from several symbols, without each consumer growing its
    own vocabulary.
    """
    return frozenset(name for name, sym in _load()[0].items()
                     if sym.kind == "sequence"
                     or (sym.step_token_axis and sym.kind != "batch"))


def get(name: str) -> Optional[DimSymbol]:
    return _load()[0].get(name)


def written_derived_error(name: str, where: str,
                          local: Optional[Mapping[str, Any]] = None
                          ) -> Optional[str]:
    """Why `where` may not write `name` as a `dims:` key, or None: a
    `derived:` name is never written — every frame derives it from its
    sources, so the repair is to bind those."""
    sym = {**base_symbols(), **local_symbols(local or {})}.get(name)
    if sym is None or not sym.derived:
        return None
    sources = ", ".join(f"`{r}`" for r in sorted(sym.reads))
    return (f"{where}: `{name}` is derived (`{name} = {sym.derived}`) and a "
            f"spec never writes it. Bind {sources} instead; every frame "
            f"derives `{name}` from them (a call or a block assignment that "
            f"rebinds `n_experts` gets its own `{name}`).")


def refuse_written_derived(keys: Any, where: str,
                           local: Optional[Mapping[str, Any]] = None) -> None:
    """Raise, naming the repair, if any of `keys` is a `derived:` name."""
    problems = [p for k in (keys or ()) if (p := written_derived_error(
        str(k), where, local))]
    if problems:
        raise ValueError("; ".join(problems))


def is_registered(name: str) -> bool:
    """Whether the runtime table knows `name` (dims.yaml or a live spec's
    `dim_symbols`). A gate asks `registered_for(spec's dim_symbols)`."""
    return name in _load()[0]


def meanings() -> Dict[str, str]:
    """`{symbol: meaning}` — what the Dim Bindings tooltip shows. dims.yaml
    only: a spec's own `dim_symbols` are that spec's, not the glossary's."""
    return {name: sym.meaning for name, sym in _load_base()[0].items()}


def kinds() -> Mapping[str, str]:
    """`{symbol: kind}` for every registered symbol, `whole` included."""
    return {name: sym.kind for name, sym in _load()[0].items()}


def divisible_kinds() -> Mapping[str, str]:
    """`{symbol: kind}` for the symbols some rank group splits.

    `whole` is dropped because the evaluator's tables
    read membership as "a degree splits this", and a symbol classified whole
    answers that question with the same silence as an unregistered one — the
    difference between them is what the registry is for, and it belongs at
    the coverage gate rather than in the shard policy.
    """
    return {name: sym.kind for name, sym in _load()[0].items() if sym.shardable_by}


def divided_by(mesh: str, named: bool = True) -> FrozenSet[str]:
    """Every symbol living in a world on the named mesh (`tp`/`cp`/`ep`).

    `named` restricts the answer to worlds a `ParallelConfig` field can set,
    which is what a caller asking "does this deployment split it" wants: a
    world with no field runs at 1 for every deployment that exists, so an axis
    in it is whole by arithmetic. Pass `named=False` for the registry's own
    answer — whether such a split is EXPRESSIBLE — which is the question the
    coverage gate asks when a spec declares an axis sharded.
    """
    all_worlds = _load()[2]
    return frozenset(
        name
        for name, sym in _load()[0].items()
        if any(
            all_worlds[world].mesh == mesh and (all_worlds[world].degree or not named)
            for world in sym.shardable_by
        )
    )


def degree_fields(*meshes: str) -> Mapping[str, str]:
    """`{symbol: ParallelConfig field}` for worlds on the named meshes.

    The axis→knob map, derived rather than listed. Restricting it by mesh is
    the caller saying WHICH CUT it is asking about: the degrees that divide a
    weight's own axes (`tp`, `ep`) are not the ones that divide the token
    stream (`dp`, `cp`), and a caller that wants one of those questions
    answered must not be handed the other's degrees.
    """
    out: Dict[str, str] = {}
    for name in _load()[0]:
        world = world_of(name)
        if world is not None and world.mesh in meshes and world.degree:
            out[name] = world.degree
    return out


@memoize(maxsize=1)
def world_by_label() -> Mapping[str, World]:
    """`{World.label: World}` — the registry keyed the way the placement
    algebra names a world. The join, derived rather than written down a
    second time. Reads the base only: a spec's `dim_symbols` declare
    symbols, never worlds, so a layer change leaves it valid."""
    return {w.label: w for w in _load_base()[2].values()}


def world_labels(*meshes: str) -> Mapping[str, str]:
    """`{symbol: placement label}` for every world, or only those on the named
    meshes.

    What `degree_fields` is to a knob, this is to a placement: the same
    symbol→world map, under the name the state algebra uses. It keeps the
    worlds `degree_fields` drops, because a world with no field still runs at
    whatever `world_degrees` sets and a tensor split by it still has to say
    so.

    Naming no mesh asks for all of them, and that is what a caller building the
    deployment's symbol→world table wants: a mesh list is an allowlist, and an
    allowlist written out by hand drops whichever mesh is added after it. One
    was. `dp` went unlisted while `dp_attention` only ever appeared as a
    RESTING cut, and the day a mode re-partitioned onto it — `combine` returning
    expert results to their requests' ranks — the destination axis `B` had no
    world to be moved to, so `_worlds_of` left the tokens grouped by owner and
    the return half of the MoE exchange derived as nothing.
    """
    out: Dict[str, str] = {}
    for name in _load()[0]:
        world = world_of(name)
        if world is not None and (not meshes or world.mesh in meshes):
            out[name] = world.label
    return out


def render_markdown() -> str:
    """The registry as the glossary document agents and humans read.

    Rendered rather than stored: a second copy of 110 meanings is a second
    place to edit and a second place to be wrong, and this one is derived by
    construction.
    """
    by_group: Dict[str, list] = {}
    for sym in _load_base()[0].values():
        by_group.setdefault(sym.group, []).append(sym)

    out = [
        "# Dimension symbols glossary",
        "",
        "Generated from `backend/arch/dims.yaml` — the registry is "
        "the source, this is the reading copy. Each model's `dims:` block "
        "binds a subset of these symbols to integers via `config.<field>` "
        "references; what each symbol counts and which rank groups may "
        "split it live in the registry.",
        "",
        "## Rank groups",
        "",
        "A symbol's `shardable_by` lists groups allowed to split it; an "
        "endpoint or propagation chooses the actual layout. A group's "
        "degree is the deployment's, and it is 1 unless a deployment says "
        "otherwise. Every world divides a SHAPE — the tensor gets smaller "
        "along one axis — and the mesh says which axis: `tp` a weight's own, "
        "`ep` an expert bank's leading axis, `cp` the positions of one "
        "sequence, `dp` whole requests. Pipeline parallelism is no world: "
        "it partitions the layer index space unevenly, so no degree "
        "describes a stage.",
        "",
        "| World | Mesh | Degree | Meaning |",
        "|---|---|---|---|",
    ]
    out.extend(
        f"| `{world.slug}` | {world.mesh} "
        f"| {f'`{world.degree}`' if world.degree else '— (1)'} "
        f"| {world.meaning} |"
        for world in _load()[2].values()
    )
    out.append("")
    for group in _load()[1]:
        members = by_group.get(group.slug) or []
        if not members:
            continue
        out.append(f"## {group.title}")
        out.append("")
        if group.intro:
            out.append(group.intro)
            out.append("")
        out.append("| Symbol | Kind | Shardable by | Meaning | Binding |")
        out.append("|---|---|---|---|---|")
        for sym in members:
            binding = sym.binding_hint
            if sym.used_by:
                binding = f"{sym.used_by} · {binding}".strip(" ·")
            capabilities = ", ".join(sym.shardable_by) or "—"
            out.append(
                f"| `{sym.name}` | {sym.kind} | {capabilities} "
                f"| {sym.meaning} | {binding} |"
            )
        out.append("")
    return "\n".join(out)
