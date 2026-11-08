"""Ulysses / pcp / layersplit / DCP: the closed `(scheme, phase)` table and the geometry it implies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import sympy as sp

from backend.arch import dims
from backend.arch.axis_propagation import dim_map, signature_for
from backend.arch.eval.dim import (Dim, common_divisor, equal, expr_of,
                                   require_multiple, value_of)
from backend.arch.eval.transition import KernelGeometry
from backend.arch.eval.refusal import LayoutRefusal
from backend.arch.ops import OpSpec, ShapeKind, build_custom_op_spec, get_op

from .edge import EdgeState
from .groups import ProcessGroupFamily, orthogonal, same
from .layout import (
    AxisLayout,
    Block,
    Product,
    Replicated,
    block,
    physical_local_extent,
    product,
    sharded,
)
from .attention import (
    ATTENTION_APPLY_OPS,
    ATTENTION_SCORE_OPS,
    AttentionForm,
    indexer_heads,
)
from .resolver import layout_for_symbol

#: What the attention kernel does when sequence parallelism is live.
#: Keys are (plan name, phase). Prefill names are `CpScheme` values;
#: decode is always `"dcp"` — not a CpScheme. A missing row is unmodeled.
ATTENTION_PLANS = {
    ("ulysses", "prefill"): AttentionForm.HEAD_REPARTITION,
    ("pcp", "prefill"): AttentionForm.CONTEXT_GATHER,
    ("layersplit", "prefill"): AttentionForm.CONTEXT_GATHER,
    ("dcp", "decode"): AttentionForm.CACHE_SHARD,
}


class IndivisibleHeadRepartition(LayoutRefusal):
    """Ulysses needs the sequence degree to divide the query heads on one
    TP shard. The deployment asked for a degree that does not.

    `reason` is the stable code a page reads; `legal_cp_degrees` is the
    same interior check solved across every head-repartition attention
    node (query and gatherable K/V) — `legal_cp_degrees` over the model's
    walk, attached by the deployment that ran the walk (`solved`) — so a
    picker lists them without a second copy of the rule.
    """

    reason = "head_repartition_not_divisible"

    def __init__(self, where: str, what: str, heads_per_tp_shard: int,
                 degree: int,
                 legal_cp_degrees: Optional[Tuple[int, ...]] = None):
        self.where = where
        self.what = what
        self.heads_per_tp_shard = heads_per_tp_shard
        self.degree = degree
        self.legal_cp_degrees = legal_cp_degrees
        legal = ("" if legal_cp_degrees is None else
                 f" (legal CP degrees: "
                 f"{', '.join(str(d) for d in legal_cp_degrees)})")
        super().__init__(
            f"{where}: {what} carries {heads_per_tp_shard} heads "
            f"per tensor-parallel shard, which does not divide the "
            f"sequence degree {degree} — the ulysses head "
            f"repartition needs divisibility{legal}"
        )

    def solved(self, legal_cp_degrees: Tuple[int, ...]
               ) -> "IndivisibleHeadRepartition":
        """The same refusal, carrying the model's legal CP degrees."""
        return IndivisibleHeadRepartition(
            self.where, self.what, self.heads_per_tp_shard, self.degree,
            legal_cp_degrees)


@dataclass(frozen=True)
class HeadRepartitionKVRegion:
    """Whole-cache views and the row-wise producers behind them."""

    views: FrozenSet[int] = frozenset()
    producers: FrozenSet[int] = frozenset()


def degrees_dividing(extent: int) -> Tuple[int, ...]:
    """Positive integers that divide `extent`. Degree 1 always divides."""
    n = int(extent)
    if n <= 0:
        return (1,)
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _interior_accepts(per_shard: int, degree: int,
                      gather_if_too_few: bool) -> bool:
    """Whether `AttentionPlan.interior` would accept `degree` on this edge."""
    if degree <= 0:
        return False
    if per_shard % degree == 0:
        return True
    return gather_if_too_few and per_shard < degree


def _intersect_interior_degrees(
        constraints: Sequence[Tuple[int, bool]]) -> Tuple[int, ...]:
    """Degrees every (per_shard, gather_if_too_few) edge would accept."""
    finite: Optional[set] = None
    gather_caps: List[int] = []
    for per_shard, gather_if_too_few in constraints:
        shard = int(per_shard)
        if shard <= 0:
            continue
        if gather_if_too_few:
            gather_caps.append(shard)
            continue
        divs = set(degrees_dividing(shard))
        finite = divs if finite is None else finite & divs
    if finite is None:
        return (1,)
    for cap in gather_caps:
        finite = {d for d in finite if _interior_accepts(cap, d, True)}
    if not finite:
        return (1,)
    return tuple(sorted(finite))


def replicated_encoders(program: Any, parallel: Any, phase: str
                        ) -> FrozenSet[str]:
    """The encoder stacks whose query heads the attention TP degree does not
    divide: vLLM runs such a tower data parallel, every weight whole on
    every rank (`vision.is_vit_use_data_parallel(num_heads)`), and splits
    the others as declared. A stack's query heads are the `n_heads` its
    attention scores declare; KV heads fewer than the degree replicate as
    GQA does, and leave the stack split. A stack with no attention score
    keeps its declared placement.
    """
    degree = parallel.degree_of_world("tp_attention", phase=phase)
    # A stack is split or replicated as one: every head count its scores
    # declare must take the degree, or the whole stack runs replicated (a
    # pipeline's text encoder and the transformer's token refiner it feeds
    # are one stack with two counts).
    heads: Dict[str, List[Any]] = {}
    for uid in program.encoder_uids:
        node = program.by_uid(uid)
        if node.op not in ATTENTION_SCORE_OPS:
            continue
        stack = program.stack_of(node)
        count = node.bound("n_heads")
        if count is None:
            raise ValueError(
                f"{node.display_path}: an encoder attention score whose "
                "frame binds no `n_heads`")
        heads.setdefault(stack, []).append(count)
    return frozenset(stack for stack, counts in heads.items()
                     if any(not equal(common_divisor(count, degree), degree)
                            for count in counts))


def head_repartition_constraints(analysis: Any, parallel: Any
                                 ) -> Tuple[Tuple[int, bool], ...]:
    """`(per_shard, gather_if_too_few)` for every Ulysses interior edge.

    Read off a prefill walk of the model (`analysis`): each attention
    score/apply node that takes the head-repartition plan (not encoder
    stacks, not indexer scores, as the node's walked frame binds its head
    axes), the head extent each operand arrives with, and that axis's
    identity — the symbol the walked extent carries — placed on one
    tensor-parallel rank of `parallel`. Query edges require divisibility;
    K/V edges keep `gather_if_too_few`. Head extents are global and the
    same at any workload, so any prefill walk of the program answers.
    """
    program = analysis.program
    encoder = program.encoder_uids
    out: List[Tuple[int, bool]] = []
    for node in program.nodes():
        op = node.op
        if op not in ATTENTION_SCORE_OPS and op not in ATTENTION_APPLY_OPS:
            continue
        shape = analysis.shapes.values.get(node.uid)
        if node.uid in encoder or shape is None or indexer_heads(
                node, analysis.node_dims.values.get(node.uid) or {}):
            continue
        edges = (((0, False), (1, True)) if op in ATTENTION_SCORE_OPS
                 else ((1, True),))
        for index, gather in edges:
            out.append((_heads_on_one_tp_shard(
                shape.ins[index][1], parallel), gather))
    return tuple(out)


def _heads_on_one_tp_shard(head: Any, parallel: Any) -> int:
    """Heads one tensor-parallel rank holds of a walked head axis — the
    `per_shard` `AttentionPlan.interior` checks against the sequence
    degree. An axis no symbol names is placed by no world: whole."""
    extent = int(value_of(head))
    symbol = expr_of(head)
    if not isinstance(symbol, sp.Symbol):
        return extent
    layout = layout_for_symbol(symbol.name, extent, parallel, "prefill")
    if layout is None:
        return extent
    return physical_local_extent(layout, extent)


def query_heads_per_tp_shard(analysis: Any, parallel: Any) -> int:
    """Tightest query per-shard head count among head-repartition nodes."""
    query = [h for h, gather in head_repartition_constraints(
        analysis, parallel) if not gather]
    return min(query) if query else 0


def _declared_spec(node: Any) -> Optional[Any]:
    """The `OpSpec` a node runs: its declaration, built when it is the
    spec's own custom op (`SpecIRNode.declaration`), else the registry's."""
    declared = getattr(node, "declaration", None)
    if isinstance(declared, OpSpec):
        return declared
    if declared is not None:
        return build_custom_op_spec(node.op, declared)
    try:
        return get_op(node.op or "")
    except KeyError:
        return None


def _sequence_axis(spec: Any, node: Any,
                   input_index: int) -> Optional[Tuple[int, int]]:
    """The one attended-sequence axis of an attention operand.

    The op's index signature, not its display identity, says which input axis
    is which. A K or V operand has exactly one axis the dim registry classes
    as a sequence; batch, head-count, and feature axes are the other kinds.
    """
    signature = signature_for(spec, node, len(spec.shape_out or ()))
    if signature is None or input_index >= len(signature.ins):
        return None
    found = [
        axis for axis, label in enumerate(signature.ins[input_index])
        if _axis_kind(label) == "sequence"
    ]
    return (found[0], len(signature.ins[input_index])) if len(found) == 1 else None


def _head_axis(spec: Any, node: Any,
               input_index: int) -> Optional[Tuple[int, int, str]]:
    """The operand's head axis and the extent the node's frame binds it to
    (`SpecIRNode.bound`): this runs before the walk."""
    signature = signature_for(spec, node, len(spec.shape_out or ()))
    if signature is None or input_index >= len(signature.ins):
        return None
    found = [
        axis for axis, label in enumerate(signature.ins[input_index])
        if _axis_kind(label) == "head_count"
    ]
    if len(found) != 1:
        return None
    axis = found[0]
    label = signature.ins[input_index][axis]
    extent = node.bound(label)
    if extent is None:
        return None
    return axis, int(value_of(extent)), label


def _axis_kind(label: str) -> str:
    symbol = dims.get(label)
    return symbol.kind if symbol is not None else ""


def _role_input(spec: Any, role: str) -> Optional[int]:
    for index, declared in enumerate(spec.input_roles or ()):
        if (declared if isinstance(declared, str) else declared.key) == role:
            return index
    return None


def _is_whole_cache_view(spec: Any, node: Any) -> bool:
    """A marker that publishes every cached row, not a bounded selection."""
    return (
        spec.kv_stream_marker
        and spec.shape_kind is ShapeKind.AXIS_REPLACE
        and "read_topk" not in (node.dims or {})
        and "read_topk" not in (node.dims_raw or {})
    )


def _backward_sequence_sources(
        node: Any, spec: Any, output_axis: int, output_rank: int
) -> Tuple[Tuple[int, int, int], ...]:
    """Input axes that row-wise supply this output sequence axis.

    The admission is semantic, not a producer-name list: a view, a broadcast,
    a permutation, a one-operand feature rewrite, or a concat on another axis
    keeps each row local. A coupled sequence axis (softmax, recurrence,
    selection, contraction) has no backward row correspondence and stops.
    """
    signature = signature_for(spec, node, output_rank)
    if signature is not None:
        label = signature.out[output_axis]
        if not label or label in signature.concat or label in signature.selects:
            return ()
        use = (spec.axis_use or {}).get(label)
        if use is not None and use.category != "separable":
            return ()
        if label in (spec.couples_on or ()):
            return ()
        kind = spec.shape_kind
        rowwise = (
            spec.is_view_op
            or kind is ShapeKind.PRESERVE
            or kind is ShapeKind.BROADCAST
            or kind is ShapeKind.PERMUTE
            or (kind is ShapeKind.AXIS_REPLACE
                and len(node.inputs_obj or ()) == 1)
            or (kind is ShapeKind.AXIS_CONCAT
                and label not in signature.concat)
        )
        if not rowwise:
            return ()
        return tuple(
            (input_index, input_axis, len(signature.ins[input_index]))
            for input_index, input_axis in signature.positions(label)
        )

    # Explicit view declarations (`view_split`, `broadcast`) state their
    # correspondence in DimMap rather than in a signature. Infer the only
    # input rank whose declared output rank matches, then read that one map.
    if not spec.is_view_op or len(node.inputs_obj or ()) != 1:
        return ()
    candidates: List[Tuple[Tuple[int, int, int], ...]] = []
    for input_rank in range(1, output_rank + 1):
        mapped = dim_map(node.op, node, in_ranks=(input_rank,))
        if mapped is None or len(mapped) != output_rank:
            continue
        source = mapped[output_axis]
        if source.kind == "from" and source.refs:
            candidates.append(tuple(
                (index, axis, input_rank)
                for index, axis in source.refs
            ))
    if len(candidates) != 1:
        return ()
    return candidates[0]


def _trace_kv_cache_views(
        program: Any, node: Any, port: str, output_axis: int,
        output_rank: int, seen: FrozenSet[Tuple[int, str, int]],
) -> Tuple[FrozenSet[int], FrozenSet[int], bool]:
    key = (node.uid, port, output_axis)
    if key in seen:
        return frozenset(), frozenset(), False
    seen = seen | {key}
    if node.children:
        leaf, leaf_port = program.output_leaf(node, port)
        return _trace_kv_cache_views(
            program, leaf, leaf_port, output_axis, output_rank, seen)

    spec = _declared_spec(node)
    if spec is None:
        return frozenset(), frozenset(), False
    if _is_whole_cache_view(spec, node):
        return frozenset((node.uid,)), frozenset((node.uid,)), False
    if (spec.kv_stream_marker
            and ("read_topk" in (node.dims or {})
                 or "read_topk" in (node.dims_raw or {}))):
        return frozenset(), frozenset(), True

    views: set = set()
    producers: set = {node.uid}
    sparse = False
    for input_index, input_axis, input_rank in _backward_sequence_sources(
            node, spec, output_axis, output_rank):
        refs = node.inputs_obj or ()
        if input_index >= len(refs) or refs[input_index].src is None:
            continue
        ref = refs[input_index]
        found, found_producers, found_sparse = _trace_kv_cache_views(
            program, ref.src, ref.port or "out", input_axis,
            input_rank, seen)
        # A producer belongs to the invocation only on a branch that reaches
        # a whole-cache view.  Branches that stop without one (a GQA cached
        # producer reached directly from attention, or its residual norm)
        # describe persistent SSA dataflow, not this invocation's row cut.
        if found:
            views.update(found)
            producers.update(found_producers)
        sparse = sparse or found_sparse
    return frozenset(views), frozenset(producers), sparse


def head_repartition_kv_cache_region(program: Any, parallel: Any
                                     ) -> HeadRepartitionKVRegion:
    """Whole-cache views whose invocation rows Ulysses can compute locally.

    The walk starts at each main attention's declared K and V operands and
    follows actual dataflow backwards only through row-wise producers. It
    does not classify producers by module name, and it excludes encoders,
    DSA indexer scores, sparse reads, and K/V heads too few to repartition.
    """
    degree = parallel.degree_of_world("cp", phase="prefill")
    if degree <= 1:
        return HeadRepartitionKVRegion()
    out: set = set()
    producer_region: set = set()
    for node in program.nodes():
        spec = _declared_spec(node)
        if spec is None or spec.attention_kernel is None:
            continue
        if (node.uid in program.encoder_uids
                or indexer_heads(node, node.dims or {})):
            continue
        role = "k" if spec.attention_kernel == "score" else "v"
        input_index = _role_input(spec, role)
        if input_index is None:
            continue
        sequence_axis = _sequence_axis(spec, node, input_index)
        head = _head_axis(spec, node, input_index)
        if sequence_axis is None or head is None:
            continue
        _axis, extent, label = head
        # Reuse the one head-placement law also used after a walk.  The
        # backward dataflow below runs while the rule is being constructed,
        # so the head extent comes from the graph's resolved frame; no local
        # head extent is derived here a second time.
        #
        # Both operands of the divisibility below are numbers at every
        # workload: a head count is architectural, and `_head_axis` already
        # read it as one (`int(value_of(...))`, which refuses an expression
        # rather than deciding), while `degree` is the world's configured
        # size.  So this one is decided here rather than carried as a
        # condition on the point, and `tests/meta/test_symbolic_decisions.py`
        # holds that exemption with the same reason.
        per_shard = _heads_on_one_tp_shard(Dim(label, extent), parallel)
        if not _interior_accepts(per_shard, degree, gather_if_too_few=False):
            continue
        refs = node.inputs_obj or ()
        if input_index >= len(refs) or refs[input_index].src is None:
            continue
        ref = refs[input_index]
        views, producers, sparse = _trace_kv_cache_views(
            program, ref.src, ref.port or "out", sequence_axis[0],
            sequence_axis[1], frozenset())
        if views and not sparse:
            out.update(views)
            producer_region.update(producers)
    return HeadRepartitionKVRegion(frozenset(out),
                                   frozenset(producer_region))


def legal_cp_degrees(analysis: Any, parallel: Any, *,
                     scheme: Optional[str] = None
                     ) -> Optional[Tuple[int, ...]]:
    """Legal prefill CP degrees under `scheme` for the walked model.

    Ulysses interior is solved across every head-repartition attention
    node: the intersection of the query cut and each gatherable K/V edge.
    Schemes that gather context do not impose that cut (`None`).
    """
    name = str(scheme if scheme is not None else parallel.cp_scheme)
    form = ATTENTION_PLANS.get((name, "prefill"))
    if form is not AttentionForm.HEAD_REPARTITION:
        return None
    constraints = head_repartition_constraints(analysis, parallel)
    if not constraints:
        return (1,)
    return _intersect_interior_degrees(constraints)


def added_cut(before: AxisLayout, after: AxisLayout) -> Optional[Block]:
    """`after` == `before` with exactly one more inner cut, or None."""
    if isinstance(before, Replicated) and isinstance(after, Block) \
            and after.replication_degree == 1 and after.pad_to is None:
        return after
    if isinstance(before, Block) and isinstance(after, Product) \
            and len(after.cuts) == 2 and after.cuts[0] == before:
        return after.cuts[1]
    return None


def cut_swap(a: Sequence[AxisLayout], b: Sequence[AxisLayout]
             ) -> Optional[Tuple[int, Block, int, Block]]:
    """A cut-migration PATTERN: one axis gains a cut, exactly one other
    loses one, everything else equal. Returns
    (gained_axis, gained, lost_axis, lost)."""
    gained_axis = gained = lost_axis = lost = None
    for axis, (x, y) in enumerate(zip(a, b, strict=True)):
        if x == y:
            continue
        added = added_cut(x, y)
        removed = added_cut(y, x)
        if added is not None and gained is None:
            gained_axis = axis
            gained = added
        elif removed is not None and lost is None:
            lost_axis = axis
            lost = removed
        else:
            return None
    if gained is None or lost is None:
        return None
    return gained_axis, gained, lost_axis, lost


@dataclass(frozen=True)
class AttentionPlan:
    """Live (scheme, phase) form plus the families it speaks.

    None `form` means no sequence world above degree one — score/apply
    run without interior/exit exchanges.
    """

    name: Optional[str]
    form: Optional[AttentionForm]
    phase: str
    universe: frozenset
    resident: Optional[ProcessGroupFamily]
    cache: Optional[ProcessGroupFamily]
    attention_tp: Optional[ProcessGroupFamily]
    parallel: Any
    families: Mapping[str, Any]

    @classmethod
    def lookup(cls, parallel: Any, phase: str, families: Mapping[str, Any],
               universe: frozenset, resident: Optional[ProcessGroupFamily],
               cache: Optional[ProcessGroupFamily]) -> "AttentionPlan":
        name = None
        form = None
        live_label = "cp" if phase == "prefill" else "dcp"
        live_degree = parallel.degree_of_world(live_label, phase=phase)
        if live_degree > 1:
            name = (str(parallel.cp_scheme) if phase == "prefill" else "dcp")
            form = ATTENTION_PLANS.get((name, phase))
            if form is None:
                raise NotImplementedError(
                    f"attention execution plan {name!r} has no "
                    f"({name!r}, {phase!r}) row — the plan table is closed; "
                    "declare the scheme's row before running this "
                    "deployment")
        return cls(
            name=name, form=form, phase=phase, universe=universe,
            resident=resident, cache=cache,
            attention_tp=families.get("tp_attention"),
            parallel=parallel, families=families,
        )

    @classmethod
    def from_rule(cls, rule: Any) -> "AttentionPlan":
        parallel = rule.parallel
        if parallel is None:
            return cls(
                name=None, form=None, phase=rule.phase,
                universe=rule.universe,
                resident=rule.resident_sequence_family,
                cache=rule.cache_family,
                attention_tp=(rule.families or {}).get("tp_attention"),
                parallel=None, families=rule.families or {},
            )
        return cls.lookup(
            parallel, rule.phase, rule.families or {}, rule.universe,
            rule.resident_sequence_family, rule.cache_family)

    def replicated(self) -> Replicated:
        return Replicated(scope=self.universe)

    def pairing_whole(self, node: Any, axis: AxisLayout, what: str) -> None:
        """An attention pairing axis — whole, or owned by the active plan."""
        if isinstance(axis, Replicated):
            return
        if (self.form is not None
                and self.resident is not None
                and isinstance(axis, Block)
                and axis.is_sharded_on(self.resident)):
            return
        raise NotImplementedError(
            f"TODO: {node.display_path}: sharded {what} is the "
            "attention-kernel plan's business (online/pcp/ulysses) and "
            "no active plan row owns this cut")

    def interior(self, node: Any, state: Any, head_extent: int,
                 what: str, *, gather_if_too_few: bool = False,
                 ) -> Any:
        """Ulysses interior: head axis gains the sequence family as an inner cut."""
        family = self.resident
        head, sequence = state.layout[1], state.layout[2]
        per_shard = physical_local_extent(head, head_extent)
        if per_shard % family.group_size:
            if gather_if_too_few and per_shard < family.group_size:
                # Whole already when the producer published it so — a
                # cached K/V leaves its leaf gathered on this family —
                # else cut on this family and gathered here.
                resident = (
                    isinstance(sequence, Block)
                    and sequence.is_sharded_on(family)
                )
                if not resident and not isinstance(sequence, Replicated):
                    raise ValueError(
                        f"{node.display_path}: {what} cannot gather a "
                        f"non-resident sequence layout {sequence!r}")
                layout = list(state.layout)
                layout[2] = self.replicated()
                return EdgeState(
                    tuple(layout), state.value, state.workload,
                    state.provenance,
                )
            raise IndivisibleHeadRepartition(
                node.display_path, what, per_shard, family.group_size)
        if isinstance(head, Replicated):
            joint: AxisLayout = sharded(family)
        elif isinstance(head, Block) and head.pad_to is None:
            joint = product((head, sharded(family)))
        else:
            raise ValueError(
                f"{node.display_path}: {what} head layout {head!r} cannot "
                "take the sequence repartition")
        resident = (isinstance(sequence, Block)
                    and sequence.is_sharded_on(family))
        if not resident and not isinstance(sequence, Replicated):
            raise NotImplementedError(
                f"TODO: {node.display_path}: {what} sequence cut "
                f"{sequence!r} is not the resident sequence family")
        return EdgeState(
            (state.layout[0], joint, self.replicated(), state.layout[3]),
            state.value, state.workload, state.provenance)

    def gather_resident_axis(self, node: Any, state: Any,
                             axis: int, what: str) -> Any:
        sequence = state.layout[axis]
        if isinstance(sequence, Replicated):
            return state
        family = self.resident
        if not (isinstance(sequence, Block)
                and sequence.is_sharded_on(family)):
            raise ValueError(
                f"{node.display_path}: {what} expected the resident "
                f"sequence cut, got {sequence!r}")
        layout = list(state.layout)
        layout[axis] = self.replicated()
        return EdgeState(
            tuple(layout), state.value, state.workload, state.provenance
        )

    def gather_context(self, node: Any, state: Any, what: str) -> Any:
        return self.gather_resident_axis(node, state, 2, what)

    def exit(self, node: Any, interior_head: AxisLayout
             ) -> Tuple[AxisLayout, AxisLayout]:
        """Strip the sequence cut off the interior head; hand it back to rows."""
        family = self.resident
        if isinstance(interior_head, Product) \
                and len(interior_head.cuts) == 2 \
                and same(interior_head.cuts[1].group, family):
            return interior_head.cuts[0], sharded(family)
        if isinstance(interior_head, Block) \
                and same(interior_head.group, family):
            return self.replicated(), sharded(family)
        raise ValueError(
            f"{node.display_path}: attention interior head layout "
            f"{interior_head!r} does not carry the sequence repartition")

    def kv_side_context(self, node: Any, state: Any,
                        in_shapes: Sequence[Any], what: str) -> AxisLayout:
        context = state.layout[2]
        if self.form == AttentionForm.CACHE_SHARD:
            if not isinstance(context, Replicated):
                raise ValueError(
                    f"{node.display_path}: the decode K/V leaf arrived cut "
                    f"({context!r}); the new-token write is per-rank whole")
            require_multiple(
                in_shapes[1][2], self.cache.group_size,
                lambda extent: ValueError(
                    f"{node.display_path}: cache length {extent} does not "
                    f"divide the dcp degree {self.cache.group_size}"))
            return sharded(self.cache)
        if isinstance(context, Replicated):
            return context
        raise NotImplementedError(
            f"TODO: {node.display_path}: {what} cut {context!r} has no "
            "active plan row")

    def cache_read_state(self, node: Any, state: Any,
                         in_shapes: Sequence[Any], what: str) -> Any:
        context = self.kv_side_context(node, state, in_shapes, what)
        layout = list(state.layout)
        layout[2] = context
        return EdgeState(
            tuple(layout), state.value, state.workload, state.provenance
        )

    def kernel_local_axes(
        self, node: Any, shape: Sequence[Any], state: Any, what: str,
        project,
    ) -> Tuple[Any, ...]:
        if len(shape) != len(state.layout):
            raise ValueError(
                f"{node.display_path}: {what} shape rank {len(shape)} "
                f"does not match state rank {len(state.layout)}"
            )
        out = []
        for extent, layout in zip(shape, state.layout, strict=True):
            local = project(extent, layout)
            out.append(
                Dim(expr_of(extent), local)
                if isinstance(extent, Dim) else local
            )
        return tuple(out)

    @staticmethod
    def expanded_axis(axes: Sequence[Any], axis: int, factor: int
                      ) -> Tuple[Any, ...]:
        out = list(axes)
        extent = out[axis]
        value = int(value_of(extent)) * factor
        out[axis] = (
            Dim(expr_of(extent), value)
            if isinstance(extent, Dim) else value
        )
        return tuple(out)

    def dcp_head_buffer_expands(self) -> bool:
        attention = self.attention_tp
        return attention is None or orthogonal(self.cache, attention)

    def dcp_score_geometry(
        self, node: Any, in_shapes: Sequence[Any],
        query: Any, key: Any, gathered_head: AxisLayout,
        key_head_shared: bool, project,
    ) -> KernelGeometry:
        query_axes = self.kernel_local_axes(
            node, in_shapes[0], query, "query kernel input", project
        )
        if self.dcp_head_buffer_expands():
            query_axes = self.expanded_axis(
                query_axes, 1, self.cache.group_size
            )
        else:
            query_axes = list(query_axes)
            head_extent = in_shapes[0][1]
            local_head = project(head_extent, gathered_head)
            query_axes[1] = (
                Dim(expr_of(head_extent), local_head)
                if isinstance(head_extent, Dim) else local_head
            )
            query_axes = tuple(query_axes)
        key_axes = self.kernel_local_axes(
            node, in_shapes[1], key, "key cache input", project
        )
        if key_head_shared:
            key_axes = list(key_axes)
            key_axes[1] = query_axes[1]
            key_axes = tuple(key_axes)
        output = (query_axes[0], query_axes[1], query_axes[2], key_axes[2])
        return KernelGeometry((query_axes, key_axes), output)

    def dcp_apply_geometry(
        self, node: Any, in_shapes: Sequence[Any],
        scores: Any, value: Any, value_head_shared: bool, project,
    ) -> KernelGeometry:
        score_axes = self.kernel_local_axes(
            node, in_shapes[0], scores, "score kernel input", project
        )
        if self.dcp_head_buffer_expands():
            score_axes = self.expanded_axis(
                score_axes, 1, self.cache.group_size
            )
        value_axes = self.kernel_local_axes(
            node, in_shapes[1], value, "value cache input", project
        )
        if value_head_shared:
            value_axes = list(value_axes)
            value_axes[1] = score_axes[1]
            value_axes = tuple(value_axes)
        output = (
            score_axes[0], score_axes[1], score_axes[2], value_axes[3]
        )
        return KernelGeometry((score_axes, value_axes), output)

    def dcp_gathered_head(
            self, node: Any, query: AxisLayout, kv: AxisLayout,
            ) -> AxisLayout:
        attention = self.attention_tp
        if attention is None or orthogonal(self.cache, attention):
            return query

        if isinstance(kv, Block):
            for group in self.cache.groups:
                shards = {kv.shard_of_rank(rank) for rank in group}
                if len(shards) != 1:
                    raise ValueError(
                        f"{node.display_path}: dcp group {group!r} crosses "
                        "KV head shards; decode context parallelism may only "
                        "split TP replica cells left by KV-head placement")
        elif not isinstance(kv, Replicated):
            raise ValueError(
                f"{node.display_path}: DCP requires a replicated or "
                f"replica-cell KV head axis, got {kv!r}")

        if isinstance(query, Replicated):
            return query
        if not isinstance(query, Block) or query.pad_to is not None:
            raise ValueError(
                f"{node.display_path}: DCP cannot gather query head layout "
                f"{query!r}")

        gathered = {
            rank: frozenset(
                query.shard_of_rank(member)
                for member in self.cache.group_of(rank)
            )
            for rank in query.group.ranks()
        }
        parts = sorted(set(gathered.values()), key=min)
        flat = [shard for part in parts for shard in sorted(part)]
        if flat != list(range(query.num_shards)) or any(
                tuple(sorted(part)) != tuple(
                    range(min(part), max(part) + 1)
                ) for part in parts):
            raise ValueError(
                f"{node.display_path}: DCP query gather produces a "
                "non-contiguous head owner map")

        patterns = [
            tuple(
                tuple(index for index, rank in enumerate(group)
                      if gathered[rank] == part)
                for part in parts
            )
            for group in query.group.groups
        ]
        if any(pattern != patterns[0] for pattern in patterns[1:]) \
                or any(not cell for cell in patterns[0]):
            raise ValueError(
                f"{node.display_path}: DCP head replication is not uniform "
                "across tensor-parallel groups")
        return block(query.group, patterns[0])

    def declared_query_head(self, extent: int) -> AxisLayout:
        layout = layout_for_symbol(
            "n_heads", extent, self.parallel, self.phase,
            dict(self.families),
        )
        return layout if layout is not None else self.replicated()

    def is_ulysses_entry(
            self, gained: Block, lost: Block,
            gained_axis: int, lost_axis: int,
            node: Any, input_index: Optional[int]) -> bool:
        return (
            self.form == AttentionForm.HEAD_REPARTITION
            and self.resident is not None
            and same(gained.group, self.resident)
            and (gained_axis, lost_axis) == (1, 2)
            and gained == lost
            and node is not None
            and ((node.op in ATTENTION_SCORE_OPS and input_index in (0, 1))
                 or (node.op in ATTENTION_APPLY_OPS and input_index == 1)))

    def ulysses_exit(
            self, required: Sequence[Any], produced: Any, node: Any,
            ) -> List[Tuple[str, Any, Tuple[Tuple[str, str], ...]]]:
        if self.form != AttentionForm.HEAD_REPARTITION \
                or node is None or node.op not in ATTENTION_APPLY_OPS:
            return []
        if not required or required[0] is None or produced is None:
            return []
        entered = required[0]
        if len(entered.layout) != len(produced.layout):
            return []
        swap = cut_swap(entered.layout, produced.layout)
        if swap is None:
            return []
        gained_axis, gained, lost_axis, lost = swap
        if gained != lost or self.resident is None \
                or not same(gained.group, self.resident) \
                or (gained_axis, lost_axis) != (2, 1):
            return []
        return [("out", entered, ((gained.group.id, "all_to_all"),))]
