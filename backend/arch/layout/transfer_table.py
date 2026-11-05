"""The exhaustive geometry transfer table for static axis layouts.

The table answers one question: given an axis relation already derived by the
shape system, what output-axis layouts follow from the input layouts?  A cell
either derives those layouts (and, only for contraction, a pending reduction)
or explains why this rewrite is not representable.  It never prescribes
communication; redistribution is the propagator's decision after a rejection.

There are exactly 42 keys.  Five unary rows have four constructor cells each,
MERGE has all sixteen ordered constructor pairs, CONCAT is constructor-agnostic,
and BROADCAST has four counterpart cells plus the no-counterpart case.  Unknown
is bottom and is rejected before lookup.  Relation arguments are existing shape
facts (row-major split extents, slice interval, repeat factor), not declarations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import sympy as sp

from backend.arch.eval.dim import require, require_equal, same_law
from backend.arch.eval.refusal import LayoutRefusal

from .groups import cross_replica_family, orthogonal, same
from .layout import (
    AxisLayout,
    Block,
    Product,
    Replicated,
    Segmented,
    Unknown,
    _is_int,
    block,
    physical_local_extent,
    product,
    replicated,
    segmented,
    whole_interval,
    slice_axis,
)
from .value import PENDING_MERGE_KINDS, Partial, join_orthogonal_partials


class Relation(str, Enum):
    PRESERVE = "preserve"
    CONTRACT = "contract"
    SPLIT = "split"
    MERGE = "merge"
    CONCAT = "concat"
    SLICE = "slice"
    BROADCAST = "broadcast"
    REPEAT = "repeat"


PRESERVE = Relation.PRESERVE
CONTRACT = Relation.CONTRACT
SPLIT = Relation.SPLIT
MERGE = Relation.MERGE
CONCAT = Relation.CONCAT
SLICE = Relation.SLICE
BROADCAST = Relation.BROADCAST
REPEAT = Relation.REPEAT


class Arity(str, Enum):
    ZERO = "zero"
    ONE = "one"
    FROM_MAP = "from_map"


NO_PENDING = False
PENDING_ALLOWED = True

#: Per-row contract for successful cells: output arity and whether the value
#: side may carry a pending reduction.  Rejected cells have no output contract.
ROW_CONTRACT = {
    PRESERVE: (Arity.ONE, NO_PENDING),
    CONTRACT: (Arity.ZERO, PENDING_ALLOWED),
    SPLIT: (Arity.FROM_MAP, NO_PENDING),
    MERGE: (Arity.ONE, NO_PENDING),
    CONCAT: (Arity.ONE, NO_PENDING),
    SLICE: (Arity.ONE, NO_PENDING),
    BROADCAST: (Arity.ONE, NO_PENDING),
    REPEAT: (Arity.ONE, NO_PENDING),
}

# 不可折叠：Flattened 故意不进表。它的 index map 不是 SPLIT/MERGE
# 格子能表达的几何；给六个构造器统一加 contract/split 会把一张缺格表
# 假装成全总对象代数。
LAYOUT_CONSTRUCTORS = (Replicated, Block, Segmented, Product)


@dataclass(frozen=True)
class Derived:
    """Layouts derived by a cell, plus contraction's optional value debt."""

    out_axes: Tuple[AxisLayout, ...]
    pending_reduction: Optional[Partial] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "out_axes", tuple(self.out_axes))
        for axis in self.out_axes:
            if isinstance(axis, Unknown):
                raise ValueError("a transfer cell may not derive Unknown")
            if not isinstance(axis, LAYOUT_CONSTRUCTORS):
                raise TypeError(f"not an AxisLayout: {axis!r}")
        if self.pending_reduction is not None and not isinstance(
            self.pending_reduction, Partial
        ):
            raise TypeError("pending_reduction must be a Partial or None")

    def require(self, what: str, where: str) -> "Derived":
        del what, where
        return self


@dataclass(frozen=True)
class Rejected:
    """This geometry rewrite is not representable for the input layout."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("Rejected requires a useful reason")

    def require(self, what: str, where: str) -> Derived:
        raise LayoutRefusal(f"{where}: {what}: {self.reason}")


CellResult = Union[Derived, Rejected]
Cell = Callable[[Tuple[AxisLayout, ...], Mapping[str, Any]], CellResult]


def validate_cell_result(
    relation: Relation,
    result: CellResult,
    *,
    expected_outputs: Optional[int] = None,
) -> CellResult:
    """Enforce the row contract after every lookup; violations are code bugs."""
    try:
        relation = Relation(relation)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown axis relation {relation!r}") from exc
    if isinstance(result, Rejected):
        return result
    if not isinstance(result, Derived):
        raise TypeError(f"transfer cell returned foreign result {result!r}")

    arity, pending_allowed = ROW_CONTRACT[relation]
    wanted = {
        Arity.ZERO: 0,
        Arity.ONE: 1,
        Arity.FROM_MAP: expected_outputs,
    }[arity]
    if wanted is None:
        raise ValueError(f"{relation.value} contract requires expected_outputs")
    if len(result.out_axes) != wanted:
        raise ValueError(
            f"{relation.value} cell violated output arity: "
            f"expected {wanted}, got {len(result.out_axes)}"
        )
    if result.pending_reduction is not None and not pending_allowed:
        raise ValueError(
            f"{relation.value} cell returned a pending reduction; "
            "only contract may create one"
        )
    return result


def _scope_of(layout: AxisLayout) -> frozenset:
    if isinstance(layout, Replicated):
        return layout.scope
    if isinstance(layout, Block):
        return layout.group.ranks()
    if isinstance(layout, Product):
        return layout.cuts[0].group.ranks()
    if isinstance(layout, Segmented):
        scopes = {_scope_of(child) for _, _, child in layout.segments}
        if len(scopes) != 1:
            raise ValueError("one axis cannot mix different rank universes")
        return next(iter(scopes))
    if isinstance(layout, Unknown):
        raise ValueError(
            f"Unknown layout reached transfer: {layout.reason or 'no reason recorded'}"
        )
    raise TypeError(layout)


def _require_args(
    relation: Relation, axes: Tuple[AxisLayout, ...], args: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = dict(args)
    if relation is BROADCAST:
        expected = {"scope"} if not axes else set()
    else:
        expected = {
            PRESERVE: set(),
            CONTRACT: {"reducer"},
            SPLIT: {"extent", "output_extents"},
            MERGE: set(),
            CONCAT: {"extents"},
            SLICE: {"start", "stop", "extent"},
            REPEAT: {"factor"},
        }[relation]
    if set(args) != expected:
        raise ValueError(
            f"{relation.value} expects arguments {sorted(expected)}, got {sorted(args)}"
        )

    if relation is CONTRACT and args["reducer"] not in PENDING_MERGE_KINDS:
        raise ValueError(f"unknown reducer {args['reducer']!r}")
    if relation is SPLIT:
        extent = args["extent"]
        try:
            outputs = tuple(args["output_extents"])
        except TypeError as exc:
            raise ValueError("split output_extents must be an iterable") from exc
        # Zero is a legal extent: an empty axis (DSv4's complete-window
        # count at a decode step shorter than one window) splits into an
        # empty factor. `transfer` answers every empty range the same way.
        if not _is_int(extent):
            raise ValueError("split extent must be a non-negative integer")
        require(sp.Ge(extent, 0),
               lambda at: ValueError(
                   f"split extent must be a non-negative integer, got "
                   f"{at(extent)}"))
        if not outputs or any(not _is_int(x) for x in outputs):
            raise ValueError(
                "split output_extents must be non-negative integers")
        for one in outputs:
            require(sp.Ge(one, 0),
                   lambda at, one=one: ValueError(
                       f"split output_extents must be non-negative "
                       f"integers, got {at(one)}"))
        require_equal(
            prod(outputs), extent,
            lambda at: ValueError(
                f"split index map tiles {at(prod(outputs))} elements, not "
                f"{at(extent)}"))
        normalized["output_extents"] = outputs
    if relation is CONCAT:
        try:
            extents = tuple(args["extents"])
        except TypeError as exc:
            raise ValueError("concat extents must be an iterable") from exc
        if len(extents) != len(axes):
            raise ValueError("concat needs one extent per input axis")
        if any(not _is_int(x) for x in extents):
            raise ValueError("concat extents must be positive integers")
        for one in extents:
            require(sp.Gt(one, 0),
                   lambda at, one=one: ValueError(
                       f"concat extents must be positive integers, got "
                       f"{at(one)}"))
        normalized["extents"] = extents
    if relation is SLICE:
        start, stop, extent = args["start"], args["stop"], args["extent"]
        if not all(_is_int(x) for x in (start, stop, extent)):
            raise ValueError("slice interval values must be integers")
        require(
            sp.And(sp.Le(0, start), sp.Le(start, stop), sp.Le(stop, extent)),
            lambda at: ValueError(
                f"bad interval [{at(start)}, {at(stop)}) of {at(extent)}"))
    if relation is REPEAT:
        factor = args["factor"]
        if not _is_int(factor) or factor <= 0:
            raise ValueError("repeat factor must be a positive integer")
    return normalized


def _validate_inputs(relation: Relation, axes: Tuple[AxisLayout, ...]) -> None:
    expected = {
        PRESERVE: 1,
        CONTRACT: 1,
        SPLIT: 1,
        MERGE: 2,
        SLICE: 1,
        REPEAT: 1,
    }
    if relation in expected and len(axes) != expected[relation]:
        raise ValueError(
            f"{relation.value} expects {expected[relation]} input axis/axes, "
            f"got {len(axes)}"
        )
    if relation is CONCAT and not axes:
        raise ValueError("concat requires at least one input axis")
    if relation is BROADCAST and len(axes) > 1:
        raise ValueError("broadcast accepts zero or one counterpart axis")
    for axis in axes:
        if isinstance(axis, Unknown):
            _scope_of(axis)
        if not isinstance(axis, LAYOUT_CONSTRUCTORS):
            raise TypeError(f"not an AxisLayout: {axis!r}")
        _scope_of(axis)


def _validate_input_extents(
    relation: Relation, axes: Tuple[AxisLayout, ...], args: Mapping[str, Any]
) -> None:
    if relation is SPLIT:
        extents = (args["extent"],)
    elif relation is CONCAT:
        extents = args["extents"]
    elif relation is SLICE:
        extents = (args["extent"],)
    else:
        return
    for axis, extent in zip(axes, extents, strict=True):
        physical_local_extent(axis, extent)


def _preserve(axes, _args) -> CellResult:
    return Derived((axes[0],))


def _contract_replicated(_axes, _args) -> CellResult:
    return Derived(())


def _partial_for_block(layout: Block, reducer: str) -> Optional[Partial]:
    if layout.num_shards == 1:
        return None
    group = (
        layout.group
        if layout.replication_degree == 1
        else cross_replica_family(layout.group, layout.replica_cells)
    )
    return Partial(reducer=reducer, group=group)


def _contract_block(axes, args) -> CellResult:
    return Derived((), _partial_for_block(axes[0], args["reducer"]))


def _contract_segmented(axes, args) -> CellResult:
    layout = axes[0]
    children = tuple(child for _, _, child in layout.segments)
    first = children[0]
    if isinstance(first, Replicated) and all(child == first for child in children):
        return Derived(())
    if isinstance(first, Block) and all(
        isinstance(child, Block)
        and child.group == first.group
        and child.replica_cells == first.replica_cells
        for child in children
    ):
        return Derived((), _partial_for_block(first, args["reducer"]))
    return Rejected(
        "segmented contraction needs every segment to have the same complete "
        "group and replica-cell pattern; otherwise one Partial cannot name "
        "all required reductions"
    )


def _contract_product(axes, args) -> CellResult:
    cuts = axes[0].cuts
    pending = _partial_for_block(cuts[0], args["reducer"])
    assert pending is not None
    for cut in cuts[1:]:
        other = _partial_for_block(cut, args["reducer"])
        assert other is not None
        try:
            pending = join_orthogonal_partials(pending, other)
        except ValueError as exc:
            return Rejected(
                "product contraction cannot combine its pending groups into "
                f"one reduction: {exc}"
            )
    return Derived((), pending)


def _split_cuts(
    cuts: Tuple[Block, ...], output_extents: Tuple[int, ...], scope: frozenset
) -> CellResult:
    residual = list(output_extents)
    assigned = [[] for _ in output_extents]
    for cut in cuts:
        if cut.pad_to is not None:
            return Rejected(
                "a padded Block cannot cross a split index map; padding must "
                "be resolved before reshaping"
            )
        target = next((i for i, extent in enumerate(residual) if extent != 1), None)
        if target is None or residual[target] % cut.num_shards:
            return Rejected(
                f"{cut.num_shards} shards do not divide the outermost remaining "
                "output factor; skipping it would change the row-major owner map"
            )
        assigned[target].append(cut)
        residual[target] //= cut.num_shards

    out = [replicated(scope) if not bucket else product(bucket) for bucket in assigned]
    return Derived(tuple(out))


def _split_replicated(axes, args) -> CellResult:
    outputs = tuple(args["output_extents"])
    return Derived(tuple(replicated(axes[0].scope) for _ in outputs))


def _split_block(axes, args) -> CellResult:
    layout = axes[0]
    return _split_cuts((layout,), tuple(args["output_extents"]), layout.group.ranks())


def _split_segmented(_axes, _args) -> CellResult:
    return Rejected(
        "Segmented may only be separated at declared segment boundaries; "
        "that operation is slice, not an axis-factor split"
    )


def _split_product(axes, args) -> CellResult:
    layout = axes[0]
    return _split_cuts(
        layout.cuts,
        tuple(args["output_extents"]),
        layout.cuts[0].group.ranks(),
    )


def _scope_matches(left: AxisLayout, right: AxisLayout) -> bool:
    return _scope_of(left) == _scope_of(right)


def _merge_replicated(axes, _args) -> CellResult:
    if not _scope_matches(*axes):
        return Rejected("merged axes belong to different rank universes")
    return Derived((axes[0],))


def _merge_outer_cut(axes, _args) -> CellResult:
    if not _scope_matches(*axes):
        return Rejected("merged axes belong to different rank universes")
    if isinstance(axes[0], Block) and axes[0].pad_to is not None:
        return Rejected(
            "a padded outer Block cannot cross merge without the inner extent "
            "needed to scale its padding quantum"
        )
    return Derived((axes[0],))


def _reject_inner_cut(_axes, _args) -> CellResult:
    return Rejected(
        "the inner merge factor carries a rank cut, so each rank would hold "
        "strided rather than locally contiguous intervals"
    )


def _merge_cut_sequences(axes, _args) -> CellResult:
    left, right = axes
    cuts = (left.cuts if isinstance(left, Product) else (left,)) + (
        right.cuts if isinstance(right, Product) else (right,)
    )
    if any(cut.pad_to is not None for cut in cuts):
        return Rejected("a padded Block cannot become a Product cut")
    for i, outer in enumerate(cuts):
        for inner in cuts[i + 1 :]:
            if same(outer.group, inner.group):
                return Rejected(
                    "merge would place two cuts from the same group on one "
                    "axis, creating diagonal coverage"
                )
            if not orthogonal(outer.group, inner.group):
                return Rejected(
                    "merge cuts are not orthogonal and cannot form a Product"
                )
    return Derived((product(cuts),))


def _reject_segmented_merge(_axes, _args) -> CellResult:
    return Rejected(
        "Segmented does not cross merge; slice back to a declared segment first"
    )


def _concat(axes, args) -> CellResult:
    pieces = []
    offset = 0
    for axis, extent in zip(axes, args["extents"], strict=True):
        if isinstance(axis, Segmented):
            total = sum(segment_extent for _, segment_extent, _ in axis.segments)
            require_equal(
                total, extent,
                lambda at, total=total, extent=extent: ValueError(
                    f"Segmented tiles {at(total)} elements but concat "
                    f"received {at(extent)}"))
            for _, segment_extent, child in axis.segments:
                pieces.append((offset, segment_extent, child))
                offset += segment_extent
        else:
            pieces.append((offset, extent, axis))
            offset += extent
    scopes = {_scope_of(child) for _, _, child in pieces}
    if len(scopes) != 1:
        return Rejected("concat inputs belong to different rank universes")
    return Derived((segmented(pieces),))


def _slice_replicated(axes, _args) -> CellResult:
    return Derived((axes[0],))


def _slice_block(axes, args) -> CellResult:
    whole = whole_interval(args["start"], args["stop"], args["extent"])
    if not whole:
        return Rejected(
            "a proper subinterval of a Block leaves out-of-interval shard "
            "ranks empty and breaks per-rank uniformity"
        )
    return Derived((axes[0],))


def _slice_segmented(axes, args) -> CellResult:
    try:
        out = slice_axis(axes[0], args["start"], args["stop"], args["extent"])
    except ValueError as exc:
        return Rejected(f"segment slice is not representable: {exc}")
    return Derived((out,))


def _slice_product(axes, args) -> CellResult:
    whole = whole_interval(args["start"], args["stop"], args["extent"])
    if not whole:
        return Rejected(
            "a proper Product subinterval leaves some ranks empty or gives "
            "ranks unequal local extents"
        )
    return Derived((axes[0],))


def _broadcast_new(_axes, args) -> CellResult:
    return Derived((replicated(args["scope"]),))


def _broadcast_counterpart(axes, _args) -> CellResult:
    return Derived((axes[0],))


def _repeat_replicated_or_block(axes, args) -> CellResult:
    layout = axes[0]
    if isinstance(layout, Block) and layout.pad_to is not None:
        layout = block(
            layout.group,
            layout.replica_cells,
            pad_to=layout.pad_to * args["factor"],
        )
    return Derived((layout,))


def _repeat_segmented(_axes, _args) -> CellResult:
    return Rejected("element repeat over Segmented is not defined")


def _repeat_product(_axes, _args) -> CellResult:
    return Rejected("element repeat over nested Product cuts is not defined")


# Explicitly written as a table so constructor coverage is reviewable.  The
# expected-key set below is independently generated and pinned by tests.
TRANSFER: dict[tuple, Cell] = {
    (PRESERVE, Replicated): _preserve,
    (PRESERVE, Block): _preserve,
    (PRESERVE, Segmented): _preserve,
    (PRESERVE, Product): _preserve,
    (CONTRACT, Replicated): _contract_replicated,
    (CONTRACT, Block): _contract_block,
    (CONTRACT, Segmented): _contract_segmented,
    (CONTRACT, Product): _contract_product,
    (SPLIT, Replicated): _split_replicated,
    (SPLIT, Block): _split_block,
    (SPLIT, Segmented): _split_segmented,
    (SPLIT, Product): _split_product,
    (MERGE, (Replicated, Replicated)): _merge_replicated,
    (MERGE, (Replicated, Block)): _reject_inner_cut,
    (MERGE, (Replicated, Segmented)): _reject_segmented_merge,
    (MERGE, (Replicated, Product)): _reject_inner_cut,
    (MERGE, (Block, Replicated)): _merge_outer_cut,
    (MERGE, (Block, Block)): _merge_cut_sequences,
    (MERGE, (Block, Segmented)): _reject_segmented_merge,
    (MERGE, (Block, Product)): _merge_cut_sequences,
    (MERGE, (Segmented, Replicated)): _reject_segmented_merge,
    (MERGE, (Segmented, Block)): _reject_segmented_merge,
    (MERGE, (Segmented, Segmented)): _reject_segmented_merge,
    (MERGE, (Segmented, Product)): _reject_segmented_merge,
    (MERGE, (Product, Replicated)): _merge_outer_cut,
    (MERGE, (Product, Block)): _merge_cut_sequences,
    (MERGE, (Product, Segmented)): _reject_segmented_merge,
    (MERGE, (Product, Product)): _merge_cut_sequences,
    (CONCAT,): _concat,
    (SLICE, Replicated): _slice_replicated,
    (SLICE, Block): _slice_block,
    (SLICE, Segmented): _slice_segmented,
    (SLICE, Product): _slice_product,
    (BROADCAST, None): _broadcast_new,
    (BROADCAST, Replicated): _broadcast_counterpart,
    (BROADCAST, Block): _broadcast_counterpart,
    (BROADCAST, Segmented): _broadcast_counterpart,
    (BROADCAST, Product): _broadcast_counterpart,
    (REPEAT, Replicated): _repeat_replicated_or_block,
    (REPEAT, Block): _repeat_replicated_or_block,
    (REPEAT, Segmented): _repeat_segmented,
    (REPEAT, Product): _repeat_product,
}


EXPECTED_TRANSFER_KEYS = frozenset(
    [
        (relation, constructor)
        for relation in (PRESERVE, CONTRACT, SPLIT, SLICE, REPEAT)
        for constructor in LAYOUT_CONSTRUCTORS
    ]
    + [
        (MERGE, (outer, inner))
        for outer in LAYOUT_CONSTRUCTORS
        for inner in LAYOUT_CONSTRUCTORS
    ]
    + [(CONCAT,)]
    + [(BROADCAST, counterpart) for counterpart in (None, *LAYOUT_CONSTRUCTORS)]
)


def _is_empty_range(relation: Relation, args: Mapping[str, Any]) -> bool:
    """Whether the relation's result indexes zero elements by its
    declaration — one law for both ends, an extent that is the number zero.
    One empty at some points only is laid out like any other, and holds no
    element at the points where it comes to nothing."""
    if relation is SLICE:
        return same_law(args["start"], args["stop"])
    if relation is SPLIT:
        return same_law(args["extent"], 0)
    return False


def transfer(
    relation: Relation,
    in_axes: Tuple[AxisLayout, ...],
    *,
    routed: bool = False,
    **args: Any,
) -> CellResult:
    """Look up one cell and enforce its input and output contracts."""
    try:
        relation = Relation(relation)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown axis relation {relation!r}") from exc
    axes = tuple(in_axes)
    _validate_inputs(relation, axes)
    args = _require_args(relation, axes, args)
    if _is_empty_range(relation, args):
        # An empty range holds no element, so no rank owns a piece of it and
        # no layout of the input constrains it: every output axis is
        # Replicated over the input's rank universe.
        scope = _scope_of(axes[0])
        outputs = (len(tuple(args["output_extents"]))
                   if relation is SPLIT else 1)
        return validate_cell_result(
            relation, Derived(tuple(replicated(scope) for _ in range(outputs))),
            expected_outputs=outputs if relation is SPLIT else None)

    _validate_input_extents(relation, axes, args)
    if not isinstance(routed, bool):
        raise TypeError("routed must be bool")
    if routed and relation not in (PRESERVE, BROADCAST):
        return Rejected(
            "routed rows may only be preserved or broadcast; combine is the "
            "only operation allowed to consume that axis"
        )

    if relation in (PRESERVE, CONTRACT, SPLIT, SLICE, REPEAT):
        key = (relation, type(axes[0]))
    elif relation is MERGE:
        key = (relation, (type(axes[0]), type(axes[1])))
    elif relation is CONCAT:
        key = (relation,)
    else:
        key = (relation, None if not axes else type(axes[0]))
    try:
        cell = TRANSFER[key]
    except KeyError as exc:
        raise AssertionError(f"transfer table is missing key {key!r}") from exc
    result = cell(axes, args)
    expected_outputs = len(tuple(args["output_extents"])) if relation is SPLIT else None
    return validate_cell_result(relation, result, expected_outputs=expected_outputs)
