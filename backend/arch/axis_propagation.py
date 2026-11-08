"""Per-op DimMap — the dim-correspondence of every op, the analogue of PyTorch
DTensor's view-op DimMap.

The walk's `infer_shape` propagates SIZES (how big each output dim is)
but not the CORRESPONDENCE (which output dim came from which input dim). That
correspondence is what a complete shape propagation needs — DTensor uses one
DimMap to drive BOTH shape and placement propagation. This module supplies it:

    dim_map(op, node) -> tuple[DimSource, ...]   # one entry per OUTPUT dim

Each `DimSource` says where an output dim comes from: a 1:1 `from` an input
dim, a `split` part of an input dim (view_split's n_heads/d_head), a `merge` of several
input dims (view_merge), a `concat` axis, or a `new` dim (a rewritten linear
feature / a source op). It is POSITIONAL — it never reads an axis symbol, so a
head axis concretized to `64` (a headwise-gate binding `n_heads` as a feature dim)
behaves identically to one labelled `n_heads`.

Consumers:
  * The evaluator walk — `infer_shape` is how an op produces `shape_out`.
  * Layout geometry — view_split / view_merge / transpose apply the same
    map through the transfer table.
  * Shape validation — `shape_consistency_errors` checks size relations
    (split parts multiply, merge is a product).
  * `axis_transfer` inverts the map to follow a sharded axis forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from backend.arch.ops import (
    OpSpec,
    ShapeKind,
    elementwise_broadcast_axes,
    get_op,
)
from backend.arch.op_names import op_keyed
from backend.arch.shape_node import ShapeProbe


@dataclass(frozen=True)
class DimSource:
    """How ONE output dim is produced from the op's inputs.

    kind:
      "from"     — 1:1 copy of input dim `refs[0]`.
      "split"    — one part of a split input dim `refs[0]`; `part` 0 = the
                   leading/head-count slot, 1 = the trailing/per-head slot.
      "merge"    — product of the input dims in `refs` (flattened together).
      "concat"   — `refs[0]` axis whose size is the SUM across inputs (concat).
      "broadcast"— trailing-aligned input axes whose size is the non-unit
                   operand extent.
      "new"      — not derived from any input (rewritten linear feature, source).
      "select"   — new selection index, bounded by an input extent and a dim key.
      "const"    — a written constant (`part`, default 1); `dim_key` is its name.
      "ceil_div" — `ceil(refs[0] / dims[dim_key]**exp)`; `part` is the exponent.
      "div"      — `refs[0] // dims[dim_key]**exp`.
      "mul"      — `refs[0] * dims[dim_key]**exp`.
      "extend"   — `refs[0] + part`: the same index grown by `part` entries
                   (a history `stack_append` adds one entry to), or cut
                   by `-part` (a slice from `start`).
      "grid"     — the product over `refs` of each extent divided by
                   `dims[dim_key]`: the patch grid of an image cut into
                   `dim_key`-sided patches.
      "frames"   — the product of `refs[:-1]` times `ceil(refs[-1] /
                   dims[dim_key])`: a video's items times its temporal
                   patches, its frames taken `dim_key` at a time (the last
                   repeated to a whole patch).
      "frame_grid" — `ceil(refs[0] / dims["temporal"])` times the grid of
                   `refs[1:]` in `dims["patch"]`-sided patches: one video's
                   temporal patches times its patch grid.
    refs: tuple of (input_index, input_dim) positions.
    """
    kind: str
    refs: Tuple[Tuple[int, int], ...] = ()
    part: int = 0
    # For "split" / "new": the dim symbol whose resolved value gives this
    # output dim's size. For "select", it supplies the upper bound combined
    # with the referenced input extent. Lets `infer_shape` work from DimMap alone.
    dim_key: Optional[str] = None


def _rank_out(node) -> Optional[int]:
    sh = node.out_shape
    return len(sh) if sh else None


def _input_rank(node, input_index: int,
                in_ranks: Optional[Tuple[int, ...]] = None) -> Optional[int]:
    if in_ranks is not None and input_index < len(in_ranks):
        return in_ranks[input_index]
    refs = node.inputs_obj
    if input_index < len(refs) and refs[input_index].src is not None:
        return _rank_out(refs[input_index].src)
    return None


def _input_shape(node, input_index: int,
                 in_shapes: Optional[Sequence] = None) -> Optional[Tuple[Any, ...]]:
    if in_shapes is not None and input_index < len(in_shapes) and in_shapes[input_index]:
        return tuple(in_shapes[input_index])
    refs = node.inputs_obj
    if input_index >= len(refs) or refs[input_index].src is None:
        return None
    shape = refs[input_index].src.out_shape
    return tuple(shape) if shape else None


def _norm(axis: int, rank: int) -> int:
    return axis if axis >= 0 else rank + axis


def _lookup(op) -> Optional[Any]:
    if isinstance(op, OpSpec):
        return op
    if isinstance(op, str):
        try:
            return get_op(op)
        except KeyError:
            return None
    # A spec-scoped custom op is not in the registry; it still declares
    # `shape_kind` and the DimMap compiles that the same way.
    if getattr(op, "shape_kind", None) is not None:
        return op
    return None


def _shape_kind(spec) -> Optional[ShapeKind]:
    if spec is None:
        return None
    kind = spec.shape_kind
    if isinstance(kind, ShapeKind):
        return kind
    if kind is None:
        return None
    try:
        return ShapeKind(str(kind))
    except ValueError:
        return None


# --- explicit DimMaps for the irregular reshape / attention ops -------------
def _dm_view_split(node, r_in0, rout) -> List[DimSource]:
    # Split the declared input axis into the two output axes at its
    # position — `axis: -1` is [..., feat] → [..., n_heads, d_head]. There is
    # no default: the schema rejects a view_split without `axis:`, and a
    # probe that arrives without one is refused here rather than guessed.
    # Factor names come from the node's declared dims in authored
    # order (outer, inner), so a KV split (`n_kv_heads`) is not silently read
    # as `n_heads`, and a window fold names its own (slots, window). Exact
    # product — `shape_consistency_errors` checks the parts multiply back to
    # the split axis; a ragged tail is the call site's explicit `slice`,
    # never rounding here.
    factors = [k for k in node.dims_raw if k != "in"]
    if len(factors) < 2:
        factors = ["n_heads", "d_head"]
    if node.axis is None:
        raise ValueError("view_split declares no `axis:`")
    ax = _norm(node.axis, r_in0)
    if not 0 <= ax < r_in0:
        raise ValueError(
            f"view_split axis {node.axis} outside input rank {r_in0}")
    return (
        [DimSource("from", ((0, j),)) for j in range(ax)]
        + [
            DimSource("split", ((0, ax),), part=0, dim_key=factors[0]),
            DimSource("split", ((0, ax),), part=1, dim_key=factors[1]),
        ]
        + [DimSource("from", ((0, j),)) for j in range(ax + 1, r_in0)]
    )


#: The axes `view_merge` flattens, as positions of the tensor that arrives:
#: its last two (`[..., n_heads, d_head] -> [..., n_heads * d_head]`). The
#: merged width is their product and nothing else, so a call site names no
#: width. Positional, because the two axes are a different pair of labels at
#: every call (`n_heads / d_head`, `n_hidden_copies / d_model`, ...).
#: `layout.axis_use.GEOMETRY_POSITIONS` reads this same constant.
VIEW_MERGE_AXES: Tuple[int, int] = (-2, -1)


def _dm_view_merge(node, r_in0, rout) -> List[DimSource]:
    # The axes at `VIEW_MERGE_AXES` merge into the last output axis, whose
    # extent is their product; every other axis copies through.
    del node, rout
    merged = tuple(_norm(axis, r_in0) for axis in VIEW_MERGE_AXES)
    out = [DimSource("from", ((0, j),)) for j in range(r_in0)
           if j not in merged]
    out.append(DimSource("merge", tuple((0, j) for j in merged)))
    return out


def _dm_axis_insert(node, r_in0, rout) -> List[DimSource]:
    """Insert one new axis and shift every existing input axis around it."""
    position = node.axis
    position = -1 if position is None else position
    position = _norm(position, rout)
    return [
        DimSource("new", dim_key="width")
        if output_axis == position
        else DimSource(
            "from",
            ((0, output_axis if output_axis < position else output_axis - 1),),
        )
        for output_axis in range(rout)
    ]


def _dm_attention_score(node, r_in0, rout) -> List[DimSource]:
    # [B,n_heads,s_query,d_head]·[B,n_heads,s_kv,d_head] →
    # [B,n_heads,s_query,s_kv]: B,n_heads,s_query from input0; s_kv from input1's
    # sequence axis (second-to-last); the per-head d_head is contracted (absent).
    r_in1 = _input_rank(node, 1) or r_in0
    return [
        DimSource("from", ((0, 0),)),       # B
        DimSource("from", ((0, 1),)),       # n_heads
        DimSource("from", ((0, 2),)),       # s_query
        DimSource("from", ((1, r_in1 - 2),)),  # s_kv (from K's seq axis)
    ][:rout]


def _dm_windowed_score(node, r_in0, rout) -> List[DimSource]:
    # [B,n_heads,s_query,d_head]·[B,n_kv_heads,s_kv,d_head] →
    # [B,n_heads,s_query,s_attend]: the key axis is what ONE query attends,
    # the window the walk bound as `s_attend` — not K's rows, which are
    # every key the step reads.
    return [
        DimSource("from", ((0, 0),)),
        DimSource("from", ((0, 1),)),
        DimSource("from", ((0, 2),)),
        DimSource("new", dim_key="s_attend"),
    ][:rout]


def _dm_windowed_apply(node, r_in0, rout) -> List[DimSource]:
    # [B,n_heads,s_query,s_attend]·[B,n_kv_heads,s_kv,d_head] →
    # [B,n_heads,s_query,d_head]: the leading three from the probabilities,
    # d_head from V's last axis. The banded contraction pairs one query's
    # window (`s_attend`) with V's rows (`s_kv`, every key the step reads),
    # two different extents, so no label is shared to contract over.
    r_in1 = _input_rank(node, 1) or r_in0
    return [
        DimSource("from", ((0, 0),)),          # B
        DimSource("from", ((0, 1),)),          # n_heads
        DimSource("from", ((0, 2),)),          # s_query
        DimSource("from", ((1, r_in1 - 1),)),  # d_head (V's feature axis)
    ][:rout]


def _dm_preserve_in0(node, r_in0, rout) -> List[DimSource]:
    # Output copies input0's shape; later inputs are non-shape side data (a
    # softmax's per-head sink bias [n_heads]). Used by the sink and key-0-bias
    # softmaxes.
    return [DimSource("from", ((0, j),)) for j in range(rout)]


def _dm_dsa_score(node, r_in0, rout) -> List[DimSource]:
    # Leading axes copy the query. The last axis is the sparse read length
    # the walk already bound as `s_kv` (`Min(s_kv, n_topk_keys)`), not K's
    # stored sequence — that stored axis is the full cache, and treating
    # them as the same index would make a correspondence the op does not
    # declare.
    return [
        DimSource("from", ((0, 0),)),
        DimSource("from", ((0, 1),)),
        DimSource("from", ((0, 2),)),
        DimSource("new", dim_key="s_kv"),
    ][:rout]


def _dm_head_repeat(node, r_in0, rout) -> List[DimSource]:
    # [B, N, S, D] → [B, n_heads, S, D]: only the head-count axis is rewritten.
    return [
        DimSource("from", ((0, 0),)),
        DimSource("new", dim_key="n_heads"),
        DimSource("from", ((0, 2),)),
        DimSource("from", ((0, 3),)),
    ][:rout]


def _dm_copy_input(input_index: int):
    """Copy every axis of one input — the written correspondence."""

    def _dm(node, r_in0, rout) -> List[DimSource]:
        del node, r_in0
        return [DimSource("from", ((input_index, j),)) for j in range(rout)]

    return _dm


def _dm_stack_init(node, r_in0, rout) -> List[DimSource]:
    # [B, S, D] -> [B, S, 1, D]: a history of one entry, the new axis
    # before the feature axis.
    return [DimSource("from", ((0, 0),)), DimSource("from", ((0, 1),)),
            DimSource("const", part=1),
            DimSource("from", ((0, 2),))][:rout]


def _dm_stack_append(node, r_in0, rout) -> List[DimSource]:
    # [B, S, m, D] ++ [B, S, D] -> [B, S, m + 1, D]: the entry lands at the
    # end of the history axis; B, S and D are one index across both inputs.
    return [DimSource("from", ((0, 0), (1, 0))),
            DimSource("from", ((0, 1), (1, 1))),
            DimSource("extend", ((0, 2),), part=1),
            DimSource("from", ((0, 3), (1, 2)))][:rout]


def _view_merge_dims(dims: Mapping[str, Any]) -> Optional[str]:
    if "out" in dims:
        return ("`view_merge` merges the two axes at "
                f"{list(VIEW_MERGE_AXES)} of the tensor it reads, and its "
                "width is their product; delete `out:` from its `dims:`")
    return None


def _slice_dims(dims: Mapping[str, Any]) -> Optional[str]:
    if "out" not in dims and "start" not in dims:
        return ("`slice` keeps a window of one axis and states it at the "
                "call site: `dims: { out: <kept width> }`, with `start: "
                "<offset>` beside it for a window that does not begin at 0")
    return None


#: What a call site's `dims:` must and must not say to the DimMap of the
#: ops whose DimMap reads the call site's own `dims:` (`_dm_slice`) or
#: derives an extent a call site could restate (`_dm_view_merge`). Asked at
#: load (`call_site_dims_error`) of every graph node and model layer.
_CALL_SITE_DIMS = op_keyed({
    "view_merge": _view_merge_dims,
    "slice": _slice_dims,
})


def call_site_dims_error(op: Optional[str],
                         dims: Mapping[str, Any]) -> Optional[str]:
    """Why the `dims:` a call site of `op` writes do not fit that op's
    DimMap, or None."""
    check = _CALL_SITE_DIMS.get(op or "")
    return None if check is None else check(dims)


def _dm_slice(node, r_in0, rout) -> List[DimSource]:
    # A window of one axis: the same index with fewer entries, so a shard
    # of any other axis rides through. `out` is the extent it keeps (a
    # fused projection's q / k / v part, with `start` its offset). A slice
    # that gives `start` and no `out` keeps the rest of the axis from
    # there — `hs[:, start:]`, whatever the axis holds: the crop a
    # whole-stream join takes of each element (`Program.lowered`).
    del r_in0
    at = _norm(node.axis if node.axis is not None else -1, rout)
    own = node.dims_raw or {}
    rest = "out" not in own and "start" in own
    return [
        (DimSource("extend", ((0, j),), part=-int(own["start"])) if rest
         else DimSource("from", ((0, j),), dim_key="out"))
        if j == at else DimSource("from", ((0, j),))
        for j in range(rout)]


def _dm_sum_reduce(node, r_in0, rout) -> List[DimSource]:
    ax = node.axis
    ax = r_in0 - 1 if ax is None else _norm(ax, r_in0)
    return [DimSource("from", ((0, j),)) for j in range(r_in0) if j != ax]


def _dm_hc_split(node, r_in0, rout) -> List[DimSource]:
    return [DimSource("from", ((0, j),)) for j in range(r_in0 - 1)] + [
        DimSource("new", dim_key="n_hidden_copies"),
    ]


def _dm_hc_mix(node, r_in0, rout) -> List[DimSource]:
    return [
        DimSource("from", ((0, 0),)),
        DimSource("from", ((0, 1),)),
        DimSource("from", ((0, 2),)),
        DimSource("new", dim_key="d_model"),
    ][:rout]


def _dm_indexer_logits(node, r_in0, rout) -> List[DimSource]:
    # [q: B, H, s_query, d], [k: B, s_kv, d], [weights: B, s_query, H] →
    # [B, s_query, s_kv]: the batch and the query axis are the query's and
    # the head weights'.
    return [
        DimSource("from", ((0, 0), (2, 0))),
        DimSource("from", ((0, 2), (2, 1))),
        DimSource("new", dim_key="s_kv"),
    ][:rout]


def _dm_conv2d(node, r_in0, rout) -> List[DimSource]:
    # [B, S, C_in] → [B, ceil(S / stride^2), C_out]. The output length is a
    # ceil of the incoming sequence (a strided conv on a flattened square grid
    # halves each spatial side), and the channel is the new C_out.
    del r_in0
    return [
        DimSource("from", ((0, 0),)),
        DimSource("ceil_div", ((0, 1),), part=2, dim_key="stride"),
        DimSource("new", dim_key="C_out"),
    ][:rout]


#: The axes `spatial_merge` regroups, as positions of the tensor that
#: arrives: its rows fold m x m windows into its features. Positional, as
#: `VIEW_MERGE_AXES` is; `layout.axis_use.GEOMETRY_POSITIONS` reads it.
SPATIAL_MERGE_AXES: Tuple[int, int] = (-2, -1)

#: The axis a row scatter writes along, in both operands: x's positions and
#: v's rows (`ops_registry` `scatter_rows_*`, `layout.geometry.scatter_rows`).
ROW_SCATTER_AXES: Tuple[int] = (1,)


def _dm_spatial_merge(node, r_in0, rout) -> List[DimSource]:
    # [.., S, C] → [.., S/m², C·m²]. Both trailing axes are the input's,
    # scaled by the same merge factor.
    rows, features = (_norm(axis, r_in0) for axis in SPATIAL_MERGE_AXES)
    return (
        [DimSource("from", ((0, j),)) for j in range(max(0, r_in0 - 2))]
        + [
            DimSource("div", ((0, rows),), part=2, dim_key="n_spatial_merge"),
            DimSource("mul", ((0, features),), part=2, dim_key="n_spatial_merge"),
        ]
    )[:rout]



def _dm_patchify(node, r_in0, rout) -> List[DimSource]:
    # An image [N, n, H, W] → [N·n, (H/P)·(W/P), C·Tp·P²]: one segment per
    # item, one row per patch, one patch's values per row. A video
    # [N, n, T, H, W] has ceil(T/Tp) temporal patches: one segment each
    # (`segment: temporal_patch`) → [N·n·⌈T/Tp⌉, (H/P)(W/P), …], or the
    # whole video one segment (`segment: item`) → [N·n, ⌈T/Tp⌉(H/P)(W/P), …].
    width = DimSource("new", dim_key="channels * temporal * patch * patch")
    if r_in0 == 5:
        if node.segment == "item":
            return [
                DimSource("merge", ((0, 0), (0, 1))),
                DimSource("frame_grid", ((0, 2), (0, 3), (0, 4))),
                width,
            ][:rout]
        return [
            DimSource("frames", ((0, 0), (0, 1), (0, 2)), dim_key="temporal"),
            DimSource("grid", ((0, 3), (0, 4)), dim_key="patch"),
            width,
        ][:rout]
    return [
        DimSource("merge", ((0, 0), (0, 1))),
        DimSource("grid", ((0, 2), (0, 3)), dim_key="patch"),
        width,
    ][:rout]


def _dm_row_gather(node, r_in0, rout) -> List[DimSource]:
    # table [B, rows, dim] picked by ids [B, s] → [B, s, dim]: the ids'
    # axes, then the table's width.
    del r_in0
    return [
        DimSource("from", ((1, 0),)),
        DimSource("from", ((1, 1),)),
        DimSource("from", ((0, 2),)),
    ][:rout]


def _dm_view_rows(node, r_in0, rout) -> List[DimSource]:
    # [B, R, rows·w] → [B, R·rows, w].
    del r_in0
    return [
        DimSource("from", ((0, 0),)),
        DimSource("mul", ((0, 1),), part=1, dim_key="rows"),
        DimSource("div", ((0, 2),), part=1, dim_key="rows"),
    ][:rout]


def _dm_conv_transpose1d(node, r_in0, rout) -> List[DimSource]:
    # [B, S, C_in] → [B, S·stride, C_out].
    del r_in0
    return [
        DimSource("from", ((0, 0),)),
        DimSource("mul", ((0, 1),), part=1, dim_key="stride"),
        DimSource("new", dim_key="C_out"),
    ][:rout]


def _dm_append_tokens(node, r_in0, rout) -> List[DimSource]:
    # [M, R, D] + [1, k₁, D] + … → [M, R + k₁ + …, D].
    del r_in0
    return [
        DimSource("from", ((0, 0),)),
        DimSource("concat", ((0, 1),)),
        DimSource("from", ((0, 2),)),
    ][:rout]


def _dm_declared_rows(node, r_in0, rout) -> List[DimSource]:
    # A re-tiling whose three output axes the call site declares (`tiles` /
    # `items`, `rows`, `channels`): the tiles overlap, so no output axis is
    # one of the input's.
    del r_in0
    lead = "tiles" if node.op == "tile_video" else "items"
    return [
        DimSource("new", dim_key=lead),
        DimSource("new", dim_key="rows"),
        DimSource("new", dim_key="channels"),
    ][:rout]


def _dm_group_items(node, r_in0, rout) -> List[DimSource]:
    # [M·k, R, D] → [M, k·R, D]; M is declared (`groups`), so k = 0 holds.
    del r_in0
    return [
        DimSource("new", dim_key="groups"),
        DimSource("mul", ((0, 1),), part=1, dim_key="items"),
        DimSource("from", ((0, 2),)),
    ][:rout]


def _dm_pad_rows(node, r_in0, rout) -> List[DimSource]:
    # [M, V, ...] → [M, rows, ...].
    return [
        DimSource("from", ((0, 0),)),
        DimSource("new", dim_key="rows"),
        *(DimSource("from", ((0, j),)) for j in range(2, r_in0)),
    ][:rout]


def _dm_unpad(node, r_in0, rout) -> List[DimSource]:
    # x [M, R, D], rows [M, V, C] → [M, V, D].
    del r_in0
    return [
        DimSource("from", ((0, 0),)),
        DimSource("from", ((1, 1),)),
        DimSource("from", ((0, 2),)),
    ][:rout]


# ---------------------------------------------------------------------------
# Index signatures — the form DimMap is a view of
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexSignature:
    """A label on every axis of every tensor at a node, inputs and output.

    `dim_map` is a BACKWARD function — output dim to its source — and a backward
    function cannot express two things, because its domain is the output:

      * whether two INPUT axes are the same index. Inputs appear only as values,
        and the constructors name one representative (input 0), which is
        sufficient for a size question and wrong for an identity question. A
        pointwise `mul` reported its second operand's axes as consumed.
      * whether an input axis was consumed at all. "Not mentioned" and
        "mentioned via the representative" are indistinguishable.

    Labelling every axis makes both trivial: same index is the same label, and
    consumed is a label present in the inputs and absent from the output. They
    stop being mechanisms and become queries.

    The labels come from the OP's templates, never from the node's dims — so a
    head axis a headwise gate concretised to `64` still carries the label its op
    gave it. That is the same robustness `dim_map` gets from being positional,
    kept for the same reason.
    """

    ins: Tuple[Tuple[str, ...], ...] = ()
    out: Tuple[str, ...] = ()
    #: labels whose extent comes from `dims` rather than from an input — a
    #: rewritten feature axis, and also a NARROWING, where the label is shared
    #: with an input but the extent is not.
    sizes: Mapping[str, str] = field(default_factory=dict)
    #: labels whose extent SUMS across the inputs carrying them
    concat: FrozenSet[str] = frozenset()
    #: label → the dim key bounding a SELECTION along it: the output keeps the
    #: index (a top-k picks positions of the axis it ranks) and its extent is
    #: min(input extent, dims[key])
    selects: Mapping[str, str] = field(default_factory=dict)
    #: label → the labels whose product it equals (a merge, or a split read the
    #: other way round)
    products: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def positions(self, label: str) -> Tuple[Tuple[int, int], ...]:
        return tuple((i, j) for i, t in enumerate(self.ins)
                     for j, other in enumerate(t) if other == label)

    def consumed(self) -> FrozenSet[str]:
        """Input labels the output does not carry. This is the query placement
        needs and the one a backward map cannot answer."""
        survives = set(self.out)
        for label, parts in self.products.items():
            if label in survives:
                survives |= set(parts)
        return frozenset(
            label for t in self.ins for label in t if label not in survives)


def signature_for(spec, node, rank_out: Optional[int]) -> Optional[IndexSignature]:
    """Derive the signature from what the op already declares.

    No op is re-declared. `shape_in` is the input half — it has been an einsum
    signature all along, which is what made flops derivable from shape — and
    `shape_kind` plus `out_key` / `axis_locked` / `permute_to` is the output
    half in compressed form. This expands the compression.

    Returns None when the templates do not describe these tensors (a rank
    mismatch, a variadic op, no template at all), and the caller keeps the
    positional dispatch. Refusing is the point: a signature that does not fit is
    not evidence about the node.
    """
    if rank_out is None:
        return None
    for candidate in (getattr(spec, "shape_in", None) or (),
                      *tuple(spec.alt_shape_in or ())):
        sig = _signature_from_template(spec, node, rank_out, candidate)
        if sig is not None:
            return sig
    return None


def _signature_from_template(spec, node, rank_out, raw):
    """One candidate template's signature, or None when it does not fit."""
    if not raw and _shape_kind(spec) is ShapeKind.PERMUTE:
        # A permutation's mapping is already fully declared — by `axes` at the
        # call site. `transpose` therefore carries no template of its own, and
        # asking it for one is asking it to restate the rank `axes` implies.
        # Positional labels are the whole content of a permutation: output
        # axis j comes from input axis `perm[j]`, which is what a caller of
        # this signature reads. 337 of the corpus's 746 refusals were this op
        # alone, every one of them for having nothing to say rather than for
        # saying something that did not fit.
        perm = tuple(node.axes or spec.permute_to or ())
        if not perm:
            return None
        raw = tuple(f"a{i}" for i in range(len(perm)))
    if not raw:
        return None
    templates = (tuple(tuple(str(x) for x in t) for t in raw)
                 if all(isinstance(t, (list, tuple)) for t in raw)
                 else (tuple(str(x) for x in raw),))
    if any(not t or any(not lbl.isidentifier() for lbl in t) for t in templates):
        return None
    in0 = templates[0]
    kind = _shape_kind(spec)
    sizes: Dict[str, str] = {}
    concat: set = set()
    selects: Dict[str, str] = {}

    if kind is ShapeKind.PRESERVE:
        out_labels = in0
    elif kind is ShapeKind.AXIS_REPLACE:
        locked = (spec.axis_locked if spec.axis_locked is not None
                  else (node.axis if node.axis is not None
                        else -1))
        lp = _norm(locked, len(in0))
        if not spec.out_key or lp >= len(in0):
            return None
        if spec.is_view_op:
            # A narrowing keeps the label and takes a new extent.
            out_labels = in0
            sizes[in0[lp]] = spec.out_key
        else:
            out_labels = (*in0[:lp], spec.out_key, *in0[lp + 1:])
            sizes[spec.out_key] = spec.out_key
    elif kind is ShapeKind.AXIS_SELECT:
        # A selection keeps the index it selects along — top-k's k picks are
        # positions of the axis it ranks — at extent min(input axis, out_key).
        locked = (spec.axis_locked if spec.axis_locked is not None
                  else (node.axis if node.axis is not None
                        else -1))
        lp = _norm(locked, len(in0))
        if not spec.out_key or lp >= len(in0):
            return None
        out_labels = in0
        selects = {in0[lp]: spec.out_key}
    elif kind is ShapeKind.AXIS_CONCAT:
        locked = (spec.axis_locked if spec.axis_locked is not None
                  else (node.axis if node.axis is not None
                        else -1))
        ax = _norm(locked, len(in0))
        # The concat axis is ONE index. The templates give it a different name
        # per input — `in_a` against `in_b` — because the extents differ, which
        # is what `concat` already means: same index, extents summed. Leaving
        # them distinct reports every operand after the first as consumed.
        label = in0[ax]
        templates = tuple(
            (*t[:ax], label, *t[ax + 1:]) if ax < len(t) else t
            for t in templates)
        in0 = templates[0]
        out_labels = in0
        concat.add(label)
    elif kind is ShapeKind.BROADCAST:
        actual_ranks = tuple(
            _input_rank(node, index) for index in range(len(templates))
        )
        widest = max(templates, key=len)
        if any(rank is not None and rank > len(widest) for rank in actual_ranks):
            return None
        # Elementwise operands are trailing-aligned.  A channel vector paired
        # with a rank-3 activation therefore carries only the activation
        # signature's final label; it does not need a separately enumerated
        # rank-1 template.
        templates = tuple(
            template if rank is None else widest[-rank:]
            for rank, template in zip(actual_ranks, templates, strict=True)
        )
        in0 = templates[0]
        out_labels = widest
    elif kind is ShapeKind.CONTRACT:
        if len(templates) < 2 or not templates[1]:
            return None
        out_labels = (*in0[:-1], templates[1][-1])
    elif kind is ShapeKind.PERMUTE:
        perm = tuple(node.axes or spec.permute_to or ())
        if len(perm) != len(in0):
            return None
        out_labels = tuple(in0[p] for p in perm)
    elif kind is ShapeKind.EXPLICIT:
        # The compressed forms above derive the output half from the input
        # half. An EXPLICIT op cannot be derived — that is what the kind
        # means — so it writes its output labels down, and most already do:
        # `indexer_logits` says `(B, s_query, s_kv)`. Reading them is the
        # whole branch.
        #
        # It was missing, so an EXPLICIT op was refused whether or not its
        # templates fit, even where they match their nodes exactly.
        out_labels = tuple(str(x) for x in (spec.shape_out or ()))
        if not out_labels or any(not lbl.isidentifier() for lbl in out_labels):
            return None
        for lbl in out_labels:
            if not any(lbl in t for t in templates):
                sizes.setdefault(lbl, lbl)
    else:
        return None

    if len(out_labels) != rank_out:
        return None
    return IndexSignature(ins=templates, out=out_labels, sizes=sizes,
                          concat=frozenset(concat), selects=selects)


def dim_map_from_signature(sig: IndexSignature) -> Tuple[DimSource, ...]:
    """`DimMap` as a VIEW of the signature.

    Every `from` now names every input position carrying the label, not one
    representative, so the cross-input identity a backward map could not express
    is present without a second mechanism.
    """
    out: List[DimSource] = []
    for label in sig.out:
        refs = sig.positions(label)
        size = sig.sizes.get(label)
        if not refs:
            out.append(DimSource("new", dim_key=size or label))
        elif label in sig.selects:
            out.append(DimSource("select", refs, dim_key=sig.selects[label]))
        elif label in sig.concat:
            out.append(DimSource("concat", refs, dim_key=size))
        else:
            out.append(DimSource("from", refs, dim_key=size))
    return tuple(out)

_EXPLICIT_DM = op_keyed({
    "view_split": _dm_view_split,
    "view_merge": _dm_view_merge,
    "broadcast": _dm_axis_insert,
    "attention_score": _dm_attention_score,
    "swa_attention_score": _dm_windowed_score,
    "swa_attention_apply": _dm_windowed_apply,
    "swa_softmax_with_sink": _dm_preserve_in0,
    "swa_softmax_key0_bias": _dm_preserve_in0,
    "softmax_with_sink": _dm_preserve_in0,
    "dsa_attention_score": _dm_dsa_score,
    "head_repeat": _dm_head_repeat,
    "gated_delta_rule": _dm_copy_input(2),
    "mamba2_scan": _dm_copy_input(0),
    "position_table_add": _dm_preserve_in0,
    # `permute` / `combine` are NOT here. Their geometry is what their
    # templates say — `(B, s_ctx, d_model) -> (n_expert_copies, M, d_model)` and the
    # inverse, with the token axes coming from the routing operand — so the
    # EXPLICIT branch of `signature_for` derives their DimMap like any other
    # op's. The two hand-written maps that used to sit here were where their
    # shapes actually came from, and they said `M` (one expert's slice) while
    # the ops move the whole buffer.
    "sum_reduce": _dm_sum_reduce,
    "hc_split_sinkhorn": _dm_hc_split,
    "hc_mix": _dm_hc_mix,
    "indexer_logits": _dm_indexer_logits,
    "conv2d": _dm_conv2d,
    "spatial_merge": _dm_spatial_merge,
    "patchify": _dm_patchify,
    "pad_rows": _dm_pad_rows,
    "group_items": _dm_group_items,
    "unpad": _dm_unpad,
    "stack_init": _dm_stack_init,
    "stack_append": _dm_stack_append,
    "slice": _dm_slice,
    "row_gather": _dm_row_gather,
    "view_rows": _dm_view_rows,
    "conv_transpose1d": _dm_conv_transpose1d,
    "conv_transpose1d_weight_norm": _dm_conv_transpose1d,
    "snake_beta": _dm_preserve_in0,
    "append_tokens": _dm_append_tokens,
    "tile_video": _dm_declared_rows,
    "untile_video": _dm_declared_rows,
})


# ---------------------------------------------------------------------------
# Output rank
# ---------------------------------------------------------------------------
# `dim_map` READS the output rank off the node, which is fine for checking an
# already-computed shape but circular for producing one: to drive shape
# propagation the rank has to come from the inputs, not from the answer. These
# rules close that loop — rank is a much simpler function than shape, so most
# ops fall out of their `shape_kind` alone.
#
# Rank rules for the ops whose `shape_kind` is EXPLICIT (the kind says
# nothing about rank). Keyed by op name;
# `(in_ranks, node) -> Optional[int]`. Only ops whose rank relation is a
# property of the op's definition belong here — anything whose rank depends on
# a dim value stays absent and reports None rather than guessing.
_EXPLICIT_RANK = op_keyed({
    # One declared axis splits in two.
    "view_split": lambda ir, node: ir[0] + 1 if ir else None,
    # [..., n_heads, d_head] → [..., feat]: the trailing pair merges.
    "view_merge": lambda ir, node: ir[0] - 1 if ir and ir[0] >= 2 else None,
    # Insert one call-site-positioned axis.
    "broadcast": lambda ir, node: ir[0] + 1 if ir else None,
    # A lookup appends the table's row width to the ids it reads; a hashed
    # lookup appends its rows per token before it.
    "embedding": lambda ir, node: ir[0] + 1 if ir else None,
    "hash_embedding": lambda ir, node: ir[0] + 2 if ir else None,
    # A hash route appends the expert slots to the token ids it reads.
    "hash_route": lambda ir, node: ir[0] + 1 if ir else None,
    # [B,n_heads,s_query,d_head]·[B,n_heads,s_kv,d_head] →
    # [B,n_heads,s_query,s_kv]: d_head contracted, s_kv appended.
    "attention_score": lambda ir, node: ir[0] if ir else None,
    "swa_attention_score": lambda ir, node: ir[0] if ir else None,
    "swa_attention_apply": lambda ir, node: ir[0] if ir else None,
    # [B, S, C_in] → [B, ceil(S/stride^2), C_out]: sequence shrinks and channel
    # changes, but the rank is unchanged.
    "conv2d": lambda ir, node: ir[0] if ir else None,
    # Sink bias is per-head side data, not a shape input.
    "swa_softmax_with_sink": lambda ir, node: ir[0] if ir else None,
    "swa_softmax_key0_bias": lambda ir, node: ir[0] if ir else None,
    "softmax_with_sink": lambda ir, node: ir[0] if ir else None,
    # DSA sparse family: leading axes carry through, the last axis is the op's
    # own sparse/latent feature — rank is preserved either way.
    "dsa_attention_score": lambda ir, node: ir[0] if ir else None,
    "dsa_softmax": lambda ir, node: ir[0] if ir else None,
    "dsa_attention_apply": lambda ir, node: ir[0] if ir else None,
    # Output rank equals the copied input.
    "head_repeat": lambda ir, node: ir[0] if ir else None,
    "gated_delta_rule": lambda ir, node: ir[2] if len(ir) > 2 else None,
    "mamba2_scan": lambda ir, node: ir[0] if ir else None,
    "position_table_add": lambda ir, node: ir[0] if ir else None,
    "permute": lambda ir, node: ir[0] if ir else None,
    "combine": lambda ir, node: ir[0] if ir else None,
    "sum_reduce": lambda ir, node: ir[0] - 1 if ir and ir[0] else None,
    "hc_split_sinkhorn": lambda ir, node: ir[0] if ir else None,
    "hc_mix": lambda ir, node: 4 if ir else None,
    "indexer_logits": lambda ir, node: 3 if ir else None,
    "spatial_merge": lambda ir, node: ir[0] if ir else None,
    # An image's four axes become segments, rows and a patch's values.
    "patchify": lambda ir, node: 3 if ir and ir[0] in (4, 5) else None,
    "pad_rows": lambda ir, node: ir[0] if ir else None,
    "group_items": lambda ir, node: ir[0] if ir else None,
    "unpad": lambda ir, node: ir[0] if ir else None,
    # A history axis appears before the feature axis, or grows.
    "stack_init": lambda ir, node: ir[0] + 1 if ir else None,
    "stack_append": lambda ir, node: ir[0] if ir else None,
    # A picked table row per id: the ids' axes and the row's width.
    "row_gather": lambda ir, node: ir[1] + 1 if len(ir) > 1 else None,
    "view_rows": lambda ir, node: ir[0] if ir else None,
    "conv_transpose1d": lambda ir, node: ir[0] if ir else None,
    "conv_transpose1d_weight_norm": lambda ir, node: ir[0] if ir else None,
    "snake_beta": lambda ir, node: ir[0] if ir else None,
    "append_tokens": lambda ir, node: ir[0] if ir else None,
    # A latent video's five axes become tiles, rows and channels; tiles
    # become items, pixels and channels.
    "tile_video": lambda ir, node: 3 if ir and ir[0] == 5 else None,
    "untile_video": lambda ir, node: 3 if ir else None,
})


def rank_out(op, node, in_ranks: Tuple[int, ...]) -> Optional[int]:
    """Output rank derived FORWARD from the input ranks — the piece `dim_map`
    needs before it can drive shape propagation instead of checking it.

    Returns None when the op's rank isn't a function of its input ranks plus its
    declared metadata (an unwritten EXPLICIT rank rule, a SOURCE op with no
    declared shape). None means "ask the engine", never "assume preserved" —
    a wrong rank silently reinterprets every axis position downstream.
    """
    spec = _lookup(op)
    if spec is None:
        return None

    if getattr(spec, "name", None) in _EXPLICIT_RANK:
        return _EXPLICIT_RANK[spec.name](tuple(in_ranks), node)

    kind = _shape_kind(spec)

    if kind is ShapeKind.PERMUTE:
        # Rank is the permutation's length; it must also match the input, and a
        # disagreement is a mis-wired call site rather than a rank to report.
        perm = tuple(node.axes or spec.permute_to or ())
        if not perm:
            return None
        if in_ranks and len(perm) != in_ranks[0]:
            return None
        return len(perm)

    if kind in (ShapeKind.PRESERVE, ShapeKind.AXIS_REPLACE,
                ShapeKind.AXIS_SELECT,
                ShapeKind.AXIS_CONCAT):
        return in_ranks[0] if in_ranks else None

    if kind is ShapeKind.BROADCAST:
        return max(in_ranks) if in_ranks else None

    if kind is ShapeKind.CONTRACT:
        # (..., M, d_reduce)·(..., d_reduce, d_out) → (..., M, d_out): input0's rank survives, its
        # last axis swapped for input1's.
        return in_ranks[0] if in_ranks else None

    if kind is ShapeKind.SOURCE:
        # Zero-input: the rank is whatever the call site declared, else the
        # op's own output template.
        declared = node.out_shape
        if declared:
            return len(declared)
        return len(spec.shape_out) or None

    return None


def dim_map(op, node, *,
            in_ranks: Optional[Tuple[int, ...]] = None,
            in_shapes: Optional[Sequence] = None,
            ) -> Optional[Tuple[DimSource, ...]]:
    """Per-output-dim source map, or None when it can't be derived (missing
    ranks / unhandled kind). Positional — robust to symbol concretization.

    Forward mode (no `shape_out` on the node yet) supplies `in_ranks` /
    `in_shapes` from the walk; checking mode reads ranks off the node.

    Ops whose geometry is not what `shape_kind` says — `dsa_attention_score`
    is EXPLICIT because the last axis is the bound sparse `s_kv`, not a copy
    of K — get an explicit `_EXPLICIT_DM` handler. Softmax / apply keep
    PRESERVE / CONTRACT; those kinds already describe them."""
    spec = _lookup(op)
    if spec is None:
        return None
    if in_ranks is None and in_shapes is not None:
        in_ranks = tuple(len(s) if s else 0 for s in in_shapes)
    rout = _rank_out(node)
    if rout is None:
        rout = rank_out(op, node, in_ranks or ())
    if rout is None:
        return None
    r_in0 = _input_rank(node, 0, in_ranks)

    if getattr(spec, "name", None) in _EXPLICIT_DM:
        if r_in0 is None:
            return None
        return tuple(_EXPLICIT_DM[spec.name](node, r_in0, rout))

    kind = _shape_kind(spec)
    if kind is ShapeKind.BROADCAST:
        refs = node.inputs_obj
        n_ports = len(refs) or (len(in_shapes) if in_shapes else 0)
        shapes = tuple(_input_shape(node, index, in_shapes)
                       for index in range(n_ports))
        if shapes and all(shape is not None for shape in shapes):
            try:
                axes = elementwise_broadcast_axes(shapes)
            except ValueError:
                return None
            if len(axes) != rout:
                return None
            return tuple(
                DimSource("broadcast", axis.sources) for axis in axes
            )
        ranks = tuple(_input_rank(node, index, in_ranks)
                      for index in range(n_ports))
        if not ranks or any(rank is None for rank in ranks):
            return None
        return tuple(
            DimSource(
                "broadcast",
                tuple(
                    (index, output_axis - (rout - rank))
                    for index, rank in enumerate(ranks)
                    if output_axis >= rout - rank
                ),
            )
            for output_axis in range(rout)
        )

    # The signature path, where the op's templates describe these tensors. It
    # subsumes the dispatch below — every kind here is a way of writing labels —
    # and gives cross-input identity for free, which the dispatch cannot. Falls
    # through when the templates do not fit, because a signature that does not
    # fit is not evidence.
    sig = signature_for(spec, node, rout)
    if sig is not None:
        return dim_map_from_signature(sig)

    if kind is ShapeKind.PRESERVE:
        return tuple(DimSource("from", ((0, j),)) for j in range(rout))

    if kind is ShapeKind.AXIS_CONCAT:
        # `axis_locked` first, as in AXIS_REPLACE and AXIS_SELECT below: an op
        # whose joined axis is part of its definition — `kv_stream_concat`
        # joins the sequence — says so once, rather than relying on every call
        # site to repeat it.
        locked = (spec.axis_locked if spec is not None
                  and spec.axis_locked is not None
                  else (node.axis if node.axis is not None
                        else -1))
        ax = _norm(locked, rout)
        # EVERY input contributes, not just the first. A concat's axes are
        # shared across its operands by definition — the concat axis sums over
        # them and the others must agree — so naming only input 0 reports every
        # later operand's axes as unreferenced. A consumer asking "did my
        # sharded axis survive this concat" would be told it was consumed, and
        # put an all_gather on an edge that needs nothing.
        n_in = (len(node.inputs_obj)
                or (len(in_ranks) if in_ranks else 1))
        return tuple(
            DimSource("concat" if j == ax else "from",
                      tuple((i, j) for i in range(n_in)))
            for j in range(rout))

    if kind is ShapeKind.PERMUTE:
        perm = tuple(node.axes or spec.permute_to or ())
        if len(perm) != rout:
            return None
        return tuple(DimSource("from", ((0, perm[j]),)) for j in range(rout))

    if kind is ShapeKind.AXIS_REPLACE:
        locked = (spec.axis_locked if spec.axis_locked is not None
                  else (node.axis if node.axis is not None else -1))
        lp = _norm(locked, rout)
        # `AXIS_REPLACE` covers two different things, and `is_view_op` is
        # already the declaration that tells them apart:
        #
        #   a rewrite   a linear's output axis is a NEW feature space; the
        #               input axis was contracted away and nothing of it
        #               survives.
        #   a narrowing a slice's output axis is a SUBSET of the input's — the
        #               same index with fewer elements. A sharded axis stays
        #               sharded through it, and calling it `new` reported it as
        #               consumed, which is three of DeepSeek-V3's collectives
        #               that should not exist.
        #
        # Rewrites and gather are not view ops and keep `new`: their output axis
        # is a new index space. Bounded selections such as top-k use
        # `AXIS_SELECT` below so the input extent remains part of shape inference.
        narrowing = bool(spec.is_view_op)
        replaced = (DimSource("from", ((0, lp),), dim_key=spec.out_key)
                    if narrowing else DimSource("new", dim_key=spec.out_key))
        return tuple(
            replaced if j == lp else DimSource("from", ((0, j),))
            for j in range(rout))

    if kind is ShapeKind.AXIS_SELECT:
        locked = (spec.axis_locked if spec.axis_locked is not None
                  else (node.axis if node.axis is not None else -1))
        lp = _norm(locked, rout)
        selected = DimSource("select", ((0, lp),), dim_key=spec.out_key)
        return tuple(
            selected if j == lp else DimSource("from", ((0, j),))
            for j in range(rout))

    if kind is ShapeKind.CONTRACT:
        # (..., M, d_reduce)·(..., d_reduce, d_out) → (..., M, d_out): leading + M
        # from input0, d_out from input1's last axis; d_reduce contracted.
        r_in1 = _input_rank(node, 1, in_ranks)
        last = (DimSource("from", ((1, r_in1 - 1),)) if r_in1
                else DimSource("new"))
        return (*tuple(DimSource("from", ((0, j),)) for j in range(rout - 1)), last)

    if kind is ShapeKind.SOURCE:
        labels = tuple(str(x) for x in (spec.shape_out or ()))
        if labels and len(labels) == rout and all(lbl.isidentifier() for lbl in labels):
            return tuple(DimSource("new", dim_key=lbl) for lbl in labels)
        # parameter_tensor: rank from the call-site `shape:`, extents from
        # `_shape` in infer_shape.
        return tuple(DimSource("new") for _ in range(rout))

    return None


def infer_shape(op, node, in_shapes, dims=None) -> Optional[Tuple]:
    """Infer the output shape from the input shapes via the DimMap. A `from`
    copies the input dim, a `merge` multiplies its inputs, a `concat` sums the
    axis across inputs, a `split`/`new` reads its size from `dims[dim_key]`, and
    a `select` takes the minimum of its input extent and configured bound.

    Returns None when the DimMap can't determine a dim (an unmapped op, or a
    `split`/`new` whose `dim_key` is absent from `dims`). The evaluator
    applies this map."""
    spec = _lookup(op)
    if spec is not None and _shape_kind(spec) is ShapeKind.BROADCAST:
        from backend.arch.ops import elementwise_broadcast_shape
        return elementwise_broadcast_shape(in_shapes)
    in_ranks = tuple(len(s) if s else 0 for s in in_shapes)
    dm = dim_map(op, node, in_ranks=in_ranks, in_shapes=in_shapes)
    if dm is None:
        return None
    out = []
    for src in dm:
        # `dim_key` outranks the kind wherever it is set. It is documented as
        # "the dim symbol whose resolved value gives this output dim's SIZE",
        # and a narrowing needs both halves at once: `from` says the output axis
        # is the SAME INDEX as its input (so a shard rides through it), while
        # the key says the extent is smaller. Reading only the ref made a slice
        # infer its input's width.
        if src.kind == "select":
            if (not src.refs or not dims or src.dim_key is None
                    or src.dim_key not in dims):
                return None
            i, j = src.refs[0]
            if i >= len(in_shapes) or j >= len(in_shapes[i]):
                return None
            from backend.arch.eval.dim import minimum
            out.append(minimum(in_shapes[i][j], dims[src.dim_key]))
        elif src.kind == "const":
            from backend.arch.eval.dim import Dim
            value = src.part if src.part else 1
            name = src.dim_key
            out.append(Dim.symbol(name, value) if name else Dim.of(value))
        elif src.kind in ("ceil_div", "div", "mul"):
            extent = _scaled_extent(src, in_shapes, dims)
            if extent is None:
                return None
            out.append(extent)
        elif src.kind == "grid":
            if not dims or src.dim_key not in dims:
                return None
            cells = 1
            for i, j in src.refs:
                if i >= len(in_shapes) or j >= len(in_shapes[i]):
                    return None
                cells = cells * _scaled_extent(
                    DimSource("div", ((i, j),), dim_key=src.dim_key),
                    in_shapes, dims)
            out.append(cells)
        elif src.kind in ("frames", "frame_grid"):
            if not dims or "temporal" not in dims or (
                    src.kind == "frame_grid" and "patch" not in dims):
                return None
            if any(i >= len(in_shapes) or j >= len(in_shapes[i])
                   for i, j in src.refs):
                return None
            extents = [in_shapes[i][j] for i, j in src.refs]
            # The temporal patches: `ceil(frames / temporal)`.
            frames_ref = src.refs[-1] if src.kind == "frames" else src.refs[0]
            patches = _scaled_extent(
                DimSource("ceil_div", (frames_ref,), dim_key="temporal"),
                in_shapes, dims)
            if src.kind == "frames":
                count = patches
                for extent in extents[:-1]:
                    count = count * extent
                out.append(count)
            else:
                cells = patches
                for ref in src.refs[1:]:
                    cells = cells * _scaled_extent(
                        DimSource("div", (ref,), dim_key="patch"),
                        in_shapes, dims)
                out.append(cells)
        elif src.kind == "extend":
            i, j = src.refs[0]
            if i >= len(in_shapes) or j >= len(in_shapes[i]):
                return None
            out.append(in_shapes[i][j] + src.part)
        elif src.dim_key is not None and dims and src.dim_key in dims:
            out.append(dims[src.dim_key])
        elif src.kind == "from":
            i, j = src.refs[0]
            if i >= len(in_shapes) or j >= len(in_shapes[i]):
                return None
            out.append(in_shapes[i][j])
        elif src.kind == "merge":
            prod = 1
            for i, j in src.refs:
                if i >= len(in_shapes) or j >= len(in_shapes[i]):
                    return None
                prod *= in_shapes[i][j]
            out.append(prod)
        elif src.kind == "concat":
            _i, j = src.refs[0]
            total = 0
            for s in in_shapes:
                if j >= len(s):
                    return None
                total += s[j]
            out.append(total)
        elif src.kind == "new" and dims and "_shape" in dims:
            # parameter_tensor: the call site wrote the extents in order.
            shape = dims["_shape"]
            if len(out) >= len(shape):
                return None
            out.append(shape[len(out)])
        else:  # split / new — size from dims
            if not dims or src.dim_key is None:
                return None
            if src.dim_key in dims:
                out.append(dims[src.dim_key])
                continue
            if src.dim_key.isidentifier():
                return None
            # An extent the op defines over its declared dims (a patch's
            # `channels * temporal * patch * patch` values).
            from backend.arch.eval.dim import eval_dim_expr
            out.append(eval_dim_expr(src.dim_key, dict(dims)))
    return tuple(out)


def compute_shape_out(
    op,
    shape_in: Sequence[Sequence[Any]],
    dims: Optional[Dict[str, Any]] = None,
    axis: Optional[int] = None,
    axes: Optional[Sequence[int]] = None,
) -> Optional[Tuple[Any, ...]]:
    """Output sizes from the DimMap, with a stand-in node for tests.

    The walk calls `infer_shape` with the real graph node. Tests that only
    have an op and upstream extents come here; OpSpec does not call back.
    """
    dims = dims or {}
    spec = _lookup(op)
    listed = (dims.get("_shape") if spec is not None
              and _shape_kind(spec) is ShapeKind.SOURCE else None)
    node = ShapeProbe.probe(
        axis=axis if axis is not None else getattr(op, "axis_locked", None),
        axes=axes if axes is not None else getattr(op, "permute_to", None),
        out_shape=listed if listed else None)
    return infer_shape(op, node, shape_in, dims)


def _scaled_extent(src: DimSource, in_shapes, dims) -> Any:
    """`refs[0]` scaled by `dims[dim_key] ** exp` — the written pool / merge."""
    if (not src.refs or not dims or src.dim_key is None
            or src.dim_key not in dims):
        return None
    i, j = src.refs[0]
    if i >= len(in_shapes) or j >= len(in_shapes[i]):
        return None
    exp = src.part if src.part else 1
    factor = dims[src.dim_key] ** exp if exp != 1 else dims[src.dim_key]
    ref = in_shapes[i][j]
    if src.kind == "mul":
        return ref * factor
    # A division that drops its remainder keeps the floor on both halves
    # (`dim.floor_div`), as `ceil_div` keeps its ceiling: the axis's
    # identity is the law's own shape, `floor(img_h/s_patch)`.
    from backend.arch.eval.dim import ceil_div, floor_div
    if src.kind == "div":
        return floor_div(ref, factor)
    return ceil_div(ref, factor)


# Ops with a genuine fixed-rank input contract — a wrong rank here IS a
# mis-wired edge / missing reshape. The attention score/apply family (4-D
# Q/K/V) and matmul. Everything else is rank-polymorphic or irregular.


def edge_consistency_errors(op, in_shapes) -> List[str]:
    """Connectivity check: for a RANK-SENSITIVE op (one whose logic reads
    specific axis positions — the EXPLICIT attention / CONTRACT families), each
    wired input must have the rank the op's declared `shape_in` template
    expects. A wrong rank means a producer of the wrong shape is wired, or a
    reshape is missing upstream — the mis-wired-edge bug that the size checks
    miss (they only inspect whatever IS wired) and that arity validation (load
    time) doesn't see (the count can be right while a rank is wrong).

    Enforced only where the op says its `shape_in` rank is a CONTRACT rather
    than an illustration, via `OpSpec.input_rank_is_exact`. That distinction
    is real and measurable: eight ops in the corpus receive a rank their own
    `shape_in` does not show — `rmsnorm` declares three and is applied to
    rank-3 and rank-4 tensors alike, `mul` to rank 1, 3 and 4, `sum_reduce`
    to rank 4. Checking those would be nothing but false positives.

    It used to be a name list kept here, which meant the two readings of
    `shape_in` were distinguished nowhere the op could see, and an attention
    variant registered without being added to the list silently got the loose
    one."""
    spec = _lookup(op)
    if spec is None or not spec.input_rank_is_exact:
        return []
    tmpl = spec.shape_in
    if not tmpl:
        return []
    expected = ([len(t) for t in tmpl] if isinstance(tmpl[0], (tuple, list))
                else [len(tmpl)])
    errs: List[str] = []
    for i, rank in enumerate(expected):
        if i < len(in_shapes) and len(in_shapes[i]) != rank:
            errs.append(
                f"{spec.name} input {i} has rank {len(in_shapes[i])}, op expects "
                f"{rank} (mis-wired edge / missing reshape)")
    return errs


def axis_transfer(op, node, in_axis: int, input_index: int = 0) -> Optional[int]:
    """Output axis position carrying a shard entering on `in_axis` of
    `input_index`, or None when the op removes it. Derived by inverting
    `dim_map`: a head shard follows a `from`, the leading `split` part (whole
    heads), or a `merge`; it is dropped on contraction / per-head split / a
    rewritten feature."""
    dm = dim_map(op, node)
    if dm is None:
        return in_axis  # conservative pass-through when unmappable
    rin = _input_rank(node, input_index)
    ax = _norm(in_axis, rin) if rin else in_axis
    for j, src in enumerate(dm):
        if (input_index, ax) not in src.refs:
            continue
        if src.kind in ("from", "merge", "concat", "broadcast", "extend"):
            return j
        if src.kind == "split":
            return j if src.part == 0 else None  # head shard rides the count slot
    return None


def _gt(a, b) -> bool:
    """`a > b` for extents that may be symbolic; unknown comparisons are not
    errors."""
    try:
        return bool(a > b)
    except TypeError:
        return False


def shape_consistency_errors(op, node, in_shapes, out_shape) -> List[str]:
    """Check the DimMap's size relations against resolved shapes — the
    dim-correspondence cross-check the size-only `infer_shape` lacks.

    Returns a list of human-readable mismatches (empty = consistent). `from`
    must copy the input dim exactly; `merge` must equal the product of its
    inputs; `split` parts of one input dim must multiply back to it; `select`
    cannot exceed its input extent. `new` / `concat` are skipped."""
    spec = _lookup(op)
    if spec is not None and _shape_kind(spec) is ShapeKind.BROADCAST:
        from backend.arch.ops import elementwise_broadcast_shape
        try:
            expected = elementwise_broadcast_shape(in_shapes)
        except ValueError as error:
            return [str(error)]
        return ([] if expected == tuple(out_shape) else [
            f"broadcast output {tuple(out_shape)} != expected {expected}"
        ])
    dm = dim_map(op, node)
    if dm is None:
        return []
    errs: List[str] = []

    def _in(ii, idx):
        if ii < len(in_shapes) and idx < len(in_shapes[ii]):
            return in_shapes[ii][idx]
        return None

    # group split parts by their source input dim to check the product.
    split_groups: dict = {}
    for j, src in enumerate(dm):
        if j >= len(out_shape):
            continue
        o = out_shape[j]
        if src.kind == "from":
            ii, idx = src.refs[0]
            v = _in(ii, idx)
            if v is None or o is None:
                pass
            elif src.dim_key is not None:
                # A narrowing: the same index with fewer elements, which is what
                # a slice is. Equality is the wrong check here and the right one
                # is stronger — a narrowing that GREW is a real error, and
                # nothing said so before.
                if _gt(o, v):
                    errs.append(
                        f"dim {j}: narrowing in[{ii}][{idx}]={v} -> out={o} "
                        f"grew")
            elif v != o:
                errs.append(f"dim {j}: from in[{ii}][{idx}]={v} != out={o}")
        elif src.kind == "merge":
            prod = 1
            ok = True
            for ii, idx in src.refs:
                v = _in(ii, idx)
                if v is None:
                    ok = False
                    break
                prod *= v
            if ok and o is not None and prod != o:
                errs.append(f"dim {j}: merge {src.refs}={prod} != out={o}")
        elif src.kind == "split":
            split_groups.setdefault(src.refs[0], []).append(o)
        elif src.kind == "select":
            ii, idx = src.refs[0]
            v = _in(ii, idx)
            if v is not None and o is not None and _gt(o, v):
                errs.append(
                    f"dim {j}: selection from in[{ii}][{idx}]={v} "
                    f"grew to out={o}")
    for (ii, idx), parts in split_groups.items():
        v = _in(ii, idx)
        if v is None or any(p is None for p in parts):
            continue
        prod = 1
        for p in parts:
            prod *= p
        if prod != v:
            errs.append(f"split parts {parts} of in[{ii}][{idx}]={v} product={prod}")
    return errs
