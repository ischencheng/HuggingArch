"""Transpose, view, concat, slice, and broadcast layout handlers."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from backend.arch.spec_tree.nodes import SpecIRNode
    from .propagation import LayoutPropagation

from backend.arch import dims
from backend.arch.axis_propagation import dim_map
from backend.arch.eval.dim import (
    floored,
    is_whole,
    require_multiple,
    same_law,
    value_of,
)
from backend.arch.eval.refusal import LayoutRefusal

from .attention import AttentionForm
from .edge import EdgeState
from .layout import (
    Block,
    Replicated,
    Segmented,
    axis_fully_replicated,
    block,
    sharded,
)
from .transfer_table import (
    BROADCAST,
    Derived,
    CONCAT,
    MERGE,
    PRESERVE,
    SLICE,
    SPLIT,
    transfer,
)
from .weight_endpoints import _eval_tracked, _slot_source



def _whole(extent: Any) -> Any:
    """An axis extent as the whole number a layout indexes — or, for an
    extent in the workload's symbols, the expression it is."""
    return floored(extent)

def geometry_from_dim_map(
        rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], EdgeState]:
    """Apply the op's DimMap through the transfer table.

    ShapeKind compiled the correspondence; this is the same map the
    walk used for extents, now asked of AxisLayouts.
    """
    in_ranks = tuple(
        len(st.layout) if st is not None else (len(s) if s else 0)
        for st, s in zip(
            states,
            list(in_shapes) + [()] * max(0, len(states) - len(in_shapes)),
            strict=False,
        )
    )
    dm = dim_map(node.op, node, in_ranks=in_ranks, in_shapes=in_shapes)
    if not dm:
        raise ValueError(
            f"{node.display_path}: no DimMap for geometry op {node.op!r}")
    states = list(states)
    binding = rule.env.get(node.uid) or {}
    reference = next((s for s in states if s is not None), None)
    if reference is None:
        raise ValueError(
            f"{node.display_path}: geometry op arrived with no state")
    layouts: List[Any] = []
    i = 0
    while i < len(dm):
        src = dm[i]
        if src.kind == "from":
            inp, ax = src.refs[0]
            st = states[inp]
            if st is None:
                raise ValueError(
                    f"{node.display_path}: DimMap from input {inp} "
                    "has no arriving state")
            cell = rule.cell(
                node, f"preserve axis {ax}",
                transfer(PRESERVE, (st.layout[ax],)))
            layouts.append(cell.out_axes[0])
            i += 1
        elif src.kind == "split":
            inp, ax = src.refs[0]
            parts = []
            while (i < len(dm) and dm[i].kind == "split"
                   and dm[i].refs == src.refs):
                parts.append(dm[i])
                i += 1
            try:
                values = tuple(value_of(binding[p.dim_key]) for p in parts)
            except KeyError as exc:
                raise ValueError(
                    f"{node.display_path}: split factor {exc} not in "
                    "the node's dim environment") from exc
            st = states[inp]
            if not all(is_whole(v) for v in values):
                # An average extent — the windows a decode step completes,
                # `s_ctx / compress_ratio` — indexes no element, so no cut
                # of it is a layout: the folded axis arrives whole and every
                # factor stays whole.
                if not axis_fully_replicated(st.layout[ax]):
                    raise ValueError(
                        f"{node.display_path}: split into average extents "
                        f"{values} of a cut axis {ax}")
                layouts.extend(rule.replicated() for _ in parts)
                continue
            extents = tuple(_whole(v) for v in values)
            split_extent = _whole(in_shapes[inp][ax])
            cell = rule.cell(
                node, "split",
                transfer(SPLIT, (st.layout[ax],), extent=split_extent,
                         output_extents=extents))
            layouts.extend(cell.out_axes)
        elif src.kind == "merge":
            # The merged axis is row-major, so a cut on an inner factor
            # under a whole outer one would leave each rank strided
            # intervals, which no layout names. The inner factor is
            # required whole first — the all-gather a column-split
            # producer performs before a flatten (vLLM's
            # `gather_output`) — and the walk prices it.
            axes_in = []
            for inp, ax in src.refs:
                axis = states[inp].layout[ax]
                if axis_fully_replicated(axis):
                    axis = rule.replicated()
                elif (axes_in and not isinstance(axis, Segmented)
                      and all(isinstance(outer, Replicated)
                              for outer in axes_in)):
                    states[inp] = rule.whole_axis(
                        node, states[inp], ax, None,
                        f"axis {ax} (inner merge factor)")
                    axis = rule.replicated()
                axes_in.append(axis)
            cell = rule.cell(
                node, "merge", transfer(MERGE, tuple(axes_in)))
            layouts.extend(cell.out_axes)
            i += 1
        elif src.kind in ("new", "grid", "frame_grid"):
            # A grid counts cells of input axes the handler required whole.
            layouts.append(rule.replicated())
            i += 1
        elif src.kind == "frames":
            # An item axis times a video's temporal patches, which the
            # handler required whole: the item axis's layout, when whole.
            if not all(axis_fully_replicated(states[inp].layout[ax])
                       for inp, ax in src.refs[1:]):
                raise ValueError(
                    f"{node.display_path}: a video's frames arrive cut")
            cell = rule.cell(node, "merge", transfer(MERGE, tuple(
                states[inp].layout[ax] if k == 0 else rule.replicated()
                for k, (inp, ax) in enumerate(src.refs[:2]))))
            layouts.extend(cell.out_axes)
            i += 1
        elif src.kind in ("mul", "div", "ceil_div"):
            # A scaled axis is a whole axis scaled: the handler required
            # its input whole (`group_items`' rows grow by the items laid
            # end to end).
            (inp, ax), = src.refs
            if not axis_fully_replicated(states[inp].layout[ax]):
                raise ValueError(
                    f"{node.display_path}: axis {ax} is scaled "
                    f"({src.kind}) but arrives cut")
            layouts.append(rule.replicated())
            i += 1
        else:
            raise ValueError(
                f"{node.display_path}: geometry DimMap kind "
                f"{src.kind!r} is not a layout transfer")
    return states, EdgeState(
        tuple(layouts), reference.value, reference.workload)


def transpose(rule: LayoutPropagation, node: SpecIRNode, states: List[Optional[EdgeState]],
              in_shapes: Sequence[Any]
              ) -> Tuple[List[EdgeState], EdgeState]:
    _index, state = rule.sole(node, states)
    perm = tuple(node.axes or ())
    if sorted(perm) != list(range(len(state.layout))):
        raise ValueError(
            f"{node.display_path}: transpose permutation {perm!r} does "
            f"not match rank {len(state.layout)}")
    if state.workload is not None and perm[:2] != (0, 1):
        raise ValueError(
            f"{node.display_path}: routed tensors keep semantic axes "
            f"(experts, rows, feature); transpose {perm!r} "
            "reorders them")
    return geometry_from_dim_map(rule, node, states, in_shapes)


def view_split(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """Split the declared axis (`node.axis`; there is no default) in two.

    The trailing axis is a feature axis factoring into (heads, head width):
    the SPLIT cell tracks a shard of it onto the outer factor. Any other
    axis is a sequence folding into (slots, window) — DSv4's compressor —
    and a cut of it falls wherever the ranks' chunks end, not on window
    boundaries (and the overlap variant's windows read the previous one),
    so that axis is required whole first, the gather priced by the walk,
    exactly as `spatial_merge` treats its patch window. Either way the DimMap
    names the split axis and the factors come out of the same SPLIT cell.
    """
    index, state = rule.sole(node, states)
    if state.workload is not None:
        raise ValueError(
            f"{node.display_path}: a routed tensor cannot change rank "
            "inside the dispatch/combine zone")
    rank = len(state.layout)
    if node.axis is None:
        raise ValueError(
            f"{node.display_path}: view_split declares no `axis:`")
    ax = int(node.axis)
    ax = ax + rank if ax < 0 else ax
    if not 0 <= ax < rank:
        raise ValueError(
            f"{node.display_path}: view_split axis {node.axis} outside "
            f"rank {rank}")
    if ax != rank - 1:
        states = list(states)
        states[index] = rule.whole_axis(
            node, state, ax, None, f"axis {ax} (folded windows)")
    return geometry_from_dim_map(rule, node, states, in_shapes)


def shift_axis(rule: LayoutPropagation, node: Any,
               states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """`y[.., i, ..] = x[.., i - 1, ..]` along `node.axis`: position i reads
    its predecessor on the same axis, so a cut of that axis would leave each
    rank's first position on its neighbour. The axis is required whole — the
    gather priced by the walk — and every other axis passes through."""
    index, state = rule.sole(node, states)
    if state.workload is not None:
        raise ValueError(
            f"{node.display_path}: a routed tensor cannot be shifted inside "
            "the dispatch/combine zone")
    rank = len(state.layout)
    if node.axis is None:
        raise ValueError(f"{node.display_path}: shift declares no `axis:`")
    ax = int(node.axis)
    ax = ax + rank if ax < 0 else ax
    if not 0 <= ax < rank:
        raise ValueError(
            f"{node.display_path}: shift axis {node.axis} outside rank {rank}")
    need = rule.whole_axis(node, state, ax, None, f"axis {ax} (shifted)")
    required = list(states)
    required[index] = need
    return required, EdgeState(need.layout, need.value, need.workload,
                               need.provenance)


def view_merge(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    _index, state = rule.sole(node, states)
    if state.workload is not None:
        raise ValueError(
            f"{node.display_path}: a routed tensor cannot change rank "
            "inside the dispatch/combine zone")
    return geometry_from_dim_map(rule, node, states, in_shapes)


def spatial_merge(rule: LayoutPropagation, node: Any,
                  states: List[Optional[EdgeState]],
                  in_shapes: Sequence[Any]
                  ) -> Tuple[List[EdgeState], EdgeState]:
    """Fold each contiguous m-by-m patch window into the feature axis."""
    index, state = rule.sole(node, states)
    if len(state.layout) < 2 or not in_shapes or not in_shapes[0] \
            or len(in_shapes[0]) != len(state.layout):
        raise ValueError(
            f"{node.display_path}: spatial_merge needs an input with "
            "sequence and feature axes")
    binding = rule.env.get(node.uid) or {}
    factor = binding.get("n_spatial_merge")
    if factor is None:
        raise ValueError(
            f"{node.display_path}: spatial_merge has no resolved "
            "n_spatial_merge dimension")
    window = int(value_of(factor)) ** 2
    if window <= 0:
        raise ValueError(
            f"{node.display_path}: spatial window must be positive, "
            f"got {window}")

    need = rule.whole_axis(
        node, state, len(state.layout) - 2, None,
        "rows (m*m patch window)")
    # Pixel-unshuffle forms [..., S/m^2, m^2, C] before flattening the
    # two trailing axes.  A C cut would become strided after that merge,
    # which the contiguous Block vocabulary deliberately cannot express.
    if not isinstance(need.layout[-1], Replicated):
        need = EdgeState(
            (*need.layout[:-1], rule.replicated()), need.value,
            need.workload, need.provenance)
    feature = rule.cell(
        node, "merge of spatial window and feature axes",
        transfer(MERGE, (rule.replicated(), need.layout[-1])))
    required = list(states)
    required[index] = need
    return required, EdgeState(
        (*need.layout[:-1], *feature.out_axes), need.value,
        need.workload, need.provenance)


def slice_axis(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """`(.., in, ..)` -> `(.., out, ..)`: SLICE of `node.axis` (default the
    trailing axis), the same axis the DimMap narrows.

    `dims.start` may declare the exact zero-based interval offset.  Older
    graphs that declare only the slice width remain representable when
    segment identity cannot affect placement: every contiguous segment
    run of that width is tried and all candidates must derive the same
    layout. Distinguishable candidates without `start` are a hard error,
    never a guess. A whole-axis slice is the identity for every
    constructor. A proper subinterval of a Block/Product cut cannot stay
    cut (the ranks would hold unequal pieces), so the axis is required
    whole — the gather is priced by the walk — and the slice is taken of
    the replicated axis. An empty interval is Replicated (the table's
    empty-range rule).
    """
    _index, state = rule.sole(node, states)
    if not in_shapes or not in_shapes[0] \
            or len(in_shapes[0]) != len(state.layout):
        raise ValueError(
            f"{node.display_path}: slice input shape does not match "
            "the arriving state's rank")
    ax = -1 if node.axis is None else int(node.axis)
    ax = ax + len(state.layout) if ax < 0 else ax
    if not 0 <= ax < len(state.layout):
        raise ValueError(
            f"{node.display_path}: slice axis {node.axis} outside rank "
            f"{len(state.layout)}")
    in_extent = _whole(in_shapes[0][ax])
    # A slice may be empty (no complete window in a decode step), so its
    # width is a non-negative length, not a weight endpoint's positive one.
    # One that gives `start` and no `out` keeps the rest of the axis
    # (`axis_propagation._dm_slice`).
    own = node.dims or {}
    if "out" not in own and "start" in own:
        out_extent = in_extent - _whole(own["start"])
    else:
        out_extent = _whole(_eval_tracked(
            _slot_source(node, "out"), rule.env.get(node.uid) or {}))
    if out_extent < 0:
        raise ValueError(
            f"{node.display_path}: slice width {out_extent} is negative")
    axis = state.layout[ax]
    binding = rule.env.get(node.uid) or {}
    declared_start = binding.get("start")
    if declared_start is not None:
        start = _whole(declared_start)
        if start < 0 or start + out_extent > in_extent:
            raise ValueError(
                f"{node.display_path}: slice interval [{start}, "
                f"{start + out_extent}) exceeds extent {in_extent}")
        result = transfer(SLICE, (axis,), start=start,
                          stop=start + out_extent, extent=in_extent)
        if not isinstance(result, Derived) and not isinstance(axis, Segmented):
            # The same lowering as an undeclared offset below: a proper
            # subinterval of an evenly cut axis (an all-gathered
            # column-parallel output split into its parts) is taken of the
            # axis gathered whole.
            state = rule.whole_axis(
                node, state, ax, None, f"axis {ax} (uneven slice)")
            states = list(states)
            states[_index] = state
            result = transfer(SLICE, (state.layout[ax],), start=start,
                              stop=start + out_extent, extent=in_extent)
        cell = rule.cell(node, f"slice of axis {ax}", result)
        (out_axis,) = cell.out_axes
    elif isinstance(axis, Segmented) and not same_law(out_extent, in_extent):
        starts = []
        for i in range(len(axis.segments)):
            total = 0
            for _offset, extent, _child in axis.segments[i:]:
                total += extent
                if total >= out_extent:
                    break
            if total == out_extent:
                starts.append(axis.segments[i][0])
        derived = {
            rule.cell(node, "segment slice",
                       transfer(SLICE, (axis,), start=start,
                                stop=start + out_extent,
                                extent=in_extent)).out_axes
            for start in starts}
        if not derived:
            raise ValueError(
                f"{node.display_path}: slice width {out_extent} does "
                "not tile any run of the declared segments of "
                f"{axis!r}")
        if len(derived) > 1:
            raise ValueError(
                f"{node.display_path}: slice width {out_extent} "
                "matches segment runs with different layouts and the "
                "spec declares no offset to pick one — cannot identify "
                "the segment")
        (out_axis,) = next(iter(derived))
    else:
        result = transfer(SLICE, (axis,), start=0, stop=out_extent,
                          extent=in_extent)
        if not isinstance(result, Derived) and not isinstance(axis, Segmented):
            # A prefix of an evenly cut axis (DSv4's complete windows of a
            # sequence under CP) leaves the ranks uneven; the representable
            # lowering gathers the axis whole and slices that.
            index, _ = rule.sole(node, states)
            state = rule.whole_axis(
                node, state, ax, None, f"axis {ax} (uneven slice)")
            states = list(states)
            states[index] = state
            result = transfer(SLICE, (state.layout[ax],), start=0,
                              stop=out_extent, extent=in_extent)
        cell = rule.cell(node, f"slice of axis {ax}", result)
        (out_axis,) = cell.out_axes
    layout = list(state.layout)
    layout[ax] = out_axis
    return list(states), \
        EdgeState(tuple(layout), state.value, state.workload)


def broadcast(rule: LayoutPropagation, node: SpecIRNode, states: List[Optional[EdgeState]],
              in_shapes: Sequence[Any]
              ) -> Tuple[List[EdgeState], EdgeState]:
    del in_shapes
    _index, state = rule.sole(node, states)
    if state.workload is not None:
        raise ValueError(
            f"{node.display_path}: a routed tensor cannot change rank "
            "inside the dispatch/combine zone")
    rank_out = len(state.layout) + 1
    position = node.axis
    position = -1 if position is None else int(position)
    if position < 0:
        position += rank_out
    if not 0 <= position < rank_out:
        raise ValueError(
            f"{node.display_path}: broadcast axis {position} outside "
            f"output rank {rank_out}")
    binding = rule.env.get(node.uid) or {}
    width = binding.get("width")
    names = tuple(getattr(width, "names", ()) or ())
    world_symbols = [name for name in names if dims.world_of(name)]
    if len(world_symbols) > 1:
        raise ValueError(
            f"{node.display_path}: broadcast width {names!r} names more than "
            f"one rank-group axis {world_symbols!r}; which one the inserted "
            "axis is owned by cannot be inferred — declare the width on one "
            "symbol")
    # No world symbol means no rank group splits the inserted axis: it is
    # replicated (universe). That is the declaration's meaning, not a guess.
    counterpart = None
    if len(world_symbols) == 1:
        symbol = world_symbols[0]
        extent = binding.get(symbol, width)
        counterpart = rule.symbol_layout(symbol, _whole(extent))
    cell = (transfer(BROADCAST, (), scope=rule.universe)
            if counterpart is None
            else transfer(BROADCAST, (counterpart,)))
    inserted = rule.cell(
        node, "broadcast insertion", cell).out_axes[0]
    axes = (*state.layout[:position], inserted,
            *state.layout[position:])
    return list(states), EdgeState(axes, state.value, state.workload)


def one_hot(rule: LayoutPropagation, node: Any,
            states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
            ) -> Tuple[List[EdgeState], EdgeState]:
    """`(.., in)` -> `(.., out)`: a constant whose leading axes are the
    input's. They keep the input's layout — each rank builds the rows it
    holds — and the filled trailing axis is whole on every rank, as an
    axis `broadcast` inserts with no rank group. No value of the input is
    read, so the output is complete."""
    del in_shapes
    _index, state = rule.sole(node, states)
    filled = rule.cell(node, "one-hot axis",
                       transfer(BROADCAST, (), scope=rule.universe)
                       ).out_axes[0]
    return list(states), EdgeState(
        (*state.layout[:-1], filled), None, state.workload)


def concat(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
           in_shapes: Sequence[Any]
           ) -> Tuple[List[EdgeState], EdgeState]:
    """Join `node.axis` (default the trailing one) — `concat`, packing
    features, or DSv4's overlap compressor stacking two row runs on its window
    axis. The shape law reads the same `axis:`; a layout that joined a
    different axis than the shape would describe a tensor nobody computed."""
    return _concat_on(rule, node, states, in_shapes,
                      -1 if node.axis is None else int(node.axis))


def kv_stream_concat(rule: LayoutPropagation, node: Any,
                     states: List[Optional[EdgeState]],
                     in_shapes: Sequence[Any]
                     ) -> Tuple[List[EdgeState], EdgeState]:
    """Join the SEQUENCE axis — `kv_stream_concat`, unioning cache streams.

    Cached tensors leave their producer whole under prefill CP, so both
    operands arrive replicated on that family. A cut here means the
    producer gather did not run.
    """
    return _concat_on(rule, node, states, in_shapes, -2,
                      collapse_replicated=True)


def media_rows(rule: LayoutPropagation, node: Any,
               states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """`patchify`, `pad_rows`, `unpad`, `group_items`: a processor (or the
    model, laying items' rows end to end) regrouping one item's own
    axes — its image sides into a patch grid, its rows padded or stripped.
    Those axes arrive whole (an encoder's rows belong to no sequence world);
    the item axis keeps the request's layout, and the rest follow the
    DimMap."""
    whole = {"patchify": tuple(range(1, len(in_shapes[0]))),
             "pad_rows": (1,), "unpad": (1,),
             "group_items": (0, 1)}[node.op]
    states = list(states)
    for index, state in enumerate(states):
        if state is None:
            continue
        for axis in whole:
            if axis < len(state.layout):
                states[index] = rule.whole_axis(
                    node, states[index], axis, None,
                    f"axis {axis} (an item's own axis)")
    return geometry_from_dim_map(rule, node, states, in_shapes)


def sequence_window(rule: LayoutPropagation, node: Any,
                    states: List[Optional[EdgeState]],
                    in_shapes: Sequence[Any]
                    ) -> Tuple[List[EdgeState], EdgeState]:
    """`(.., s_ctx, d)` -> `(.., Min(s_ctx, s_window), d)`: eviction.

    Prefill CP gathers a cached tensor at its producer, so the window
    sees whole positions. A cut here is not a local narrowing — except the
    one invocation seam Ulysses owns: a main-attention K/V view takes its
    resident rows, computes the decompression chain locally, and hands the
    sequence cut to the attention interior's all-to-all.
    """
    _index, state = rule.sole(node, states)
    if len(state.layout) < 2:
        raise ValueError(
            f"{node.display_path}: a windowed KV stream needs a sequence "
            f"axis, and the arriving state has rank {len(state.layout)}")
    axis = state.layout[-2]
    if node.uid in rule.head_repartition_kv_views:
        family = rule.resident_sequence_family
        if (rule.attention.form is not AttentionForm.HEAD_REPARTITION
                or family is None):
            raise ValueError(
                f"{node.display_path}: a head-repartition K/V view has no "
                "active Ulysses plan")
        if not axis_fully_replicated(axis):
            raise ValueError(
                f"{node.display_path}: the cached producer did not publish "
                f"whole rows ({axis!r})")
        input_shape = in_shapes[0] if in_shapes else None
        if not input_shape:
            raise ValueError(
                f"{node.display_path}: a head-repartition K/V view has no "
                "input shape")
        require_multiple(
            input_shape[-2], family.group_size,
            lambda extent: ValueError(
                f"{node.display_path}: cache length {extent} does not "
                f"divide the ulysses degree {family.group_size}"))
        layout = list(state.layout)
        layout[-2] = sharded(family)
        return list(states), EdgeState(
            tuple(layout), state.value, state.workload,
            state.provenance)
    if not axis_fully_replicated(axis):
        raise LayoutRefusal(
            f"{node.display_path}: the window's sequence axis is cut "
            f"({axis!r}); a cached tensor leaves its producer whole, so "
            "eviction is local")
    return list(states), state


def scatter_rows(rule: LayoutPropagation, node: Any,
                 states: List[Optional[EdgeState]],
                 in_shapes: Sequence[Any]
                 ) -> Tuple[List[EdgeState], EdgeState]:
    """`x: [B, S, D]`, `v: [B, N, D]` (one or more, each media input's rows,
    as one masked scatter writes them all) -> `[B, S, D]` in `x`'s layout,
    for `scatter_rows_replace` and `scatter_rows_add` alike.

    Every operand arrives in one reduction state. `replace` has no law in
    `foldlaw`, so the walk has finalized any pending sum before this runs;
    `add` has one, so the same pending sum on both sides rides through.
    B and D follow the add law: each `v` must arrive placed as `x` is. The
    row axes are the op's own. With `x`'s sequence whole, `v`'s rows are
    whole. With the sequence cut over a world W, rank r holds the
    placeholder rows that fall in its sequence block — under the
    uniform-density co-shard projection, `v`'s N axis cut in the same world
    as a padded Block (`N_rank = ceil(N / W)`). A replicated `v` reaches
    that by a local slice. Any other placement is refused, not
    redistributed.
    """
    if len(states) < 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: a row scatter reads x and v, and every "
            "operand must arrive with a layout")
    x, *vs = states
    if len(in_shapes) != len(states) or any(
            shape is None or len(shape) != 3 for shape in in_shapes) or any(
            len(state.layout) != 3 for state in states):
        raise ValueError(
            f"{node.display_path}: a row scatter reads x [B, S, D] and "
            "v [B, N, D]")
    required = [x]
    for v in vs:
        if x.workload is not None or v.workload is not None:
            raise ValueError(
                f"{node.display_path}: a row scatter cannot run inside a "
                "routed zone")
        if x.value != v.value:
            raise ValueError(
                f"{node.display_path}: x and v arrive in different reduction "
                f"states ({x.value!r} vs {v.value!r})")
        rule.align(node, x.layout[0], v.layout[0], "batch axis")
        rule.align(node, x.layout[2], v.layout[2], "feature axis")
        sequence = x.layout[1]
        if axis_fully_replicated(sequence):
            rows = v.layout[1]
            if not axis_fully_replicated(rows):
                raise ValueError(
                    f"{node.display_path}: v's rows are cut ({rows!r}) while "
                    "x's sequence is whole")
        elif isinstance(sequence, Block) and sequence.replication_degree == 1:
            rows = block(sequence.group, sequence.replica_cells, pad_to=1)
            if not axis_fully_replicated(v.layout[1]) and v.layout[1] != rows:
                raise ValueError(
                    f"{node.display_path}: v's rows arrive as "
                    f"{v.layout[1]!r}; the co-shard with x's sequence needs "
                    f"{rows!r} or a replicated v")
        else:
            raise ValueError(
                f"{node.display_path}: x's sequence placement {sequence!r} "
                "has no declared row co-shard")
        required.append(EdgeState((v.layout[0], rows, v.layout[2]), v.value,
                                  v.workload, v.provenance))
    return required, EdgeState(x.layout, x.value, x.workload, x.provenance)


def history_of(rule: LayoutPropagation, entry: EdgeState) -> EdgeState:
    """The layout of a history of `[B, S, D]` entries, `[B, S, m, D]`: each
    entry's own layout, with a whole history axis before the feature axis.
    What an entry is split by, the history is split by; nothing splits the
    entries apart, because every reader combines all of them."""
    if len(entry.layout) != 3:
        raise ValueError(
            f"a history entry is [B, S, D]; this one has rank "
            f"{len(entry.layout)}")
    return EdgeState(
        (*entry.layout[:2], rule.replicated(), entry.layout[2]),
        entry.value, entry.workload, entry.provenance)


def stack_init(rule: LayoutPropagation, node: Any,
               states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """`[B, S, D]` -> `[B, S, 1, D]`: a history of the one entry."""
    _index, entry = rule.sole(node, states)
    return list(states), history_of(rule, entry)


def stack_append(rule: LayoutPropagation, node: Any,
                 states: List[Optional[EdgeState]],
                 in_shapes: Sequence[Any]
                 ) -> Tuple[List[EdgeState], EdgeState]:
    """`[B, S, m, D]` ++ `[B, S, D]` -> `[B, S, m + 1, D]`, laid out as the
    history: the entry must arrive placed as the history's entries are."""
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: stack_append reads a history and an "
            "entry, and both must arrive with a layout")
    history, entry = states
    if len(history.layout) != 4 or len(entry.layout) != 3:
        raise ValueError(
            f"{node.display_path}: stack_append reads a history "
            "[B, S, m, D] and an entry [B, S, D]")
    if not axis_fully_replicated(history.layout[2]):
        raise ValueError(
            f"{node.display_path}: the history axis is cut "
            f"({history.layout[2]!r}); a history is whole along it")
    if (history.value != entry.value
            or history.workload != entry.workload):
        raise ValueError(
            f"{node.display_path}: the history and the entry arrive in "
            "different reduction states or workloads")
    for history_axis, entry_axis, what in (
            (0, 0, "batch axis"), (1, 1, "sequence axis"),
            (3, 2, "feature axis")):
        rule.align(node, history.layout[history_axis],
                   entry.layout[entry_axis], what)
    return list(states), EdgeState(
        history.layout, history.value, history.workload, history.provenance)


def _concat_on(rule: LayoutPropagation, node: Any,
               states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any], joined: int,
               collapse_replicated: bool = False
               ) -> Tuple[List[EdgeState], EdgeState]:
    if len(states) < 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: {node.op} expects resolved inputs")
    inputs = list(states)
    rank = len(inputs[0].layout)
    if any(len(state.layout) != rank for state in inputs):
        raise ValueError(f"{node.display_path}: {node.op} rank mismatch")
    if any(state.value != inputs[0].value for state in inputs[1:]):
        raise ValueError(
            f"{node.display_path}: {node.op} inputs carry mixed values")
    if any(state.workload != inputs[0].workload
           for state in inputs[1:]):
        raise ValueError(
            f"{node.display_path}: {node.op} inputs carry mixed workloads")

    # Every axis but the joined one has to AGREE across the operands; the
    # joined one is settled below. Which index that is depends on the caller,
    # so this is a skip rather than a `rank - 1` stop.
    ax = joined % rank
    required = list(states)
    out_layout: List[Any] = []
    for axis in range(rank):
        if axis == ax:
            out_layout.append(None)
            continue
        choices = [state.layout[axis] for state in inputs]
        cut = next((choice for choice in choices
                    if not axis_fully_replicated(choice)), None)
        if cut is None:
            out_layout.append(choices[0])
            continue
        if any(choice != cut and not axis_fully_replicated(choice)
               for choice in choices):
            raise ValueError(
                f"{node.display_path}: {node.op} unjoined axis {axis} has "
                f"incompatible owner maps {choices!r}")
        out_layout.append(cut)
        for index, choice in enumerate(choices):
            if choice == cut:
                continue
            if not isinstance(choice, Replicated):
                raise ValueError(
                    f"{node.display_path}: {node.op} unjoined axis {axis} "
                    f"cannot locally slice {choice!r} to {cut!r}")
            localized = list(required[index].layout)
            localized[axis] = cut
            required[index] = EdgeState(
                tuple(localized), required[index].value,
                required[index].workload)

    if len(in_shapes) != len(inputs) or any(not shape for shape in in_shapes):
        raise ValueError(
            f"{node.display_path}: {node.op} needs every input shape")
    extents = tuple(_whole(shape[ax]) for shape in in_shapes)
    joined_axes = [state.layout[ax] for state in inputs]
    # An operand empty along the joined axis contributes no position (a
    # vision tower's per-row newline tokens when the step carries no image):
    # the join is of the others, and of the first when every one is empty.
    # Empty is the number zero; an operand empty at some points only (a
    # pool of complete windows) is joined like any other, and holds no
    # position at the points where it comes to zero.
    present = [(axis, extent) for axis, extent in zip(joined_axes, extents,
                                                      strict=True)
               if not same_law(extent, 0)]
    if len(present) < len(joined_axes):
        present = present or [(joined_axes[0], extents[0])]
        joined_axes = [axis for axis, _extent in present]
        extents = tuple(extent for _axis, extent in present)
        if len(joined_axes) == 1:
            out_layout[ax] = joined_axes[0]
            return required, EdgeState(
                tuple(out_layout), inputs[0].value, inputs[0].workload)
    if (collapse_replicated
            and all(axis_fully_replicated(axis) for axis in joined_axes)
            and all(axis == joined_axes[0] for axis in joined_axes[1:])):
        # A segment run records WHICH operand each position came from, which
        # is what lets a later `slice` take MLA's rope half back out. A KV
        # union is never taken apart again — the attention reads it whole —
        # and every position lives on every rank either way, so the run would
        # be an identity nobody consults and a `Segmented` the KV-side plan
        # has no row for.
        out_axis = joined_axes[0]
    else:
        out_axis = rule.cell(
            node, "concat of joined axes",
            transfer(CONCAT, tuple(joined_axes), extents=extents)
        ).out_axes[0]
    out_layout[ax] = out_axis
    return required, EdgeState(
        tuple(out_layout), inputs[0].value, inputs[0].workload)
