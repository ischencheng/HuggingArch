"""axis_use — the tagged upgrade of `couples_on`, per (op, axis).

`couples_on` says WHICH axes an op couples along; it deliberately does
not say HOW. Layout needs the HOW — contract produces a Partial, normalize
demands the whole axis unless an online plan exists, window/recurrent
gate context sharding, route marks the seam pair — so every coupled axis
gets a category here. The registration GATE form is inherited unchanged:
the completeness tests assert this registry and `couples_on` agree
exactly (an axis declared coupled but untagged, or tagged but not
declared, is a hard failure), and custom ops are inside the gate.

Categories are the closed six of the design. An explicit `SEPARABLE`
entry is a reviewed decision that the coupling is NOT a dataflow
dependence of this op — attention's pairing axes (a joint [s_q, s_kv]
output over two sharded inputs) couple geometrically and are governed by
the attention kernel plan, not by axis_use. Geometry ops (concat/slice/
view_*/head_repeat) are exempt as a NAMED set: their coupling is the
transfer matrix's business.

The `param` of contract/normalize names the whole-axis statistic in op
terms (sum, softmax, rms, sinkhorn, topk…). This namespace is NOT the
Value-reducer vocabulary: a Partial only ever carries one of the closed
three reducers, and only an execution plan declaring an online form
turns a normalize into one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from backend.arch.axis_propagation import (ROW_SCATTER_AXES,
                                          SPATIAL_MERGE_AXES,
                                          VIEW_MERGE_AXES, dim_map)
from backend.arch.op_names import op_keyed
from backend.arch.schema.custom_ops import AXIS_USE_CATEGORIES
from backend.arch.shape_node import ShapeProbe

SEPARABLE = "separable"
CONTRACT = "contract"
NORMALIZE = "normalize"
WINDOW = "window"
RECURRENT = "recurrent"
ROUTE = "route"
CATEGORIES = (SEPARABLE, CONTRACT, NORMALIZE, WINDOW, RECURRENT, ROUTE)
assert CATEGORIES == AXIS_USE_CATEGORIES, "one vocabulary, declared in schema"


#: Contract statistics whose mergeable-summary schema is REGISTERED in the
#: closed payload vocabulary; only these may become a generic Partial under
#: sharding. A top-2 summary is mergeable too (algebraic in Gray et al.'s
#: aggregate classification — MaxN/MinN), but it has no registered schema,
#: so it demands the whole axis or a dedicated plan — structurally, not
#: via prose.
REGISTERED_MERGE_STATISTICS = ("sum", "online_attention")


@dataclass(frozen=True)
class AxisUse:
    category: str
    param: str = ""      # statistic / state schema / window reach
    plan: str = ""       # owning plan family, when distributed behavior
                         # is plan-owned (machine-readable, not prose)
    note: str = ""       # why, when the tag alone would surprise

    @property
    def admits_deferred_reduction(self) -> bool:
        """Whether the generic rule (sharded contract -> Partial) applies.
        False for exotic statistics and plan-owned axes: they require the
        whole axis or their plan, never a generic accumulator."""
        return (self.category == CONTRACT
                and self.param in REGISTERED_MERGE_STATISTICS
                and not self.plan)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"axis_use category {self.category!r} is not "
                             "in the closed vocabulary")
        if self.category in (CONTRACT, NORMALIZE) and not self.param:
            raise ValueError(f"{self.category} requires the whole-axis "
                             "statistic as param")
        if self.category == WINDOW and not self.param:
            raise ValueError("window requires its reach as param — conv "
                             "kernel, sliding window, pool ratio and RoPE "
                             "pairing are different windows")


def _c(param, note=""):
    return AxisUse(CONTRACT, param, note=note)


def _n(param, note=""):
    return AxisUse(NORMALIZE, param, note=note)


_R = AxisUse(ROUTE)


def _w(param, note=""):
    return AxisUse(WINDOW, param, note=note)
_PAIR = AxisUse(SEPARABLE, plan="attention_kernel",
                note="pairing axis of a joint attention output; sharding "
                     "it is the attention kernel plan's business "
                     "(online/pcp/ulysses), not a dataflow dependence of "
                     "this op")

#: Geometry ops: coupling consumed by the transfer matrix, no axis_use.
#: The exemption is an explicit AXIS MAP, not a name skip — if an exempted
#: op's couples_on drifts, the gate fires.
GEOMETRY_COUPLES = {
    "concat": ("in_a", "in_b"),
    "kv_stream_concat": ("s_a", "s_b"),
    "slice": ("in", "out"),
    "view_split": ("in",),
    "head_repeat": ("n_heads",),
}
#: Geometry ops that couple axes by POSITION of the arriving tensors, not by
#: label: their `couples_on` is empty, and the positions are the op's own
#: declaration (imported, never restated) — the axes its DimMap regroups, or,
#: for an op that regroups none, the axis at which its operands' templates
#: differ (a row scatter's x positions against v's rows: the row co-shard is
#: `geometry.scatter_rows`'s law).
GEOMETRY_POSITIONS = op_keyed({
    "view_merge": VIEW_MERGE_AXES,
    "spatial_merge": SPATIAL_MERGE_AXES,
    "scatter_rows_replace": ROW_SCATTER_AXES,
    "scatter_rows_add": ROW_SCATTER_AXES,
})
GEOMETRY_OPS = (*GEOMETRY_COUPLES, *GEOMETRY_POSITIONS)

#: op -> axis -> AxisUse, covering exactly the non-geometry couples_on set.
AXIS_USE = {
    # ---- contract family ------------------------------------------------
    "linear": {"in": _c("sum")},
    "linear_param": {"in": _c("sum")},
    "linear_external_weight": {"in": _c("sum")},
    "per_head_linear": {"in": _c("sum")},
    "embedding": {"n_vocab": _c("sum", "one-hot table contraction; "
                                       "vocab-parallel yields Partial(sum)")},
    "hash_embedding": {"n_vocab": _c("sum", "one-hot table contraction per "
                                            "hashed row; vocab-parallel "
                                            "yields Partial(sum)")},
    "sum_reduce": {"in": _c("sum")},
    "group_top2_sum": {"in": _c("top2_sum", "the axis disappears but the "
                                            "statistic is NOT plain sum: "
                                            "local-top2-then-sum of shards "
                                            "!= global top2 — sharded "
                                            "arrival demands the whole axis "
                                            "or a dedicated merge plan, "
                                            "never Partial(sum)")},
    "group_max": {"in": _c("group_max", "the axis disappears into one "
                                        "maximum per group: no registered "
                                        "merge schema, so a sharded arrival "
                                        "demands the whole axis")},
    "hc_mix": {"n_hidden_copies": _c("sum")},
    "attn_res_combine": {"Sar": _n("softmax"), "d_model": _c("sum")},
    "indexer_logits": {
        "d_indexer_head": _c("sum"),
        "n_indexer_heads": _c("sum", "head-weighted score folds the head "
                                     "axis in"),
        "s_query": _PAIR,
        "s_kv": _PAIR,
    },
    # ---- attention pairing + online chain --------------------------------
    "attention_score": {"d_head": _c("sum"), "s_query": _PAIR,
                        "s_kv": _PAIR},
    "attention_apply": {
        "s_kv": AxisUse(CONTRACT, "online_attention",
                        plan="attention_kernel",
                        note="consumed by the online chain; sharded s_kv "
                             "is legal exactly when the kernel plan "
                             "declares the online form"),
        "s_query": _PAIR,
    },
    "dsa_attention_score": {"d_head": _c("sum"), "s_query": _PAIR,
                            "s_kv": _PAIR},
    "dsa_attention_apply": {"s_kv": AxisUse(CONTRACT, "online_attention",
                                            plan="attention_kernel"),
                            "s_query": _PAIR},
    # ---- normalize family -------------------------------------------------
    # The cast's scale is an amax, and the strongest case is one per TOKEN:
    # the whole row decides the scale, exactly as the RMS does. A per-group
    # scale (`activation_granularity: per-group:128`) is weaker — a cut of
    # `dim` into multiples of the group leaves every group whole — but which
    # of the two a cell runs is the deployment's
    # (`ActivationFormat.granularity`), not the op's, so the declaration here
    # is the one that holds either way.
    "quantise_activation": {"dim": _n("amax")},
    "rmsnorm": {"dim": _n("rms")},
    "rmsnorm_unscaled": {"dim": _n("rms")},
    "l2norm": {"dim": _n("l2")},
    "hadamard_transform": {"dim": _n("hadamard", "every output channel is a signed sum over the whole axis")},
    "layernorm": {"dim": _n("mean_var")},
    "softmax": {"s_kv": AxisUse(NORMALIZE, "softmax",
                                plan="attention_kernel",
                                note="the online form is the attention "
                                     "plan's LSE stream — without the plan "
                                     "this axis must arrive whole")},
    "softmax_with_sink": {"s_kv": AxisUse(NORMALIZE, "softmax",
                                          plan="attention_kernel",
                                          note="same LSE stream plus a "
                                               "per-head sink")},
    "dsa_softmax": {"s_kv": AxisUse(NORMALIZE, "softmax",
                                    plan="attention_kernel")},
    "softmax_act": {"dim": _n("softmax", "the normalized axis — node.axis, "
                                         "default last (a router's n_experts)")},
    "normalize_sum_one": {"in": _n("sum")},
    "hc_split_sinkhorn": {"d_hc_mix": _n("sinkhorn")},
    "topk": {"in": _n("topk", "whole-axis selection, axis survives "
                              "TRUNCATED (extent k) — no generic online "
                              "form: never produces an accumulator; the "
                              "axis is n_experts, never sharded "
                              "(decision 6)")},
    "dsa_topk_mask": {"s_kv": AxisUse(NORMALIZE, "topk",
                                      plan="attention_kernel",
                                      note="decision 6: under dcp the "
                                           "top-k-of-top-k merge is "
                                           "plan-internal communication, "
                                           "and NO accumulator ever leaves "
                                           "the plan")},
    # ---- window family ----------------------------------------------------
    "conv1d": {"C_in": _c("sum"), "s_ctx": _w("kernel_size-1 halo")},
    "conv2d": {"C_in": _c("sum"), "s_ctx": _w("kernel window halo")},
    "conv1d_weight_norm": {"C_in": _c("sum"), "s_ctx": _w("kernel_size-1 halo")},
    "conv_transpose1d": {"C_in": _c("sum"), "s_ctx": _w("kernel_size taps "
                                                         "scattered per input row")},
    "conv_transpose1d_weight_norm": {"C_in": _c("sum"),
                                     "s_ctx": _w("kernel_size taps scattered "
                                                 "per input row")},
    "swa_attention_score": {"d_head": _c("sum"), "s_kv": _w("s_window"),
                            "s_query": _PAIR},
    "swa_attention_apply": {
        "s_kv": AxisUse(CONTRACT, "online_attention",
                        plan="attention_kernel",
                        note="contract WITHIN the sliding window (phase-0: "
                             "window + contract composite); the window "
                             "reach s_window is the plan's fact"),
        "s_attend": AxisUse(CONTRACT, "online_attention",
                            plan="attention_kernel",
                            note="the probabilities' side of the same "
                                 "banded contraction: one query's window"),
        "s_query": _PAIR,
    },
    "swa_softmax": {"s_attend": AxisUse(NORMALIZE, "softmax",
                                    plan="attention_kernel",
                                    note="window-bounded")},
    "swa_softmax_with_sink": {"s_attend": AxisUse(NORMALIZE, "softmax",
                                              plan="attention_kernel",
                                              note="window-bounded + "
                                                   "per-head sink")},
    "swa_softmax_key0_bias": {"s_attend": AxisUse(NORMALIZE, "softmax",
                                              plan="attention_kernel",
                                              note="window-bounded + "
                                                   "per-head key-0 bias")},
    # The window as a property of the TENSOR, not of whoever scores it:
    # same reach as `swa_attention_score.s_kv`, carried on the stream.
    "sliding_window_kv": {"s_in": _w("s_window")},
    "kv_cache_view": {"s_in": _w("s_kv")},
    "rope": {"dim": _w("pairing dim/2", "rotation couples element "
                                       "pairs, not a sliding window")},
    "rope_emb": {"dim": _w("pairing dim/2")},
    "rope_emb_inverse": {"dim": _w("pairing dim/2")},
    "mrope": {"dim": _w("pairing dim/2")},
    "rope_2d": {"dim": _w("pairing dim/2")},
    "rope_3d": {"dim": _w("pairing dim/2")},
    "xdrope": {"dim": _w("pairing dim/2")},
    # ---- recurrent family ---------------------------------------------------
    "mamba2_scan": {"s_ctx": AxisUse(RECURRENT, "d_ssm_state"),
                    "d_ssm_state": AxisUse(RECURRENT, "state")},
    "gated_delta_rule": {"s_ctx": AxisUse(RECURRENT, "d_lin_k_head"),
                         "d_lin_k_head": AxisUse(RECURRENT, "state")},
    # ---- route family -------------------------------------------------------
    "permute": {"B": _R, "s_ctx": _R},
    "combine": {"n_expert_copies": _R, "M": _R},
    "hash_route": {"n_vocab": _R},
    "gather": {"in": _R},
    "group_mask_apply": {"in": _R, "n_groups_active": _R},
}

def custom_axis_use(spec: Any) -> Dict[str, AxisUse]:
    """The tags of an op the walk lays out by its declared family, read off
    the declaration it hands over — `layout_axis_use()` of a `CustomOpSpec`
    or of an `OpSpec` that declares `layout_family`: derived for the linear
    family, authored otherwise. The declaration is the source; there is no
    second table here to drift."""
    return {axis: AxisUse(**decl.model_dump())
            for axis, decl in spec.layout_axis_use().items()}


def _position_violations(name: str, spec: Any,
                         positions: Tuple[int, ...]) -> list:
    """Whether the op couples exactly the declared positions, probed at every
    rank its DimMap is defined for: the input-0 axes its DimMap does not copy
    1:1, or — where it copies every axis — the axes at which its operands'
    templates name different labels."""
    out = []
    templates = (tuple(spec.shape_in)
                 if spec.shape_in and isinstance(spec.shape_in[0], tuple)
                 else ())
    for rank in range(max(len(positions), 2), 6):
        sources = dim_map(spec, ShapeProbe.probe(),
                          in_ranks=(rank,) * max(1, len(templates)))
        coupled = tuple(sorted({ref for source in sources or ()
                                if source.kind != "from"
                                for ref in source.refs if ref[0] == 0}))
        if not coupled and len(templates) > 1:
            if any(len(t) != rank for t in templates):
                continue  # the op takes its operands at their own rank
            coupled = tuple((0, j) for j in range(rank)
                            if len({t[j] for t in templates}) > 1)
        declared = tuple(sorted((0, axis % rank) for axis in positions))
        if coupled != declared and (coupled or templates):
            out.append(f"{name}: at input rank {rank} the op couples "
                       f"{list(coupled)}, the declaration says "
                       f"{list(declared)}")
    return out


def registry_violations(op_registry) -> list:
    """The completeness gate, as data: every violation is one string.
    Empty list == the registry and `couples_on` agree exactly."""
    out = []
    for name, spec in op_registry.items():
        coupled = tuple(spec.couples_on or ())
        if name in GEOMETRY_POSITIONS:
            if name in AXIS_USE:
                out.append(f"{name}: geometry ops are exempt, not tagged")
            if coupled:
                out.append(f"{name}: couples axes by position "
                           f"{GEOMETRY_POSITIONS[name]}, but couples_on "
                           f"names {sorted(coupled)}")
            out.extend(_position_violations(name, spec,
                                            GEOMETRY_POSITIONS[name]))
            continue
        if name in GEOMETRY_COUPLES:
            if name in AXIS_USE:
                out.append(f"{name}: geometry ops are exempt, not tagged")
            if set(coupled) != set(GEOMETRY_COUPLES[name]):
                out.append(f"{name}: geometry couples_on drifted to "
                           f"{sorted(coupled)}, exemption declared "
                           f"{sorted(GEOMETRY_COUPLES[name])}")
            continue
        if spec.layout_family is not None:
            # A declared layout family carries its own tags; OpSpec has
            # already checked they are exactly `couples_on`.
            if name in AXIS_USE:
                out.append(f"{name}: declares layout_family "
                           f"{spec.layout_family!r} and is tagged here too")
            continue
        tagged = AXIS_USE.get(name)
        if coupled and tagged is None:
            out.append(f"{name}: couples_on {coupled} but no axis_use entry")
            continue
        if tagged is not None and not coupled:
            out.append(f"{name}: tagged but couples_on is empty")
            continue
        if coupled and set(tagged) != set(coupled):
            out.append(f"{name}: tagged axes {sorted(tagged)} != "
                       f"couples_on {sorted(coupled)}")
    out.extend(f"{name}: tagged but not in OP_REGISTRY"
               for name in sorted(set(AXIS_USE) - set(op_registry)))
    return out
