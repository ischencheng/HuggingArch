"""
Op Registry — Primitive compute operations with FLOPs/Params formulas.

Only truly atomic operations live here. Everything else (attention, FFN, MoE)
is defined as sub-DAGs in the YAML spec's `components` section.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple,
    Union,
)

import sympy as sp
from pydantic import BaseModel, model_validator

from backend.arch.dtypes import canonical
from backend.arch.eval import dim as _dim
from backend.arch.eval.refusal import SpecRefusal
from sympy import Expr, Integer

# Shape transform taxonomy. Replaces fixed-rank `shape_in` / `shape_out`
# templates as the source of truth for downstream shape inference. The
# template fields stay for documentation / fallback display; the kind
# (and its associated parameters) drives `infer_shape`, which
# propagates the actual rank from each upstream node's output.
#
# Each member documents its semantics; pick the most specific one when
# registering a new op so the engine can derive shape from the kind
# (or a DimMap for an irregular case).
class ShapeKind(str, Enum):
    PRESERVE = "preserve"
    """Output identical to first input (rmsnorm, silu, …)."""

    BROADCAST = "broadcast"
    """Standard trailing-axis broadcast across every input (add, mul)."""

    AXIS_REPLACE = "axis_replace"
    """One axis dim swapped to dims[out_key] (linear, slice, …)."""

    AXIS_SELECT = "axis_select"
    """One axis becomes a bounded selection index:
    `min(input_axis, dims[out_key])` (top-k)."""

    AXIS_CONCAT = "axis_concat"
    """One axis dim = sum of inputs' axis dims (concat)."""

    PERMUTE = "permute"
    """Axes reordered, dims unchanged (transpose)."""

    CONTRACT = "contract"
    """2-input bilinear contraction over a shared axis:
    `(..., M, d_reduce) · (..., d_reduce, d_out) → (..., M, d_out)`. Generalizes matmul
    and `attention_apply` (A · V). The contracted d_reduce axis is
    eliminated; output prefix comes from input[0], last dim from
    input[1]. (Note: `attention_score` looks like contract but
    needs an implicit Kᵀ — registered as `EXPLICIT` instead.)
    """

    EXPLICIT = "explicit"
    """Irregular geometry written as a DimMap (view_split, permute,
    attention_score, …)."""

    SOURCE = "source"
    """Zero-input op whose shape is the bound dims named by `shape_out`
    (parameter_vector, rotary_inv_freq_buffer)."""


# Accept either the enum member or the legacy string when registering
# an op, so existing `shape_kind="preserve"` call sites keep working
# without a sweep through 60+ register_op calls.
ShapeKindLike = Union[ShapeKind, str]

# Edge semantics vocabulary (7.1): what flows along an edge, not just its
# shape. `activation` is the default — a tensor in the forward stream;
# `state` is a recurrent/cache state read or written; `mask` is a
# selection (top-k indices, attention mask, position ids); `cache` is a
# KV cache tensor. Declared on OpSpec (`input_kinds` per input position,
# `output_kind` for the op's product); BlockGraph validation checks every
# edge's producer kind against the consumer's expectation.
EDGE_KIND_VOCABULARY = frozenset({"activation", "state", "mask", "cache"})

# Parts an operand can play (OpSpec.input_roles). Attention: the score reads a
# query `q` against keys `k`, optionally gated by a `select`ion; the apply
# weighs values `v` by the normalized scores `probs`. MoE: `permute` sends
# `tokens` to experts by a `routing` decision; `combine` merges `expert_out`
# back by the same routing. Linear attention: a recurrence reads q/k/v plus a
# `decay` of its state and a `write` strength (gated delta rule's g and beta).
@dataclass(frozen=True)
class InputRole:
    """One part an operand can play, and how a reader names that operand."""
    key: str
    label: str


#: The closed vocabulary, in reading order (the order structure views list
#: the paths that compute each operand). A new role is one line here.
INPUT_ROLES: Tuple[InputRole, ...] = (
    InputRole("q", "Q"), InputRole("k", "K"), InputRole("v", "V"),
    InputRole("select", "Selection"), InputRole("probs", "Scores"),
    InputRole("tokens", "Tokens"), InputRole("routing", "Routing"),
    InputRole("expert_out", "Expert output"),
    InputRole("decay", "State decay"), InputRole("write", "Write strength"),
)
INPUT_ROLE_VOCABULARY = frozenset(role.key for role in INPUT_ROLES)


#: The closed categories of `axis_use` — how an op couples along an axis:
#: contract produces a partial sum, normalize wants the whole axis unless an
#: online plan exists, window / recurrent gate context sharding, route marks
#: a dispatch seam, separable is a reviewed "not a dataflow dependence".
#: `backend/arch/layout/axis_use.py` reads this tuple; the layout tables for
#: registered ops use the same vocabulary.
AXIS_USE_CATEGORIES = (
    "separable", "contract", "normalize", "window", "recurrent", "route")


class AxisUseDecl(BaseModel):
    """How a custom op couples along one axis — the author's answer to the
    question `couples_on` only poses. `param` names the whole-axis
    statistic (contract / normalize), the window reach (window) or the
    state axis (recurrent)."""
    model_config = {"extra": "forbid"}

    category: str
    param: str = ""
    plan: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _closed_and_parameterized(self) -> "AxisUseDecl":
        if self.category not in AXIS_USE_CATEGORIES:
            raise ValueError(
                f"axis_use category {self.category!r} is not one of "
                f"{list(AXIS_USE_CATEGORIES)}")
        if self.category in ("contract", "normalize", "window") and not self.param:
            raise ValueError(
                f"axis_use category {self.category!r} names its statistic / "
                "reach in `param`")
        return self



@dataclass(frozen=True)
class Lookup:
    """A row gather from a stored table indexed by ids — the stored-table
    case of the law `OpSpec.gather_source` states for an activation edge.

    `ids` is the input port carrying the ids. `table` is where the table
    lives: None when it is the op's own weight (`embedding`, `hash_embedding`
    — an author fact, since `weight_shape` + SOURCE would also describe a bare
    parameter or a precomputed buffer), or the input port that carries a
    stored parameter (DeepSeek-V4 `hash_route`, `tid2eid[input_ids]`).

    One geometry for both: the table's leading axis is its rows, its last
    axis is the row width, and the output's last axis is that width. What it
    declares about traffic: a step reads the rows it gathers, one per
    lookup, capped at the table (`traffic._table_read_fractions`). A table
    split along its rows (vocab-parallel) is read under the BALANCED-LOOKUP
    convention: lookups are taken to fall uniformly over the real rows, so a
    rank serves the share of lookups that its share of rows is, and its
    figure is the long-run average expected useful bytes — not one step's
    physical bytes on the busiest rank. A table split along its width, or
    not split, is exact: every rank reads one row per lookup.
    """
    ids: int = 0
    table: Optional[int] = None

    @property
    def owns_table(self) -> bool:
        """The table is the op's own weight, not an input."""
        return self.table is None

    def check(self, op: "OpSpec") -> None:
        """The ports exist on `op`'s signature, and the table is in exactly
        one place."""
        arity = op.arity()
        ports = [("ids", self.ids)] + (
            [] if self.table is None else [("table", self.table)])
        for what, port in ports:
            if not isinstance(port, int) or not 0 <= port < arity:
                raise ValueError(
                    f"OpSpec({op.name!r}): lookup {what} port {port!r} is not "
                    f"one of its {arity} input(s)")
        if self.table is None and not op.has_weight():
            raise ValueError(
                f"OpSpec({op.name!r}): a lookup into its own table declares "
                "the table as its weight")
        if self.table is not None:
            if self.table == self.ids:
                raise ValueError(
                    f"OpSpec({op.name!r}): lookup ids and table share port "
                    f"{self.ids}")
            # The row's whole weight read is the table's: `phases` restates
            # it at another step's lookups.
            if op.has_weight():
                raise ValueError(
                    f"OpSpec({op.name!r}): a lookup reads its table off input "
                    f"{self.table} and stores a weight of its own; the table "
                    "is one or the other")


@dataclass(frozen=True)
class BroadcastAxis:
    """One output axis of a standard trailing broadcast.

    ``canonical`` supplies the output extent and semantic identity. ``peers``
    are equal-size inputs that carry the same output index; ``broadcasts`` are
    unit inputs expanded over it.  Keeping those roles separate prevents a
    unit operand from claiming that its index or sharding survives the op.
    """

    extent: Any
    canonical: Tuple[int, int]
    peers: Tuple[Tuple[int, int], ...]
    broadcasts: Tuple[Tuple[int, int], ...]

    @property
    def sources(self) -> Tuple[Tuple[int, int], ...]:
        """Input axes that carry the output index.

        When every extent is one, the widest operand is the stable identity
        source; the remaining unit operands are numerically compatible but do
        not get to rename the output axis.
        """
        return self.peers if _dim.value_of(self.extent) != 1 \
            else (self.canonical,)


def elementwise_broadcast_axes(
    shape_in: Sequence[Optional[Sequence[Any]]],
) -> Tuple[BroadcastAxis, ...]:
    """Resolve every axis of a standard trailing broadcast.

    Operand rank is the canonical tie-breaker.  This matches the output
    signature's widest-template identity and makes the result independent of a
    lower-rank gate or bias appearing before the activation in operand order.
    """
    present = [
        (index, shape) for index, shape in enumerate(shape_in) if shape
    ]
    if not present:
        return ()
    rank = max(len(shape) for _index, shape in present)
    out: List[BroadcastAxis] = []
    for output_axis in range(rank):
        candidates = [
            (
                len(shape),
                (input_index, output_axis - (rank - len(shape))),
                shape[output_axis - (rank - len(shape))],
            )
            for input_index, shape in present
            if output_axis >= rank - len(shape)
        ]
        non_unit = [
            candidate for candidate in candidates
            if not _dim.is_unit(candidate[2])
        ]
        eligible = non_unit or candidates
        selected = max(eligible, key=lambda candidate: candidate[0])
        selected_value = _dim.value_of(selected[2])

        def refusal(at, output_axis=output_axis, candidates=candidates):
            values = [at(_dim.value_of(candidate[2]))
                      for candidate in candidates]
            return SpecRefusal(
                f"elementwise broadcast axis {output_axis} has incompatible "
                f"extents {values}",
                declaration="in",
                fix=("make the operands this node reads agree on every "
                     "non-unit axis: name the same dim symbol on both "
                     "sides, not a literal standing for its value, and "
                     "check each operand is the tensor the forward source "
                     "combines here"))

        # A non-unit operand is a peer of the selected one. Two laws of one
        # extent are one axis; two different laws must come to one number
        # wherever the step runs, a condition on the point
        # (`require_equal`), not a branch of the walk.
        peers = []
        for _operand_rank, ref, extent in candidates:
            if _dim.is_unit(extent) and non_unit:
                continue
            if not _dim.same_law(extent, selected_value):
                _dim.require_equal(extent, selected_value, refusal)
            peers.append(ref)
        peers = tuple(peers)
        broadcasts = tuple(
            ref for _operand_rank, ref, extent in candidates
            if not _dim.is_unit(selected[2]) and _dim.is_unit(extent)
        )
        out.append(BroadcastAxis(
            extent=selected[2],
            canonical=selected[1],
            peers=peers,
            broadcasts=broadcasts,
        ))
    return tuple(out)


def elementwise_broadcast_shape(
    shape_in: Sequence[Optional[Sequence[Any]]],
) -> Optional[Tuple[Any, ...]]:
    """Return the standard trailing-axis broadcast of ``shape_in``."""
    if not shape_in:
        return None
    axes = elementwise_broadcast_axes(shape_in)
    return tuple(axis.extent for axis in axes) if axes else None


def _flag(d: Dict[str, Any], key: str, default: int = 0) -> Any:
    """Read a dim flag, treating None / missing as `default`. Returns
    int when the dim binding is concrete, SymPy Expr when symbolic.
    Used in op lambdas instead of Python `if d.get(key):` truthy checks
    that fail for non-grounded Symbols."""
    v = d.get(key)
    return v if v is not None else default


class PhaseBindingMissing(KeyError):
    """A binding without the phase triple asked an attention op for its cost
    or its cache law. Distinct from a plain `KeyError` so a KV-law reader,
    which treats a missing op-specific dim (`s_window`) as "this op has no
    such law", does not read a missing phase as no cache at all."""


def _phase_dim(d: Dict[str, Any], key: str) -> Any:
    """One of the step extents `Phase.resolve` declares. A binding
    without it is not a phase, and answering with a stand-in (`s_ctx` for
    `s_query`, `s_query` for `s_kv`, the full matrix for `s_pairs`) priced every
    caller that skipped `Phase.bind` as full-matrix prefill without a word.
    The evaluator and the per-rank projection both bind through
    `Deployment`, so the only callers this refuses are the ones that were
    getting a wrong number."""
    v = d.get(key)
    if v is None:
        raise PhaseBindingMissing(
            f"{key}: not in this binding — attention cost needs the phase's "
            f"step extents (s_query, s_kv, s_past, s_pairs) from Phase.bind"
        )
    return v


def _s_query(d: Dict[str, Any]) -> Any:
    """Query-axis sequence length, from the phase binding."""
    return _phase_dim(d, "s_query")


def _s_kv(d: Dict[str, Any]) -> Any:
    """Attended-to (key/value-axis) sequence length, from the phase binding."""
    return _phase_dim(d, "s_kv")


def _s_pairs(d: Dict[str, Any]) -> Any:
    """Live (query, key) pairs of the matrix this node scores, per batch row
    and head: the phase's dense causal count (`binding.STEP_LAWS`), or what
    the node's op rebinds it to in its own frame (a top-k, a window, a
    pooled stream, a union of them)."""
    return _phase_dim(d, "s_pairs")


def _pairs(d: Dict[str, Any]) -> Any:
    """The pairs a dense attention scores: the frame's `s_pairs` where the
    attention is causal, and its whole `s_query × s_kv` matrix where it is
    bidirectional (`is_causal: "0"` on the node — HF `is_causal=False`),
    which an encoder runs in a phase that binds no step."""
    is_causal, _ = mask_flags(d)
    if _dim.value_of(is_causal) == 0:
        return _s_query(d) * _s_kv(d)
    return _s_pairs(d)


def mask_flags(d: Mapping[str, Any]) -> Tuple[Any, Any]:
    """`(is_causal, is_chunked)` as an attention node declares them, with the
    defaults an unstated declaration means: causal, a sliding band."""
    return _flag(d, "is_causal", 1), _flag(d, "is_chunked", 0)


def _causal_context_sum(n: Any, cap: Any = None, ratio: Any = None) -> Any:
    """`Σ_{c=1..n} min(⌊c / ratio⌋, cap)` in closed form: what the queries
    whose causal contexts are 1, 2, …, n keys see, summed. `cap=None` is no
    cap; `ratio=1` is no pooling.

    Uncapped, the blocks of `ratio` contexts that share a quotient give
    `P(n) = ratio·t(t−1)/2 + t·(n − ratio·t + 1)` with `t = ⌊n/ratio⌋`
    (`n(n+1)/2` at `ratio = 1`). A cap subtracts what exceeds it: a context
    of `c` sees `⌊c/ratio⌋ − cap` too many exactly when `c ≥ cap·ratio`, and
    those excesses are `P` again over the contexts shifted by `cap·ratio`."""
    half = sp.Rational(1, 2)

    def pooled(m: Any) -> Any:
        if ratio is None:
            return m * (m + 1) * half
        t = _dim.floor_div(m, ratio)
        return ratio * t * (t - 1) * half + t * (m - ratio * t + 1)

    if cap is None:
        return pooled(n)
    shift = cap if ratio is None else cap * ratio
    return pooled(n) - pooled(_dim.maximum(n - shift, 0))


def _selected_pairs(d: Dict[str, Any], cap: Any = None,
                    ratio: Any = None) -> Any:
    """Live (query, key) pairs of one step, per batch row and head, when a
    query whose causal context is `c` keys sees `min(⌊c / ratio⌋, cap)` of
    them — a top-k selection (`cap = k`), a sliding window (`cap = W`), a
    pooled stream (`ratio = m`, every complete block), a selection over a
    pooled stream (both).

    The step's queries have contexts `s_past + 1 … s_past + s_query`
    (`Phase.resolve`), so the count is a difference of two prefix sums. One
    query (`s_query = 1`, decode) gives its own per-query count; a dense
    triangle (`cap ≥ s_kv`, `ratio = 1`) gives the phase's `s_pairs`."""
    past = _phase_dim(d, "s_past")
    return (_causal_context_sum(past + _s_query(d), cap, ratio)
            - _causal_context_sum(past, cap, ratio))


# Canonical Symbol-valued dim dict used to derive display strings from
# `flops_fn` / `params_fn` lambdas. Each lambda is called once at OpSpec
# creation time with these Symbols; the resulting SymPy Expression is
# formatted into the per-node tooltip's symbolic form. Auto-derivation
# eliminates the manual sync between the lambda math and the display
# string (`flops_expr`); callers that need a custom display still get
# `flops_expr_override` / `params_expr_override`.
_CANONICAL_DIMS: Dict[str, Any] = {
    name: sp.Symbol(name) for name in (
        # Engine-supplied runtime dims. `s_ctx` = fresh-token sequence axis
        # (the one that flows through linears, norms, RoPE, …). For attention
        # ops the query and key sequence axes can decouple — `s_query` is
        # the query length, `s_kv` is the attended-to length (cache + new).
        # Phase binding sets all three; pure-prefill keeps s_query=s_kv=s_ctx.
        # `s_pairs` is the live (query, key) pairs of the matrix a node
        # scores, which the QKᵀ / AV / softmax counts are written in.
        "B", "s_ctx", "s_query", "s_kv", "s_past", "n_verify", "s_pairs", "s_pairs_kept",
        # Linear in/out
        "in", "out",
        # Standard model dims (size + flag)
        "d_model", "n_heads", "d_head", "n_kv_heads", "d_ffn", "n_vocab", "n_layers",
        "n_experts_active", "s_window", "dim", "c", "bias",
        # The `k` of a top-K, as the selecting ops name it (torch.topk's own
        # parameter). The model symbol a call site binds to it is whichever
        # thing was selected — `n_experts_active` for a router, `n_groups_active`
        # for its group stage, `n_topk_keys` for DSA. All three used to bind the
        # slot named `n_experts_active`, so two of the three call sites read as
        # a statement about experts that was not one.
        "k",
        # Quant / fused-weight flags
        "Fb", "Qb", "Ob", "n_router_bias_params",
        # MoE / routing
        "n_experts", "n_shared_experts", "d_ffn_expert", "d_ffn_shared", "n_expert_groups", "n_groups_active",
        # MLA / DSA
        "d_kv_lora", "d_q_lora", "d_qk_rope", "d_v_head", "n_indexer_heads", "d_indexer_head", "n_topk_keys",
        # SSM / linear-attn
        "n_lin_k_heads", "d_lin_k_head", "n_lin_v_heads", "d_lin_v_head", "s_conv_kernel",
        # conv1d's kernel-width SLOT (the model symbol a call site binds to it
        # is `s_conv_kernel`; this is the op's own name for the axis).
        "kernel_size",
        # Mamba-2 selective SSM (Nemotron-H, Zamba2, …)
        "n_ssm_heads", "d_ssm_head", "d_ssm_state", "n_ssm_groups", "s_ssm_conv_kernel",
        # Conv1d / vision
        "C", "C_in", "C_out", "groups", "n_vis_channels", "s_patch", "s_patch_temporal", "n_spatial_merge", "d_vis_model", "d_vis_ffn", "n_vis_layers",
        # MTP / sliding window / partial RoPE
        "n_mtp_layers", "d_rope", "d_nope", "d_vis_head", "d_proj_out", "n_vis_pos_vocab", "d_proj_hidden", "d_proj_in",
        # Hyper-Connections (V4-Pro): n_hidden_copies is the parallel-copy
        # count (hc_mult); d_hc_mix = (2+n_hidden_copies)·n_hidden_copies.
        "n_hidden_copies", "d_hc_mix",
        # Misc
        "rope_inv_freq_len", "flash_kv_shards", "n_zero_experts",
    )
}


# Names that should appear at the front of every `*`-product term in
# the order listed below. `B` (batch) and `s_ctx` (sequence) are runtime
# dims — convention is to read them first when scanning a formula.
# Numeric literals stay in front of all symbols (SymPy's default).
_LEADING_DIM_ORDER = ("B", "s_ctx", "s_query", "n_verify", "s_kv")


class _DotPrinter(sp.printing.str.StrPrinter):
    """SymPy str printer that emits `*` for multiplication (copy-paste
    safe back into Python / sympy) and forces a stable operand order in
    `Mul` expressions: numeric coefficient first (default SymPy
    behavior), then `B`, then `S`, then everything else in SymPy's
    default lexicographic order. Without this, `2*B*s_ctx*d_model*s_kv` would
    render as `2*B*d_model*s_ctx*s_kv` (alphabetical), which breaks
    the convention of putting batch/seq dims up front.
    """

    def _print_Mul(self, expr):
        # Split numeric coefficient from symbolic factors using SymPy's
        # canonical decomposition; this keeps the `2` (or other coeff)
        # rendered first and handles negative / fractional coefficients
        # the same way SymPy normally would.
        coeff, rest = expr.as_coeff_Mul()
        factors = sp.Mul.make_args(rest)

        def sort_key(f):
            name = str(f)
            if name in _LEADING_DIM_ORDER:
                return (0, _LEADING_DIM_ORDER.index(name), name)
            return (1, 0, name)

        ordered = sorted(factors, key=sort_key)
        parts = [self._print(f) for f in ordered]
        if coeff != 1:
            parts.insert(0, self._print(coeff))
        return "*".join(parts)

    def _print_Pow(self, expr):
        # Render `dim**2` as `dim^2` (no space, caret) to match the
        # project's display convention. Defer to default for everything
        # else (negative / fractional exponents are uncommon here).
        b, e = expr.as_base_exp()
        return f"{self._print(b)}^{self._print(e)}"


def _format_display(expr: Any) -> str:
    """SymPy Expr / int → display string for per-node tooltips.

    `**` → `^`, simplify away `+ 0` / `* 1`, lowercase `Min`/`Max` to
    match the project's display convention. Multiplication renders as
    `*` so users can copy formulas back into Python / sympy. Operand
    order in each product term: numeric coefficient, then `B`, `S`,
    then remaining symbols alphabetically. Returns "" when the lambda
    couldn't be auto-derived (e.g. raised on missing canonical dim)."""
    if expr is None:
        return ""
    if isinstance(expr, (int, Integer)):
        return str(int(expr))
    try:
        # `expand` keeps the additive form (`2*B*S*in*out + B*S*out*bias`)
        # readable instead of the factored form `B*S*out*(bias + 2*in)`
        # that `simplify` would produce. Both are mathematically
        # equivalent; the expanded form matches the convention used in
        # the original hand-written display strings.
        simp = sp.expand(expr)
    except (TypeError, AttributeError):
        simp = expr
    s = _DotPrinter().doprint(simp)
    s = s.replace("Min", "min").replace("Max", "max")
    return s


def _derive_display(fn: Callable) -> str:
    """Call `fn(_CANONICAL_DIMS)` and pretty-print the resulting Expr.

    Returns "" when the lambda needs dims absent from the canonical set
    or raises on symbolic input — caller is expected to provide a manual
    override string in that case. Used only by `OpSpec.flops_expr` /
    `params_expr` for the op-registry table; the analyzer / tooltip path
    renders each node's symbolic formula layer instead.
    """
    try:
        expr = fn(_CANONICAL_DIMS)
    except (KeyError, TypeError, AttributeError):
        return ""
    return _format_display(expr)


@dataclass(frozen=True)
class TensorDecl:
    """One tensor an op owns: its checkpoint suffix, its shape, and the dim that
    gates its existence.

    `when` names a dim the spec sets per node (`bias: "1"`), so whether a module
    has a bias is the spec's statement, exactly as its width is. `None` means
    unconditional. `when_unset` is the op's own answer for a node that leaves
    the dim unset, the default of the module it mirrors: a `linear` writes no
    bias unless it says `bias: "1"`; a `layernorm` stores its beta unless it
    says `bias: "0"` (`nn.LayerNorm(bias=False)`, or a checkpoint that stores
    only the gain).
    """

    suffix: str
    shape: Tuple[Any, ...] = ()
    when: Optional[str] = None
    when_unset: bool = False
    kind: Optional[str] = None      # override the rank-derived StorageKind
    # Projection from this tensor's axes to the primary weight's axes.
    # ``None`` means the primary tensor itself (identity).  A linear bias is
    # ``(0,)`` because it follows only the output axis.  This is op geometry,
    # not endpoint placement, and prevents consumers guessing sibling
    # provenance from equal numeric extents.
    axis_refs: Optional[Tuple[int, ...]] = None

    def flag(self, dims: Mapping[str, Any]) -> Any:
        """The `when` switch at a node's dims: the bound value, or the op's
        `when_unset` where the node leaves it unset. 1 for an unconditional
        tensor."""
        if self.when is None:
            return 1
        value = dims.get(self.when)
        return int(self.when_unset) if value is None else value


#: The two ops that bound a MoE dispatch region: tokens leave the dense stream
#: at `permute` (routed to experts) and rejoin it at `combine`. Declared once
#: here, next to the registry that defines them; `layout.zone` re-exports them
#: and every dispatch judgement reads these names rather than spelling them.
DISPATCH_OP = "permute"
COMBINE_OP = "combine"


#: The layout families an op can declare instead of a hand-written handler,
#: each the law `layout/propagation.py` `family_handler` dispatches to.
LAYOUT_FAMILIES = ("linear", "recurrent")

class ProcessorStep(str, Enum):
    """The steps of a multimodal processor's work an op can be
    (`OpSpec.processor_step`). The vocabulary's one definition; `OpSpec`
    refuses any other word."""

    CUT = "cut"
    """Cuts each media item (an image, a video) into its patch rows."""

    PAD = "pad"
    """Pads each item's rows to a fixed count."""

    UNPAD = "unpad"
    """Strips the padding a pad added."""

    GROUP = "group"
    """Lays consecutive items' rows end to end as one item's."""


class OpView(Protocol):
    """The uniform view of an op that both a registered primitive (`OpSpec`)
    and a spec-scoped custom op (`CustomOpSpec`) present — the "one protocol,
    two implementations" that `eval.op_ref.OpRef` resolves once at the node.

    `OpSpec` carries every member as a field. `CustomOpSpec` carries the first
    eight as fields and the rest (which a YAML op genuinely lacks) as honest
    read-only `@property`. Conformance is STRUCTURAL — neither type inherits
    this Protocol. `name` and `shape_in` are deliberately NOT members: a custom
    op's identity is the node's `op:` string and its input template lives in
    `signature:`, not on the type.
    """

    shape_kind: Any
    weight_tensor_suffix: str
    weight_shape: Any
    secondary_outputs: Mapping[str, Any]
    out_key: Optional[str]
    axis_locked: Optional[int]
    carries_recurrent_state: bool
    input_roles: Any
    is_view_op: bool
    is_buffer: bool
    processor_step: Optional[str]
    tensors: Tuple
    alt_shape_in: Tuple
    permute_to: Optional[Tuple[int, ...]]


@dataclass
class OpSpec:
    """A primitive compute operation.

    `flops_expr` / `params_expr` (display strings shown in per-node
    tooltips) are auto-derived from the corresponding `_fn` lambda by
    calling it with `_CANONICAL_DIMS` Symbols and formatting the
    resulting SymPy Expression. Override only when the auto-derived
    form is wrong for the op (e.g. conv1d's depthwise conditional
    can't be expressed without picking one branch).
    """
    name: str
    flops_fn: Callable[[Dict[str, Any]], Expr]
    params_fn: Callable[[Dict[str, Any]], Expr]
    # Shape templates — list of dim key names. Engine substitutes concrete
    # values: ["B", "s_ctx", "in"] with dims {B:1, s_ctx:2048, in:3584} → [1, 2048, 3584].
    #
    # These document the op's canonical I/O signature and are consumed by
    # FOUR runtime paths besides display:
    #   • spec_worker prompt: dim symbols → required-dims column in the
    #     primitive ops table the agent reads.
    #   • rendering._node_build: `if not shape_in` marks zero-input ops
    #     as DAG sources (parameter_vector, rotary_inv_freq_buffer, …)
    #     so the SVG renderer doesn't draw spurious incoming edges.
    #   • rendering._shapes._port_label: multi-input ops' incoming-edge
    #     labels come from the per-input template's last dim.
    #   • engine display fallback: when zero-input ops have no upstream
    #     shape, `shape_in_list = self._resolve_shape(shape_in, dims)`.
    # Runtime SHAPE INFERENCE (the per-node `shape_out` value the engine
    # computes) is driven by `shape_kind` + `infer_shape` against
    # actual upstream shapes — not template substitution.
    shape_in: Tuple[str, ...] = ()
    shape_out: Tuple[str, ...] = ()
    #: Further input templates this op accepts, each in `shape_in`'s encoding.
    #:
    #: An op applied at more than one rank has more than one index signature,
    #: and they are different MAPPINGS rather than one mapping with a
    #: tolerance: `slice` after a `head_split` reads
    #: `(B, n_heads, s_ctx, in)`, and the `n_heads` label is what tells a
    #: consumer that axis 1 of the output is axis 1 of the input. A wildcard
    #: would match the rank and lose the label, which is the half that
    #: answers identity questions.
    #:
    #: Only the input half is declared: the output follows from `shape_kind`
    #: exactly as it does for the canonical template, so an alternative that
    #: restated it could disagree with itself.
    alt_shape_in: Tuple[Tuple, ...] = ()
    # --- Edge semantics (7.1) ---
    # What flows along each input edge, aligned with the input positions of
    # `shape_in` (missing trailing positions default to `activation`), and
    # what the op's output edge carries. Shape says how big a tensor is;
    # kind says what it IS — a top-k index list is int32[B,Sq,K]-shaped like
    # an activation, and only the kind stops it being wired where an
    # activation belongs. BlockGraph validation checks every edge against
    # this pair (producer's output_kind vs consumer's input kind at that
    # position).
    input_kinds: Tuple[str, ...] = ()
    output_kind: str = "activation"
    # What each input IS TO THE OP, aligned with graph input positions like
    # `input_kinds` (which says what flows on the edge). Score reads its
    # query at 0 and its keys at 1; `permute` routes its tokens (0) by a
    # routing decision (1). Declared only on ops whose operands play
    # distinct parts that a reader of the graph needs to tell apart — the
    # structure views group a component by the operand each node feeds.
    # Positions may exceed `shape_in`: `permute` is wired with two inputs
    # but declares one shape template, on purpose (see its registration).
    # Closed vocabulary; a `mask` edge must be the `select` role. None marks
    # a position that plays no part a reader tells apart (a per-head
    # parameter wired between operands, as in `mamba2_scan`).
    input_roles: Tuple[Optional[str], ...] = ()
    # --- Shape transform kind ---
    # See ShapeKind for the taxonomy. Accepts either the enum member or
    # its string value for ergonomic registration; normalized to the
    # enum in __post_init__ so downstream comparisons can use `is`.
    # Every registered op declares one of the documented kinds.
    shape_kind: ShapeKindLike = ShapeKind.EXPLICIT
    # axis_replace: dim key whose resolved value becomes the new axis size.
    out_key: Optional[str] = None
    # axis_replace: pin the axis to a fixed value. If None, the call
    # site's `node.axis` (default -1) is used. Set this for ops where
    # the axis is part of the op's mathematical definition rather than
    # a call-site choice.
    axis_locked: Optional[int] = None
    # The op has no default axis: every call site declares `axis:` and the
    # schema rejects a node without one. For a reshape whose axis is the
    # whole meaning (`view_split`: which axis factors), a silent default is a
    # derived fact standing in for a declared one.
    axis_required: bool = False
    # permute: 0-based positional permutation of axes. e.g. (0, 2, 1, 3)
    # for the (B, s_ctx, n_heads, d_head) → (B, n_heads, s_ctx, d_head)
    # transpose used after view_split.
    # Length must equal input rank. Per-call-site override comes from
    # `node.axes`; this field provides a default when the OpSpec wraps a
    # fixed permutation.
    permute_to: Optional[Tuple[int, ...]] = None
    # Manual override for the display string. Empty (default) means
    # auto-derive from `flops_fn` / `params_fn`. Use only when the
    # lambda's symbolic execution can't represent the intended display
    # form — e.g. conv1d's `if groups == C: depthwise else: standard`
    # branch picks one path under symbolic execution and loses the other.
    flops_expr_override: str = ""
    params_expr_override: str = ""
    # Computation semantics (distinct from flops/params cost) — e.g. "W·x + b"
    # for linear, or "TopK_K(w·ReLU(Q_idx·K_idxᵀ))" for dsa_topk_mask. Rendered in
    # node UI so the user can see *what* the op computes, not just *how much*.
    formula: str = ""
    description: str = ""
    # On-disk weight tensor shape, derived from a node's resolved dims
    # (the spec_tree.iter_leaves leaf-emission path). Only set for ops
    # that actually store a learnable weight on disk: linear / per_head_linear /
    # embedding / norm / conv1d / parameter_vector. Returns None when
    # required dims are unbound. spec_tree calls this to filter
    # "weight-bearing leaves" and derive shape — replaces the prior
    # hardcoded `_LINEAR_OPS` / `_NORM_OPS` / `_CONV_OPS` frozensets in
    # spec_tree.py: adding a new param-storing op now means registering
    # one entry here, not editing spec_tree.
    #
    # DECLARED as a template, not written as code. Each entry is an op-slot
    # name or a dim expression over slots (`"C_in // groups"`), resolved
    # against the node's dims by the same evaluator every other shape in this
    # file uses. Custom ops already declared it this way (`CustomOpSpec.
    # weight_shape`); primitives were the outlier, and it cost the usual:
    # eight of the nine were a name tuple wrapped in an identical "return None
    # if unresolved" guard — one policy, copied eight times — and a lambda's
    # names cannot be checked against the slot vocabulary, rendered, or read
    # by anything that wants the shape rather than a call.
    #
    # `_shape` as a single slot means the call site's `shape:` list
    # (parameter_tensor).
    weight_shape: Tuple[Any, ...] = ()
    # On-disk tensor-name suffix appended to the spec_id when emitting a
    # weight LeafTensor. Default `.weight` matches `nn.Linear` / `nn.Embedding`
    # / `nn.LayerNorm` storage convention (the underlying nn.Module hangs
    # its parameter under the `.weight` attribute). Empty string means
    # "store under the spec_id directly" — used by ops whose weight is a
    # naked `nn.Parameter` declared on the parent module (no enclosing
    # nn.Module wraps it). Examples: `parameter_vector` (`e_score_correction_bias`,
    # `attention_sinks`, …) and `linear_param` (V4-Pro hyper-connection
    # `hc_attn_fn` is `nn.Parameter`, called via `F.linear(x, hc_fn)`,
    # checkpoint name is the bare `hc_attn_fn` not `hc_attn_fn.weight`).
    weight_tensor_suffix: str = ".weight"
    # Multi-output port specifications. Empty (default) means single-output —
    # the op produces one tensor referenced as `<node>` or `<node>.out`.
    # Populated only when an op produces multiple distinct tensors that
    # downstream nodes reference independently (e.g. `hc_split_sinkhorn`
    # → `pre / post / comb`). Each entry maps a port name to a shape
    # template of dim symbols, same encoding as YAML `secondary_outputs`.
    # The main `out` port still comes from `shape_kind` / DimMap.
    secondary_outputs: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    # Explicit list of dim keys that `flops_fn` / `params_fn` rely on
    # (read via `dims["<key>"]` or `dims.get("<key>")`). The schema's
    # BlockGraph validator
    # cross-checks at load time that every node using this op
    # supplies all required keys via `dim:` / `dims:`, surfacing
    # a friendly "node X using op Y is missing dim Z" error
    # instead of a runtime `KeyError: 'Z'` deep in shape inference.
    #
    # Universal axes that the engine binds globally — `B`, `s_ctx`,
    # `s_query`, `s_kv`, `M`, `s_pairs` — are NOT listed here
    # (they're always available in `dims`); only op-local keys
    # the call site must bind.
    required_dims: Tuple[str, ...] = ()
    # Model-scope dims the op reads off its scope's binding rather than off
    # its node's `dims:` — a dispatch's routing counts. Declared so a static
    # read of a graph that runs the op (the Playground's unbound-dim report,
    # `closure_dim_refs`) sees what the walk will ask for.
    binding_dims: Tuple[str, ...] = ()

    # HBM-cost classification for inference rooftop modeling. True iff
    # this op is pure stride/layout manipulation that moves zero HBM
    # bytes in any sane fused impl (slice / view_split / view_merge /
    # transpose / broadcast / expand — vLLM treats these as PyTorch
    # views, no kernel launched). False (default) for ops that always
    # touch memory (concat with cache write, MoE permute, every
    # compute op). `access_bytes.annotate` uses this to decide
    # whether to count input+output activation traffic. Aligned with
    # vLLM — concat in particular is FALSE because the K/V
    # concat-into-cache path is a real `concat_and_cache_mla` kernel.
    is_view_op: bool = False
    # A view whose output REPEATS its input's elements: a stride-0 expansion
    # (`.expand()` / `np.broadcast_to`), never a narrowing or a reordering.
    # A reader's kernel loads the stored tensor once and fans it out in
    # registers, so the bytes behind the view are the input's, not the
    # output's (`traffic._fanout` divides by the ratio). Declared because the
    # ratio alone cannot say it: a `slice` narrows, and a `kv_cache_view`
    # spans more positions than its producer wrote this step for a reason
    # the cache law prices instead.
    repeats_input: bool = False
    # The op reads its last declared input any number of times (one or more),
    # each a further operand of the same kind: a row scatter writing several
    # media inputs' rows into one sequence in one masked scatter.
    repeats_last_input: bool = False
    # Which step of a multimodal processor's work the op reads as rows
    # (`ProcessorStep`), or None for every op that is no processor step.
    # Only the op's author knows it: a layer of such an op cites the
    # processor (`source_anchor:`); only a `cut` reads an image or video
    # model input and declares its attention `segment:`; and the rows an
    # encoder computes are the cut's, or a `pad` reading them
    # (`Analysis.media_rows`).
    processor_step: Optional[Union[ProcessorStep, str]] = None
    # The value an operand stands for when the graph declares its producer
    # absent (`repeat: 0`), if the op has one: `0` for a sum, `1` for a
    # product. One value for every operand, so it is declared only by an op
    # whose operands are interchangeable; a step whose other operands are
    # all absent then hands its readers its one live operand and is not
    # built. Undeclared means an absent operand has no value here — a gate
    # whose gate is missing is not the ungated tensor — and the builder
    # refuses the step. An op whose operands differ declares per operand,
    # in a field of its own.
    absent_operand_identity: Optional[str] = None
    # The op's storage is a precomputed BUFFER, not a learnable parameter: it
    # occupies bytes on disk and in HBM, and it is excluded from the
    # HF-comparable parameter total. Declared here because nothing else implies
    # it — `engine._is_quant_meta` reaches the same exclusion by name, and by
    # the same predicate it uses for quantisation scales, which is why the two
    # cannot be told apart downstream.
    is_buffer: bool = False
    # A precomputed SIDE BUFFER that branches off the backbone instead of
    # advancing the residual stream — a position table or precomputed RoPE.
    # Declared because the top-level compute graph wires it as a standalone
    # source (no backbone inputs); nothing else implies it — the pre-declaration
    # name set held `rotary_emb`, which matched nothing.
    is_side_buffer: bool = False
    # A dispatch-zone BOUNDARY — the `permute` that splits tokens by expert and
    # the `combine` that reduces them back. The nodes between the two carry
    # `in_dispatch_zone`; the boundaries themselves are excluded from it, yet
    # their memory-bound time is part of the MoE block's real MFU. Declared so
    # the block aggregation reads the fact off the op instead of matching the
    # `permute` / `combine` names.
    is_dispatch_boundary: bool = False
    # The op IS a recurrence: it reads last step's state and writes this
    # one's, through the kernel's own channel rather than through an edge.
    # Nothing else can say so. A `state:` cache entry declares the bytes and
    # names the node carrying them, since no edge reads a state; this flag is
    # the other half of that tie — every node running such an op must be
    # some state's `carrier:` (`ArchSpec._check_recurrent_state_carriers`),
    # or a Mamba layer's decode, which is that read almost entirely, is
    # billed to nobody. An op that only sometimes keeps a state (a causal
    # `conv1d` keeps its window in a recurrent layer and nothing in an audio
    # encoder) does not set it; the entry naming it is the tie.
    carries_recurrent_state: bool = False

    # The layout law of an op the layout walk has no hand-written handler
    # for — an op an op proposal adds. `linear`: one `[Out, In]` weight
    # contracting the operand's last axis (`weight_shape: (Out, In)` with
    # `out_key: Out`), its axis use derived. `recurrent`: a recurrence over
    # heads (`carries_recurrent_state`), its `axis_use` tagging exactly two
    # axes `recurrent`, one of them the state (`param: "state"`). None: the
    # op's law is its entry in `layout/propagation.py` `_HANDLERS` and its
    # axis use is `layout/axis_use.py` `AXIS_USE`.
    layout_family: Optional[str] = None
    # How the op couples along each `couples_on` axis, for the recurrent
    # family (the linear family's is derived).
    axis_use: Mapping[str, AxisUseDecl] = field(default_factory=dict)

    # Every tensor this op owns on disk, in checkpoint order. A module with a
    # bias puts TWO entries in the file under two names, and which axis the bias
    # follows is a fact about the op — `linear` biases the first axis of
    # `(out, in)`, `headwise_linear` the first two, `conv1d` the first of
    # `(channels_out, channels_in, kernel_width)`. There is no positional rule,
    # which is the argument for
    # declaring it rather than for recovering it by arithmetic.
    #
    # `params_fn` is then a PROJECTION of this list rather than a second,
    # independent declaration. Two independent declarations is how a bias could
    # be counted in `params` and absent from storage at the same time.
    #
    # Empty means the single-tensor case, taken from `weight_shape`.
    tensors: Tuple["TensorDecl", ...] = ()

    # The op's OWN dim slots along which it is NOT separable: split the axis
    # across ranks and "split then compute" stops equalling "compute then
    # split", so the difference has to be paid in communication. A shard on
    # any OTHER axis is orthogonal to what the op does and passes through it
    # untouched, on any input, in any combination.
    #
    # Not `contract_on`: contraction is only one of the four ways an axis
    # couples, and the narrow name would make half these declarations false
    # under it.
    #   * contraction    — the axis disappears, summed. `linear`'s `in`,
    #                      `attention_score`'s `d_head`. Repair: keep the
    #                      partial sum, all_reduce once.
    #   * reduce-and-broadcast — the axis survives, but every output element
    #                      depends on all of it. `softmax`'s `s_kv`,
    #                      `rmsnorm`'s `dim`. Repair: an online merge (LSE)
    #                      or an all_gather. `concat` sums nothing and is
    #                      still not separable along the axis it joins.
    #   * join           — `concat`'s `in_a` / `in_b`. Repair: all_gather.
    #   * re-partition   — same axis, different partitioning on the way out.
    #                      `permute`'s `s_ctx` becoming expert-local.
    #                      Repair: all_to_all.
    # WHICH kind it is decides the repair collective, and that is already
    # axis_use / layout's job. This field answers only "is there an account to
    # settle here", so it stays a flat set: two declarations of one fact is
    # where the next silent divergence grows.
    #
    # A mode used to name a state per input POSITION, so an op that is
    # transparent to an axis had to enumerate every combination its
    # operands can arrive in: MLA packs a head-sharded nope part against a replicated
    # decoupled RoPE key, and covering that pair alone means writing the
    # cross product. Naming the axis instead says it once.
    #
    # Three states, not two. `()` is the positive claim that the op couples
    # along nothing — every pointwise activation, `add`, `mul`. `None` is
    # the absence of a claim, and a distributed tensor reaching an op that
    # has not made one raises: an op author who says nothing must not be
    # read as having said "transparent".
    #
    # Authoring rule: name every axis the op is not UNCONDITIONALLY
    # separable along. `attention_score` is separable along `s_query` alone
    # and along `s_kv` alone, and along neither once both are cut — which is
    # ring attention, and is not free. If separability depends on which
    # operands are cut, the axis is coupled and the catalog decides what to
    # do about it. Over-declaring costs a recorded assumption whose eventual
    # behaviour is still pass-through; under-declaring costs a silent zero.
    #
    # The entries are slot names (`dim`, `in_a`, `in_b`), not dim symbols,
    # because an op does not know what a caller binds to them. They resolve
    # per call site against the labels the arriving tensors bind (see
    # `eval/flops.py::bind_labels`) and against the node's `dims_raw`, so
    # the comparison is concrete-symbol against concrete-symbol.
    couples_on: Optional[Tuple[str, ...]] = None

    # True iff this op IS a communication primitive (its semantics
    # are data movement across ranks, not local compute). When True,
    # `flops_fn` returns 0 and `access_bytes` accounting follows the
    # NCCL ring formula (factored on the runtime world size at
    # inference time, not at spec-engine time). False for every
    # compute op (the default).
    is_comm_op: bool = False
    #: Display name the renderer shows for this op (`RMSNorm`, `Q @ K^T`).
    #: The author's word; `rendering/_consts.OP_LABEL` is a projection of it.
    label: str = ""
    #: The op's forward is a row gather from a stored table indexed by ids
    #: (`Lookup`): an embedding table, or a parameter table handed in on an
    #: edge. None for every op that gathers from no table.
    lookup: Optional["Lookup"] = None
    #: The input port whose ACTIVATION the output is an index gather of —
    #: one of its elements per element written (`gather`'s per-token pick,
    #: `permute`'s routed rows). A launch reads at most what it writes of
    #: that input, and at most the input once, so the edge is billed
    #: `min(input, output)` elements (`traffic._activation_reads`). The same
    #: law `Lookup` states for a stored table, on an edge: which rows the
    #: index picks no workload carries, so a picked row is counted per pick
    #: (an upper bound) and capped at the tensor. Whether the cap binds is
    #: the op's geometry: `gather` writes `k < in` per token, so it binds
    #: under every deployment; `permute` writes `topk/ep` rows per token, so
    #: only under expert parallelism. None for every op that reads its
    #: inputs whole.
    gather_source: Optional[int] = None
    #: How many position streams the rotation reads: 1 for a rotary position
    #: embedding over one position per token (`rope`, `rope_emb`, its
    #: inverse), 3 for multimodal RoPE (`mrope`: temporal / height / width
    #: positions, the cos/sin of each channel section chosen from one of
    #: them), 0 for every op that rotates nothing. The arithmetic is the same
    #: for every value; what differs is the kernel a serving framework runs —
    #: vLLM `MRotaryEmbedding` → `triton_mrope`, and sglang leaves it out of
    #: its fused qk-norm+rope kernel — so the fusion plans and the config
    #: check (`validate._rotary_positions`) read this, not an op name.
    rotary_position_axes: int = 0
    #: The rope-parameter key whose section list assigns this rotation's
    #: channel sections to its position streams — the key a framework's
    #: dispatch builds the kernel from: `mrope_section` for `mrope` (vLLM
    #: `MRotaryEmbedding`), `xdrope_section` for `xdrope` (vLLM
    #: `XDRotaryEmbedding`). Empty for a rotation no section list
    #: describes (one position stream, or an encoder's image grid).
    rotary_sections_key: str = ""
    #: The weight's placement under tensor parallelism follows which of its
    #: axes a TP knob divides — a weight carrying no such axis is replicated
    #: (`sharding.leaf_class`'s demotion). False for the vocab-parallel
    #: table and for bare parameters, whose placement is decided elsewhere.
    tp_axis_placement: bool = False
    #: The op attends over a sliding window (`s_window` bounds its keys) —
    #: the fused attention span that absorbs it is measured as the windowed
    #: kernel cell. Author fact, not a name prefix.
    windowed: bool = False
    #: The op's softmax carries a per-head sink term in its denominator —
    #: the span that absorbs it is measured with `n_attn_sinks`.
    sink: bool = False
    #: For a collective: what it does to the sequence-parallel axis of its
    #: operand — `"divide"` (reduce_scatter), `"multiply"` (all_gather), or
    #: `""` (shape-preserving). Read through `sp_axis_effect`.
    sp_axis_effect: str = ""
    #: For a collective: it is a REDUCTION — its output combines the members'
    #: values (`all_reduce`, `reduce_scatter` sum partial results) rather
    #: than moving them (`all_gather`, `all_to_all`, `send_recv` deliver each
    #: element some member already held). A moved tensor is still the tensor
    #: it was, so a cached K/V carried across ranks is still read from the
    #: cache (`eval/cache.py::context_sources`); a reduced one is new data.
    #: Read by `eval/cache.py::_moves`. None on every op that is no
    #: collective; a collective declares it either way (`__post_init__`), so
    #: a new one cannot be filed as a move by omission.
    reduction: Optional[bool] = None
    #: For a collective: the time schedule it may run under when its group
    #: straddles nodes — `"hierarchical"` (intra-RS · inter-AR · intra-AG,
    #: all_reduce only), `"point_to_point"` (one hop between two members —
    #: a pipeline send/recv) or `""` (one ring). Read through
    #: `collective_schedule`.
    collective_schedule: str = ""
    #: The op's math is a (batched) matmul against its stored weight — an
    #: author fact like `is_activation`, not a guess from `weight_shape`'s
    #: rank (`per_head_linear` is 3-D and is a GEMM). Read through
    #: `is_gemm_op`; it replaced two hand-kept `_GEMM_OPS` sets whose
    #: members never agreed.
    gemm: bool = False
    #: This op is a DeepSeek Sparse Attention compute op — its presence in an
    #: attention component's graph means the layer runs DSA, whatever its
    #: (MLA/GQA-shaped) cache class. Read through `is_dsa_op`; replaced the
    #: hand-kept `_DSA_OPS` set in the validator (one place naming the fact).
    dsa: bool = False
    #: Which half of the main attention kernel geometry this op is —
    #: `"score"` (QKᵀ) or `"apply"` (PV) — for the score/apply pairs that
    #: share one kernel (dense, sliding-window, sparse). Replaced two
    #: name-prefix sets in `layout/attention.py`.
    attention_kernel: Optional[str] = None
    #: A pointwise nonlinearity — an FFN gate, a router scoring function.
    #:
    #: Declared rather than derived, and that is not laziness: every
    #: shape-level proxy fails. "PRESERVE, weightless, one input" is true of
    #: `rope`, of every attention softmax and of all four collectives. What
    #: makes something an activation is what it is FOR, and only the op's
    #: author knows that.
    #:
    #: Set by `_act_op`, which is how activations are registered — so an
    #: activation added the normal way cannot forget to say so. That is the
    #: whole difference from a list kept at the consumer, which is what this
    #: replaced: it named three activations that do not exist, and missed
    #: `relu_squared` and `situ`, which do.
    is_activation: bool = False
    #: The names a Hugging Face config gives exactly this function in
    #: `hidden_act` / `hidden_activation` (transformers `ACT2FN` keys):
    #: `gelu` is the erf form, `gelu_pytorch_tanh` / `gelu_new` / `gelu_fast`
    #: the tanh approximation, which is a different op. Declared on the op so
    #: that reading a config's activation is a lookup, not a family guess;
    #: an activation no config names (neg_exp, softmax_act) leaves it empty.
    #: `register_op` builds the reverse map and refuses a name two ops declare.
    config_names: Tuple[str, ...] = ()
    #: The real numbers this op's math takes beside its operands, by the
    #: symbol each is read under: `clamp` takes `clamp_limit`, SiTU
    #: `situ_beta` and `situ_linear_beta`. Each is a `dims.yaml` symbol of
    #: kind `constant`, and the op says only THAT it takes one — never where
    #: a model publishes it. That is the spec's: its `dims:` binds the symbol
    #: to its config's field, in the scope it lives in, and a node or call
    #: site rebinds it with `dims:` where one part of the model reads
    #: another scope (a vision tower's clamp reads `vis_clamp_limit`). The
    #: walk reads each from the node's own frame, as it reads `dim`, and
    #: records it per node (`Analysis.consts`); a fused kernel carries the
    #: ones its origins took.
    #:
    #: They do not enter cost — an elementwise function costs the same at
    #: every value of its constant. They are declared because they change
    #: what the function IS, which matters to anything that has to COMPUTE
    #: it: a measured kernel and its oracle both ran SiTU at beta 1 and
    #: agreed with each other about a gate no model has. A frame that binds
    #: none leaves the constant absent, not defaulted.
    #:
    #: `scale_const` and `add_const` take a constant too, but each node's is
    #: an expression over extents (`d_model**-0.5`, `1/n_hidden_copies`)
    #: that no consumer computes from; their nodes say it in `formula`.
    consts: Tuple[str, ...] = ()
    #: Whether the RANK of `shape_in` is a contract or an illustration.
    #:
    #: `shape_in` is one field with two readings, and that is the problem it
    #: names. For most ops it is a canonical example: `rmsnorm` declares
    #: `(B, s_ctx, dim)` and is applied to rank-3 and rank-4 tensors alike,
    #: `mul` to rank 1, 3 and 4, `sum_reduce` to rank 4 — eight ops in the
    #: corpus receive a rank their own `shape_in` does not show. For the
    #: attention family it is binding: `attention_score` reading a rank-3
    #: query is a mis-wired edge or a missing reshape, every time.
    #:
    #: Only the op's author can say which, so it is said here. Asserting it
    #: from outside — the list this replaced — meant the two readings were
    #: distinguished nowhere the op could see, and an op registered without
    #: being added to that list silently got the loose one.
    input_rank_is_exact: bool = False

    # The dtype this op's OUTPUT is written in where that is not the role's
    # activation dtype — a dtype NAME (`DTYPES` spelling), whose width
    # traffic bills and whose name the measurement key states, so the two
    # cannot disagree (a byte count named neither fp32 nor int32: both are
    # 4). `dsa_topk_mask` emits `topk_indices` as I32 regardless of the
    # surrounding model's bf16/fp16/fp8 activation dtype, and counting it
    # at activation dtype would silently miscount stage-2 HBM traffic;
    # `indexer_logits` writes F32 logits.
    # `None` (the default) means "use the role's activation dtype".
    # Applies only to the output side; a CONSUMER that reads an index tensor
    # from outside the graph declares that itself via `input_dtype` below.
    output_dtype: Optional[str] = None
    # An in-place update reads the tensor it writes: `acc.add_(x)` loads the
    # accumulator's current value as well as `x`. Traffic bills one read of
    # the output's shape per invocation beside the input edges.
    reads_own_output: bool = False
    # The op reads its inputs' shapes and none of their elements: torch's
    # `*_like` factories (`zeros_like`, `full_like`, DeepSeek-V4.1's
    # `make_identity_pre_mix(h, …)`) build a constant shaped like `h`.
    # Traffic bills the write and no input read.
    reads_input_shape_only: bool = False
    # A join with one non-empty operand is that operand: the framework hands
    # it on without copying (vLLM v0.30.0 models/step3_vl.py:704-707
    # `torch.cat(cur_feature) if len(cur_feature) > 1 else cur_feature[0]`,
    # an image with no crops). Traffic bills such a call nothing.
    passes_a_lone_operand: bool = False

    # Same, for this op's INPUT. Set for ops whose input is an index tensor
    # rather than an activation — `embedding` reads token IDs, which the
    # runtimes hold as integers, not at the model's bf16/fp16/fp8
    # activation dtype. I32 matches vLLM v1's input_ids buffer
    # (`gpu_model_runner.py`: `self.input_ids = self._make_buffer(
    # max_num_tokens, dtype=torch.int32)`). Note sglang holds input_ids as
    # int64 instead (`model_executor/input_buffers.py`), so this is
    # framework-dependent; we follow the default vLLM binding. Either way the
    # term is a rounding error next to the embedding table read — it is here
    # for fidelity, not magnitude. An op that declares it reads indices and
    # no data operand, so its kernel computes at its role's dtype
    # (`access`'s operand).
    # `None` (the default) means "use the role's activation dtype".
    input_dtype: Optional[str] = None

    # Effective KV-cache positions read per query position, as a
    # SymPy-valued function of the same `dims` dict that `flops_fn`
    # receives. Populated on attention ops whose effective s_kv differs
    # from the global s_kv — the full sparse-attention chain (score +
    # softmax + apply): dense `attention_score`, `swa_attention_score`
    # / `swa_attention_apply`, `dsa_attention_score` /
    # `dsa_attention_apply`, …
    #
    # Two consumers read this:
    #
    # 1. The engine, when evaluating each op's `shape_in` / `flops_fn`
    #    / `params_fn`. The fn's result substitutes the per-op local
    #    `s_kv`, so an SWA score's K input shows shape
    #    (B,n_heads,s_window,d_head) and an SWA apply's V input shows the
    #    same shape — per-leaf
    #    access_bytes / shape rendering reflect the windowed read.
    # 2. The KV-cache traffic consumer (`_resolve_kv_positions_per_query`
    #    in `inference/kv_topology.py`), which finds the FIRST node in
    #    the block graph that declares this fn and uses its expression
    #    as the canonical per-step KV-read cardinality for the whole
    #    block. Score ops are declared before apply ops in component
    #    YAML, so score is the de-facto canonical marker, but apply
    #    declaring the same expression is consistent — both stages of
    #    the same attention kernel really do read the same Min(s_kv, …)
    #    positions per query.
    #
    # Dense attention reads all `s_kv` positions; SWA caps at `s_window`; DSA
    # caps at `n_topk_keys` (top-k). The declaration sits next to `flops_fn` on
    # the same op so the FLOPs-side coefficient (which already counts
    # `s_kv` / `Min(s_kv,s_window)` / `Min(s_kv,n_topk_keys)`) and the KV-read side stay
    # *structurally* in sync — adding a new sparse-attention variant
    # means registering one op with two parallel lambdas, with no
    # downstream consumer code change.
    kv_positions_per_query_fn: Optional[Callable[[Dict[str, Any]], Expr]] = None

    # Effective KV-cache positions *stored* per query — parallel to
    # `kv_positions_per_query_fn` but for the *storage* side. The two
    # are NOT the same: DSA still caches every latent K position so its
    # indexer can score them at decode time (storage = `s_kv`, read =
    # `Min(s_kv, n_topk_keys)`), while true sliding-window attention can evict
    # positions outside the window (storage = `Min(s_kv, s_window)`, read = same).
    # Consumers downstream of capacity / max-batch math read this field
    # to compute per-request effective KV footprint — used by SWA models
    # to express that they don't need budget for the full sequence, only
    # for the window. Dense / DSA leave this as `s_kv` (full cache).
    kv_storage_positions_per_query_fn: Optional[Callable[[Dict[str, Any]], Expr]] = None

    # Distinct KV positions ONE STEP reads — the union over the step's
    # queries of what each attends. A decode query reads what it attends; a
    # prefill step's queries together reach further, and the kernel loads
    # every key of that union at least once. A sliding window of W over the
    # last `s_query` positions reaches `Min(s_kv, s_query + W − 1)`. The
    # walk binds it as the K / V operands' rows (`s_kv`), while the per-query
    # count above is the score matrix's key axis (`s_attend`). Unset: the
    # step's K / V operands hold what one query attends (dense: `s_kv`;
    # top-k: the per-query count). A top-k's cache read is not this: its
    # kernel gathers every query's selection, the pairs below.
    kv_positions_per_step_fn: Optional[Callable[[Dict[str, Any]], Expr]] = None

    # Live (query, key) pairs ONE STEP scores, per batch row and head, under
    # the op's selection: the per-query law above summed over the step's
    # queries, each at its own causal context (`_selected_pairs`). A decode
    # query's pairs are its read; a prefill step's are not `s_query` times
    # the last query's read, because the early queries see fewer keys than
    # the cap. The walk rebinds the node's `s_pairs` to these pairs, and a
    # stream marker hands them on with its length, so a dense attention over
    # a stream union prices each stream's mask. Unset: the phase's dense
    # pairs already are the op's (dense causal attention).
    #
    # The same pairs are the cache read of an op that selects them itself (a
    # top-k attention) or of one reading a `kv_stream_union` of index lists:
    # their kernels gather each query's listed rows by index (FlashMLA
    # sparse; vLLM `_gqa_sparse_fwd_kernel`, sglang
    # `_sparse_gqa_chunk_prefill` — a grid over queries), so a step reads
    # one cache position per live pair, the walk records the reader in
    # `Analysis.gathered`, and `traffic` bills it `CachedTensor.
    # gathered_bytes`. Queries share rows only when several requests share a
    # prompt and a shared-prefix kernel reads it once — a deployment the
    # engine does not model; for distinct requests the per-pair read is
    # exact. A stream marker's own pairs (a window, every pooled block) are
    # its mask, not a list: a reader of that one stream (an indexer scoring
    # every pooled key) streams it as a dense kernel does.
    kv_pairs_per_step_fn: Optional[Callable[[Dict[str, Any]], Expr]] = None

    # Live (query, key) pairs the op's own matrix holds, for its cost alone:
    # a banded attention that streams its window (`swa_attention_*`, whose
    # pairs are `_window_pairs`) rather than gathering an index list. The
    # walk rebinds the node's `s_pairs` to them; they are no cache read and
    # no selection, so nothing else reads them.
    cost_pairs_fn: Optional[Callable[[Dict[str, Any]], Expr]] = None

    # Whether this op may serve as the BLOCK-level KV cardinality anchor —
    # i.e. the single (read, storage) expr that summarizes "how a query
    # touches this layer's KV", used as the fallback for cached nodes that
    # don't resolve their own per-node expr, and by the single-class decode
    # read. The MAIN attention-score ops are anchors (True). Auxiliary
    # full-scan ops that ALSO carry position fns but must NOT define the
    # layer's headline cardinality — DSA's `indexer_logits` does a dense
    # O(s_kv) scan of the indexer key, but the layer's main read is the
    # top-K MLA attention — set this False so they contribute their cache's
    # cardinality PER NODE (via consumer lookup) without hijacking the
    # block summary. Default True keeps every existing op unchanged.
    kv_block_anchor: bool = True

    # Whether this op JOINS several KV streams into one tensor. Two halves of
    # the same fact, which is why they are one flag: the op's own cardinality
    # is the SUM of its inputs' (they describe disjoint positions of the same
    # cache, unlike a feature-axis `concat` whose operands describe the same
    # positions), and that sum is the length every consumer downstream of it
    # reads — not the layer's `s_kv`. DeepSeek-V4 is the case: the forward
    # cats the sliding window and the compressed latents on the sequence axis
    # and runs one attention over the union. The union of masked streams is
    # their joined per-query index lists (DeepSeek `inference/model.py`
    # `get_window_topk_idxs` + `get_compress_topk_idxs` into `sparse_attn`;
    # vLLM `combine_topk_swa_indices`), which its reader gathers.
    kv_stream_union: bool = False

    # Whether this op's KV law counts the positions of the tensor it EMITS
    # rather than of one it reads: a marker on a KV stream (a pooled latent,
    # a window). The length it declares is then the length every reader of
    # that stream reads, as a union's sum is. A reader's law — an attention
    # score's `Min(s_kv, n_topk_keys)` — describes its own read and says
    # nothing to its consumers. GLM-5.3's indexer is the case: it scores
    # ⌊s_kv/m⌋ pooled keys, not `s_kv` raw ones, and only the pooled stream
    # knows that.
    kv_stream_marker: bool = False

    # What the arriving tensors must satisfy beyond what the template labels
    # already say. A label bound twice is checked for equality by the walk
    # (`bind_labels`); a relation between two DIFFERENT labels is not a label
    # fact, and only the op's author knows it — a row scatter writes `rows`
    # rows into a sequence of `s_ctx` positions, so `rows <= s_ctx`. Called
    # with the arriving shapes in input order; returns the violation, or None
    # when the tensors satisfy it. A comparison the extents cannot decide
    # (the symbolic domain) is not a violation. `None` means the templates
    # say everything.
    # Called with the node's dims as well, for a relation the op's declared
    # dims take part in (a `patchify` side must be a multiple of `align`).
    input_contract: Optional[Callable[[Sequence[Sequence[Any]],
                                       Mapping[str, Any]],
                                      Optional[str]]] = None

    def __post_init__(self) -> None:
        self._check_layout_family()
        for declared in (self.output_dtype, self.input_dtype):
            if declared is not None:
                canonical(declared)          # an unknown name fails here
        if self.is_comm_op != (self.reduction is not None):
            raise ValueError(
                f"OpSpec({self.name!r}): `reduction` is declared by a "
                f"collective (True: it combines the members' values; False: "
                f"it moves them) and by nothing else — is_comm_op="
                f"{self.is_comm_op}, reduction={self.reduction!r}")
        if self.processor_step is not None:
            try:
                self.processor_step = ProcessorStep(self.processor_step)
            except ValueError as e:
                raise ValueError(
                    f"OpSpec({self.name!r}): unknown processor_step "
                    f"{self.processor_step!r}. Valid: "
                    f"{[k.value for k in ProcessorStep]}") from e
        # Accept either ShapeKind member or its string value during
        # registration; canonicalize to the enum so the DimMap
        # dispatch can use identity comparisons.
        if isinstance(self.shape_kind, str):
            try:
                self.shape_kind = ShapeKind(self.shape_kind)
            except ValueError as e:
                raise ValueError(
                    f"OpSpec({self.name!r}): unknown shape_kind "
                    f"{self.shape_kind!r}. Valid: "
                    f"{[k.value for k in ShapeKind]}"
                ) from e
        # `weight_shape` and the primary `TensorDecl` are one fact. Empty
        # `tensors` is the single-tensor case and reads `weight_shape`; a
        # non-empty list must lead with that same shape, or storage and
        # params would be describing two different tensors.
        if self.tensors and self.weight_shape \
                and tuple(self.tensors[0].shape) != tuple(self.weight_shape):
            raise ValueError(
                f"OpSpec({self.name!r}): weight_shape {self.weight_shape!r} "
                f"disagrees with tensors[0].shape {self.tensors[0].shape!r}")
        # Operand roles: closed vocabulary, no duplicates, and consistent
        # with the edge kinds — a position that carries a mask is the op's
        # selection, whatever else it is.
        named = [r for r in self.input_roles if r is not None]
        for r in named:
            if r not in INPUT_ROLE_VOCABULARY:
                raise ValueError(
                    f"OpSpec({self.name!r}): unknown input role {r!r}. "
                    f"Valid: {sorted(INPUT_ROLE_VOCABULARY)}")
        if len(set(named)) != len(named):
            raise ValueError(
                f"OpSpec({self.name!r}): duplicate input role in {self.input_roles!r}")
        for i, k in enumerate(self.input_kinds):
            if k == "mask" and i < len(self.input_roles) and self.input_roles[i] != "select":
                raise ValueError(
                    f"OpSpec({self.name!r}): input {i} is a mask edge but its "
                    f"role is {self.input_roles[i]!r}, not 'select'")
        # Edge-kind vocabulary is a closed set (see the fields above).
        for k in (*self.input_kinds, self.output_kind):
            if k not in EDGE_KIND_VOCABULARY:
                raise ValueError(
                    f"OpSpec({self.name!r}): unknown edge kind {k!r}. "
                    f"Valid: {sorted(EDGE_KIND_VOCABULARY)}"
                )
        if self.lookup is not None:
            self.lookup.check(self)
        if self.gather_source is not None and not (
                isinstance(self.gather_source, int)
                and 0 <= self.gather_source < self.arity()):
            raise ValueError(
                f"OpSpec({self.name!r}): gather_source {self.gather_source!r} "
                f"is not one of its {self.arity()} input(s)")

    @property
    def linear_labels(self) -> Optional[Tuple[str, str]]:
        """`(out, in)` for an op of the linear layout family, else None —
        the labels its `[Out, In]` weight carries."""
        if self.layout_family != "linear":
            return None
        return (str(self.weight_shape[0]), str(self.weight_shape[1]))

    def layout_axis_use(self) -> Dict[str, AxisUseDecl]:
        """The op's axis tags for its layout family: the in axis contracts
        by sum for the linear family; declared for the recurrent family."""
        labels = self.linear_labels
        if labels is not None:
            return {labels[1]: AxisUseDecl(category="contract", param="sum")}
        return dict(self.axis_use)

    def _check_layout_family(self) -> None:
        """A declared layout family is one the layout can place, stated
        completely."""
        family = self.layout_family
        where = f"OpSpec({self.name!r})"
        if family is None:
            if self.axis_use:
                raise ValueError(
                    f"{where}: axis_use is declared for an op of a layout "
                    "family; declare `layout_family` too")
            return
        if family not in LAYOUT_FAMILIES:
            raise ValueError(
                f"{where}: layout_family {family!r} has no layout law; "
                f"declare one of {list(LAYOUT_FAMILIES)}")
        if family == "linear":
            shape = tuple(str(x) for x in (self.weight_shape or ()))
            if len(shape) != 2 or shape[0] != str(self.out_key):
                raise ValueError(
                    f"{where}: the linear family applies one `[Out, In]` "
                    "weight; declare `weight_shape=(Out, In)` with "
                    "`out_key=Out`")
            if self.axis_use:
                raise ValueError(
                    f"{where}: a linear-family op's axis use is derived (In "
                    "contracts by sum); leave `axis_use` undeclared")
        else:
            recurrent = {axis: use for axis, use in self.axis_use.items()
                         if use.category == "recurrent"}
            states = [axis for axis, use in recurrent.items()
                      if use.param == "state"]
            if not self.carries_recurrent_state or len(recurrent) != 2 \
                    or len(states) != 1:
                raise ValueError(
                    f"{where}: the recurrent family runs a state over heads; "
                    "declare `carries_recurrent_state=True` and tag exactly "
                    "two axes `recurrent` in `axis_use`, one of them with "
                    "`param='state'`")
        tagged = set(self.layout_axis_use())
        if tagged != set(self.couples_on or ()):
            raise ValueError(
                f"{where}: couples_on {sorted(self.couples_on or ())} must "
                f"name exactly the axes its axis use tags {sorted(tagged)}")

    def input_role(self, position: int) -> Optional[str]:
        """The part input `position` plays in this op, or None when the op
        declares no role for it."""
        if 0 <= position < len(self.input_roles):
            return self.input_roles[position]
        return None

    def input_kind(self, position: int) -> str:
        """Kind of the edge this op expects at input `position`
        (default `activation` — most edges in the graph)."""
        if 0 <= position < len(self.input_kinds):
            return self.input_kinds[position]
        return "activation"

    # Display string in canonical Symbols, used by `spec_worker.py` for
    # the LLM-facing op-registry table. Evaluation does not read this;
    # per-node tooltips render the formula layer instead, whose expressions carry
    # model-level Symbols (`d_model`, `n_heads`, `d_head`, …) populated during the symbolic
    # mode pass that runs alongside the numeric traversal.
    @property
    def flops_expr(self) -> str:
        return self.flops_expr_override or _derive_display(self.flops_fn)

    @property
    def params_expr(self) -> str:
        return self.params_expr_override or _derive_display(self.params_fn)

    # -- Questions consumers ask -------------------------------------------
    #
    # Derived, so they are answered here once instead of re-derived at each
    # call site from the fields above. The distinction from a field is
    # whether the answer is already implied: `is_activation` is a fact only
    # the op's author has, `has_weight` is a reading of `weight_shape`.
    # Every one of these replaced a name list kept at a consumer, and every
    # list was missing an op by the time it was replaced.

    def has_weight(self) -> bool:
        """Whether this op puts a learnable tensor on disk."""
        return bool(self.weight_shape)

    def reads_mask(self) -> bool:
        """Whether this op's cost depends on an attention mask — whether its
        `flops_fn` reads the mask declaration `is_causal`. Asked of the
        lambda itself (on the canonical symbols), so an op that prices a
        masked matrix is found without a list of their names."""
        read: Set[str] = set()

        class _Reads(dict):
            def get(self, key, default=None):
                read.add(key)
                return super().get(key, default)

            def __getitem__(self, key):
                read.add(key)
                return super().__getitem__(key)

        try:
            self.flops_fn(_Reads(_CANONICAL_DIMS))
        except (KeyError, TypeError, AttributeError, ValueError):
            # A lambda that needs a dim the canonical set lacks stops early;
            # what it read before stopping is still what it reads.
            return "is_causal" in read
        return "is_causal" in read


    def is_tensor_source(self) -> bool:
        """A zero-input op that stores a tensor: `is_identity_source` plus
        the precomputed buffers. What arrives from one is a stored tensor,
        never a token-domain activation — which is the question the
        dispatch-zone check asks."""
        return self.arity() == 0 and self.has_weight()

    def bias_decl(self) -> Optional["TensorDecl"]:
        """The op's `.bias` tensor declaration, if it stores one: gated on
        a per-node dim when `when` is set (`linear`), unconditional otherwise
        (`layernorm`). One fact, read by both paths that ask whether a bias
        is on disk; the name list they used to consult is gone."""
        for decl in self.tensors:
            if decl.suffix == ".bias":
                return decl
        return None

    def is_identity_source(self) -> bool:
        """Whether its output IS the tensor it stores, so it moves nothing.

        `parameter_tensor` and `parameter_vector` answer yes, and the three
        clauses are why rather than a list of their names. Zero inputs:
        there is no index to gather with, so nothing is selected out of the
        tensor — which is what excludes `embedding`, whose table is stored
        but whose output is a slice it really does move. Owns a weight:
        `stack_init` reads its input but owns nothing on disk. Not a buffer:
        `rotary_inv_freq_buffer` is precomputed rather than learned, and
        whether its read is a weight read is a separate question from this
        one.

        Two consumers ask it, and they ask the same thing. `eval.traffic`
        bills the read to whoever consumes such a node rather than to the
        node itself; `layout` does not take an arriving EDGE STATE from
        one, because what arrives on that edge is a weight and a weight's
        placement is not the activation's.
        """
        return (self.arity() == 0 and self.has_weight()
                and not self.is_buffer)

    def is_bare_parameter(self) -> bool:
        """Whether its weight is a naked `nn.Parameter` rather than an
        `nn.Module`'s `.weight`, so its checkpoint name carries no suffix.

        `linear_param` answers yes and was missing from the hand-kept list
        that this replaced — DSv4's `Gate` holds exactly such a parameter.
        """
        return self.has_weight() and self.weight_tensor_suffix == ""

    def arity(self) -> int:
        """How many inputs it reads.

        `shape_in` encodes one input as a flat tuple of axis names and two or
        more as a tuple of those, so asking this question requires knowing
        the encoding. Asked here, the encoding is known in one place.

        An empty `shape_in` is two different declarations: `shape_kind:
        SOURCE` says nothing arrives — a parameter, a buffer, a state the op
        produces from itself — while a view or permute op leaves it empty
        because its one input's shape is whatever arrives.
        """
        if self.shape_in:
            return (len(self.shape_in)
                    if isinstance(self.shape_in[0], tuple) else 1)
        return 0 if self.shape_kind is ShapeKind.SOURCE else 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OP_REGISTRY: Dict[str, OpSpec] = {}


def resolve_shape_template(template, dims: Dict[str, Any]):
    """A declared shape template against a node's dims, or None.

    ONE resolver and ONE guard, shared by `OpSpec.weight_shape` and
    `CustomOpSpec.weight_shape` — the same template form. The guard is what
    eight primitives had each written out for themselves: a shape is None when
    an entry fails to resolve or lands non-positive, because a weight of width
    0 is not a weight, it is an unbound dim that would otherwise be stored as
    a real tensor of no size.

    A bare slot name is a LOOKUP, not a parse. `linear`'s slots are `in` and
    `out`, and `in` is not a legal expression in any language sympy will read —
    the lambdas this replaces did `d["in"]` and never met the question. Only an
    entry that is not a slot is evaluated as an expression, which is what makes
    `"C_in // groups"` expressible at all.

    Every name an entry mentions must be a dim of this node. A name that is
    not — a misspelt `weight_shape:` on a custom op — is a `KeyError` with the
    entry in it, not a free symbol: sympy would happily carry `nonexistent_dim`
    through as a symbol, the guard below would let it pass (it cannot be
    compared to 0), and the tensor would be stored with a shape nobody wrote.
    """
    out = []
    for entry in template:
        if entry == "_shape":
            listed = dims.get("_shape")
            if not listed:
                return None
            for item in listed:
                try:
                    if item is None or int(item) <= 0:
                        return None
                except (TypeError, ValueError):
                    pass
                out.append(item)
            continue
        if isinstance(entry, int):
            v = entry
        elif entry in dims:
            v = dims[entry]
        else:
            unknown = [name for name in re.findall(r"[A-Za-z_]\w*", str(entry))
                       if name not in dims]
            if unknown:
                raise KeyError(
                    f"weight_shape entry {entry!r} names {unknown!r}, which "
                    f"this node does not bind (dims: {sorted(dims)!r})"
                )
            if "//" in str(entry):
                left, right = str(entry).replace(" ", "").split("//", 1)
                lv = dims[left] if left in dims else _dim.eval_dim_expr(left, dims)
                rv = dims[right] if right in dims else _dim.eval_dim_expr(right, dims)
                v = _dim.exact_div(lv, rv)
            else:
                v = _dim.eval_dim_expr(str(entry), dims)
        try:
            if v is None or int(v) <= 0:
                return None
        except (TypeError, ValueError):
            pass
        out.append(v)
    return tuple(out)


def reject_divisible_parameter_names(name: str, cop,
                                      local: Optional[Mapping[str, Any]] = None) -> None:
    """A custom op may not name a parameter after a symbol some parallelism
    divides.

    The shard policy divides on READ, under the name a value is bound to — so
    the key in `dims: {…}` decides whether an axis shards, and the expression
    only supplies the extent. That is the right rule for a model symbol, where
    the name IS the axis's identity. It is the wrong rule for an op's own
    parameter, which means whatever each call site binds to it: `In` is the
    contraction width of THIS call, and two calls may bind a width one
    parallelism splits and a width none does.

    Both custom ops that borrowed `n_indexer_heads` for that parameter were
    wrong in the same way and neither said so. GPT-OSS's `mxfp4_fused_linear`
    bound it to `d_model`; the policy divides a DSA head count, so at TP=8 the
    expert `gate_up_proj` reported an eighth of its FLOPs and an eighth of its
    weight read, while `down_proj` — bound to `d_ffn_expert`, which happens to
    split by the same factor — came out right and made the pair look
    consistent. Inkling's `packed_expert_linear` repeated it verbatim.

    An unregistered name has no such second identity: it resolves against the
    per-rank frame, so `In: n_heads` still fans out to this rank's heads and
    `In: d_model` stays whole. Renaming is therefore always at least as
    correct, which is why this is a rejection and not a warning.

    Exempt are the engine-supplied symbols — `B`, `s_ctx`, `s_query`, `s_kv`
    and friends. Those are not parameters: no call site binds them, they arrive
    from the enclosing frame, and a signature that writes `[B, s_ctx, In]` is
    naming the real sequence axis, which CP really does divide. Each symbol
    declares that about itself, so this reads the registry rather than keeping
    a list.

    Scoped to the shape declarations — signature, `weight_shape`, `out_key` —
    and not to `flops` / `params`. A shape symbol is an axis of the op's own
    tensors, so a call has to say how wide it is; a cost expression may also
    read a model quantity the frame already holds (`n_experts_active`, a head
    count nobody rebinds), and rejecting those would be a rule about where a
    number comes from rather than about what an axis is called.
    """
    import backend.arch.dims as _dims

    # dims.yaml's symbols and, when the spec's own `dim_symbols` are given
    # (`SpecLoader` checks each custom op once with them), those; never
    # another loaded spec's.
    known = {**_dims.base_symbols(), **_dims.local_symbols(local or {})}
    divisible = {n: s.kind for n, s in known.items()
                 if s.kind not in ("whole", "constant")
                 and not s.engine_supplied}
    names: set = set()
    for shape in (cop.signature or {}).values():
        names.update(str(x) for x in shape if isinstance(x, str))
    names.update(str(x) for x in (cop.weight_shape or ()) if isinstance(x, str))
    if cop.out_key:
        names.add(str(cop.out_key))
    clash = sorted(n for n in names if n in divisible)
    if clash:
        raise ValueError(
            f"custom op '{name}' names parameter(s) {clash} after registered "
            f"dim symbol(s) a parallelism divides "
            f"({', '.join(f'{c}={divisible[c]}' for c in clash)}). An op "
            f"parameter means what the call site binds to it; a name the shard "
            f"policy knows divides the axis by the NAME instead, and the two "
            f"disagree wherever a call site binds a width that parallelism "
            f"does not split. Rename it (`In` / `Out` / `H`) and bind the "
            f"model symbol at the call site."
        )


def build_custom_op_spec(name: str, cop) -> Any:
    """Construct a real `OpSpec` from a `CustomOpSpec`'s declarations, so a
    spec-scoped custom op is a first-class equivalent of a registered
    primitive. SymPy-string fields (flops / params) become callables via
    `eval_dim_expr`; dim-name templates (weight_shape / secondary_outputs)
    stay templates; the signature's operand shapes become the op's
    positional `shape_in`. Every OpSpec field an author cannot reach from YAML
    keeps its OpSpec default here — a custom op that needs another value is an
    op for `register_op`.
    """
    reject_divisible_parameter_names(name, cop)

    def _expr_fn(s, zero=False):
        if s is None or s == "":
            return (lambda d: 0) if zero else None
        return lambda d, _s=str(s): _dim.eval_dim_expr(_s, d)

    sec = {
        port: tuple(str(x) for x in tmpl)
        for port, tmpl in (cop.secondary_outputs or {}).items()
    }

    # The engine reads positional input shapes; the signature is where the
    # author states them, one per named operand, in call order. A single
    # operand's shape is the flat tuple, matching how registered primitives
    # declare `shape_in` — nesting is what marks arity > 1.
    shapes = [tuple(shape) for shape in cop.signature.values()]
    shape_in = tuple(shapes) if len(shapes) > 1 else shapes[0]

    return OpSpec(
        name=name,
        flops_fn=_expr_fn(cop.flops, zero=True),
        params_fn=_expr_fn(cop.params, zero=True),
        shape_in=shape_in,
        shape_kind=cop.shape_kind,
        out_key=cop.out_key,
        axis_locked=cop.axis_locked,
        formula=cop.formula,
        description=cop.description,
        weight_shape=tuple(cop.weight_shape or ()),
        weight_tensor_suffix=cop.weight_tensor_suffix,
        secondary_outputs=sec,
        carries_recurrent_state=cop.carries_recurrent_state,
        couples_on=tuple(cop.couples_on),
        input_roles=tuple(cop.input_roles or ()),
    )


def weight_leading_axes(op: Any, dims: Mapping[str, Any]) -> Tuple[Any, ...]:
    """The axes a node's declared weight shape carries ahead of the op's own.

    A node may state the shape of the weight it stores (`shape:`, bound as
    `dims["_shape"]`). Its trailing axes must be the op's own weight template
    (`linear`'s `(out, in)`) and whatever precedes them is the node's: an
    expert bank's `linear` declares `[n_expert_copies, d_ffn, d_model]`. That
    is `torch.matmul`'s batch rule — the matrix is the last two axes, every
    axis before them is a batch axis — and it is the whole of it: the leading
    axes stack every tensor the op stores (`eval.storage`), and the
    activation's batch axes must equal them position by position
    (`eval.op_ref.OpRef.input_contract_violation`).

    `()` for a node that declares no shape, and for an op whose own template
    IS the node's shape (`parameter_tensor`'s `_shape`). A declared shape
    whose trailing axes are not the op's weight is refused: it describes some
    other tensor.
    """
    declared = dims.get("_shape")
    decls = op.tensors or ()
    template = tuple(decls[0].shape) if decls else tuple(op.weight_shape or ())
    if not declared or not template or "_shape" in template:
        return ()
    core = resolve_shape_template(template, dims)
    width = len(core or ())
    if (core is None or len(declared) < width
            or any(not _dim.equal(a, b)
                   for a, b in zip(tuple(declared)[len(declared) - width:],
                                   core, strict=True))):
        raise SpecRefusal(
            f"the declared weight shape "
            f"{[_dim.value_of(x) for x in declared]!r} does not end in the "
            f"op's own weight {[_dim.value_of(x) for x in core or ()]!r}",
            declaration="shape",
            fix="write the stored weight's full shape: any leading axes, "
                "then the op's own weight axes in its order")
    return tuple(declared[:len(declared) - width])


def declaration_is_gemm(declaration: Any) -> bool:
    """Whether a node carrying `declaration` multiplies its stored weight as
    a matrix: a primitive's `OpSpec.gemm`, a spec's own custom op's
    `CustomOpSpec.is_linear_map` (the scoreboard's `op_gemm_family` fact).
    False for a node with no declaration (a fused span asks its origins)."""
    if declaration is None:
        return False
    if isinstance(declaration, OpSpec):
        return bool(declaration.gemm)
    return bool(declaration.is_linear_map)


def weight_shape_of(op: str, declaration: Any,
                    dims: Optional[Dict[str, Any]]) -> Optional[tuple]:
    """The on-disk weight shape of a node carrying `declaration` with `dims`,
    or None.

    The join of two facts each owned elsewhere: the declaration carries the
    template (`OpSpec.weight_shape`, or a custom op's `weight_shape:`), the
    node supplies the resolved slot dims. Neither is the harness's, which is
    why input construction should read this rather than rebuild a weight out
    of the measurement scalars.

    `declaration` is what the leaf carries (`SpecIRNode.declaration`): a
    registered primitive, the spec's own custom op, or None for a node with
    no declaration (a fused span). Asking by name used to answer "no weight"
    for a spec-scoped op, which kept gpt-oss's four packed expert banks — the
    largest tensors in the model — out of every per-rank weight geometry.

    None for an op with no weight, and for a node whose dims resolve the
    template to a non-positive width. A custom op whose declaration does not
    build, or a template naming a dim the node does not bind, raises: the
    old `except Exception: return None` here read as "no weight" and dropped
    the tensor from every per-rank geometry.
    """
    if declaration is None or not dims:
        return None
    if isinstance(declaration, OpSpec):
        spec = declaration
    else:
        if not declaration.weight_shape:
            return None
        spec = build_custom_op_spec(op, declaration)
    template = spec.weight_shape
    if not template:
        return None
    shape = resolve_shape_template(template, dict(dims))
    if shape is None:
        return None
    return (*weight_leading_axes(spec, dims), *shape)


#: Hugging Face `hidden_act` string → the activation op that computes it,
#: built from every op's `config_names` as the op registers. One config name
#: names one function, so two ops declaring the same name is a registry error
#: raised at import, not a lookup that quietly takes whichever came first.
CONFIG_NAME_TO_OP: Dict[str, str] = {}


def register_op(spec: OpSpec) -> None:
    previous = OP_REGISTRY.get(spec.name)
    if previous is not None:
        for name in previous.config_names:
            CONFIG_NAME_TO_OP.pop(name, None)
    for name in spec.config_names:
        owner = CONFIG_NAME_TO_OP.get(name)
        if owner is not None and owner != spec.name:
            raise ValueError(
                f"op {spec.name!r} declares config name {name!r}, which op "
                f"{owner!r} already declares; a config activation name names "
                f"exactly one function")
        if name != name.lower():
            raise ValueError(
                f"op {spec.name!r}: config name {name!r} must be lower-case "
                f"(config strings are matched lower-cased)")
    OP_REGISTRY[spec.name] = spec
    for name in spec.config_names:
        CONFIG_NAME_TO_OP[name] = spec.name


def op_for_config_name(name: Optional[str]) -> Optional[str]:
    """The registered op a config's activation string names (`gelu` → the
    erf form, `gelu_pytorch_tanh` → `gelu_tanh`); None when no op declares
    it — a custom activation the spec can only approximate."""
    if not name:
        return None
    return CONFIG_NAME_TO_OP.get(str(name).lower())

