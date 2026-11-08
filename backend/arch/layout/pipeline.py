"""Layout laws of the diffusion pipeline ops (`ops_registry`, "Diffusion
pipelines"): what a rank holds of a tensor a diffusion transformer builds
for its step, and of its autoencoders' decoder tensors.

The laws are the conservative ones. A buffer the forward builds from nothing
is whole on every rank but for the request's own batch; an op whose output axes are no input axis one to one
— overlapping tiles, a feature row read as several rows, appended tokens, a
table row picked per id — reads every operand whole and writes a whole
output; the autoencoders' convolutions and activations read their rows
whole, the weights replicated. No serving framework this engine models
splits these tensors (vLLM-Omni runs the MiniMax-H3 autoencoders on one
rank), so whole is what a rank holds; a deployment that splits them is a
law to write here, not a gather this module invents.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

from backend.arch.axis_propagation import rank_out
from backend.arch.ops import get_op

from .edge import EdgeState
from .layout import axis_fully_replicated

if TYPE_CHECKING:
    from .propagation import LayoutPropagation


def _whole(rule: "LayoutPropagation", rank: int) -> EdgeState:
    return EdgeState(tuple(rule.replicated() for _ in range(rank)), None)


def _out_rank(node: Any, in_shapes: Sequence[Any]) -> int:
    rank = rank_out(get_op(node.op), node,
                    tuple(len(shape or ()) for shape in in_shapes))
    if rank is None:
        rank = len(get_op(node.op).shape_out)
    if not rank:
        raise ValueError(f"{node.display_path}: {node.op} has no output rank")
    return rank


def _require_whole_weights(rule: "LayoutPropagation", node: Any) -> None:
    placement = rule.weights.get(node.uid)
    if placement is None:
        return
    for axis in placement.axes:
        if not axis_fully_replicated(axis):
            raise ValueError(
                f"{node.display_path}: {node.op} reads its weight whole; "
                "its `weight_axes` cut one (no law here splits it)")


def built(rule: "LayoutPropagation", node: Any,
          states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
          ) -> Tuple[List[EdgeState], EdgeState]:
    """`zero_rows`, `sinusoidal_embedding`, `filter_buffer`: built from
    nothing (or a stored table), complete. The batch axis is the request
    owner's, every other whole."""
    owner = rule.request_owner()
    layout = [(owner.layout if name == "B" and owner is not None
               else rule.replicated())
              for name in get_op(node.op).shape_out]
    return list(states), EdgeState(tuple(layout), None)


def step_rows(rule: "LayoutPropagation", node: Any,
              states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
              ) -> Tuple[List[EdgeState], EdgeState]:
    """`zeros`: the step's row buffer `[B, s_ctx, dim]`, laid out as the
    step's own inputs arrive (`model_input_state`); the scatters that fill
    it read their rows against it."""
    env = rule.env[node.uid]
    state = rule.model_input_state(
        node.display_path, (env["B"], env["s_ctx"], env["dim"]), encoder=False)
    return list(states), state


def whole(rule: "LayoutPropagation", node: Any,
          states: List[Optional[EdgeState]], in_shapes: Sequence[Any]
          ) -> Tuple[List[EdgeState], EdgeState]:
    """Every operand whole, the output whole: the regroupings
    (`tile_video`, `untile_video`, `view_rows`, `append_tokens`,
    `row_gather`) and the decoders' convolutions and activations
    (`conv_transpose1d`, `snake_beta`)."""
    _require_whole_weights(rule, node)
    settled = []
    for state in states:
        if state is not None:
            for index in range(len(state.layout)):
                state = rule.whole_axis(node, state, index, None,
                                        f"axis {index}")
        settled.append(state)
    return settled, _whole(rule, _out_rank(node, in_shapes))
