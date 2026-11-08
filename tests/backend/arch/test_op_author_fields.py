"""Author facts on `OpSpec` that replaced consumer-side op-name lists.

Each field's extension is pinned so the declaration cannot quietly widen
or shrink past what the consumer it replaced meant.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.ops import (
    OP_REGISTRY, collective_schedule, op_labels, sp_axis_effect,
)


def _ops_where(pred):
    return {name for name, spec in OP_REGISTRY.items() if pred(spec)}


def test_the_lookup_table_is_the_embedding():
    # ...and the hashed multi-row lookup into one table (Qwen3.8 n-gram);
    # DeepSeek-V4's hash router gathers from the `tid2eid` parameter handed
    # to it on input 1.
    lookups = {name: (spec.lookup.ids, spec.lookup.table)
               for name, spec in OP_REGISTRY.items() if spec.lookup}
    assert lookups == {"embedding": (0, None), "hash_embedding": (0, None),
                       "hash_route": (0, 1)}


def test_tp_axis_placement_is_the_six_weight_ops_the_demotion_covered():
    # The extension of the name list `sharding.leaf_class` used to consult:
    # the vocab-parallel table and the bare parameters stay out. `conv2d`
    # joins as the vision downsampler's weight-bearing op (like conv1d), and
    # `conv1d_weight_norm` as conv1d with its weight stored weight-normed.
    assert _ops_where(lambda s: s.tp_axis_placement) == {
        "linear", "rmsnorm", "layernorm", "conv1d", "conv2d", "per_head_linear",
        "conv1d_weight_norm"}
    assert all(OP_REGISTRY[n].has_weight() for n in _ops_where(lambda s: s.tp_axis_placement))


def test_windowed_and_sink_ops_are_the_ones_the_span_readers_named():
    assert _ops_where(lambda s: s.windowed) == {
        "swa_attention_score", "swa_softmax", "swa_softmax_with_sink",
        "swa_softmax_key0_bias", "swa_attention_apply", "sliding_window_kv"}
    assert _ops_where(lambda s: s.sink) == {"softmax_with_sink", "swa_softmax_with_sink"}


def test_collective_effects_are_declared_on_the_collectives():
    assert {n: sp_axis_effect(n) for n in ("reduce_scatter", "all_gather", "all_reduce", "all_to_all")} == {
        "reduce_scatter": "divide", "all_gather": "multiply", "all_reduce": "", "all_to_all": ""}
    assert _ops_where(lambda s: s.collective_schedule) == {"all_reduce", "send_recv"}
    assert collective_schedule("all_reduce") == "hierarchical"
    assert collective_schedule("send_recv") == "point_to_point"
    assert sp_axis_effect("not_an_op") == "" and collective_schedule("not_an_op") == ""
    assert all(OP_REGISTRY[n].is_comm_op for n in _ops_where(lambda s: s.sp_axis_effect or s.collective_schedule))


def test_labels_cover_what_the_render_table_used_to_spell():
    labels = op_labels()
    assert labels["rmsnorm"] == "RMSNorm" and labels["attention_score"] == "Q @ K^T"
    assert set(labels) == {"rmsnorm", "layernorm", "embedding", "linear", "add", "mul", "rope",
                           "rope_emb", "rope_emb_inverse", "mrope", "rope_2d", "rope_3d", "xdrope", "hash_embedding", "softmax", "attention_score", "attention_apply",
                           "hadamard_transform", "indexer_logits", "dsa_topk_mask",
                           "dsa_attention_score",
                           "dsa_attention_apply", "dsa_softmax"}
    from backend.rendering._consts import OP_LABEL
    assert dict(OP_LABEL) == labels


def test_the_rotations_and_their_position_streams():
    """`rotary_position_axes` replaced the rotation names spelled in the
    structural diff and the view-absorb hosts: one stream for RoPE and its
    inverse, two for a vision tower's (h, w) rotation, three for multimodal
    RoPE and for a video grid's (t, h, w) rotation, four for XD-RoPE — and
    the two sectioned ones name the rope parameter their kernel is built
    from."""
    assert {n: s.rotary_position_axes for n, s in OP_REGISTRY.items()
            if s.rotary_position_axes} == {
        "rope": 1, "rope_emb": 1, "rope_emb_inverse": 1, "rope_2d": 2,
        "mrope": 3, "rope_3d": 3, "xdrope": 4}
    assert {n: s.rotary_sections_key for n, s in OP_REGISTRY.items()
            if s.rotary_sections_key} == {
        "mrope": "mrope_section", "xdrope": "xdrope_section"}
