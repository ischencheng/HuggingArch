"""The diffusion-pipeline primitives: their shapes, their costs and the
tensors they store, as MiniMax-H3's transformer and decoders run them."""
from __future__ import annotations

from backend.arch.axis_propagation import compute_shape_out
from backend.arch.layout.axis_use import registry_violations
from backend.arch.layout.propagation import layout_handler
from backend.arch.ops import OP_REGISTRY

NEW_OPS = ("zeros", "zero_rows", "sinusoidal_embedding", "view_rows",
           "row_gather", "rope_3d", "conv1d_weight_norm", "conv_transpose1d",
           "conv_transpose1d_weight_norm", "filter_buffer", "snake_beta",
           "append_tokens", "tile_video", "untile_video")


def test_every_new_op_has_a_layout_law_and_its_axis_use():
    assert [op for op in NEW_OPS
            if layout_handler(op, OP_REGISTRY[op]) is None] == []
    assert [v for v in registry_violations(OP_REGISTRY)
            if v.split(":")[0] in NEW_OPS] == []


def test_a_transposed_conv_upsamples_rows_by_its_stride():
    dims = {"B": 2, "s_ctx": 40, "C_in": 1024, "C_out": 512, "groups": 1,
            "kernel_size": 9, "stride": 5, "bias": 1}
    assert compute_shape_out("conv_transpose1d", [(2, 40, 1024)], dims) == (
        2, 200, 512)
    op = OP_REGISTRY["conv_transpose1d"]
    # Each input row scatters k taps into every output channel; the bias
    # lands on every output row.
    assert op.flops_fn(dims) == 2 * 2 * 40 * 1024 * 512 * 9 + 2 * 200 * 512
    assert op.params_fn(dims) == 1024 * 512 * 9 + 512


def test_weight_norm_stores_a_direction_and_one_magnitude_per_slice():
    conv = OP_REGISTRY["conv1d_weight_norm"]
    assert [t.suffix for t in conv.tensors] == [".weight_v", ".weight_g", ".bias"]
    dims = {"B": 1, "s_ctx": 10, "C_in": 2048, "C_out": 1024, "groups": 1,
            "kernel_size": 7, "dilation": 1, "bias": 1}
    assert conv.params_fn(dims) == 1024 * 2048 * 7 + 1024 + 1024
    # A transposed conv's magnitudes run along its input channels, the
    # first axis of its weight, and its bias along the second.
    up = OP_REGISTRY["conv_transpose1d_weight_norm"]
    g = next(t for t in up.tensors if t.suffix == ".weight_g")
    bias = next(t for t in up.tensors if t.suffix == ".bias")
    assert g.shape == ("C_in",) and bias.axis_refs == (1,)


def test_the_adaln_table_is_read_as_one_row_per_noise_level_and_modality():
    # [B, 2 noise levels, 3 · 6 · 5376] → [B, 6, 6 · 5376]
    assert compute_shape_out("view_rows", [(1, 2, 3 * 32256)],
                             {"rows": 3}) == (1, 6, 32256)
    # one table row picked per packed row
    assert compute_shape_out("row_gather", [(1, 6, 32256), (1, 40000)],
                             {"dim": 32256}) == (1, 40000, 32256)


def test_registers_and_a_zero_token_are_appended_to_every_tile():
    assert compute_shape_out(
        "append_tokens", [(56, 1792, 2048), (1, 4, 2048), (1, 1, 2048)],
        {"dim": 2048}) == (56, 1797, 2048)
