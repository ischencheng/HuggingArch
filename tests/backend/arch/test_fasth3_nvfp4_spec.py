"""FastH3 V2 8-step NVFP4 Consumer against its checkpoint, read the way
FastVideo loads it: the spec validates against the part of the repo T2VA reads,
each stage holds the header's bytes, the sparse attention is priced on the
tiles it keeps, and what is stored but never run is declared. An overlay, as
MiniMax-H3's (`docs/diffusion_pipelines.md`)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from backend.arch.eval.dim import value_of

pytestmark = pytest.mark.regression  # a whole-model walk and validation

MODEL = "FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer"
OVERLAY = str(Path(__file__).resolve().parents[3] / "overlays" / "fasth3_v2_nvfp4")


@pytest.fixture(scope="module")
def validated():
    from backend.arch.validate import validate
    return validate(MODEL, overlay_dir=OVERLAY)


@pytest.fixture(scope="module")
def resolved():
    from backend.arch.loader import SpecLoader, overlay_kwargs
    from backend.hf.metadata import fetch_config, fetch_metadata
    meta = fetch_metadata(MODEL, OVERLAY)
    return SpecLoader(**overlay_kwargs(OVERLAY)).match(
        fetch_config(MODEL, OVERLAY), model_id=MODEL, header=meta["header"])


def test_the_spec_validates_against_the_read_part_of_the_repo(validated):
    assert validated["valid"], validated["errors"][:5]
    assert validated["model_storage_bytes"]["diff"] == 0
    assert validated["quantization_metadata_bytes"]["diff"] == 0


def test_each_stage_holds_the_bytes_of_the_file_it_reads(resolved):
    from backend.arch.eval.projection import project_model
    totals = defaultdict(int)
    for row in project_model(resolved, checkpoint=True).analysis.iter_checkpoint():
        totals[row.name.split(".", 1)[0]] += (
            int(row.storage_bytes) * int(value_of(row.count)))
    # Summed from the safetensors headers' own byte spans, file by file.
    assert dict(totals) == {
        # the six BF16 shards and the NVFP4 export
        "transformer": 39_632_637_892,
        # all but the final norm and the vision tower
        "text_encoder": 15_270_283_640,
        # the INT8 decoder and the dense file's post_quant_conv
        "vae": 1_776_660_976,
        "audio_vae": 259_757_412,
    }


def test_the_dit_runs_nvfp4_with_its_gates_and_the_decoder_runs_int8(resolved):
    from backend.arch.eval.projection import project_model
    methods = defaultdict(set)
    for row in project_model(resolved, checkpoint=True).analysis.checkpoint_tensors():
        if row.tensor.matrix_dims is not None:
            methods[row.method or "none"].add(row.name)
    export = methods["nvfp4-fastvideo-export"]
    assert ("transformer.transformer_blocks.X.attn.to_gate_compress"
            "::_nvfp4_weight") in export
    assert "transformer.transformer_blocks.X.ff.fc_in::_nvfp4_weight" in export
    assert not any("adaln_proj" in name for name in export)
    assert len(methods["nvfp4-packed"]) == 7   # q, k, v, o, up, gate, down
    assert len(methods["int8-convrot"]) == 4   # qkv, out, w1, w2
    # The refiner keeps BF16 beside the NVFP4 text encoder.
    assert any("token_refiner" in n for n in methods["none"])


def test_sparse_attention_is_priced_on_the_tiles_it_keeps(resolved):
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    # 1344x768, 124 frames: 37 · 24 · 42 video rows, 414 audio, a 256-row prompt.
    prefix, video = 414 + 256, 37 * 24 * 42
    rows = prefix + video
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=rows).checkpoint()
    costs = {}
    for op, value in analysis.cost.leaves().map_with_node(
            lambda node, v: (node.op, v)).values.values():
        costs[op] = int(value_of(value.flops))
    # every row reads the non-video rows, a non-video row reads all, a video
    # row keeps ceil(20%) of the video rows
    kept = rows * prefix + video * (prefix + 7460)
    assert costs["block_sparse_attention"] == (4 * 128 + 5) * 56 * kept
    # at this clip: 0.228 of the dense matrix, FastVideo's 0.222 at 243 frames
    assert kept / rows ** 2 == pytest.approx(0.228, abs=0.001)


def test_what_the_path_never_runs_is_declared_unread(validated):
    unread = {p["module"]: p["bytes"] for p in validated["checkpoint_unread"]}
    # the dense decoder the INT8 export overwrites
    assert unread["vae.decoder"] == 3_504_187_392
    assert unread["text_encoder.model.visual"] == 1_190_533_600
    assert "transformer" not in " ".join(unread)
