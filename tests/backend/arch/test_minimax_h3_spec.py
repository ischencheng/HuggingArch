"""MiniMax-H3 (text to video and audio) against its checkpoint: the spec
validates, each stage holds the parameters the header gives it, and the
modules text-to-video never runs are accounted as unread.

The spec and its snapshot sit in an overlay (`overlays/minimax_h3`), outside
the shared corpus: a model admitted there owes every corpus pin its keys
(`tools/corpus_admit`), and H3's waits on a person's TP-split decision
(`docs/diffusion_pipelines.md`). The tests at the end walk what admission
measures, so a change to the engine that would block it again shows here."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from backend.arch.eval.dim import value_of

pytestmark = pytest.mark.regression  # a whole-model walk and validation

MODEL = "MiniMaxAI/MiniMax-H3"
OVERLAY = str(Path(__file__).resolve().parents[3] / "overlays" / "minimax_h3")


@pytest.fixture(scope="module")
def validated():
    from backend.arch.validate import validate
    return validate(MODEL, overlay_dir=OVERLAY)


def test_the_spec_validates_against_the_read_part_of_the_checkpoint(validated):
    assert validated["valid"], validated["errors"][:5]
    assert validated["model_storage_bytes"]["diff"] == 0


def test_each_stage_holds_the_parameters_the_header_gives_it():
    from backend.arch.eval.projection import project_model
    from backend.arch.loader import SpecLoader, overlay_kwargs
    from backend.hf.metadata import fetch_config, fetch_metadata
    meta = fetch_metadata(MODEL, OVERLAY)
    resolved = SpecLoader(**overlay_kwargs(OVERLAY)).match(
        fetch_config(MODEL, OVERLAY), model_id=MODEL, header=meta["header"])
    totals = defaultdict(int)
    for row in project_model(resolved, checkpoint=True).analysis.checkpoint_tensors():
        n = int(value_of(row.elements)) * int(value_of(row.count))
        stage = next(prefix for prefix in (
            "transformer.", "text_encoder.model.language_model.embed_tokens",
            "text_encoder.model.language_model.layers", "vae.", "audio_vae.")
            if row.name.startswith(prefix))
        totals[stage] += n
        if ".adaln_proj." in row.name:
            totals["adaln"] += n
    assert dict(totals) == {
        "transformer.": 33_122_992_896,
        "adaln": 13_010_457_600,
        "text_encoder.model.language_model.layers": 24_379_916_800,
        "text_encoder.model.language_model.embed_tokens": 777_912_320,
        "vae.": 2_423_531_096,
        # the 252 + 2 stored 12-tap resampling filters (3,048 values) included
        "audio_vae.": 64_939_353,
    }


def test_what_text_to_video_never_runs_is_declared_unread(validated):
    unread = {(p["module"], p["layers"]): p["params"]
              for p in validated["checkpoint_unread"]}
    assert unread[("text_encoder.model.language_model.layers", "50-63")] == (
        14 * 487_598_336)
    assert unread[("transformer_ref", None)] == 33_122_992_896


@pytest.fixture(scope="module")
def admitted():
    from backend.arch.loader import SpecLoader, overlay_kwargs
    from backend.hf.metadata import fetch_config, fetch_metadata
    meta = fetch_metadata(MODEL, OVERLAY)
    config = fetch_config(MODEL, OVERLAY)
    resolved = SpecLoader(**overlay_kwargs(OVERLAY)).match(
        config, model_id=MODEL, header=meta["header"])
    return resolved, config, meta


def test_the_saved_bytes_reconcile_with_the_part_of_the_file_it_reads(admitted):
    from backend.arch.eval.checkpoint import Reconciliation
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    resolved, _config, meta = admitted
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=128).checkpoint()
    result = Reconciliation.of(analysis, meta, MODEL)
    assert result.delta == 0 and result.measured_delta == 0


def test_the_overview_is_a_connected_graph(admitted):
    from tools.corpus_pins.checks.arch_view_graph import view_problems
    resolved, config, _meta = admitted
    assert view_problems(resolved, config) == []


def test_every_media_input_has_a_reference_of_its_own_cut(admitted):
    from backend.arch.eval.program import Program
    from tools.corpus_pins.pins.encode_surface import reference_media
    media = reference_media(Program.of(admitted[0]), "one")
    # 7 units of the transformer's 2-latent patch, read and decoded alike
    assert media["video_latents"]["vlat_h"] == 14
    assert media["denoised_video"]["vden_h"] == 14


def test_the_pack_follows_the_request_owner_under_data_parallel(admitted):
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.step_eval import evaluate_step
    step = evaluate_step(
        admitted[0], ParallelConfig(dp_attention=8, ep_moe=8),
        phase="prefill", batch=1, input_tokens=8, output_tokens=0)
    assert step is not None


def test_a_model_with_no_cache_is_bounded_by_its_activations(admitted):
    from backend.inference.capacity import estimate_pending
    from backend.inference.parallel_config import ParallelConfig
    resolved, config, _meta = admitted
    result = estimate_pending(
        config=config, resolved=resolved, num_gpus=1,
        parallel=ParallelConfig(), gpu_sku="H100-SXM",
        precision={"preset": "as_shipped"}, avg_input_tokens=4096,
        avg_output_tokens=1024, selection_mode="theoretical_only").finish()
    assert result["max_tokens_per_gpu"] is None
    assert result["max_batch_per_gpu"] > 0
    assert result["kv_cache"]["per_token_bytes"] == 0
