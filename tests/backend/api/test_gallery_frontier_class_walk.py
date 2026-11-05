"""The gallery frontier's class walk (`gallery_frontier._layout_walk`) priced
on a card (`_restamp_layout`) is that card's own walk: the parity oracle.

A model's frontier is walked once per walk class (`framework_bindings.card_class`:
the cards that walk a model alike) and read on every card of the class
without walking again (`capacity.ClassSweep`). For that to be a cache and
not a second estimator, the entry it gives a card must be the entry a sweep
walked ON that card gives (`estimate_capacity_sweep`, through
`_grid_points`), for every card of the class — not one representative:

  - the same points, (context, batch) for (context, batch): the card's own
    capacity decides which batches it serves and what its ceiling batch is;
  - each point's step equal as a function of the floors (`FloorForm.at`)
    at several floor pairs, and at the sweep's floors equal to the step
    time the card's own walk published (1e-9 relative);
  - the same measures, keyed on the card;
  - under speculative decoding, every K's points and steps likewise;
  - a dtype the class has no peak for refused on every card, in the card's
    own words (`NoPeakForDtype`), never priced at zero;
  - every card's layouts those its own frontier enumerates.

The fresh side folds its rows with `FloorForm.of` — the card's own rows at
their own peaks — while the class walk binds the walk card's rows to the
card's sheet row (`capacity.floor_form_on`): the two share the fold, not
the rows, the peaks or the fabric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import backend.api.gallery_frontier as f
from backend.api.estimate_env import resolve_estimate_environment
from backend.api.gallery_overview import _grid_points
from backend.arch.eval.precision import AS_SHIPPED, PRESETS
from backend.inference.capacity import served_points
from backend.inference.comm_cost import Fabric
from backend.inference.gpu_specs import NoPeakForDtype, get_gpu
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import class_cards, frontier_layouts, sweep_caps
from backend.inference.phases import Speculation
from backend.inference.row_roofline import FloorForm, LawBook
from backend.inference.sweep_grid import (FRONTIER_ACCEPTANCE_DEFAULT,
                                          FRONTIER_BATCH_GRID,
                                          FRONTIER_DRAFT_TOKENS,
                                          FRONTIER_MAX_GPUS,
                                          GALLERY_SEQ_LENS)
from backend.inference.defaults import GPU_MEMORY_FRACTION_DEFAULT
from tests.snapshots import resolve_snapshot

#: Link latency and launch floor pairs (µs) every step is compared at.
FLOORS_US = ((2.0, 1.0), (0.0, 0.0), (5.0, 0.5), (20.0, 4.0))
SEQ_LENS = list(GALLERY_SEQ_LENS)
GRID = list(FRONTIER_BATCH_GRID)


def _fresh_points(swept: dict, sku: str, book: LawBook) -> dict:
    """`{(context, batch): (form, decode measure, decode phase)}` of a sweep
    walked on `sku`, each served point's step folded off its own rows."""
    out = {}
    for seq, per, pt in served_points(swept, SEQ_LENS, GRID):
        dec = per["phases"]["decode"]
        draft = dec.get("speculative_draft") or {}
        form = FloorForm.of(per["per_op_timing_decode"], book=book,
                            peak_flops=per["peak_flops_per_sec"],
                            peak_bw=per["peak_hbm_bw_bytes_per_sec"],
                            fabric=Fabric.of(get_gpu(sku)),
                            draft_rows=per.get("per_op_timing_draft") or (),
                            draft_forwards=int(draft.get("forwards") or 0))
        out[(seq, dec["batch"])] = (form, (pt.get("measures") or {}).get("decode"),
                                    dec)
    return out


def _fresh(model: str, sku: str, parallel: dict, precision: dict,
           drafts: bool) -> dict:
    """What a sweep walked on `sku` alone gives the frontier: the plain
    points, and per K the speculative ones (`_fresh_points`)."""
    book = LawBook()
    req = f._layout_request(model, sku, GPU_MEMORY_FRACTION_DEFAULT, precision,
                            parallel)
    plain = _fresh_points(_grid_points(resolve_estimate_environment(req),
                                       seq_lens=SEQ_LENS, batch_grid=GRID),
                          sku, book)
    out = {"plain": plain, "draft": {}}
    if not (drafts and plain):
        return out
    ks = list(FRONTIER_DRAFT_TOKENS)
    spec = req.model_copy(update={
        "speculative_draft_tokens": ks[0], "speculative_draft": "recursive",
        "acceptance_rate": FRONTIER_ACCEPTANCE_DEFAULT,
        "selection_mode": "theoretical_only"})
    swept = _grid_points(resolve_estimate_environment(spec),
                         seq_lens=SEQ_LENS * len(ks), batch_grid=GRID,
                         s_verifies=[Speculation(k).s_verify for k in ks
                                     for _ in SEQ_LENS])
    n = len(SEQ_LENS)
    for i, k in enumerate(ks):
        at_k = {key: pts[i * n:(i + 1) * n] for key, pts in swept.items()}
        out["draft"][k] = _fresh_points(at_k, sku, book)
    return out


def _assert_same_step(priced_form: dict, fresh_form: FloorForm, where) -> None:
    got = FloorForm.from_json(priced_form)
    for comm_us, launch_us in FLOORS_US:
        want = fresh_form.at(comm_us * 1e-6, launch_us * 1e-6)
        assert got.at(comm_us * 1e-6, launch_us * 1e-6) == pytest.approx(
            want, rel=1e-9, abs=0), (*where, comm_us, launch_us)


def _assert_priced_is_fresh(priced: dict, fresh: dict, sku: str) -> None:
    assert "theory" in priced, (sku, priced)
    plain = {(t["seq"], t["sol"]["phases"]["decode"]["batch"]): t
             for t in priced["theory"]
             if not t["sol"]["phases"]["decode"].get("speculative_draft")}
    assert sorted(plain) == sorted(fresh["plain"]), sku
    for key, (form, measure, dec) in fresh["plain"].items():
        item = plain[key]
        got = item["sol"]["phases"]["decode"]
        _assert_same_step(item["floors"], form, (sku, *key))
        # At the sweep's own floors: the time the card's walk published.
        assert got["t_sol_s"] == pytest.approx(dec["t_sol_s"], rel=1e-9, abs=0)
        assert got["throughput"]["sol"]["per_gpu"] == pytest.approx(
            dec["throughput"]["sol"]["per_gpu"], rel=1e-9, abs=0)
        assert (got["batch"], got["binding"]) == (dec["batch"], dec["binding"])
        assert item["measures"].get("decode") == measure, (sku, *key)
    assert priced["draft_errors"] == [], (sku, priced["draft_errors"])
    for k, points in fresh["draft"].items():
        drafted = {(t["seq"], t["sol"]["phases"]["decode"]["batch"]): t["draft"][str(k)]
                   for t in priced["theory"] if str(k) in (t.get("draft") or {})}
        assert sorted(drafted) == sorted(points), (sku, k)
        for key, (form, _measure, _dec) in points.items():
            _assert_same_step(drafted[key], form, (sku, k, *key))


#: (a card of the class, model, layout). First: the case a regression of
#: the class walk turns red — a class of three cards with different HBM
#: (each its own ceiling batch and capacity), a model with an MTP draft
#: (every K's sweep, the draft's weighted collectives) and a layout with a
#: collective (its algorithm range re-decided on each card's fabric).
CASES = [
    ("H100-SXM", "XiaomiMiMo/MiMo-7B-RL", {"tp_attention": 2, "tp_dense_ffn": 2}),
    # Blackwell: the NVL72 card walks for the 8-GPU node card — sixteen
    # GPUs are one NVLink domain on one and two nodes on the other, so the
    # experts' all-to-all crosses the inter-node fabric on B200 only.
    ("B200", "Qwen/Qwen3-30B-A3B", {"tp_attention": 2, "tp_dense_ffn": 2,
                                    "dp_attention": 8, "ep_moe": 16}),
    ("H20", "Qwen/Qwen3-30B-A3B", {"tp_attention": 8, "tp_dense_ffn": 8,
                                   "dp_attention": 2, "ep_moe": 16}),
    # A class of ONE card (B300 is the Gallery's only SM103 part): the walk
    # card is the card, so the restamp must still be the fresh walk.
    ("B300", "meta-llama/Llama-3.2-1B", {}),
]


@pytest.mark.parametrize("gpu,model,parallel", CASES)
def test_a_class_walk_priced_on_each_card_is_that_cards_own_walk(gpu, model,
                                                                  parallel):
    cards = class_cards(gpu)
    precision = AS_SHIPPED.canonical()
    walked = f._layout_walk((model, cards, GPU_MEMORY_FRACTION_DEFAULT,
                             precision, parallel))
    assert "plain" in walked, walked
    drafts = f.draft_refusal(model) is None
    assert (walked.get("draft") is not None) == drafts
    cfg = ParallelConfig.from_dict(parallel)
    for sku in cards:
        fresh = _fresh(model, sku, parallel, precision, drafts)
        # Every case's layout serves on every card of its class.
        assert fresh["plain"] and (fresh["draft"] or not drafts), sku
        _assert_priced_is_fresh(f._restamp_layout(walked, sku, cfg), fresh, sku)


def test_a_dtype_the_class_has_no_peak_for_is_refused_on_each_card():
    """FP4 matmuls on Hopper: the walk refuses, and each card says so in
    its own name — what a sweep walked on that card raises."""
    precision = PRESETS["w_nvfp4_w4a4_kv_fp8"].canonical()
    model, parallel = "meta-llama/Llama-3.2-1B", {}
    cards = class_cards("H100-SXM")
    walked = f._layout_walk((model, cards, GPU_MEMORY_FRACTION_DEFAULT,
                             precision, parallel))
    assert "no_peak" in walked, walked
    for sku in cards:
        with pytest.raises(NoPeakForDtype) as fresh:
            req = f._layout_request(model, sku, GPU_MEMORY_FRACTION_DEFAULT,
                                    precision, parallel)
            _grid_points(resolve_estimate_environment(req), seq_lens=SEQ_LENS,
                         batch_grid=GRID)
        assert f._restamp_layout(walked, sku, ParallelConfig()) == {
            "error": str(fresh.value)}


@pytest.mark.parametrize("gpu", ["B200", "H100-SXM"])
def test_each_card_prices_the_layouts_its_own_frontier_enumerates(gpu):
    """The model walk covers every frontier layout of the card's walk class
    — that class's cards only, since a law's rows are one class's — and
    publishes them all for every card of the class (`layoutsByCard`: the
    bundle is every consumer's), while each card's Gallery entry is priced
    over its own (`galleryLayoutsByCard`: `frontier_layouts` on it)."""
    resolved = resolve_snapshot("Qwen/Qwen3-30B-A3B")
    plan = f._walk_plan(resolved, f.FrontierView(gpu))
    caps = sweep_caps({}, resolved=resolved)
    assert plan["cards"] == class_cards(gpu) and gpu in plan["cards"]
    for sku in plan["cards"]:
        own = [cfg.to_dict() for cfg in frontier_layouts(caps, sku)]
        assert plan["galleryLayoutsByCard"][sku] == own
        assert plan["layoutsByCard"][sku] == plan["layouts"]
        assert all(p in plan["layouts"] for p in own)
    assert all(ParallelConfig.from_dict(cfg).cluster_size <= FRONTIER_MAX_GPUS
               for own in plan["galleryLayoutsByCard"].values() for cfg in own)
    assert all(ParallelConfig.from_dict(cfg).cluster_size <= FRONTIER_MAX_GPUS
               for cfg in plan["layouts"])
    assert len(plan["layouts"]) == len({str(sorted(p.items()))
                                        for p in plan["layouts"]})


def test_a_view_at_another_hbm_fraction_walks_its_own_card_alone():
    view = f.FrontierView("H100-SXM", 0.8)
    assert not view.keeps_walk() and view.walk_cards() == ["H100-SXM"]
    assert f.FrontierView("H100-SXM").walk_cards() == class_cards("H100-SXM")


#: (card, model, layout) for the canonical law road. First: the case a law
#: walked on another class turns red — DeepSeek-V4-Flash, whose Blackwell
#: walk casts four activations Hopper's does not (`quantise_activation`),
#: priced on a Hopper card. Then a model whose rows no class changes.
LAW_CASES = [
    ("H100-SXM", "deepseek-ai/DeepSeek-V4-Flash",
     {"tp_attention": 8, "tp_dense_ffn": 8, "ep_moe": 8}),
    ("B200", "deepseek-ai/DeepSeek-V4-Flash",
     {"tp_attention": 8, "tp_dense_ffn": 8, "ep_moe": 8}),
    ("H100-SXM", "meta-llama/Llama-3.2-3B", {"tp_attention": 2,
                                            "tp_dense_ffn": 2}),
]

#: What identifies a per-op row for the row-set comparison: the row, its op
#: and every precision it is priced at (§1.5: compare the set, its order
#: and its dtypes, not only the time).
_ROW_FIELDS = ("uid", "op", "act_dtype", "weight_dtype", "output_dtype",
               "output_format")


def _own_walk(model: str, sku: str, parallel: dict, precision: dict) -> dict:
    req = f._layout_request(model, sku, GPU_MEMORY_FRACTION_DEFAULT, precision,
                            parallel)
    grid = [{"key": "b1", "prefill_batch_size": 1, "decode_batch_size": 1}]
    return _grid_points(resolve_estimate_environment(req),
                        seq_lens=SEQ_LENS[:1], batch_grid=grid)["b1"][0]["sol"]


def _row_set(per: dict) -> dict:
    return {phase: [tuple(r.get(k) for k in _ROW_FIELDS) for r in per[key]]
            for phase, key in (("prefill", "per_op_timing"),
                               ("decode", "per_op_timing_decode"))}


@pytest.mark.regression
@pytest.mark.parametrize("gpu,model,parallel", LAW_CASES)
def test_the_law_a_card_is_priced_by_is_its_own_classs_walk(gpu, model,
                                                            parallel):
    """The oracle for the canonical law road (`_walk_plan` → `_layout_law` →
    `servingLawPoint`): the card the view's law was walked on runs the same
    per-op rows as the card itself — same rows, same order, same dtypes —
    and the law priced on the card is the card's own single estimate."""
    from backend.inference.card import Card
    from backend.inference.sol_law import servingLawPoint
    from backend.inference.sweep_grid import GALLERY_OUTPUT_TOKENS

    precision = AS_SHIPPED.canonical()
    view = f.FrontierView(gpu)
    plan = f._walk_plan(resolve_snapshot(model), view)
    record = f._layout_law((model, plan["cards"], GPU_MEMORY_FRACTION_DEFAULT,
                            precision, parallel))
    assert "phases" in record, record
    own = _own_walk(model, gpu, parallel, precision)
    walked = (own if record["walked_on"] == gpu else
              _own_walk(model, record["walked_on"], parallel, precision))
    assert _row_set(walked) == _row_set(own), (record["walked_on"], gpu)
    served = servingLawPoint(
        record, card=Card.of(gpu), context=SEQ_LENS[0],
        outputTokens=GALLERY_OUTPUT_TOKENS, decodeBatch=1, prefillBatch=1,
        comm_latency_us=view.comm_latency_us,
        launch_floor_us=view.launch_floor_us)
    for phase in ("prefill", "decode"):
        assert served["phases"][phase]["t_sol_s"] == pytest.approx(
            own["phases"][phase]["t_sol_s"], rel=1e-9, abs=0), phase
