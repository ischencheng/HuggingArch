"""The gallery's decode Pareto frontier: which points a layout draws, which
of them survive the frontier, what a refused layout reports, and how
Kernel Bench rows fold on.

The estimator is stubbed: these pin the frontier's reading of its rows —
the same questions the overview's point tests asked of the old line
charts (a batch that does not fit is not drawn, past the published
context is empty, no room for a request draws nothing).
"""
from __future__ import annotations

import copy

import pytest

import backend.api.gallery_frontier as f
from backend.code_trace.provider import recording
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import class_cards
from backend.inference.sweep_grid import FRONTIER_BATCH_GRID, GALLERY_SEQ_LENS


@pytest.fixture(autouse=True)
def _one_computation():
    """Each test drives the worker side of a Gallery job directly — the
    fold, the read, a bundle written — so each is one recorded computation,
    as the job it stands in for is (`measure.pool.submit` runs every job in
    one: `code_trace.provider.run_recorded`). What it keeps is stored with
    the code it ran."""
    with recording():
        yield


def _pt(user, gpu, batch=1):
    return {"tok_s_per_user": user, "tok_s_per_gpu": gpu, "batch": batch}


def test_the_frontier_keeps_only_points_nothing_beats_on_both_axes():
    pts = [_pt(100, 100), _pt(50, 400), _pt(40, 300), _pt(100, 90),
           _pt(10, 1000), _pt(60, 50)]
    front = f.pareto_front(pts)
    assert [(p["tok_s_per_user"], p["tok_s_per_gpu"]) for p in front] == [
        (100, 100), (50, 400), (10, 1000)]
    assert f.pareto_front([]) == []


def _row(max_bs, batch, *, beyond=False):
    tpot = 0.01 * (1 + batch / 64)
    return {"max_batch_per_gpu": max_bs, "beyond_max_context": beyond,
            "phases": {"decode": {
                "phase": "decode", "batch": batch, "t_sol_s": tpot,
                "binding": {"tokens_per_step": 1, "effective_tokens_per_step": batch,
                            "effective_tokens_formula": {
                                "symbolic": "B × 1", "substituted": f"{batch} × 1"}},
                "throughput": {"groups": 1, "num_gpus": 2,
                               "concurrent_requests": batch, "sol": {
                    "time_per_output_token_s": tpot, "per_gpu": batch / tpot}}}}}


def _constant_forms(monkeypatch, card_form_of=None):
    """The rows' laws stubbed: each point's step is its own time whatever
    the floors — a card form of one memory term that the card's bandwidth
    turns back into that time — and its measures are carried as they are."""
    import backend.inference.capacity as capacity
    from backend.inference.gpu_specs import get_gpu, peak_hbm_bw
    from backend.inference.row_roofline import CardForm

    def constant(per, gpu_sku, *, phase, book, floors_s):
        t = per["phases"][phase]["t_sol_s"]
        return CardForm(((0.0, t * peak_hbm_bw(get_gpu(gpu_sku)), 0, 0.0, -1,
                          0.0, 1),), ("",))
    monkeypatch.setattr(capacity, "card_form_of", card_form_of or constant)
    monkeypatch.setattr(capacity, "measures_on_card", lambda m, sku: m)
    return constant


def _layout_points(item):
    """One layout of a model walked for its GPU alone and priced there
    (`_layout_walk`, `_restamp_layout`)."""
    model_id, view, parallel = item
    walked = f._layout_walk((model_id, [view.gpu_sku],
                             view.gpu_memory_fraction,
                             f._GALLERY_PRECISION.canonical(), parallel))
    return f._restamp_layout(walked, view.gpu_sku, ParallelConfig.from_dict(parallel))


def _theory_at(res, seq):
    return [p for p in res["theory"] if p["seq"] == str(seq)]


def _batch(point):
    return point["sol"]["phases"]["decode"]["batch"]


def test_a_layout_draws_the_batches_that_fit_and_max_once(monkeypatch):
    # 100 requests fit per GPU at every context but the last, which is past
    # the model's context.
    def grid(env, *, seq_lens, batch_grid, s_verifies=None):
        out = {}
        for bp in batch_grid:
            b = bp["decode_batch_size"] or 100
            out[bp["key"]] = [{"sol": _row(100, b, beyond=(s == seq_lens[-1]))}
                              for s in seq_lens]
        return out
    monkeypatch.setattr(f, "resolve_estimate_environment", lambda req: {})
    monkeypatch.setattr(f, "_grid_points", grid)
    _constant_forms(monkeypatch)
    res = _layout_points(("test/Model", f.FrontierView("B200"), ParallelConfig(tp_attention=2).to_dict()))
    pts = _theory_at(res, GALLERY_SEQ_LENS[0])
    assert sorted(_batch(p) for p in pts) == [1, 2, 4, 8, 16, 32, 64, 100]
    assert all(p["label"] == "TP2" and p["num_gpus"] == 2 for p in pts)
    assert _theory_at(res, GALLERY_SEQ_LENS[-1]) == []
    assert len(FRONTIER_BATCH_GRID) == 12


def test_a_batch_that_does_not_fit_is_not_drawn(monkeypatch):
    """62 requests fit per GPU: 64 and above are absent, max is 62 once.
    Same question the overview asked of b64 at 16K."""
    def grid(env, *, seq_lens, batch_grid, s_verifies=None):
        out = {}
        for bp in batch_grid:
            b = bp["decode_batch_size"] or 62
            out[bp["key"]] = [{"sol": _row(62, b)} for _s in seq_lens]
        return out
    monkeypatch.setattr(f, "resolve_estimate_environment", lambda req: {})
    monkeypatch.setattr(f, "_grid_points", grid)
    _constant_forms(monkeypatch)
    res = _layout_points(
        ("test/Model", f.FrontierView("B200"), ParallelConfig(tp_attention=2).to_dict()))
    pts = _theory_at(res, GALLERY_SEQ_LENS[0])
    assert sorted(_batch(p) for p in pts) == [1, 2, 4, 8, 16, 32, 62]


def test_a_layout_with_no_room_for_a_request_draws_nothing(monkeypatch):
    """Weights / KV leave max_batch 0: every context is empty, the page
    names the model under the chart."""
    why = "weights (500.00 GiB) exceed usable HBM."
    def grid(env, *, seq_lens, batch_grid, s_verifies=None):
        dead = _row(0, 1)
        dead["infeasibility_reason"] = why
        dead["memory"] = {"fits": False}
        return {bp["key"]: [{"sol": dead} for _s in seq_lens]
                for bp in batch_grid}
    monkeypatch.setattr(f, "resolve_estimate_environment", lambda req: {})
    monkeypatch.setattr(f, "_grid_points", grid)
    _constant_forms(monkeypatch)
    res = _layout_points(
        ("test/Model", f.FrontierView("B200"), ParallelConfig().to_dict()))
    assert res["theory"] == []


def test_a_refused_layout_is_reported_not_failed(monkeypatch):
    def boom(req):
        raise ValueError("does not deploy")

    def refuse(resolved, cfg):
        from backend.arch.eval.refusal import SpecRefusal
        raise SpecRefusal(
            "heads do not divide", declaration="num_heads",
            fix="divide heads across the tensor-parallel degree")
    monkeypatch.setattr(f, "resolve_estimate_environment", boom)
    monkeypatch.setattr(f, "_resolve_one", lambda mid: (None, object(), "v"))
    monkeypatch.setattr(f, "layout_verdict", refuse)
    res = f._layout_walk((
        "test/Model", ["B200"], 0.9,
        f._GALLERY_PRECISION.canonical(), ParallelConfig().to_dict()))
    assert res == {"refused": "heads do not divide"}

    sk = str(GALLERY_SEQ_LENS[0])
    entry = f._fold("test/Model", [ParallelConfig(), ParallelConfig(tp_attention=2)],
                    [res, {"theory": [{"seq": sk, "label": "TP2", "num_gpus": 2,
                                       "sol": _row(10, 10), "floors": None,
                                       "measures": {}}]}])
    assert entry["error"] is None
    assert entry["layouts"][0]["refused"] == "heads do not divide"
    assert entry["layouts"][1]["points"] == 1
    out = f._read_entry(entry, f.FrontierView("B200"))
    assert [p["batch"] for p in out["frontier"][sk]] == [10]


def test_an_engine_bug_is_not_downgraded_and_does_not_sink_the_warm(monkeypatch):
    """A plain ValueError from the walk is an error, not a refusal.

    The diagnostic walk may fail with the same engine bug. It is reported
    beside the original failure, but must not replace it — and it must not
    escape the per-layout boundary and cancel the remaining Gallery models.
    """
    def boom(req):
        raise ValueError("engine bug")

    def diagnostic(resolved, cfg):
        raise ValueError("same engine bug")

    monkeypatch.setattr(f, "resolve_estimate_environment", boom)
    monkeypatch.setattr(f, "_resolve_one", lambda mid: (None, object(), "v"))
    monkeypatch.setattr(f, "layout_verdict", diagnostic)
    res = f._layout_walk((
        "test/Model", ["B200"], 0.9,
        f._GALLERY_PRECISION.canonical(), ParallelConfig().to_dict()))
    assert res == {"error": "engine bug",
                   "diagnostic": "same engine bug"}


def test_measured_rows_fold_onto_the_frontier(monkeypatch):
    """Kernel Bench rows ride the same points as SOL: a complete measured
    step publishes tok/s beside the estimate, and the measured Pareto is
    folded from those. The theory half is not served."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))

    def apply(items, **_kw):
        for it in items:
            dec = it["sol"]["phases"]["decode"]
            dec["throughput"]["measured"] = {
                "time_per_output_token_s": 0.02, "per_gpu": 50.0}
            dec["coverage"] = {"measured": 3, "total": 3}

    monkeypatch.setattr(f, "measure_points", apply)
    sk = str(GALLERY_SEQ_LENS[0])
    sol = {"phases": {"decode": {"batch": 1, "throughput": {
        "concurrent_requests": 1, "sol": {
            "time_per_output_token_s": 0.01, "per_gpu": 100.0}}}}}
    out = f._read_entry({
        "model_id": "m", "frontier": {sk: [_pt(100, 100)]},
        "layouts": [], "error": None,
        "_theory": [{"seq": sk, "label": "1×GPU · TP1/EP1/CP1/DP1",
                     "num_gpus": 1, "sol": sol, "measures": {"decode": {}}}],
    }, f.FrontierView("B200"))
    assert "_theory" not in out
    pt = out["frontier"][sk][0]
    assert pt["tok_s_per_user"] == 100.0
    assert pt["tok_s_per_user_measured"] == 50.0
    assert pt["tok_s_per_gpu_measured"] == 50.0
    meas = out["frontier_measured"][sk][0]
    assert meas["tok_s_per_user_measured"] == 50.0


def test_the_measured_fold_does_not_walk(monkeypatch):
    """Joining Kernel Bench rows is a lookup over the cached theory, not
    another estimate walk."""
    walks = []
    monkeypatch.setattr(f, "_grid_points",
                        lambda *a, **k: walks.append(1) or {})
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    monkeypatch.setattr(f, "measure_points", lambda items, **k: None)
    sk = str(GALLERY_SEQ_LENS[0])
    f._read_entry({
        "model_id": "m", "frontier": {sk: []}, "layouts": [], "error": None,
        "_theory": [{"seq": sk, "label": "1×GPU", "num_gpus": 1,
                     "sol": {"phases": {"decode": {}}}, "measures": {}}],
    }, f.FrontierView("B200"))
    assert walks == []


def test_incomplete_measured_coverage_does_not_publish_tok_s(monkeypatch):
    """A mixed step has no measured tok/s: the page never draws a partial
    total. Same all-or-nothing the overview asked of `_safe_measured`."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))
    monkeypatch.setattr(f, "measure_points", lambda items, **k: None)
    sk = str(GALLERY_SEQ_LENS[0])
    sol = {"phases": {"decode": {"batch": 1, "throughput": {
        "concurrent_requests": 1,
        "sol": {"time_per_output_token_s": 0.01, "per_gpu": 100.0},
        "measured": None},
        "coverage": {"measured": 2, "total": 3, "states": {"missing": ["k"]}}}}}
    out = f._read_entry({
        "model_id": "m", "frontier": {sk: [_pt(100, 100)]},
        "layouts": [], "error": None,
        "_theory": [{"seq": sk, "label": "1×GPU", "num_gpus": 1,
                     "sol": sol, "measures": {"decode": {}}}],
    }, f.FrontierView("B200"))
    pt = out["frontier"][sk][0]
    assert "tok_s_per_user_measured" not in pt
    assert out["frontier_measured"][sk] == []


def test_the_route_refuses_a_gpu_outside_the_gallery_series(monkeypatch):
    """The picker offers `sweep_grid.GALLERY_GPUS`; another SKU is not swept
    cold — a card the sheet lists but the Gallery does not publish included.
    RTX-4090 is that second case and not a hypothetical one: it is a card of
    the SWEEP series (`parallel_sweep.DEFAULT_SWEEP_GPUS`, where Kernel
    Bench's measurements make it the comparison against theory), and the
    Gallery's narrower series is what this route admits."""
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app
    from backend.api.routes import gallery as route

    monkeypatch.setattr(route, "build_frontier", lambda *a: (_ for _ in ()).throw(
        AssertionError("swept")))
    client = TestClient(app)
    assert client.get("/gallery/frontier",
                      params={"gpu_sku": "A100-40G-PCIe"}).status_code == 400
    refused = client.get("/gallery/frontier", params={"gpu_sku": "RTX-4090"})
    assert refused.status_code == 400
    assert "RTX-4090" not in refused.json()["detail"].split(";")[0]


def _timed_item(seq, *, batch=4, plain_s=0.01, drafts=None, speculative=False):
    """A theory point as `_layout_points` stores it: its decode step, the
    plain step's floor form and each K's (`draft`), constants here."""
    dec = {"phase": "decode", "batch": batch, "t_sol_s": plain_s,
           "binding": {"tokens_per_step": 1, "effective_tokens_per_step": batch,
                       "effective_tokens_formula": {"symbolic": "B × 1",
                                                    "substituted": f"{batch} × 1"}},
           "throughput": {"groups": 1, "num_gpus": 2}}
    if speculative:
        dec["speculative_draft"] = {"forwards": 2}
    return {"seq": seq, "label": "TP2", "num_gpus": 2,
            "sol": {"phases": {"decode": dec}},
            "floors": None if speculative else {"constant": plain_s, "terms": []},
            "measures": {},
            "draft": {str(k): {"constant": t, "terms": []}
                      for k, t in (drafts or {}).items()}}


def test_a_speculative_view_reads_the_drafts_step_and_keeps_one_plus_alpha_k():
    """Under K drafted tokens a point's step is its form at that K, and each
    request keeps 1 + α·K tokens of it: tpot is the step over that."""
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk, drafts={2: 0.016})
    view = f.FrontierView("B200", draft_tokens=2, acceptance_rate=0.5)
    dec = f._at_view(item, view)["sol"]["phases"]["decode"]
    assert dec["t_sol_s"] == 0.016
    assert dec["binding"]["effective_tokens_per_step"] == 4 * (1 + 0.5 * 2)
    rates = dec["throughput"]["sol"]
    assert rates["time_per_output_token_s"] == 0.016 / 2
    assert rates["per_gpu"] == 4 * 2 / 0.016 / 2
    # Another acceptance reads the same form; the plain view is untouched.
    hi = f._at_view(item, f.FrontierView("B200", draft_tokens=2,
                                         acceptance_rate=1.0))
    assert hi["sol"]["phases"]["decode"]["throughput"]["sol"][
        "time_per_output_token_s"] == 0.016 / 3
    plain = f._at_view(item, f.FrontierView("B200"))["sol"]["phases"]["decode"]
    assert plain["throughput"]["sol"]["time_per_output_token_s"] == 0.01
    assert item["sol"]["phases"]["decode"]["t_sol_s"] == 0.01


def test_a_point_with_no_step_under_the_view_is_not_drawn():
    """No form at that K: not drawn there. A batch served only under a
    draft (its own step verifies drafts): not drawn under plain decode."""
    sk = str(GALLERY_SEQ_LENS[0])
    both = _timed_item(sk, batch=4, drafts={1: 0.012})
    plain_only = _timed_item(sk, batch=8, plain_s=0.02)
    drafted_only = _timed_item(sk, batch=6, drafts={1: 0.015}, speculative=True)
    entry = {"model_id": "m", "layouts": [], "error": None,
             "draft_unavailable": None,
             "_theory": [both, plain_only, drafted_only]}
    k1 = f._read_entry(entry, f.FrontierView("B200", draft_tokens=1))
    assert sorted(p["batch"] for p in k1["frontier"][sk]) == [4, 6]
    assert k1["measured_track"] == {"state": "none"}
    assert f._at_view(plain_only, f.FrontierView("B200", draft_tokens=2)) is None
    assert f._at_view(drafted_only, f.FrontierView("B200")) is None


def test_a_model_with_no_mtp_draft_reads_an_empty_speculative_frontier():
    sk = str(GALLERY_SEQ_LENS[0])
    entry = {"model_id": "m", "layouts": [], "error": None,
             "draft_unavailable": "this model declares no MTP stack",
             "_theory": [_timed_item(sk)]}
    out = f._read_entry(entry, f.FrontierView("B200", draft_tokens=3))
    assert all(pts == [] for pts in out["frontier"].values())
    assert out["draft_unavailable"] == "this model declares no MTP stack"
    assert len(f._read_entry(entry, f.FrontierView("B200"))["frontier"][sk]) == 1


def test_a_speculative_view_is_its_own_request_but_not_its_own_entry():
    plain, k2 = f.FrontierView("B200"), f.FrontierView("B200", draft_tokens=2)
    assert plain.walk_parts() == k2.walk_parts()
    assert plain.key_parts() != k2.key_parts()
    assert (f.FrontierView("B200", draft_tokens=2, acceptance_rate=0.6).key_parts()
            != k2.key_parts())
    # Acceptance is not a key under plain decode.
    assert f.FrontierView("B200", acceptance_rate=0.6).key_parts() == plain.key_parts()
        # One entry serves every K and acceptance, keyed by its floors.
    assert (plain.entry_parts() == k2.entry_parts() == f.FrontierView(
        "B200", draft_tokens=3, acceptance_rate=0.6).entry_parts())
    floored = f.FrontierView("B200", comm_latency_us=5.0)
    fraction = f.FrontierView("B200", gpu_memory_fraction=.8)
    assert not fraction.keeps_walk()
    assert floored.entry_parts() != plain.entry_parts()
    assert floored.entry_parts() != f.FrontierView(
        "B200", launch_floor_us=9.0, draft_tokens=2).entry_parts()


def test_a_serving_entry_draws_each_k_off_its_kept_step_time():
    """A serving entry's point keeps every K's step total (`draftTimes`),
    not a form: a K view reads it and keeps 1 + α·K tokens of it."""
    sk = str(GALLERY_SEQ_LENS[0])
    item = _timed_item(sk)
    item.pop("floors"), item.pop("draft")
    item["draftTimes"] = {"2": 0.016}
    view = f.FrontierView("B200", draft_tokens=2, acceptance_rate=0.5)
    dec = f._at_view(item, view)["sol"]["phases"]["decode"]
    assert dec["t_sol_s"] == 0.016
    assert dec["throughput"]["sol"]["time_per_output_token_s"] == 0.016 / 2
    assert f._at_view(item, f.FrontierView("B200", draft_tokens=1)) is None
    assert f._at_view(item, f.FrontierView("B200")) is item


def test_the_route_refuses_a_draft_length_off_the_pickers(monkeypatch):
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app
    from backend.api.routes import gallery as route

    monkeypatch.setattr(route, "build_frontier", lambda *a: (_ for _ in ()).throw(
        AssertionError("swept")))
    client = TestClient(app)
    for params in ({"draft_tokens": 5}, {"draft_tokens": 2, "acceptance_rate": 0},
                   {"draft_tokens": 2, "acceptance_rate": 1.5}):
        resp = client.get("/gallery/frontier", params={"gpu_sku": "B200", **params})
        assert resp.status_code in (400, 422), params


def _spec_grid(calls, *, fail=None):
    """A `_grid_points` stub: the plain sweep at every context, and the
    speculative one (`s_verifies`) at every context of every K — recorded
    in `calls`, raising `fail` when given."""
    def grid(env, *, seq_lens, batch_grid, s_verifies=None):
        calls.append({"seq_lens": list(seq_lens), "s_verifies": s_verifies})
        if s_verifies is not None and fail is not None:
            raise fail
        out = {}
        for bp in batch_grid:
            b = bp["decode_batch_size"] or 100
            row = _row(100, b)
            row["phases"]["prefill"] = {"batch": 1, "t_sol_s": 1.0}
            row["phases"]["decode"].update(
                phase="decode", t_sol_s=0.01,
                binding={"tokens_per_step": 1, "effective_tokens_per_step": b,
                         "effective_tokens_formula": {"symbolic": "B × 1",
                                                      "substituted": f"{b} × 1"}})
            row["phases"]["decode"]["throughput"].update(groups=1, num_gpus=2)
            out[bp["key"]] = [{"sol": row, "measures": {"prefill": {"m": 1},
                                                         "decode": {"m": 2}}}
                              for _s in seq_lens]
        return out
    return grid


def _speculating(monkeypatch, grid, card_form_of=None):
    monkeypatch.setattr(f, "resolve_estimate_environment", lambda req: {})
    monkeypatch.setattr(f, "_grid_points", grid)
    monkeypatch.setattr(f, "draft_refusal", lambda mid: None)
    _constant_forms(monkeypatch, card_form_of)


def test_every_k_is_one_sweep_of_every_context(monkeypatch):
    """The K's of `FRONTIER_DRAFT_TOKENS` are one speculative sweep: each
    context once per K, at that K's verify width — one walk of the layout,
    every K read off it."""
    from backend.inference.sweep_grid import FRONTIER_DRAFT_TOKENS
    calls: list = []
    _speculating(monkeypatch, _spec_grid(calls))
    res = _layout_points(("test/Model", f.FrontierView("B200"),
                            ParallelConfig(tp_attention=2).to_dict()))
    spec = [c for c in calls if c["s_verifies"] is not None]
    assert len(calls) == 2 and len(spec) == 1
    seqs = list(GALLERY_SEQ_LENS)
    assert spec[0]["seq_lens"] == seqs * len(FRONTIER_DRAFT_TOKENS)
    assert spec[0]["s_verifies"] == [k + 1 for k in FRONTIER_DRAFT_TOKENS
                                     for _s in seqs]
    assert res["draft_errors"] == []
    point = _theory_at(res, seqs[0])[0]
    assert sorted(point["draft"]) == sorted(str(k) for k in FRONTIER_DRAFT_TOKENS)


def test_a_layout_no_card_serves_walks_no_draft(monkeypatch):
    """No room for a request on any card of the class: no plain point, so
    no speculative view either, and its sweep is not walked."""
    calls: list = []
    inner = _spec_grid(calls)

    def grid(env, *, seq_lens, batch_grid, s_verifies=None):
        out = inner(env, seq_lens=seq_lens, batch_grid=batch_grid,
                    s_verifies=s_verifies)
        for pts in out.values():
            for pt in pts:
                pt["sol"]["max_batch_per_gpu"] = 0
        return out
    _speculating(monkeypatch, grid)
    walked = f._layout_walk(("test/Model", ["GB200", "B200"], 0.9,
                             f._GALLERY_PRECISION.canonical(),
                             ParallelConfig(tp_attention=2).to_dict()))
    assert "draft" not in walked
    assert all(c["s_verifies"] is None for c in calls)


def test_a_cached_point_keeps_what_the_frontier_reads(monkeypatch):
    """The frontier draws decode: a cached point keeps its decode phase and
    its decode measure, not the prefill half or the rest of the estimate."""
    _speculating(monkeypatch, _spec_grid([]))
    monkeypatch.setattr(f, "draft_refusal", lambda mid: "no draft")
    res = _layout_points(("test/Model", f.FrontierView("B200"),
                            ParallelConfig(tp_attention=2).to_dict()))
    point = _theory_at(res, GALLERY_SEQ_LENS[0])[0]
    assert point["sol"] == {"phases": {"decode": point["sol"]["phases"]["decode"]}}
    assert point["measures"] == {"decode": {"m": 2}}


def test_a_failed_speculative_sweep_is_every_ks_published_error(monkeypatch):
    """No K is dropped silently: a speculative sweep that fails is an error
    of the layout at every K, published with that K's view and not with
    another's."""
    from backend.inference.sweep_grid import FRONTIER_DRAFT_TOKENS
    _speculating(monkeypatch, _spec_grid([], fail=ValueError("verify refused")))
    cfg = ParallelConfig(tp_attention=2)
    res = _layout_points(("test/Model", f.FrontierView("B200"), cfg.to_dict()))
    assert res["theory"]
    assert [(e["layout"], e["draft_tokens"], e["error"]) for e in res["draft_errors"]] == [
        ("TP2", k, "verify refused") for k in FRONTIER_DRAFT_TOKENS]
    entry = f._fold("test/Model", [cfg], [res])
    entry["draft_unavailable"] = None
    k = FRONTIER_DRAFT_TOKENS[1]
    out = f._read_entry(entry, f.FrontierView("B200", draft_tokens=k))
    assert out["draft_errors"] == [
        {"layout": "TP2", "draft_tokens": k, "error": "verify refused"}]
    assert all(pts == [] for pts in out["frontier"].values())
    assert f._read_entry(entry, f.FrontierView("B200"))["draft_errors"] == []
    assert "_draft_errors" not in out


def test_one_k_that_fails_is_that_ks_error(monkeypatch):
    """A K whose steps do not price (its floor form refused) is that K's
    error; the other K's draw."""
    from backend.inference.sweep_grid import FRONTIER_DRAFT_TOKENS
    bad = FRONTIER_DRAFT_TOKENS[-1]
    calls: list = []

    constant = _constant_forms(monkeypatch)

    def card_form_of(per, gpu_sku, *, phase, book, floors_s):
        if per["phases"]["decode"].get("bad"):
            raise ValueError("floor form gives 1 s for a step of 2 s")
        return constant(per, gpu_sku, phase=phase, book=book,
                        floors_s=floors_s)

    def grid(env, *, seq_lens, batch_grid, s_verifies=None):
        out = _spec_grid(calls)(env, seq_lens=seq_lens, batch_grid=batch_grid,
                                s_verifies=s_verifies)
        for pts in out.values():
            for i, pt in enumerate(pts):
                pt["sol"] = copy.deepcopy(pt["sol"])
                if s_verifies is not None and s_verifies[i] == bad + 1:
                    pt["sol"]["phases"]["decode"]["bad"] = True
        return out
    _speculating(monkeypatch, grid, card_form_of)
    res = _layout_points(("test/Model", f.FrontierView("B200"),
                            ParallelConfig(tp_attention=2).to_dict()))
    assert [(e["draft_tokens"], e["error"]) for e in res["draft_errors"]] == [
        (bad, "floor form gives 1 s for a step of 2 s")]
    point = _theory_at(res, GALLERY_SEQ_LENS[0])[0]
    assert sorted(point["draft"]) == sorted(
        str(k) for k in FRONTIER_DRAFT_TOKENS if k != bad)


def _kept(monkeypatch, tmp_path, *, state="reads"):
    """Hermetic frontier caches over `tmp_path`, a model with a version, the
    measured track in `state`, and a record of which part of an entry each
    read asked for (`reads`) and how many measured folds ran (`folds`)."""
    from backend.measure.cache import (DiskBackend, KeyedCache, MemoryBackend,
                                       TieredBackend)
    from backend.inference.sol_pool import SolPool
    from backend.measure.scope import MeasurementScope
    entries = KeyedCache("f", DiskBackend(tmp_path / "e"))
    views = KeyedCache("v", TieredBackend(MemoryBackend(max_bytes=1 << 20),
                                          DiskBackend(tmp_path / "v")))
    reads: list = []
    real_get = entries.get

    def get(key, **kw):
        reads.append(key)
        return real_get(key, **kw)
    monkeypatch.setattr(entries, "get", get)
    monkeypatch.setattr(f, "_FRONTIER", entries)
    monkeypatch.setattr(f, "_FRONTIER_VIEWS", views)
    monkeypatch.setattr(f, "sharedSolPool", SolPool(tmp_path / "sol-pool"))
    monkeypatch.setattr(f, "_resolve_one", lambda mid: (None, object(), "v1"))
    scope = MeasurementScope(None, "lake", 1, 1, "r")
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        (scope, "reads") if state == "reads" else (f.NO_MEASUREMENTS, state)))
    folds: list = []

    def apply(items, **_kw):
        folds.append(len(items))
        for it in items:
            it["sol"]["phases"]["decode"]["throughput"]["measured"] = {
                "time_per_output_token_s": 0.05, "per_gpu": 20.0}
            it["sol"]["phases"]["decode"]["coverage"] = {"measured": 1, "total": 1}
    monkeypatch.setattr(f, "measure_points", apply)
    return entries, reads, folds


def _built_entry(monkeypatch, cards=("B200",), calls=None):
    """A one-layout model's class walk over `cards` (the walk card first):
    the plan `_frontier_read` hands out and each layout's walk."""
    _speculating(monkeypatch, _spec_grid([] if calls is None else calls))
    cfg = ParallelConfig(tp_attention=2).to_dict()
    plan = {"cards": list(cards), "layoutsByCard": {c: [cfg] for c in cards},
            "layouts": [cfg]}
    return plan, [f._layout_walk(("test/Model", list(cards), 0.9,
                                  f._GALLERY_PRECISION.canonical(), cfg))]


def test_a_kept_view_is_read_without_its_entry(monkeypatch, tmp_path):
    """A view drawn once is kept: the next request for it reads the
    drawing, not the entry — whatever the entry's size."""
    _entries, reads, folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    view = f.FrontierView("B200")
    built = f._finish_frontier("test/Model", view, plan, results)
    assert built["frontier_measured"][str(GALLERY_SEQ_LENS[0])]
    reads.clear()
    got = f._frontier_read("test/Model", view)
    assert got == {"drawn": built}
    assert reads == []
    assert folds == [len(f._restamp_layout(results[0], "B200",
                                         ParallelConfig(tp_attention=2))["theory"])]


def test_floors_and_k_read_the_entry_but_never_its_measures(monkeypatch, tmp_path):
    """Another floor or K draws off the entry's first part; the measured
    rates folded once per Kernel Bench snapshot are reused, so the measures
    part — most of an entry — is not read for them."""
    _entries, reads, folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model = "test/Model"
    f._finish_frontier(model, f.FrontierView("B200"), plan, results)
    stored = f._Stored(model, "v1")
    measures_key = stored.measures_key(f.FrontierView("B200"))
    for view in (f.FrontierView("B200", comm_latency_us=5.0),
                 f.FrontierView("B200", draft_tokens=2),
                 f.FrontierView("B200", draft_tokens=3, acceptance_rate=0.6)):
        reads.clear()
        got = f._frontier_read(model, view)
        assert "drawn" in got
        assert reads == [stored.entry_key(view)]
        assert measures_key not in reads
    assert len(folds) == 1
    # A new snapshot folds again, off the measures part.
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        MeasurementScope(None, "lake", 2, 2, "r"), "reads"))
    reads.clear()
    assert "drawn" in f._frontier_read(model, f.FrontierView("B200"))
    assert measures_key in reads and len(folds) == 2


def test_an_entry_missing_its_measures_part_is_priced_again(monkeypatch, tmp_path):
    """The two parts are one entry: a view that needs the measures and finds
    only the first part prices the model again rather than drawing a
    measured track of nothing."""
    entries, _reads, _folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model = "test/Model"
    stored = f._Stored(model, "v1")
    core, _measures = f._entry_parts(f._restamp(
        model, {**plan, "layouts": [{"parallel": p, "walk": r} for p, r in
                                     zip(plan["layouts"], results, strict=True)]},
        "B200"))
    entries.put(stored.entry_key(f.FrontierView("B200")), core)
    monkeypatch.setattr(f, "frontier_layouts", lambda caps, gpu: [ParallelConfig()])
    monkeypatch.setattr(f, "class_frontier_layouts", lambda caps, cards: [ParallelConfig()])
    monkeypatch.setattr(f, "sweep_caps", lambda *a, **k: {})
    monkeypatch.setattr(f, "parallel_configs_for_mode", lambda *a, **k: [])
    got = f._frontier_read(model, f.FrontierView("B200"))
    # The plan is the view card's walk class: a law never spans two.
    cards = f.FrontierView("B200").walk_cards()
    assert cards == class_cards("B200") and "B300" not in cards
    assert got == {"walk": {"cards": cards,
                            "layoutsByCard": {c: [ParallelConfig().to_dict()]
                                                for c in cards},
                            "galleryLayoutsByCard": {
                                c: [ParallelConfig().to_dict()] for c in cards},
                            "layouts": [ParallelConfig().to_dict()]}}
    # A view that reads no measures draws off the first part alone.
    assert "drawn" in f._frontier_read(model, f.FrontierView("B200", draft_tokens=1))


def test_another_gpu_of_the_class_is_priced_off_the_kept_walk(monkeypatch, tmp_path):
    """Switching GPU within a walk class walks nothing: the first GPU's
    request walked the model for every card of the class and kept it; the
    other card's entry is priced off that walk — its own capacity, its own
    price — kept, and drawn."""
    entries, _reads, _folds = _kept(monkeypatch, tmp_path)
    calls: list = []
    plan, results = _built_entry(monkeypatch, cards=("GB200", "B200"), calls=calls)
    f._finish_frontier("test/Model", f.FrontierView("B200"), plan, results)
    walked = len(calls)
    got = f._frontier_read("test/Model", f.FrontierView("GB200"))
    assert "drawn" in got and got["drawn"]["frontier"][str(GALLERY_SEQ_LENS[0])]
    assert len(calls) == walked
    assert entries.get(f._Stored("test/Model", "v1").entry_key(
        f.FrontierView("GB200"))) is not None
    # A view at another HBM fraction does not read the class walk: it walks
    # its own card alone.
    monkeypatch.setattr(f, "frontier_layouts", lambda caps, gpu: [ParallelConfig()])
    monkeypatch.setattr(f, "class_frontier_layouts", lambda caps, cards: [ParallelConfig()])
    monkeypatch.setattr(f, "sweep_caps", lambda *a, **k: {})
    got = f._frontier_read("test/Model", f.FrontierView("GB200", 0.8))
    assert got["walk"]["cards"] == ["GB200"]


def test_the_lakes_answer_reaches_the_pricing_and_never_the_walk(
        monkeypatch, tmp_path):
    """A class walk is theory: its key carries no measurement scope, and a
    card priced off a kept walk is priced under whatever the lake answered
    this request — down or up — without walking again. So a lake outage
    never re-walks a class, and never keeps a walk under an outage's key."""
    _entries, _reads, folds = _kept(monkeypatch, tmp_path)
    calls: list = []
    plan, results = _built_entry(monkeypatch, cards=("GB200", "B200"),
                                 calls=calls)
    f._finish_frontier("test/Model", f.FrontierView("B200"), plan, results)
    walked = len(calls)
    stored = f._Stored("test/Model", "v1")
    # One walk key, whatever the track: the walk tier is keyed under `none`.
    keys = set()
    for state in ("reads", "unavailable", "none"):
        _track(monkeypatch, state)
        keys.add(stored.walk_identity(f.FrontierView("GB200")).key)
    assert len(keys) == 1
    # The lake down: the other card is priced off the walk all the same,
    # its drawing SOL-only and labelled for the outage.
    _track(monkeypatch, "unavailable")
    down = f._frontier_read("test/Model", f.FrontierView("GB200"))["drawn"]
    assert down["measured_track"] == {"state": "unavailable"}
    assert down["frontier"][str(GALLERY_SEQ_LENS[0])]
    assert len(calls) == walked
    # The lake back: the same entry folds its rates, still without walking.
    _track(monkeypatch, "reads")
    up = f._frontier_read("test/Model", f.FrontierView("GB200"))["drawn"]
    assert up["measured_track"]["state"] == "reads"
    assert up["frontier_measured"][str(GALLERY_SEQ_LENS[0])]
    assert len(calls) == walked and folds


def test_the_kept_parts_carry_no_measures_in_the_first(monkeypatch, tmp_path):
    entries, _reads, folds = _kept(monkeypatch, tmp_path, state="none")
    plan, results = _built_entry(monkeypatch)
    f._finish_frontier("test/Model", f.FrontierView("B200"), plan, results)
    stored = f._Stored("test/Model", "v1")
    core = entries.get(stored.entry_key(f.FrontierView("B200")))
    measures = entries.get(stored.measures_key(f.FrontierView("B200")))
    assert all("measures" not in t for t in core["_theory"])
    assert len(measures["measures"]) == len(core["_theory"])
    assert folds == []


# --- Kernel Bench coverage of the frontier ---------------------------------

def _covered_item(seq, *, batch, tpot, coverage):
    """A theory point and the coverage the measured fold states of it; a
    fully measured point gets a measured rate."""
    rates = {"concurrent_requests": batch,
             "sol": {"time_per_output_token_s": tpot, "per_gpu": batch / tpot}}
    return {"seq": seq, "label": f"b{batch}", "num_gpus": 1,
            "sol": {"phases": {"decode": {"batch": batch, "throughput": rates}}},
            "measures": {"decode": {"coverage": coverage}}}


def _read_folded(monkeypatch, items):
    """`_read_entry` with the rows read: the stubbed fold states each
    point's coverage as its measure names it."""
    from backend.measure.scope import MeasurementScope
    monkeypatch.setattr(f, "_frontier_scope",
                        lambda: (MeasurementScope(None, "lake", 1, 1, "r"), "reads"))

    def apply(pts, **_kw):
        for it in pts:
            cov = it["measures"]["decode"]["coverage"]
            dec = it["sol"]["phases"]["decode"]
            dec["coverage"] = cov
            if cov["measured"] == cov["total"]:
                tpot = dec["throughput"]["sol"]["time_per_output_token_s"]
                dec["throughput"]["measured"] = {
                    "time_per_output_token_s": tpot * 2,
                    "per_gpu": dec["throughput"]["sol"]["per_gpu"] / 2}
    monkeypatch.setattr(f, "measure_points", apply)
    monkeypatch.setattr(f, "_at_view", lambda item, view, laws=None: item)
    return f._read_entry({"model_id": "m", "layouts": [], "error": None,
                          "_theory": items}, f.FrontierView("B200"))


def test_coverage_of_a_fully_measured_frontier(monkeypatch):
    sk = str(GALLERY_SEQ_LENS[0])
    full = {"measured": 4, "total": 4}
    out = _read_folded(monkeypatch, [
        _covered_item(sk, batch=1, tpot=0.01, coverage=full),
        _covered_item(sk, batch=8, tpot=0.02, coverage=full)])
    assert out["measured_track"]["state"] == "reads"
    assert len(out["frontier_measured"][sk]) == 2
    assert out["measured_track"]["coverage"][sk] == {
        "points": 2, "complete": 2, "best": {"measured": 4, "total": 4},
        "median": 1.0}


def test_coverage_counts_only_the_points_on_the_frontier(monkeypatch):
    """A point the frontier drops (beaten on both axes) is not drawn: its
    unmeasured keys are not listed. Each key counts the frontier points
    that need it, most needed first."""
    sk = str(GALLERY_SEQ_LENS[0])
    out = _read_folded(monkeypatch, [
        _covered_item(sk, batch=1, tpot=0.01, coverage={
            "measured": 3, "total": 4, "states": {"missing": ["gemm"]}}),
        _covered_item(sk, batch=8, tpot=0.02, coverage={
            "measured": 1, "total": 4, "states": {
                "missing": ["attn", "gemm"], "unplaced": ["rope"]}}),
        # Slower and fewer tokens per GPU than batch 8: off the frontier.
        _covered_item(sk, batch=2, tpot=0.05, coverage={
            "measured": 0, "total": 4, "states": {"ambiguous": ["norm"]}}),
    ])
    cov = out["measured_track"]["coverage"][sk]
    assert cov["points"] == 2 and cov["complete"] == 0
    assert cov["best"] == {"measured": 3, "total": 4}
    assert cov["median"] == (0.75 + 0.25) / 2
    assert cov["states"] == {
        "missing": {"keys": 2, "top": [{"key": "gemm", "points": 2},
                                       {"key": "attn", "points": 1}]},
        "unplaced": {"keys": 1, "top": [{"key": "rope", "points": 1}]}}


def test_coverage_lists_the_most_needed_keys_and_counts_them_all():
    many = [f"k{i:02d}" for i in range(f.COVERAGE_KEYS_LISTED + 5)]
    held = f._PointCoverage()
    held.add(0, {"measured": 0, "total": 20, "states": {"missing": many}})
    held.add(1, {"measured": 1, "total": 20, "states": {"missing": many[-1:]}})
    kept = held.to_json()
    # Each key is named once, however many points hold it.
    assert len(kept["keys"]) == len(many)
    cov = f.frontier_coverage(list(kept["points"].values()), kept["keys"])
    missing = cov["states"]["missing"]
    assert missing["keys"] == len(many)
    assert len(missing["top"]) == f.COVERAGE_KEYS_LISTED
    assert missing["top"][0] == {"key": many[-1], "points": 2}


def test_coverage_of_a_frontier_nothing_measures(monkeypatch):
    sk = str(GALLERY_SEQ_LENS[0])
    out = _read_folded(monkeypatch, [_covered_item(sk, batch=1, tpot=0.01, coverage={
        "measured": 0, "total": 2, "states": {
            "missing": ["gemm"],
            "declined": [{"key": "fp4 gemm", "reason": "no dtype token"}]}})])
    cov = out["measured_track"]["coverage"][sk]
    assert cov["complete"] == 0 and cov["median"] == 0.0
    assert cov["best"] == {"measured": 0, "total": 2}
    assert cov["states"]["declined"] == {"keys": 1, "top": [
        {"key": "fp4 gemm", "reason": "no dtype token", "points": 1}]}
    # A context no layout serves has no points, and says so.
    assert out["measured_track"]["coverage"][str(GALLERY_SEQ_LENS[1])] == {
        "points": 0, "complete": 0, "best": None, "median": None}


def test_a_frontier_point_the_fold_did_not_reach_is_refused(monkeypatch):
    """A partial fold drops the measured overlay instead of the whole page.

    A point no measure reached cannot be reported as zero coverage, and one
    stale/partial overlay must not 500 the entire Gallery response. The SOL
    drawing remains valid and is labelled unavailable for this read.
    """
    sk = str(GALLERY_SEQ_LENS[0])
    item = _covered_item(sk, batch=1, tpot=0.01, coverage={"measured": 0, "total": 2})
    item["measures"] = {}
    out = _read_folded(monkeypatch, [item])
    assert out["frontier"][sk]
    assert out["measured_track"] == {"state": "unavailable"}


def test_no_coverage_when_the_lake_is_unavailable(monkeypatch):
    from backend.measure.scope import NONE
    monkeypatch.setattr(f, "_frontier_scope", lambda: (NONE, "unavailable"))
    sk = str(GALLERY_SEQ_LENS[0])
    out = f._read_entry({"model_id": "m", "layouts": [], "error": None,
                         "_theory": [_timed_item(sk)]}, f.FrontierView("B200"))
    assert out["frontier"][sk]
    assert out["measured_track"] == {"state": "unavailable"}


def test_a_fold_the_lake_drops_mid_request_draws_sol_only(
        monkeypatch, tmp_path):
    """The lake answered its version — the view reads a measured track —
    but the rows' lookup failed: the model still draws, SOL-only, its track
    "unavailable", rather than the whole frontier 503ing. That drawing is
    never kept under the measured track's key: the next read the lake
    answers folds the rates again instead of serving the outage's drawing."""
    from backend.measure.lookup import MeasurementsUnavailable
    _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")
    sk = str(GALLERY_SEQ_LENS[0])

    def unavailable(items, **_kw):
        raise MeasurementsUnavailable(
            "measurements unavailable: the Kernel Bench service did not "
            "answer (ReadTimeout)")

    monkeypatch.setattr(f, "measure_points", unavailable)
    built = f._finish_frontier(model, view, plan, results)
    assert built["measured_track"] == {"state": "unavailable"}
    assert built["frontier"][sk]
    assert built["frontier_measured"][sk] == []

    folds: list = []

    def apply(items, **_kw):
        folds.append(len(items))
        for it in items:
            dec = it["sol"]["phases"]["decode"]
            dec["throughput"]["measured"] = {
                "time_per_output_token_s": 0.05, "per_gpu": 20.0}
            dec["coverage"] = {"measured": 1, "total": 1}

    monkeypatch.setattr(f, "measure_points", apply)
    got = f._frontier_read(model, view)["drawn"]
    assert folds  # the degraded drawing was not kept: the fold ran again
    assert got["measured_track"]["state"] == "reads"
    assert got["frontier_measured"][sk]


def _track(monkeypatch, state):
    """The measured track every read sees from now on: the lake answering
    (`reads`, the scope `_kept` folds under) or not (`unavailable`)."""
    from backend.measure.scope import MeasurementScope
    scope = MeasurementScope(None, "lake", 1, 1, "r")
    monkeypatch.setattr(f, "_frontier_scope", lambda: (
        (scope, "reads") if state == "reads" else (f.NO_MEASUREMENTS, state)))


def test_a_flapping_lake_reads_both_drawings_back(monkeypatch, tmp_path):
    """A lake that flaps between answering and not moves the measured
    track between "reads" and "unavailable". Each is drawn off the entry
    once and kept — the unavailable one under scope `none`, whatever the
    outage — so the next flip reads its drawing, not the entry (a model's
    entry is up to ~150 MB)."""
    _entries, reads, _folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")
    sk = str(GALLERY_SEQ_LENS[0])
    measured = f._finish_frontier(model, view, plan, results)
    _track(monkeypatch, "unavailable")
    first = f._frontier_read(model, view)["drawn"]
    assert first["measured_track"] == {"state": "unavailable"}
    def sol(drawing):
        return {k: [(p["layout"], p["batch"], p["tok_s_per_gpu"]) for p in pts]
                for k, pts in drawing["frontier"].items()}
    assert sol(first) == sol(measured)
    assert first["frontier_measured"][sk] == []
    for state, want in (("reads", measured), ("unavailable", first),
                        ("reads", measured)):
        _track(monkeypatch, state)
        reads.clear()
        assert f._frontier_read(model, view) == {"drawn": want}
        assert reads == [], state


def test_a_fold_the_lake_drops_keeps_the_sol_drawing(monkeypatch, tmp_path):
    """The SOL drawing a failed fold draws is the unavailable track's: a
    later request that finds the lake down reads it back without the
    entry."""
    from backend.measure.lookup import MeasurementsUnavailable
    _entries, reads, _folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    model, view = "test/Model", f.FrontierView("B200")

    def unavailable(items, **_kw):
        raise MeasurementsUnavailable("measurements unavailable: ReadTimeout")

    monkeypatch.setattr(f, "measure_points", unavailable)
    built = f._finish_frontier(model, view, plan, results)
    _track(monkeypatch, "unavailable")
    reads.clear()
    assert f._frontier_read(model, view) == {"drawn": built}
    assert reads == []


def test_one_request_reads_the_lake_once(monkeypatch):
    """A frontier request reads the lake's version once and draws every
    model under that answer: one response reads one lake, and a lake that
    does not answer costs one timeout, not one per model."""
    import asyncio
    asked: list = []
    track = (f.NO_MEASUREMENTS, "unavailable")

    def scope():
        asked.append(1)
        return track

    seen: list = []

    def read(model_id, view, got=None):
        seen.append(got)
        return {"drawn": {"model_id": model_id}}

    monkeypatch.setattr(f, "_frontier_scope", scope)
    monkeypatch.setattr(f, "_frontier_read", read)
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: ["a/1", "b/2", "c/3"])
    out = asyncio.run(f.build_frontier(f.FrontierView("B200")))
    assert [m["model_id"] for m in out["models"]] == ["a/1", "b/2", "c/3"]
    assert asked == [1]
    assert seen == [track] * 3


def test_no_coverage_under_a_speculative_view(monkeypatch):
    monkeypatch.setattr(f, "_frontier_scope", lambda: (_ for _ in ()).throw(
        AssertionError("a speculative view reads no rows")))
    sk = str(GALLERY_SEQ_LENS[0])
    out = f._read_entry({"model_id": "m", "layouts": [], "error": None,
                         "_theory": [_timed_item(sk, drafts={1: 0.012})]},
                        f.FrontierView("B200", draft_tokens=1))
    assert out["frontier"][sk]
    assert out["measured_track"] == {"state": "none"}


def test_a_kept_view_carries_its_coverage_and_floors_reuse_the_fold(
        monkeypatch, tmp_path):
    """The coverage is the fold's, read once per measurement scope: another
    view's floors redraw it from the kept rates without folding again, and
    a kept view carries it."""
    _entries, _reads, folds = _kept(monkeypatch, tmp_path)
    plan, results = _built_entry(monkeypatch)
    view = f.FrontierView("B200")
    first = f._finish_frontier("test/Model", view, plan, results)
    sk = str(GALLERY_SEQ_LENS[0])
    cov = first["measured_track"]["coverage"][sk]
    assert cov["points"] == len(first["frontier"][sk]) > 0
    assert cov["complete"] == cov["points"]
    n = len(folds)
    other = f._frontier_read("test/Model", view._replace(comm_latency_us=7.0))
    assert len(folds) == n
    assert other["drawn"]["measured_track"]["coverage"][sk]["points"] == len(
        other["drawn"]["frontier"][sk])
    assert f._frontier_read("test/Model", view)["drawn"]["measured_track"] == (
        first["measured_track"])


def test_a_cold_frontier_publishes_kept_models_at_once_and_computes_in_the_clients_order(
        monkeypatch):
    """While the frontier computes, each finished model is published with
    the response's head (`offload.report_partial`), so a page draws it at
    once. A model already kept lands first even when the client did not ask
    for it — its read waits behind nothing that computes — and of the models
    that compute, the one the client put first (`priority`) computes first."""
    import asyncio

    from backend.api import offload

    monkeypatch.setattr(f, "_frontier_scope", lambda: ("none-scope", "none"))
    monkeypatch.setattr(f, "_model_concurrency", lambda: 1)
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: ["a/1", "b/2", "c/3"])

    def read(model_id, view, got=None):
        if model_id == "c/3":
            return {"drawn": {"model_id": model_id}}
        return {"walk": {"cards": [], "layouts": []}}

    computed: list = []

    async def compute(model_id, view, track, got):
        computed.append(model_id)
        await asyncio.sleep(0.3)
        return {"model_id": model_id}

    monkeypatch.setattr(f, "_frontier_read", read)
    monkeypatch.setattr(f, "_compute_frontier", compute)
    view = f.FrontierView("B200")

    async def scenario():
        with pytest.raises(offload.StillComputing) as info:
            await offload.within_deadline(
                f.frontier_request_key(view), lambda: f.build_frontier(view),
                hint=["b/2"])
        partial = info.value.partial
        offload.set_deadline(None)
        whole = await offload.within_deadline(
            f.frontier_request_key(view), lambda: f.build_frontier(view))
        return partial, whole

    monkeypatch.setattr(offload, "_deadline_s", 0.15)
    try:
        partial, whole = asyncio.run(scenario())
    finally:
        offload.set_deadline(None)
    assert partial["head"]["seq_lens"] == list(GALLERY_SEQ_LENS)
    assert partial["head"]["config"]["gpu_sku"] == "B200"
    assert [m["model_id"] for m in partial["items"]][:1] == ["c/3"]
    assert computed == ["b/2", "a/1"]
    assert [m["model_id"] for m in whole["models"]] == ["a/1", "b/2", "c/3"]
    assert {k: whole[k] for k in partial["head"]} == partial["head"]
