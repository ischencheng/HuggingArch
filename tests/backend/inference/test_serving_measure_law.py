"""A serving-law record owns the measured row keys, not just SOL times."""
from __future__ import annotations

import copy
import json
import math
import time
from typing import Any
from dataclasses import replace

from backend.api._models import InferenceEstimateRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.api.gallery_frontier import (
    _GALLERY_PRECISION, _layout_law, _layout_request,
)
import backend.api.gallery_frontier as gallery
from backend.inference.batch_sweep import measure_points
from backend.inference.capacity import estimate_pending
from backend.inference.card import Card
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import class_cards
from backend.inference.sweep_grid import FRONTIER_BATCH_GRID, GALLERY_SEQ_LENS
from backend.inference.serving_view import servingLawPointView
from backend.measure.auto_sweep import PointMeasure
from backend.inference.sol_law import (
    servingMeasurePoints,
    servingMeasurePoint,
    servingLawPoint,
    servingLawRecord,
)
from tests.snapshots import snapshot_config
from tests.sweep_parity import FixtureBench

MODEL = "Qwen/Qwen3-30B-A3B"
PARALLEL = ParallelConfig(tp_attention=2, tp_dense_ffn=2, ep_moe=2)
LONG_CASES = [
    (MODEL, PARALLEL),
    ("deepseek-ai/DeepSeek-V4.1-Flash",
     ParallelConfig(ep_moe=4, dp_attention=4)),
    ("moonshotai/Kimi-K3",
     ParallelConfig(tp_attention=4, tp_dense_ffn=4, ep_moe=16,
                    dp_attention=4)),
]


def request(gpu: str) -> InferenceEstimateRequest:
    return InferenceEstimateRequest(
        model_id=MODEL, gpu_sku=gpu, avg_input_tokens=1024,
        avg_output_tokens=256, parallel=PARALLEL.to_dict(),
        framework="vllm", selection_mode="theoretical_only")


def same(got, want, path="point"):
    """Whole public shape: same keys, non-numbers equal, numbers to 1e-12."""
    if isinstance(got, dict):
        assert set(got) == set(want), f"{path}: {sorted(got)} != {sorted(want)}"
        for key in got:
            same(got[key], want[key], f"{path}.{key}")
        return
    if isinstance(got, (int, float)) and not isinstance(got, bool):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=0), path
    else:
        assert got == want, path


def without_anomaly_numbers(measure: dict[str, Any]) -> dict[str, Any]:
    """The pool's lookup semantics: anomaly numerators are not selection facts."""
    point = PointMeasure.from_json(measure["measure"])
    point = replace(point, inputs=tuple(
        None if value is None else {
            "anchor": value.get("anchor"),
            "dispatch": value.get("dispatch"),
        } for value in point.inputs), aggregation_ids=())
    return {**measure, "measure": point.to_json(),
            **({"terms": []} if "terms" in measure else {})}


def oracle(gpu: str):
    snapshot_config(MODEL)
    environment = resolve_estimate_environment(request(gpu))
    record = servingLawRecord(
        environment, gpu_sku=gpu, anchorContext=1024, outputTokens=256,
        parallel=PARALLEL, capacityCards=(gpu,), capacityContexts=(1024,))
    pool = servingLawPointView(servingLawPoint(
        record, card=Card.of(gpu), context=1024, outputTokens=256,
        decodeBatch=16, prefillBatch=1, comm_latency_us=2,
        launch_floor_us=1))
    poolMeasure = servingMeasurePoint(
        record, card=Card.of(gpu), context=1024, batch=16)

    pending = estimate_pending(**resolve_estimate_environment(
        request(gpu).model_copy(update={"decode_batch_size": 16})))
    direct = pending.finish()
    directMeasures = pending.measures(direct)

    # Before the fold, the PointMeasure itself is byte-for-byte the same fact.
    assert without_anomaly_numbers(poolMeasure) == without_anomaly_numbers(
        directMeasures["decode"])
    poolItem = {"sol": copy.deepcopy(pool), "measures": {"decode": poolMeasure}}
    directItem = {"sol": servingLawPointView(direct),
                  "measures": {"decode": directMeasures["decode"]}}
    with FixtureBench() as bench:
        measure_points(
            [poolItem, directItem], selection_mode="latest",
            measurement_scope=bench.scope)
    same(poolItem["sol"], directItem["sol"])
    return record


def test_pool_measured_points_match_a_direct_walk_on_two_card_classes():
    for gpu in ("H100-SXM", "B200"):
        record = oracle(gpu)
        assert record["schema"] == "serving-law-v6"
        assert "measurePoints" in record["phases"]["decode"]


def _point_request(model: str, gpu: str, parallel: ParallelConfig,
                   batch: int) -> InferenceEstimateRequest:
    return InferenceEstimateRequest(
        model_id=model, gpu_sku=gpu, avg_input_tokens=1024,
        avg_output_tokens=256, parallel=parallel.to_dict(),
        decode_batch_size=batch, framework="vllm",
        selection_mode="theoretical_only")


def _direct_measure(model: str, gpu: str, parallel: ParallelConfig,
                    batch: int) -> dict[str, Any]:
    pending = estimate_pending(**resolve_estimate_environment(
        _point_request(model, gpu, parallel, batch)))
    result = pending.finish()
    return pending.measures(result)["decode"]


def test_stored_measure_points_are_point_keys_after_json():
    """A v6 record's concrete keys survive JSON and equal the plan road.

    The record is walked on the class representative; the checked point is on
    another card of the class, so this also pins the GPU field of every
    stored physical key. The direct side is an estimate's own
    `PendingEstimate.measures`, which asks `point_keys` at that point.
    """
    for model, parallel in LONG_CASES:
        snapshot_config(model)
        cards = class_cards("H100-SXM")
        record = _layout_law(
            (model, cards, 0.9, _GALLERY_PRECISION.canonical(),
             parallel.to_dict()))
        assert record.get("schema") == "serving-law-v6", (model, record)
        throughJson = json.loads(json.dumps(record))
        table = throughJson["phases"]["decode"]["measurePoints"]
        assert table["cardless"]
        assert all("gpu" not in point for point in table["points"])
        publicPoints = {(int(point["context"]), int(point["batch"]))
                        for point in table["points"]}
        for fact in record["capacity"]:
            if int(fact["context"]) not in GALLERY_SEQ_LENS:
                continue
            ceiling = int(fact["max_decode_batch"] or 0)
            if (ceiling <= 0 or fact.get("infeasibility_reason")
                    or fact.get("beyond_max_context")):
                continue
            expected = {
                (int(fact["context"]), int(item["decode_batch_size"]))
                for item in FRONTIER_BATCH_GRID
                if item["decode_batch_size"] is not None
                and int(item["decode_batch_size"]) <= ceiling
            } | {(int(fact["context"]), ceiling)}
            missing = expected - publicPoints
            assert not missing, (model, fact["gpu_sku"], fact["context"],
                                 sorted(missing))
        for sku in cards:
            priced = gallery._restamp_layout(
                throughJson, sku, parallel)
            assert "error" not in priced, (model, sku, priced)
        point = next(item for item in throughJson["phases"]["decode"]
                     ["measurePoints"]["points"]
                     if item["context"] == 1024
                     and item["batch"] == 1)
        gpu = cards[0]
        started = time.monotonic()
        stored = servingMeasurePoint(
            throughJson, card=Card.of(gpu), context=1024, batch=1)
        elapsed = time.monotonic() - started
        batch = servingMeasurePoints(
            throughJson, card=Card.of(gpu), points=[(1024, 1)])
        assert batch[(1024, 1)]["measure"] == PointMeasure.from_json(
            stored["measure"])
        direct = _direct_measure(model, gpu, parallel, point["batch"])
        assert (PointMeasure.from_json(stored["measure"])
                == PointMeasure.from_json(
                    without_anomaly_numbers(direct)["measure"])), model
        assert stored["terms"] == []
        assert elapsed < 0.05, (model, elapsed)


def test_offline_serving_projection_never_asks_point_keys(monkeypatch):
    """Drawing SOL from a v6 record is offline: it reads no measurement key."""
    import backend.api.gallery_frontier as gallery
    import backend.inference.calibration_overlay as overlay
    import backend.inference.sol_law as law

    snapshot_config(MODEL)
    record = _layout_law((MODEL, ["H100-SXM"], 0.9,
                          _GALLERY_PRECISION.canonical(),
                          PARALLEL.to_dict()))
    assert record.get("schema") == "serving-law-v6", record
    for module, name in ((law, "point_keys"), (overlay, "point_keys")):
        monkeypatch.setattr(
            module, name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("point_keys called")))
    out = gallery._restamp_layout(
        json.loads(json.dumps(record)), "H100-SXM", PARALLEL)
    assert out["theory"]
