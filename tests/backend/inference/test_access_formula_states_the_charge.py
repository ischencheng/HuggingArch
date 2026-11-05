"""A row's access-bytes popover states the bytes the row charges.

The per-op row publishes `access_bytes_per_rank` and, beside it, the formula
that says how it was computed (`access_bytes_formula`). Two consumers read one
fact there; they must not read two answers. The formula used to be rebuilt
from the node's shapes — every input at its full span, one dtype — while the
charge came from `traffic`, which divides a view's repetition out (a
broadcast head-weight operand, a head-broadcast key), lets the cache law take
a decode read over, prices each edge at its own width and adds the operand
conversion. DeepSeek-V4's fused indexer at 4096 tokens printed 629.1 MB for a
charge of 612.6 MB.

Now each term is built from the derivation's own records (`Traffic.in_edges`,
`Traffic.out_terms`, the cache read, the conversion, a saturating stored
read's law), so the substituted line, evaluated, is the charge: per leaf, up
to the whole bytes each per-leaf term is truncated to when the row is
summed (`access._access_row`), times the row's count.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api._models import InferenceEstimateRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.inference.capacity import estimate_pending

#: Fused kernels fed through views (DeepSeek-V4's indexer and
#: hyper-connection mixes), pooled index caches, an expert bank, and one
#: plain dense model.
MODELS = ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4.1-Flash",
          "zai-org/GLM-5.3-Flash", "Qwen/Qwen3.8-Flash-Next", "Qwen/Qwen3-8B")
#: Each per-leaf term (weights, reads, writes, conversion) is whole bytes on
#: the row; the formula prints exact extents.
TERMS = 4


def _estimate(model_id: str, parallel: dict) -> dict:
    request = InferenceEstimateRequest(
        model_id=model_id, gpu_sku="H100-SXM", avg_input_tokens=1024,
        avg_output_tokens=1, parallel=parallel, framework="vllm")
    env = resolve_estimate_environment(request)
    # The walk's rows, not the published estimate's: a deployment that cannot
    # hold the weights (DeepSeek-V4-Flash on one H100) publishes no step
    # (`capacity.without_step`), and its rows' formulas are still the ones
    # every other card and batch is re-priced from.
    return estimate_pending(avg_input_tokens=4096, **env).finish()


def _misstated(rows) -> list:
    out = []
    for row in rows:
        formula = row.get("access_bytes_formula") or {}
        text = formula.get("substituted")
        if not text:
            continue
        stated = eval(text.replace(",", ""), {"__builtins__": {}}, {})
        cnt = int(formula.get("cnt") or 1)
        charged = row["access_bytes_per_rank"]
        if abs(stated * cnt - charged) >= TERMS * cnt:
            out.append((row["op"], row["display_path"], stated * cnt, charged))
    return out


#: The first case runs by default and is a step that exists; a weight-refused
#: layout walks no rows. The rest take minutes and are `regression`.
CASES = [pytest.param(
    MODELS[0], {"tp_attention": 8, "ep_moe": 8},
    id=f"{MODELS[0]}-tp8ep8")] + [
    pytest.param(model_id, parallel, id=f"{model_id}-{name}",
                 marks=pytest.mark.regression)
    for model_id in MODELS
    for name, parallel in (("tp1", {"tp_attention": 1}),
                           ("tp8-ep8", {"tp_attention": 8, "ep_moe": 8}))
    if (model_id, name) not in {
        (MODELS[0], "tp1"), (MODELS[0], "tp8-ep8")}]


@pytest.mark.parametrize(("model_id", "parallel"), CASES)
def test_the_formula_evaluates_to_the_charge(model_id, parallel):
    result = _estimate(model_id, parallel)
    if result.get("infeasibility_reason"):
        assert "per_op_timing" not in result
        assert "per_op_timing_decode" not in result
        return
    for key in ("per_op_timing", "per_op_timing_decode"):
        assert _misstated(result[key]) == [], key
