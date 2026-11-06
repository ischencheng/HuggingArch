"""Endpoints whose error handling was fixed answer with the same keys as
before: a bug fix there must not change the shape a caller reads. Key sets
recorded before the fix, plus what an estimate read its measurements from
(`measurements`, `measurement_scope`; docs/measured_calibration.md §7.3)."""
from __future__ import annotations

from fastapi.testclient import TestClient

ESTIMATE_KEYS = ['activation_peak', 'assumptions', 'avg_input_tokens', 'avg_output_tokens', 'beyond_max_context', 'cluster_size', 'code_version', 'encode', 'estimated_max_fraction', 'formulas', 'gpu', 'gpu_memory_fraction', 'infeasibility_reason', 'kv_cache', 'kv_replication_factor', 'max_batch_per_gpu', 'max_batch_total', 'max_context_tokens', 'max_decode_batch_per_gpu', 'max_decode_batch_total', 'max_prefill_batch_per_gpu', 'max_prefill_batch_total', 'max_tokens_per_gpu', 'max_tokens_total', 'measurement_scope', 'measurements', 'memory', 'moe_enabled', 'num_gpus', 'parallel', 'parallel_warnings', 'peak_flops_per_sec', 'peak_hbm_bw_bytes_per_sec', 'peak_source', 'per_op_timing', 'per_op_timing_decode', 'phases', 'precision', 'roles_present', 'shape_warnings', 'timings', 'unique_tokens_multiplier', 'vs_naive_delta', 'weight_breakdown', 'weight_bytes_total']
SWEEP_KEYS = ['code_version', 'measurement_scope', 'results']


def test_estimate_and_sweep_keep_their_response_keys():
    from backend.api import app
    client = TestClient(app)
    estimate = client.post("/inference/estimate", json={
        "model_id": "meta-llama/Llama-3.2-1B", "gpu_sku": "H200"})
    sweep = client.post("/inference/estimate-sweep", json={
        "model_id": "meta-llama/Llama-3.2-1B", "gpu_sku": "H200",
        "seq_lens": [256], "avg_output_tokens": 64})
    assert estimate.status_code == 200 and sweep.status_code == 200
    assert sorted(estimate.json()) == ESTIMATE_KEYS
    assert sorted(sweep.json()) == SWEEP_KEYS
