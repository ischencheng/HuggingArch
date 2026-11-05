"""The public vocabulary shared by serving-law projections and estimates."""
from __future__ import annotations

from typing import Any


SERVING_POINT_FIELDS = (
    "avg_input_tokens", "avg_output_tokens", "max_batch_per_gpu",
    "max_decode_batch_per_gpu", "max_prefill_batch_per_gpu",
    "max_decode_batch_total", "max_prefill_batch_total",
    "max_tokens_per_gpu", "max_tokens_total", "num_gpus", "cluster_size",
    "parallel", "gpu_memory_fraction",
    "beyond_max_context", "infeasibility_reason", "max_context_tokens",
    "parallel_warnings", "memory", "kv_cache", "phases",
)
SERVING_MEMORY_FIELDS = (
    "total_bytes", "usable_bytes", "weight_bytes_total",
    "weight_bytes_per_gpu", "host_offload_bytes_per_gpu", "host_offload",
    "kv_budget_bytes", "weight_deficit_bytes", "static_scale_bytes", "fits",
    "pp_stage", "weight_bytes_per_gpu_heaviest_stage",
)
SERVING_KV_FIELDS = (
    "attention_type", "attention_variant", "is_hybrid_attention",
    "per_request_kv_bytes", "effective_storage_tokens_per_request",
    "replication_factor", "shard_divisor", "shard_note",
    "static_scale_bytes", "unique_tokens_multiplier",
)
SERVING_PARALLEL_FIELDS = (
    "pp", "tp_attention", "tp_dense_ffn", "tp_moe_expert", "ep_moe", "cp",
    "dp_attention", "dp_replicas",
)
SERVING_PHASE_FIELDS = (
    "phase", "batch", "binding", "t_sol_s", "t_measured_s", "coverage",
    "throughput", "time_unpublished_reason",
)


def servingCapacityView(preparation: Any, ceilings: Any) -> dict[str, Any]:
    """The capacity facts a serving projection publishes for one context.

    This is deliberately a subset of `_capacity_fields`: memory and KV facts
    the Scaling surfaces draw, but not activation liveness, formula popovers
    or an Inference plan's report. The producer stores this beside its law;
    a complete estimate is narrowed to the same fields when served.
    """
    memory = {
        "total_bytes": preparation.mem_total_bytes,
        "usable_bytes": preparation.usable_bytes,
        "weight_bytes_total": preparation.weight_bytes_total,
        "weight_bytes_per_gpu": ceilings.weight_bytes_per_rank,
        "host_offload_bytes_per_gpu": preparation.host_offload_bytes[
            ceilings.pp_stage or 0],
        "host_offload": preparation.host_offload,
        "kv_budget_bytes": ceilings.kv_budget_bytes,
        "weight_deficit_bytes": ceilings.weight_deficit_bytes,
        "static_scale_bytes": ceilings.kv_static_per_rank,
        "fits": ceilings.fits,
        "pp_stage": ceilings.pp_stage,
        "weight_bytes_per_gpu_heaviest_stage": max(
            (stage.weight_bytes for stage in preparation.pp_stages),
            default=ceilings.weight_bytes_per_rank),
    }
    kv = ceilings.kv_per_rank
    return {
        "memory": memory,
        "kv_cache": {name: kv[name] for name in SERVING_KV_FIELDS
                     if name in kv},
        "max_tokens_per_gpu": ceilings.max_tokens,
        "max_tokens_total": ceilings.max_tokens_total,
        "max_batch_total": ceilings.max_batch_total,
        "max_decode_batch_total": ceilings.max_batch_total,
        "max_prefill_batch_total": ceilings.max_prefill_batch_total,
    }


def servingLawPointView(result: Any) -> Any:
    """One point in the serving projection's public vocabulary.

    Both producers pass through here: a complete estimate and a canonical
    StepLaw projection. The direct estimate may know its activation peak,
    memory tree and precision report, but those belong to the Inference plan
    request; carrying them here would make a sweep point a second estimate
    artifact and let the two paths drift.
    """
    if not isinstance(result, dict):
        return result
    phases = result.get("phases")
    memory = result.get("memory")
    kvCache = result.get("kv_cache")
    public = {
        name: result[name]
        for name in SERVING_POINT_FIELDS
        if name in result
    }
    if public.get("parallel") is not None:
        public["parallel"] = {
            field: public["parallel"][field]
            for field in SERVING_PARALLEL_FIELDS
            if field in public["parallel"]
        }
    if memory is not None:
        public["memory"] = {
            field: memory[field] for field in SERVING_MEMORY_FIELDS
            if field in memory
        }
    if kvCache is not None:
        public["kv_cache"] = {
            field: kvCache[field] for field in SERVING_KV_FIELDS
            if field in kvCache
        }
    public |= ({
        "phases": {
            name: {field: phase[field] for field in SERVING_PHASE_FIELDS
                   if field in phase}
            for name, phase in phases.items()
            if phase
        }
    } if phases else {})
    for phase in public.get("phases", {}).values():
        # A serving projection states that it has no measured answer; a
        # measured fold replaces both fields afterwards.
        phase["t_measured_s"] = None
        phase["coverage"] = None
        phase.setdefault("time_unpublished_reason", "")
    return public
