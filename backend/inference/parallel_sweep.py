"""Parallelism-scaling sweeps for the inference Scaling panel.

Hold model + context + batch fixed and sweep the deployment SCALE (EP/TP
degree) — one series per GPU type — reporting PER-GPU throughput. Reads how
per-GPU efficiency scales as you add GPUs and where the compute-bound plateau
lands per GPU.

Each point is an independent `estimate_capacity`: `parallel` is part of the
seq-invariant `_capacity_preparation`, so there is no prelude reuse across
parallel configs (a parallel sweep is contractually N single calls). The
`backend.measure.pool` gives real multi-core over those N calls.

Config-generation rule (X → ParallelConfig), all `dp_replicas=1`, `num_gpus=X`,
X over powers of two up to `sweep_grid.SWEEP_MAX_GPUS`; every config is
`validate()`-clean, and an EP mode's X divides every expert pool:
  dp_ep : dp_attention=X, ep_moe=X
  cp_ep : cp=X, ep_moe=X, dp_attention=1          (pcp, the default scheme)
  tp_ep : tp_attention=X, tp_dense_ffn=X, ep_moe=X
  tp_tp : tp_attention=X, tp_dense_ffn=X, tp_moe_expert=X (pure TP, ep=1)
Whether the model deploys at X is the layout's answer, not a cap on heads:
each degree is solved by the same layout walk an estimate runs
(`legal_parallel_degrees`), and a refused one says why.

Each layout is also read at every per-GPU decode batch of
`sweep_grid.SCALING_DECODE_LADDER`
(`_decode_ladder`, by substitution off the same walk): the sweep's decode
point cloud (`decode_cloud`), SOL only. A decode Pareto frontier is a view
of that cloud (`decode_frontier`): some of its dimensions held, the rest
grouped into lines, each line the points no other point of it beats.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dataclasses import dataclass, field

from backend.arch.eval.analysis import MoeFacts
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import LayoutRefusal
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import get_default_loader

from .parallel_config import ParallelConfig
from .module_rows import served
from .pipeline import NoPeakForDtype
from backend.arch.eval.analysis import Point
from backend.measure.pool import map_cells
from .card import Card
from .frontier import POINT_DIMS, PointDim, frontier_view
from .point_cloud import parallel_cloud, point_coordinates
from .gpu_specs import get_gpu, gpus_per_node, hbm_bytes
from .prefill_geometry import PrefillEndpointGeometry, PrefillChunkSchedule
from .serving_view import servingLawPointView
from .sweep_grid import (FRONTIER_MAX_GPUS, GALLERY_GPUS,
                         SCALING_DECODE_LADDER, SWEEP_MAX_GPUS)
from backend.measure.lookup import MeasurementsUnavailable
from .weight_breakdown import shard_degrades
from .batch_sweep import measure_points
from .capacity import (
    CapacityKnobs,
    _capacity_pending,
    _capacity_walk,
    _capacity_preparation,
    _fold_chunked_prefill_sol,
    _point_measures_with_chunks,
    _point_batches,
    _point_rows,
    _restamp_gpu,
    _stamp_gpu_on_result,
    without_step,
    capacity_ceilings,
    point_served,
)

logger = logging.getLogger(__name__)

# Curated GPU series for the scaling charts. All SKUs are peers — consumer and
# datacenter parts get identical treatment; each card's VRAM + interconnect
# (from gpu_specs.csv) decide what's feasible, not a category. RTX-4090 (PCIe4
# intra, 200 Gbps RDMA inter-node) is the card kernel_bench measures on, so
# this series keeps it: its measured rows are what theory is compared against
# here, and Kernel Bench coverage counts on it.
# GB200 is the rack-scale part: its NVLink domain is the 72-GPU rack, so a
# layout stays on NVLink where an 8-GPU node would cross onto RDMA.
# The Gallery publishes a NARROWER series (`sweep_grid.GALLERY_GPUS`: these
# cards less the 4090) — it shows serving frontiers, not measurements — so
# the picker, `/gallery/frontier` and the gallery warm read that one.
# WHICH cards each series shows is a product decision and is declared once;
# that each names a card is the sheet's to say, so every one is resolved
# against `gpu_specs.csv` at import — a SKU renamed there would otherwise
# leave a chart axis nothing can be walked on.
def _series(*skus: str) -> List[str]:
    """The named cards, each resolved against the sheet."""
    for sku in skus:
        if get_gpu(sku) is None:
            raise ValueError(
                f"the scaling charts' GPU series names {sku!r}, which is not "
                f"a row of gpu_specs.csv")
    return list(skus)


DEFAULT_SWEEP_GPUS = _series("RTX-4090", "H20", "H100-SXM", "H200", "B200",
                             "B300", "GB200")
# The Gallery's cards live beside the other `GALLERY_*` grid constants, which
# cannot import this module; they are resolved here, by the same rule, because
# this is where the rule is stated and `class_cards` defaults to them.
_series(*GALLERY_GPUS)

#: The request / environment fields every sweep point sets for itself: its
#: own layout and GPU count, the series' cards (`gpu_skus`), and the sweep's
#: `context_len` as the input length. Whatever the caller passed for these is
#: dropped, so they are the only fields a cached sweep may leave out of its
#: key (`backend.api.sweep_runtime.sweep_cache_key`).
PER_POINT_FIELDS = frozenset({"parallel", "gpu_sku", "num_gpus", "avg_input_tokens"})

def _mode_configs(degree: int):
    """Each scaling mode with the config that defines it, in panel order.

    This generator is the vocabulary's one definition: `ALL_MODES` reads its
    names and `config_for_degree` reads its configs, so neither can list a
    different set from the other."""
    yield "dp_ep", ParallelConfig(dp_attention=degree, ep_moe=degree)
    yield "cp_ep", ParallelConfig(cp=degree, ep_moe=degree, dp_attention=1)
    yield "tp_ep", ParallelConfig(tp_attention=degree, tp_dense_ffn=degree,
                                  ep_moe=degree, tp_moe_expert=1)
    yield "tp_tp", ParallelConfig(tp_attention=degree, tp_dense_ffn=degree,
                                  tp_moe_expert=degree, ep_moe=1)


#: The scaling modes, in the order the panel offers them, read off the one
#: generator that also builds each mode's config. `_resolve_modes` and the
#: request model refuse anything else; which of them shard experts is NOT
#: stated here but read off the configs themselves (`EP_MODES`).
ALL_MODES: Tuple[str, ...] = tuple(
    mode for mode, _config in _mode_configs(2))


def _pow2_range(lo: int, hi: int) -> List[int]:
    """Powers of two in [lo, hi] (inclusive)."""
    if hi < 2 or hi < lo:
        return []
    out: List[int] = []
    x = 1
    while x < lo:
        x *= 2
    while x <= hi:
        out.append(x)
        x *= 2
    return out


@dataclass(frozen=True)
class SweepCaps:
    """What a scaling ladder is drawn from: whether the model routes tokens
    to experts (`Program.has_dispatch_zone`) and its routed expert pools,
    which every expert-parallel degree must divide."""

    moe_enabled: bool
    #: Every routed expert pool (`Analysis.moe_pools`). Expert parallelism
    #: divides each pool by its own count.
    moe_pools: Tuple[MoeFacts, ...] = ()

    def ep_divides(self, degree: int) -> bool:
        """Whether `degree` expert-parallel ranks split every pool's copies
        evenly — a degree that leaves some pool fractional per rank is not
        a deployment."""
        return all(pool.n_expert_copies % degree == 0
                   for pool in self.moe_pools)

    @classmethod
    def of(cls, analysis: Any) -> "SweepCaps":
        """The caps of the model `analysis` walked (one rank)."""
        return cls(moe_enabled=analysis.program.has_dispatch_zone,
                   moe_pools=analysis.moe_pools)

    @property
    def ep_ceiling(self) -> int:
        """The largest degree every pool divides (their copies' gcd); 0 for
        a model with no expert pool."""
        return math.gcd(*(pool.n_expert_copies for pool in self.moe_pools))


def sweep_caps(config: dict, model_id: Optional[str] = None,
               *, resolved: Any = None) -> SweepCaps:
    """The caps for a model, from its resolved spec. Raises for a config no
    spec matches — a ladder for an unknown model is not a ladder.

    A caller that already resolved the spec (a sweep resolves once, up front)
    passes it in; otherwise it is matched by `model_id` here. Matching with no
    `model_id` no longer falls back to a scorer, so an unthreaded caller would
    get None and raise — the resolved spec is the single entry point."""
    if resolved is None:
        from backend.arch.loader import SpecLoader
        resolved = SpecLoader().match(config, model_id=model_id)
    if resolved is None:
        raise ValueError("no spec matches this config; nothing to sweep")
    return SweepCaps.of(Deployment(Program.of(resolved)).analyze())


def is_moe(config: dict, model_id: Optional[str] = None) -> bool:
    """True when the model routes tokens to experts — its graph declares a
    dispatch zone (`Program.has_dispatch_zone`), which is the fact; a bound
    `n_experts` is a dim a dense spec may also carry."""
    from backend.arch.loader import SpecLoader
    resolved = SpecLoader().match(config, model_id=model_id)
    return resolved is not None and Program.of(resolved).has_dispatch_zone


def config_for_degree(mode: str, x: int) -> ParallelConfig:
    """One valid ParallelConfig at deployment degree x (num_gpus == x)."""
    for mode_name, config in _mode_configs(x):
        if mode == mode_name:
            return config
    raise ValueError(f"unknown scaling mode {mode!r}")


#: The modes that shard experts — hidden for dense (non-MoE) models, and
#: the ones a sweep holds to an expert pool that divides. Read off the
#: config each mode builds (`ep_moe` above 1) rather than listed a second
#: time: a mode's parallelism is `config_for_degree`'s to state, and a new
#: EP mode that had to be remembered here would quietly sweep degrees no
#: expert bank divides.
EP_MODES: Tuple[str, ...] = tuple(
    mode for mode in ALL_MODES if config_for_degree(mode, 2).ep_moe > 1)


def parallel_configs_for_mode(
    mode: str, caps: SweepCaps, x_min: int = 2,
    x_max: int = SWEEP_MAX_GPUS,
) -> List[Tuple[int, ParallelConfig]]:
    """`[(x, config)]` for every degree of a mode the sweep enumerates:
    powers of two up to `SWEEP_MAX_GPUS`, `validate()`-clean, and — for the
    EP modes, which shard the experts over the `ep_moe` group — dividing
    every expert pool (a model with 288 expert copies has no EP-96/128/256
    rung). Whether the model deploys at each is `legal_parallel_degrees`."""
    out: List[Tuple[int, ParallelConfig]] = []
    for x in _pow2_range(x_min, x_max):
        if mode in EP_MODES and not caps.ep_divides(x):
            continue  # some pool's experts-per-rank would be fractional — not a deployment
        cfg = config_for_degree(mode, x)
        if not cfg.validate():  # empty error list == valid
            out.append((x, cfg))
    return out


def frontier_layouts(caps: SweepCaps, gpu_sku: str, *,
                     x_max: int = FRONTIER_MAX_GPUS) -> List[ParallelConfig]:
    """Every deployment a decode Pareto frontier on `gpu_sku` prices, up to
    `x_max` GPUs.

    One GPU; `tp_tp` and `cp_ep` at each degree, as the Scaling panel
    enumerates them; and, for a model that routes to experts, the experts
    spread over all `x` GPUs while attention and the dense layers run
    tensor-parallel `t` ways and data-parallel over the other `x / t`
    (`t` a power of two up to the GPUs of one of its nodes: past them the
    tensor-parallel collectives would cross nodes every layer). `t = 1` is
    `dp_ep` and `t = x` is `tp_ep`; the ones between are what fills the
    frontier where neither is best — `dp_ep` reads the whole attention on
    every GPU every step, `tp_tp` serves one batch with the whole group.
    Each config is `validate()`-clean; whether the model deploys under it
    is the layout walk's answer, asked when it is priced."""
    tp_max = gpus_per_node(get_gpu(gpu_sku))
    out = [ParallelConfig()]
    for x in _pow2_range(2, x_max):
        if caps.moe_enabled and caps.ep_divides(x):
            out += [ParallelConfig(tp_attention=t, tp_dense_ffn=t,
                                   dp_attention=x // t, ep_moe=x)
                    for t in (1, *_pow2_range(2, min(x, tp_max)))]
            out.append(config_for_degree("cp_ep", x))
        out.append(config_for_degree("tp_tp", x))
    return [cfg for cfg in out if not cfg.validate()]


def class_cards(gpu_sku: str,
                series: Sequence[str] = GALLERY_GPUS) -> List[str]:
    """The cards of `series` that walk a model as `gpu_sku` does — its walk
    class (`card.Card.klass`), `gpu_sku` always among them —
    the one with the most HBM first: the card the class walks on, and the
    card the others are priced off without walking again
    (`capacity._restamp_gpu`, `capacity.ClassSweep`).

    The one rule for which cards may share a walk, asked by both views that
    price a series per card. `series` is the one the view publishes: the
    Gallery's own cards by default (`sweep_grid.GALLERY_GPUS` — the cards
    that walk together are the cards one view shows side by side, and a
    card it does not offer would be walked for nobody), the Scaling sweep's
    series when that is what is being priced
    (`DEFAULT_SWEEP_GPUS`, passed in by `_estimate_layout`). A card the
    series omits answers for itself alone.

    Most HBM first because a deployment that fits no card of the class fits
    none of them, so the roomiest card is the one whose walk can price the
    rest."""
    own = Card.of(gpu_sku)
    cards = [s for s in series if own.walks_alike(Card.of(s))]
    if gpu_sku not in cards:
        cards.append(gpu_sku)
    return sorted(cards, key=lambda s: (-hbm_bytes(get_gpu(s)), cards.index(s)))


def class_frontier_layouts(caps: SweepCaps, cards: Sequence[str]
                           ) -> List[ParallelConfig]:
    """Every deployment a frontier on any of `cards` prices
    (`frontier_layouts` on each, in order, each once)."""
    out: List[ParallelConfig] = []
    for sku in cards:
        out += [cfg for cfg in frontier_layouts(caps, sku) if cfg not in out]
    return out


def layout_label(cfg: ParallelConfig) -> str:
    """How a chart names a deployment: the degrees above one, short enough
    to sit beside a point (`TP8`, `TP8·EP64·DP8`, `EP32·DP32`; one GPU is
    `TP1`). The GPU count is said beside it."""
    parts = [f"{name}{n}" for name, n in (
        ("TP", cfg.tp_attention), ("CP", cfg.cp),
        ("EP", cfg.ep_moe), ("DP", cfg.dp_attention)) if n > 1]
    return "·".join(parts) or "TP1"


def layout_verdict(resolved: Any, parallel: ParallelConfig) -> List[str]:
    """Whether the model deploys under `parallel`: the layout walk every
    estimate starts from (`capacity._capacity_walk`) either refuses —
    raising its reason — or accepts, and the accepted layout's replication
    warnings (an axis that splits fewer ways than asked; its other ranks
    hold copies) are returned."""
    return shard_degrades(_capacity_walk(resolved, parallel))


def legal_parallel_degrees(resolved: Any, mode: str,
                           caps: Optional[SweepCaps] = None, *,
                           x_max: int = SWEEP_MAX_GPUS) -> dict:
    """Each degree of `mode` the sweep enumerates, solved by the layout:
    `accepted` (with its replication warnings), `refused` (with the
    layout's reason), and `beyond_cap` — the bound past which degrees are
    not enumerated (None when no degree of the mode lies past it)."""
    caps = caps if caps is not None else sweep_caps({}, resolved=resolved)
    accepted: List[dict] = []
    refused: List[dict] = []
    for x, cfg in parallel_configs_for_mode(mode, caps, x_max=x_max):
        try:
            warnings = layout_verdict(resolved, cfg)
        except LayoutRefusal as e:
            refused.append({"x": x, "reason": str(e)})
            continue
        accepted.append({"x": x, "parallel": cfg.to_dict(),
                         "warnings": warnings})
    return {"accepted": accepted, "refused": refused,
            "beyond_cap": _beyond_cap(mode, caps)}


def _beyond_cap(mode: str, caps: SweepCaps) -> Optional[dict]:
    """The enumeration bound, when the mode has degrees past it: every
    degree for a tensor-parallel ladder, none for an expert-parallel one
    whose pools a larger degree no longer divides."""
    if mode in EP_MODES and caps.ep_ceiling <= SWEEP_MAX_GPUS:
        return None
    return {"cap": SWEEP_MAX_GPUS}


def _per_gpu_tput(result: dict, phase: str, track: str = "sol") -> Optional[float]:
    """Per-GPU throughput of one track (`sol` / `measured`), read off the
    phase's `throughput` payload (`phases.phase_throughput`, applied once by
    the estimate and again by the measured fold) — no second formula, no
    recomputation. `None` when the phase didn't run (OOM / infeasible) or
    the track has no step time."""
    p = (result.get("phases") or {}).get(phase) or {}
    rates = (p.get("throughput") or {}).get(track)
    return rates.get("per_gpu") if rates else None


#: What a point of the sweep's decode cloud is held or grouped by: the one
#: point-cloud vocabulary (`frontier.POINT_DIMS`). `concurrent_requests` is a
#: value on each point, not a coordinate.
DecodeCloudDim = PointDim


def decode_frontier(cloud: Sequence[dict], *, fix: Mapping[str, Sequence[Any]],
                    group_by: Sequence[str]) -> dict:
    """A decode Pareto frontier view of the sweep's `decode_cloud`."""
    return frontier_view(cloud, dims=POINT_DIMS, fix=fix, group_by=group_by)


def _point(x: int, parallel: ParallelConfig, result: Optional[dict],
           error: Optional[str] = None) -> dict:
    if not result:
        point = {"x": x, "parallel": parallel.to_dict(),
                 "prefill_tps_per_gpu": None,
                 "decode_tps_per_gpu": None,
                 "prefill_tps_per_gpu_measured": None,
                 "decode_tps_per_gpu_measured": None, "feasible": False}
        if error is not None:
            # A crashed estimate is not the same state as "does not fit" —
            # the cell stays out of the series but says why.
            point["error"] = error
        return point
    def hover_facts(phase: str) -> dict:
        """The batch a phase ran at and its step times, so a tooltip can say
        WHY a per-GPU curve flattens (the step reads ~the whole card) instead
        of leaving the number to look stuck."""
        p = (result.get("phases") or {}).get(phase) or {}
        return {
            f"{phase}_batch_per_gpu": p.get("batch"),
            f"{phase}_t_sol_s": p.get("t_sol_s"),
            f"{phase}_t_measured_s": p.get("t_measured_s"),
            # What the measured time is missing when there is none
            # (`auto_sweep.MeasuredTrack.coverage`).
            f"{phase}_measured_coverage": p.get("coverage"),
        }

    # Both tracks, each read from its own step time. The measured track sits
    # beside theory: None unless every row of the phase is measured, never
    # the SOL value in its place.
    feasible = bool((result.get("max_decode_batch_per_gpu") or 0) > 0)
    point = {
        "x": x,
        # The deployment this point is, so a page applies it as it stands.
        "parallel": parallel.to_dict(),
        # An axis the layout splits fewer ways than asked: its other ranks
        # hold copies (`layout_verdict`).
        "warnings": list(result.get("parallel_warnings") or ()),
        "prefill_tps_per_gpu": _per_gpu_tput(result, "prefill"),
        "decode_tps_per_gpu": _per_gpu_tput(result, "decode"),
        "prefill_tps_per_gpu_measured": _per_gpu_tput(result, "prefill", "measured"),
        "decode_tps_per_gpu_measured": _per_gpu_tput(result, "decode", "measured"),
        **hover_facts("prefill"),
        **hover_facts("decode"),
        "feasible": feasible,
        # The request (context + output) runs past the model's published
        # context: the point is still estimated, and the page must say so.
        "beyond_max_context": bool(result.get("beyond_max_context")),
        "max_context_tokens": result.get("max_context_tokens"),
    }
    if not feasible:
        # Why this card cannot serve at this degree (weights over HBM, or no
        # room for one request's KV), so the panel can say it.
        point["reason"] = result.get("infeasibility_reason")
    return point


def _resolve_modes(
    mode: Optional[str], modes: Optional[Sequence[str]],
) -> List[str]:
    """`modes` if given, else `[mode]`. Deduped, order-preserving."""
    raw = list(modes) if modes else ([mode] if mode else [])
    if not raw:
        raise ValueError("mode or modes required")
    unknown = [m for m in raw if m not in ALL_MODES]
    if unknown:
        raise ValueError(f"unknown scaling mode {unknown[0]!r}")
    seen: set[str] = set()
    out: List[str] = []
    for m in raw:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out




def _estimate_layout(payload: Dict[str, Any]) -> Dict[str, dict]:
    """One ParallelConfig, every GPU in `gpu_skus`.

    Walk + prefill/decode evaluation once per CARD CLASS and per pipeline
    stage the class's cards bind (so once per class, without pp). Each card
    is a point: the bound stage's rows restamped on its HBM / peaks and
    re-read at its batch. The graph does not mention the SKU, but it does
    mention the card's class (`framework_bindings.CardClass`), so the walk
    a card is priced off is its own class's — the rule
    `capacity._restamp_gpu` enforces and `class_cards` names, and the one
    the Gallery's frontier already follows.

    Returns `{gpu_sku: {"sol", "measures"}}`: the
    point's estimate before any measurement, and its rows' keys there as
    plain data (`batch_sweep.point_measures`) — the sweep reads every
    point's measurements in one lookup (`ParallelSweepPlan.assemble`). A
    card that cannot time the layout maps to `{_LAYOUT_ERROR_KEY: reason}`.
    Top-level so the sweep pool can pickle it into a spawned worker.
    """
    gpu_skus = list(payload["gpu_skus"])
    # The request axes this payload carries beside the knobs — `estimate_capacity`'s
    # own explicit parameters plus the sweep's list of cards; an unknown key
    # is refused rather than dropped.
    knobs = CapacityKnobs.from_kwargs(payload, envelope=frozenset(['avg_input_tokens', 'avg_output_tokens', 'config', 'decode_batch_size', 'decode_ladder', 'gpu_sku', 'gpu_skus', 'prefill_batch_size', 'prefill_chunk_size']))
    avg_in = int(payload["avg_input_tokens"])
    avg_out = int(payload["avg_output_tokens"])
    chunk = payload.get("prefill_chunk_size")
    # Evaluate the graph at B=1. Each per-op row carries its law in every
    # workload count and the card (`row_roofline.build_row_exprs`); each
    # card's max-batch point is that law read there on the card's peaks
    # (`capacity._point_rows`, `batch_sweep.phase_at`) — the batch
    # dropdown's reading.
    pre_b = payload.get("prefill_batch_size") or 1
    dec_b = payload.get("decode_batch_size") or 1
    # The chunk schedule every card's prefill is cut by: one schedule for the
    # series, because a chunk boundary is a workload count and no card moves
    # it. Its cached prefix is the request's (`CapacityKnobs`), which is also
    # every preparation's (`_capacity_preparation`), and the engine refuses a
    # schedule whose prefix is not the preparation's — so this is read off the
    # knobs rather than off whichever card happens to be prepared first.
    base_schedule = PrefillChunkSchedule.materialize(
        PrefillEndpointGeometry(avg_in, knobs.s_prefix_cached, 1, 1),
        prefill_chunk_size=chunk)
    # One walk per (card class, pipeline stage the class's cards bind).
    #
    # The class, because a walk's rows depend on the card through its class
    # and nothing else (`framework_bindings.CardClass`), so cards of one
    # class share a walk and no card reads another class's. Anchoring the
    # whole series on `gpu_skus[0]` instead published one card's fused graph
    # as every card's: with RTX-4090 (compute capability 8.9) leading the
    # default series, Kimi-K3's Hopper and Blackwell decode steps were
    # priced off a walk missing every kernel vLLM gates at 9.0.
    #
    # The stage, because without pp — or when every card of a class runs out
    # on the same stage — that is one walk for the class; a card with less
    # HBM can run out on a different stage (MiMo-V2-Flash at pp=4: stage 0
    # on an H200, stage 1 on an H100), and its rows are then that stage's
    # own, not the walk card's. The walks read no measurements: every
    # point's are looked up below, together.
    #
    # The walk card of each class is its roomiest (`class_cards`); the
    # others restamp off it. A class whose walk cannot be prepared or walked
    # fails only its own cards.
    anchors: Dict[Any, Tuple[str, Any]] = {}
    pendings: Dict[Tuple[Any, Optional[int]], Any] = {}
    bases: Dict[Tuple[Any, Optional[int]], dict] = {}
    out: Dict[str, dict] = {}
    for sku in gpu_skus:
        card = Card.of(sku)
        klass = card.klass
        # A card that lists no peak for the format a step computes in cannot
        # time this layout; that is this card's answer, not the layout's —
        # the other cards of the series still price it. Its class's walk is
        # prepared here, inside that same isolation.
        try:
            if klass not in anchors:
                walk_card = class_cards(sku, gpu_skus)[0]
                anchors[klass] = (walk_card, _capacity_preparation(
                    config=payload["config"], gpu_sku=walk_card, knobs=knobs))
            walk_card, prep = anchors[klass]
            p = prep if sku == walk_card else _restamp_gpu(prep, card)
            ceilings = capacity_ceilings(
                p, avg_input_tokens=avg_in, avg_output_tokens=avg_out,
                prefill_chunk_schedule=base_schedule)
            if p.refusal_reason is not None:
                pending = _capacity_pending(
                    p, avg_input_tokens=avg_in, avg_output_tokens=avg_out,
                    prefill_chunk_schedule=base_schedule,
                    fold_prefill_chunks=False, breakdown=False)
                out[sku] = {"sol": pending.finish(), "measures": {},
                            "ladder": []}
                continue
            walk = (klass, ceilings.pp_stage)
            if walk not in bases:
                pending = _capacity_pending(
                    p,
                    avg_input_tokens=avg_in,
                    avg_output_tokens=avg_out,
                    prefill_batch_override=pre_b,
                    decode_batch_override=dec_b,
                    prefill_chunk_schedule=base_schedule,
                    fold_prefill_chunks=False,
                    breakdown=False,
                )
                pendings[walk] = pending
                bases[walk] = pending.finish()
            pending = pendings[walk]
            base = bases[walk]
            batches = _point_batches(
                ceilings,
                # None when the request said batch="max" (payload omits the
                # key) → decode uses each card's ceiling. Prefill stays one
                # interactive request; paired prefill is a measurement axis.
                prefill_batch_override=payload.get("prefill_batch_size") or 1,
                decode_batch_override=payload.get("decode_batch_size"))
            rows = _point_rows(base, p, batches, pending, Point({}),
                               breakdown=False)
            rows, keys = pending.keyed_at(rows, gpu_sku=sku)
            sol = without_step(_stamp_gpu_on_result(
                base, p, ceilings, pending.walked_on, batches=batches, rows=rows,
                walks=pending.walk_of_rows, point=Point({}), breakdown=False),
                pending.walk_of_rows)
            endpoint_schedule = base_schedule.with_endpoint_batch(
                batches["per_op_timing"])
            if not sol.get("infeasibility_reason"):
                sol = _fold_chunked_prefill_sol(
                    sol, p, pending, endpoint_schedule)
            # `p`, not the walk's preparation: this point's chunks are priced
            # on the card the point is priced on.
            measures = ({} if keys is None or sol.get("infeasibility_reason")
                        else _point_measures_with_chunks(
                            rows, keys, p, pending, sol["phases"],
                            schedule=endpoint_schedule))
            out[sku] = {"sol": sol, "measures": measures,
                        "ladder": _decode_ladder(
                            payload.get("decode_ladder") or (), sol,
                            base=base, p=p, ceilings=ceilings, batches=batches,
                            rows=rows, pending=pending)}
        except NoPeakForDtype as exc:
            out[sku] = {_LAYOUT_ERROR_KEY: str(exc)}
    return {sku: out[sku] for sku in gpu_skus}


def _decode_ladder(ladder: Sequence[int], sol: dict, *, base: dict, p: Any,
                  ceilings: Any, batches: Dict[str, int], rows: Dict[str, list],
                  pending: Any) -> List[dict]:
    """A card's decode phase at every per-GPU batch of `ladder` this layout
    serves (`capacity.point_served`), and at its ceiling: the points a
    decode Pareto frontier is drawn through. Each is read off the layout's
    one walk by substitution (`capacity._point_rows`) — only the decode row
    lists move; the point the sweep already priced (`sol`) is reused at its
    own batch. SOL only: a point keys no measurement (`PendingEstimate.
    keyed_at` is most of a point's cost), so the frontier has no measured
    track. Each served (`module_rows.served`), no module table."""
    if sol.get("infeasibility_reason"):
        return []
    beyond = bool(sol.get("beyond_max_context"))
    ceiling = max(1, int(ceilings.max_batch or 1))
    wanted: Dict[str, int] = {
        f"b{int(batch)}": int(batch)
        for batch in ladder
        if point_served(max_batch_per_gpu=ceiling,
                        beyond_max_context=beyond, requests=int(batch))
    }
    if not beyond:
        wanted["max"] = ceiling
    out: List[dict] = []
    for batch_key, b in sorted(wanted.items(), key=lambda item: item[1]):
        if b == batches["per_op_timing_decode"]:
            dec = sol["phases"]["decode"]
        else:
            at = _point_batches(ceilings,
                                prefill_batch_override=batches["per_op_timing"],
                                decode_batch_override=b)
            moved = _point_rows(base, p, {k: v for k, v in at.items()
                                          if k != "per_op_timing"},
                                pending, Point({}), breakdown=False)
            dec = _stamp_gpu_on_result(
                base, p, ceilings, pending.walked_on, batches=at,
                rows={**moved, "per_op_timing": rows["per_op_timing"]},
                walks=pending.walk_of_rows, point=Point({}),
                breakdown=False)["phases"]["decode"]
        phase = served({"phases": {"decode": dec}})["phases"]["decode"]
        out.append({"phases": {"decode": {
            k: v for k, v in phase.items() if k != "flops_by_module"}},
            "batch_key": batch_key,
            **({"beyond_max_context": True} if beyond else {})})
    return out


# Reserved key: `_estimate_layout` returns `{gpu_sku: estimate}` (a card that
# cannot time the layout maps to `{key: reason}`) and no GPU is
# named this, so the sweep can tell an error sentinel from a real result.
_LAYOUT_ERROR_KEY = "__error__"


def _estimate_layout_safe(payload: Dict[str, Any]) -> Dict[str, Any]:
    """`_estimate_layout` that isolates a single point's failure instead of
    aborting the whole sweep. A layout the model cannot deploy (CP over a
    recurrent mixer, a degree the experts do not divide, an OOM-at-layout) is
    ONE dead cell, not a dead ladder: return a sentinel the caller renders as
    an empty point that says why, so every OTHER (mode, degree, GPU) still
    reports. Top-level so the pool can pickle it."""
    try:
        return _estimate_layout(payload)
    except MeasurementsUnavailable:
        # Not this layout's answer: the whole sweep cannot say what was
        # measured, so it fails (503) and nothing is cached.
        raise
    except Exception as e:  # a bad point must not sink the sweep
        return {_LAYOUT_ERROR_KEY: str(e)}


@dataclass(frozen=True)
class LayoutCell:
    """One layout of a sweep: its mode, its degree, its config, and the
    `_estimate_layout` payload that prices it on every card of the series."""

    mode: str
    x: int
    parallel: ParallelConfig
    payload: Dict[str, Any]


def run_layout_cell(cell: LayoutCell) -> Dict[str, Any]:
    """One cell of a parallel sweep, reduced to what the sweep keeps: the
    point per card, or the layout's refusal, or the failure that sank the
    estimate. Top-level, so a compute-pool worker can run it; the full
    per-card estimates never leave the process that computed them.

    A failed estimate whose layout the model refuses (`layout_verdict`) is a
    refused degree, not a point; any other failure is a point that says
    why."""
    blob = _estimate_layout_safe(cell.payload)
    err = blob.get(_LAYOUT_ERROR_KEY)
    gpu_skus = cell.payload["gpu_skus"]
    if err:
        resolved = get_default_loader().match(
            cell.payload["config"], model_id=cell.payload.get("model_id"))
        try:
            layout_verdict(resolved, cell.parallel)
        except LayoutRefusal as reason:
            return {"refused": str(reason)}
        # Observability: a swallowed point is a real failure. Log it
        # (a systematic regression would show as a burst, not one dead cell).
        logger.warning("sweep: %s@%d isolated (all GPUs): %s",
                       cell.mode, cell.x, err)
        return {"error": err,
                "points": {g: _point(cell.x, cell.parallel, None, error=err)
                           for g in gpu_skus}}
    # Each card's point before measurement, reduced to what a chart point
    # reads (no per-op rows, no module table); a card that could not time
    # the layout carries its own reason.
    return {"cards": {
        g: ({"error": blob[g][_LAYOUT_ERROR_KEY]}
            if _LAYOUT_ERROR_KEY in (blob.get(g) or {})
            else {"sol": _chart_view(blob[g]["sol"]), "measures": blob[g]["measures"],
                  "ladder": blob[g]["ladder"]})
        for g in gpu_skus}}


def _chart_view(estimate: dict) -> dict:
    """What a chart point (`_point`) and its measured track
    (`batch_sweep.apply_measured`) read of a card's estimate: the serving
    projection's one public point shape. A measured fold may add its measured
    fields afterwards; the SOL projection never carries an estimate plan's
    breakdown."""
    return servingLawPointView(estimate)


@dataclass(frozen=True)
class ParallelSweepPlan:
    """A parallel sweep before its cells run: the modes asked for, the card
    series, the model's caps, and one `LayoutCell` per (mode, degree). The
    cells are independent; `assemble` folds their results in cell order."""

    wanted: Tuple[str, ...]
    gpu_skus: Tuple[str, ...]
    caps: SweepCaps
    skip_ep: Tuple[str, ...]
    cells: Tuple[LayoutCell, ...]
    #: The coordinates every cell shares. `batch` is the request's rung ("max"
    #: or an explicit batch); each ladder point still names its own batch_key.
    context_len: int
    batch: Any
    ep_load_factor: Any
    draft_tokens: int
    acceptance: float
    #: What the measured track is read under — the request's measurement
    #: fields (`batch_sweep.measure_points`).
    measured_under: Dict[str, Any] = field(default_factory=dict)

    def measurement(self) -> Dict[str, Any]:
        return dict(self.measured_under)

    def coordinates(self, mode: str, num_gpus: int, gpu_sku: str, *,
                    batch_key: Any) -> Dict[str, Any]:
        """One grid point's every coordinate (`frontier.POINT_DIMS`)."""
        return point_coordinates(
            mode=mode, num_gpus=num_gpus, gpu_sku=gpu_sku,
            context=self.context_len, batch_key=batch_key,
            draft_tokens=self.draft_tokens, acceptance=self.acceptance,
            ep_load_factor=self.ep_load_factor)

    @property
    def requested_batch_key(self) -> Any:
        """The request's batch rung, for an absence that belongs to a cell."""
        if isinstance(self.batch, int) and self.batch > 0:
            return f"b{int(self.batch)}"
        return "max"

    def absent_coordinates(self, mode: str, num_gpus: int, gpu_sku: str,
                           *, batch_key: Any) -> Dict[str, Any]:
        """A cell-level absence's coordinates, at the request's batch rung."""
        return self.coordinates(mode, num_gpus, gpu_sku, batch_key=batch_key)

    @classmethod
    def of(
        cls,
        *,
        mode: Optional[str] = None,
        modes: Optional[Sequence[str]] = None,
        context_len: int,
        batch: Any = "max",
        gpu_skus: Optional[List[str]] = None,
        base_env: Dict[str, Any],
        ep_load_factor: float = 1.0,
    ) -> "ParallelSweepPlan":
        """`base_env` is the `_resolve_estimate_environment` dict; its
        `parallel` / `gpu_sku` / `num_gpus` are ignored (overridden per
        point). `batch` is the per-GPU batch for BOTH phases, or "max" (each
        config's own capacity max). `ep_load_factor` (max/mean expert load)
        is the batch's routing skew, a workload input carried into every
        point's estimate: the sweep varies topology, not routing."""
        wanted = _resolve_modes(mode, modes)
        series = list(gpu_skus or DEFAULT_SWEEP_GPUS)
        resolved = get_default_loader().match(base_env["config"],
                                              model_id=base_env["model_id"])
        caps = sweep_caps(base_env["config"], resolved=resolved)

        shared = {k: v for k, v in base_env.items() if k not in PER_POINT_FIELDS}
        batch_kw: Dict[str, Any] = {}
        if isinstance(batch, int) and batch > 0:
            batch_kw = {"prefill_batch_size": batch, "decode_batch_size": batch}

        # One cell per layout, every GPU inside. The walk is SKU-blind.
        skip_ep = [m for m in wanted if m in EP_MODES and not caps.moe_enabled]
        cells = [
            LayoutCell(mode=m, x=x, parallel=cfg, payload={
                **shared, "gpu_skus": series, "num_gpus": cfg.cluster_size,
                "parallel": cfg, "avg_input_tokens": context_len, **batch_kw,
                "ep_load_factor": ep_load_factor, "decode_ladder": SCALING_DECODE_LADDER,
            })
            for m in wanted if m not in skip_ep
            for x, cfg in parallel_configs_for_mode(m, caps)
        ]
        return cls(wanted=tuple(wanted), gpu_skus=tuple(series), caps=caps,
                   skip_ep=tuple(skip_ep), cells=tuple(cells),
                   context_len=int(context_len), batch=batch,
                   ep_load_factor=ep_load_factor,
                   draft_tokens=max(0, int(base_env.get("s_verify", 1)) - 1),
                   acceptance=float(base_env.get("acceptance_rate", 1.0)),
                   measured_under={
                       "selection_mode": base_env.get("selection_mode", "theoretical_only"),
                       "measurement_scope": base_env.get("measurement_scope"),
                       "calibration_rows": base_env.get("calibration_rows"),
                       "kernel_mix": base_env.get("kernel_mix")})

    def assemble(self, results: Sequence[Dict[str, Any]]) -> dict:
        """The sweep's answer from `run_layout_cell` over `cells`, in order.

        One mode (legacy `mode=`): `{"mode", "moe_enabled", "series": [...]}`
        plus `by_mode` with that one entry. Several modes: the same per-mode
        blobs under `by_mode`."""
        moe = self.caps.moe_enabled
        # Every card's point of every layout reads its measurements from the
        # sweep's ONE lookup, then becomes a chart point.
        cards = [c for r in results for c in (r.get("cards") or {}).values()
                 if "sol" in c]
        measure_points(cards, **self.measurement())
        # The decode point cloud: every layout on every card at every batch
        # of its ladder (`_decode_ladder`); frontiers are views of it
        # (`decode_frontier`).
        cloud = parallel_cloud(self, results, layout_label).points
        results = [r if "cards" not in r else {"points": {
            g: (_point(cell.x, cell.parallel, None, error=c["error"]) if "error" in c
                else _point(cell.x, cell.parallel, c["sol"]))
            for g, c in r["cards"].items()}}
            for cell, r in zip(self.cells, results, strict=True)]
        buckets: Dict[str, Dict[str, List[dict]]] = {
            m: {g: [] for g in self.gpu_skus}
            for m in self.wanted if m not in self.skip_ep
        }
        # Track a mode's failure reason so a mode whose every point died (e.g.
        # CP over a recurrent mixer) can be reported as unavailable, not a
        # wall of empty cells.
        mode_error: Dict[str, Optional[str]] = {m: None for m in buckets}
        # A degree the layout refuses is not a point: it is reported beside
        # the series with the layout's reason (`layout_verdict`), apart from
        # a point whose estimate failed for another reason.
        refused: Dict[str, List[dict]] = {m: [] for m in buckets}
        for cell, result in zip(self.cells, results, strict=True):
            m = cell.mode
            if "refused" in result:
                refused[m].append({"x": cell.x, "reason": result["refused"]})
                continue
            if result.get("error") and mode_error.get(m) is None:
                mode_error[m] = result["error"]
            for gpu in self.gpu_skus:
                buckets[m][gpu].append(result["points"][gpu])

        by_mode: Dict[str, dict] = {}
        for m in self.wanted:
            if m in self.skip_ep:
                by_mode[m] = {"mode": m, "moe_enabled": False, "series": []}
                continue
            # Every attempted point failed → the mode does not apply to this
            # model (a recurrent mixer under CP, etc.). Collapse to an
            # empty series carrying the reason rather than a series of
            # error-only cells.
            pts = [p for g in self.gpu_skus for p in buckets[m][g]]
            all_failed = bool(pts) and all(p.get("error") for p in pts)
            if not pts and refused[m]:
                # Every degree refused: the mode does not apply, and the first
                # refusal says why.
                mode_error[m] = refused[m][0]["reason"]
                all_failed = True
            entry = {
                "mode": m,
                "moe_enabled": moe,
                "series": ([] if all_failed else
                           [{"gpu_sku": g, "points": buckets[m][g]}
                            for g in self.gpu_skus]),
                # The degrees the layout refused, each with its reason, and
                # the bound past which degrees were not enumerated.
                "ladder": {"refused": refused[m],
                           "beyond_cap": _beyond_cap(m, self.caps)},
            }
            if all_failed and mode_error.get(m):
                entry["unavailable"] = mode_error[m]
            by_mode[m] = entry

        out: Dict[str, Any] = {
            "moe_enabled": moe, "modes": list(self.wanted), "by_mode": by_mode,
            "decode_cloud": cloud,
        }
        if len(self.wanted) == 1:
            out.update(by_mode[self.wanted[0]])
        return out


def estimate_parallel_sweep(
    *,
    mode: Optional[str] = None,
    modes: Optional[Sequence[str]] = None,
    context_len: int,
    batch: Any = "max",
    gpu_skus: Optional[List[str]] = None,
    base_env: Dict[str, Any],
    ep_load_factor: float = 1.0,
) -> dict:
    """Sweep deployment degree for one or more parallel layouts, in this
    process: the plan, its cells across the pool when this process fans
    sweeps out (`map_cells`), the assembled answer. The API server drives
    the same three steps itself so the cells spread across its compute
    workers (`backend.api.sweep_runtime`)."""
    plan = ParallelSweepPlan.of(
        mode=mode, modes=modes, context_len=context_len, batch=batch,
        gpu_skus=gpu_skus, base_env=base_env, ep_load_factor=ep_load_factor)
    return plan.assemble(map_cells(run_layout_cell, list(plan.cells)))
