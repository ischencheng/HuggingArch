"""Gallery decode Pareto frontier — tokens/s per user against tokens/s per GPU.

The InferenceMax-style view of decode: for each model on one GPU type, every
layout up to `FRONTIER_MAX_GPUS` (`parallel_sweep.frontier_layouts`: one
GPU, then 2 … 64 GPUs under every split of them it prices), each at every
per-GPU decode batch of `FRONTIER_BATCH_GRID`, at every Gallery context. A
point is one (layout, batch): how fast one request's tokens come out
(interactivity) and how many tokens each GPU outputs per second. The frontier at a context is the
points no other point beats on both.

Every number is the estimate's own: a point reads its decode phase's
`throughput.sol` (`phases.phase_throughput` — `time_per_output_token_s`,
`per_gpu`), and whether it exists is `capacity.point_served`
(`capacity.served_points`). A layout is one walk, every (context, batch)
of it by substitution (`capacity.sweep_points`), each its own compute job.

The walk does not depend on the GPU, only on its walk class
(`framework_bindings.card_class`: what a framework's dispatch reads of the
card and the dtypes it lists a peak for — what changes a walk's rows). A
model has one CSE SOL law per layout and card class
(`_layout_law`, the shared SOL pool); each card's entry substitutes that law
at its peaks, bandwidth, fabric, floors and capacity facts (`_restamp`). A
view at
another HBM fraction walks its own card alone and keeps no class walk.

The SOL half is cached under scope `none`. Kernel Bench's measured plan is an
explicit later layer, never a reason to expand the canonical law into rows.
One request reads the lake's version once and draws every
model under that answer (`build_frontier`). A track the lake cannot give —
its version unanswered, or its rows' lookup failing mid-request — draws the
SOL drawing, labelled "unavailable": one drawing per view, kept under scope
`none` (`_Stored.view_key`), so a lake that flaps reads it back rather than
drawing again off the entry, and a lake that answers again folds its rates.

The lake's answer reaches the projection, never the law. A class law carries
no measurement scope (`_Stored.walk_identity` under `none`), so a lake outage
neither re-walks a class nor leaves a law under an outage's key.

A model's frontier entry on one GPU is kept on disk and in the remote tier
(`gallery_overview._FRONTIER`), keyed on its `result_version` and the
render config, in two parts (`_entry_parts`): its points — each only what
the read path reads (`_theory_item`), its decode phase and step forms — and
their Kernel Bench measures, most of its bytes. One GPU view spans every
layout up to `FRONTIER_MAX_GPUS`. A page reads drawings: each view's, and
per Kernel Bench snapshot the entry's measured rates with each folded
point's coverage, in `gallery_overview._FRONTIER_VIEWS` (`_Stored`; for
DeepSeek-V3 on H100 a drawing is ~21 KB and the measured rates ~300 KB, most
of it the 2,545 distinct unmeasured keys the coverage panel names). A view already drawn reads
no entry; another view reads the points and not the measures
(`_frontier_read`).

Each point's step is cached as a function of the two floors (`FloorForm`),
and for a model with an MTP draft at every K of `FRONTIER_DRAFT_TOKENS`
too — priced when the entry is built (the warm builds it), one speculative
walk per layout with every K read off it (`_draft_walk`, `_draft_points`).
A view's floors,
K and acceptance are substituted when it is drawn (`_draw`); none of
them walks. A K whose step failed to price is published with the views at
that K (`draft_errors`).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import statistics
import functools
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.api.gallery_overview import (
    _FRONTIER,
    _FRONTIER_VIEWS,
    _grid_points,
    _model_concurrency,
    _resolve_one,
)
from backend.api.offload import (
    PrioritySlots,
    ensure_spec_generation,
    map_off_loop,
    report_partial,
    report_progress,
    run_off_loop,
    run_ranked,
    spec_generation,
    urgency,
)
from backend.arch.eval.precision import AS_SHIPPED
from backend.arch.snapshots import iter_snapshot_ids
from backend.inference.batch_sweep import measure_points
from backend.inference.capacity import ClassSweep, point_served, served_points
from backend.inference.card import Card
from backend.measure import lookup as _measure_lookup
from backend.measure.auto_sweep import DECLINED, STATES
from backend.measure.lookup import MeasurementsUnavailable
from backend.inference.defaults import (COMM_LATENCY_US_DEFAULT,
                                        GPU_MEMORY_FRACTION_DEFAULT,
                                        LAUNCH_FLOOR_US_DEFAULT)
from backend.inference.gpu_specs import NoPeakForDtype
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import Speculation, phase_throughput, step_tokens
from backend.inference.row_roofline import FloorForm, LawBook
from backend.inference.speculative_draft import SpeculativeDraft
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import LayoutRefusal
from backend.inference.parallel_sweep import (
    ALL_MODES,
    class_cards,
    class_frontier_layouts,
    frontier_layouts,
    layout_label,
    layout_verdict,
    parallel_configs_for_mode,
    sweep_caps,
)
from backend.inference.frontier import (POINT_DIMS, frontier_view, parallel_mode,
                                       pareto_front as _pareto_front)
from backend.inference.sweep_grid import (
    FRONTIER_ACCEPTANCE_DEFAULT,
    FRONTIER_BATCH_GRID,
    FRONTIER_DRAFT_TOKENS,
    FRONTIER_MAX_GPUS,
    GALLERY_OUTPUT_TOKENS,
    GALLERY_SEQ_LENS,
    ROUTINE_SEQ_LENS,
    SWEEP_MAX_GPUS,
)
from backend.inference.sol_data import (
    StepLawKey,
    ServingWalkFacts,
    ServingWalkKey,
    capacityBatchPoint,
    walkManifestIsValid,
    walkRecordSchema,
)
from backend.inference.sol_data import sharedSolPool
from backend.code_trace.provider import recording, run_recorded
from backend.inference.sol_law import (
    draftForwardsLaw,
    servingLawCapacity,
    servingLawPoints,
    servingLawRecord,
)
from backend.inference.point_cloud import (CloudRecord, point_coordinates,
                                           produce_cloud)
from backend.inference.serving_projection import (
    PublicServingLawRequest,
    projectServingLawPoints,
)
from backend.measure.scope import NONE as NO_MEASUREMENTS, of_render

logger = logging.getLogger(__name__)

# Compatibility for callers that learned the front function from this module;
# the implementation and every new caller read `frontier.pareto_front`.
pareto_front = _pareto_front

# Payload-schema stamps folded into the frontier's keys — bump on edit of
# the fields of what each keys (the grid constants sit under the engine
# version): the entry's two parts, and what views read off them (the
# measured rates and a drawn view). A drawing's fields change without the
# entry's, and an entry is priced again only when its own do.
_FRONTIER_SCHEMA = "gallery-frontier-v11"
_DRAWING_SCHEMA = "gallery-frontier-drawing-v3"
_GALLERY_SELECTION = "envelope"
#: The precision profile every frontier is priced under: as shipped.
_GALLERY_PRECISION = AS_SHIPPED
#: How many of a coverage state's keys a frontier's coverage lists, most
#: needed first (`frontier_coverage`; the count is of all of them).
COVERAGE_KEYS_LISTED = 10


class FrontierView(NamedTuple):
    """What one frontier is priced under besides the models: the GPU, the
    HBM fraction a deployment budgets, and the two time floors (one link
    traversal's latency and the shortest one kernel launch takes, µs;
    `defaults`). One GPU view at the defaults is what the page
    opens on and the warm fills."""
    gpu_sku: str
    gpu_memory_fraction: float = GPU_MEMORY_FRACTION_DEFAULT
    comm_latency_us: float = COMM_LATENCY_US_DEFAULT
    launch_floor_us: float = LAUNCH_FLOOR_US_DEFAULT
    #: Speculative decoding with the model's MTP draft: K drafted tokens
    #: per step (0 is plain decode; one of `FRONTIER_DRAFT_TOKENS`) kept at
    #: `acceptance_rate`. Read off the entry like the floors.
    draft_tokens: int = 0
    acceptance_rate: float = FRONTIER_ACCEPTANCE_DEFAULT

    def walk_parts(self) -> List[Any]:
        """What a model's cached entry is priced under: not the floors —
        each point carries its step as a function of them (`FloorForm`)."""
        return [self.gpu_sku, _GALLERY_PRECISION.key,
                float(self.gpu_memory_fraction)]

    def keeps_walk(self) -> bool:
        """Whether the view reads the model's kept class walk
        (`_Stored.walk_identity`): at the default HBM fraction. Each card's
        ceiling batch is walked on that card at that fraction, so a view at
        another walks its own card alone (`walk_cards`), kept nowhere."""
        return (float(self.gpu_memory_fraction)
                == ServingWalkFacts.producer().gpu_memory_fraction)

    def _servingLawEnvironment(self) -> dict[str, Any]:
        return {
            **ServingWalkFacts.producer().to_environment(),
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "comm_latency_us": self.comm_latency_us,
            "launch_floor_us": self.launch_floor_us,
        }

    def _servingLawAxes(self, *, draft_tokens: int = 0) -> dict[str, Any]:
        batches = tuple(
            point["decode_batch_size"] for point in FRONTIER_BATCH_GRID)
        return {
            "contexts": tuple(GALLERY_SEQ_LENS),
            "decode_batches": batches,
            "prefill_batches": (1,) * len(batches),
            "prefix_tokens": (0,) * len(GALLERY_SEQ_LENS),
            "verify_widths": (1 + int(draft_tokens),)
            * len(GALLERY_SEQ_LENS),
        }

    def servingLawRequest(self, walk: ServingWalkKey, layout: Any, *,
                          draft_tokens: int = 0) -> PublicServingLawRequest:
        return PublicServingLawRequest(
            walk=walk, card=self.gpu_sku, layout=layout,
            environment=self._servingLawEnvironment(),
            draft_tokens=draft_tokens,
            **self._servingLawAxes(draft_tokens=draft_tokens))

    def entry_parts(self) -> List[Any]:
        """What a model's cached entry is priced under, including its floors."""
        return [*self.walk_parts(), float(self.comm_latency_us),
                float(self.launch_floor_us)]

    def walk_cards(self) -> List[str]:
        """The cards one walk serves this view from, the walk card first:
        the GPU's walk class (`parallel_sweep.class_cards`) when the view
        keeps the walk, else the GPU alone."""
        return class_cards(self.gpu_sku) if self.keeps_walk() else [self.gpu_sku]

    def key_parts(self) -> List[Any]:
        return [*self.walk_parts(), float(self.comm_latency_us),
                float(self.launch_floor_us), int(self.draft_tokens),
                float(self.acceptance_rate) if self.draft_tokens else None]

    def floors_s(self) -> Tuple[float, float]:
        """The two floors in seconds: link latency, launch floor."""
        return self.comm_latency_us * 1e-6, self.launch_floor_us * 1e-6


def _frontier_phase(dec: dict) -> dict:
    """A point's decode phase as the frontier keeps it: what the read path
    reads of it (`decode_point`, `_at_view`, the measured fold `batch_sweep.
    apply_measured`) and nothing a card prices — the step's time and rates
    are its form's on the card the entry is for (`capacity.ClassSweep.
    on_card`), so a phase walked on one card of a class is every card's.
    A speculative step keeps what its draft is, not its forward's times."""
    out = {k: dec[k] for k in ("phase", "batch", "binding",
                               "time_unpublished_reason", "t_measured_s",
                               "measured_rows", "total_rows", "coverage")
           if k in dec}
    draft = dec.get("speculative_draft")
    if draft:
        out["speculative_draft"] = {
            k: draft[k] for k in ("style", "stacks", "draft_tokens", "forwards",
                                  "tokens_per_forward") if k in draft}
    return copy.deepcopy(out)


def _batch_key(batch: Any) -> str:
    """A decode batch's frontier-grid rung (`max` for a card's ceiling)."""
    for point in FRONTIER_BATCH_GRID:
        if point["decode_batch_size"] == batch:
            return str(point["key"])
    return "max"


def _theory_item(seq: str, label: str, mode: str, num_gpus: int,
                 batch_key: str, per: dict, measures: dict,
                 floors: Optional[dict]) -> dict:
    """One cached point: what the read path reads of it and nothing else.
    The frontier draws decode, so the point keeps its decode phase alone
    (`_frontier_phase`, priced on the entry's card) and its decode measure;
    `floors` is the step as a function of the floors. The per-op rows carry
    SymPy laws that live in the worker that walked them; shipped back raw,
    the pool's one result thread rebuilds tens of thousands of expressions
    per layout and every other request waits behind it."""
    return {"seq": seq, "label": label, "mode": mode, "num_gpus": num_gpus,
            "batch_key": batch_key,
            "sol": {"phases": {"decode": per["phases"]["decode"]}},
            "floors": floors,
            "measures": ({"decode": measures["decode"]}
                         if measures.get("decode") else {})}


def _layout_request(model_id: str, gpu_sku: str, fraction: float,
                    precision: dict, parallel: dict
                    ) -> InferenceEstimateSweepRequest:
    """The sweep one layout of a model is priced by on `gpu_sku` under the
    precision profile `precision` (canonical): every Gallery context at
    every frontier batch, at the default floors (the view's are applied
    when the entry is read, `_at_view`)."""
    defaults = FrontierView(gpu_sku)
    return InferenceEstimateSweepRequest(
        model_id=model_id,
        gpu_sku=gpu_sku,
        parallel=parallel,
        precision={"canonical": precision},
        gpu_memory_fraction=fraction,
        comm_latency_us=defaults.comm_latency_us,
        launch_floor_us=defaults.launch_floor_us,
        avg_output_tokens=GALLERY_OUTPUT_TOKENS,
        seq_lens=list(GALLERY_SEQ_LENS),
        batch_grid=[BatchPoint(**bp) for bp in FRONTIER_BATCH_GRID],
        # Keys on each point so Kernel Bench rows can fold later; the walk
        # itself reads no measurements (`measurement_scope` stays none).
        selection_mode=_GALLERY_SELECTION,
    )


def _class_grids(cards: List[str]) -> Dict[str, List[dict]]:
    """The batches each card of a class sweeps: the walk card (first) the
    whole frontier grid plus the capacity row, every other card only its own
    capacity row — the rest are the walk card's points (`capacity.ClassSweep`)."""
    return {sku: ([*FRONTIER_BATCH_GRID, capacityBatchPoint] if i == 0
                  else [capacityBatchPoint])
            for i, sku in enumerate(cards)}


def _class_sweeps(requests: Dict[str, InferenceEstimateSweepRequest], *,
                  seq_lens: List[int], s_verifies: Optional[List[int]] = None
                  ) -> Dict[str, dict]:
    """Each card's sweep of one layout (`_grid_points`, the batches
    `_class_grids` gives it), or `{"error": reason}` for a card whose sweep
    fails. The walk card's (first) failure is the layout's: raised."""
    grids = _class_grids(list(requests))
    out: Dict[str, dict] = {}
    for i, (sku, req) in enumerate(requests.items()):
        try:
            out[sku] = _grid_points(resolve_estimate_environment(req),
                                    seq_lens=seq_lens, batch_grid=grids[sku],
                                    s_verifies=s_verifies)
        except Exception as e:
            if i == 0:
                raise
            out[sku] = {"error": str(e)}
    return out


def _class_sweep(swept: Dict[str, dict], cards: List[str],
                 book: LawBook) -> dict:
    """`capacity.ClassSweep` of each card's sweep, as plain data."""
    return ClassSweep.of(swept, walk_card=cards[0],
                         batch_grid=[*FRONTIER_BATCH_GRID, capacityBatchPoint],
                         project=_walk_phase, book=book,
                         floors_s=FrontierView(cards[0]).floors_s()).to_json()


def _walk_phase(name: str, phase: dict) -> dict:
    """A canonical record's phase, kept as facts rather than as a page view."""
    return _frontier_phase(phase) if name == "decode" else {
        key: phase[key] for key in (
            "phase", "batch", "binding", "time_unpublished_reason",
            "t_measured_s", "measured_rows", "total_rows", "coverage")
        if key in phase
    }


def _layout_walk(item: Tuple[str, List[str], float, dict, dict]) -> dict:
    """One layout of one model walked for every card of a walk class
    (`parallel_sweep.class_cards`, the walk card first) at every Gallery
    context and frontier batch: `{"plain": <ClassSweep>, "draft": …}` —
    `"draft"` for a model with an MTP draft, its sweep at every K of
    `FRONTIER_DRAFT_TOKENS` (`_draft_walk`) — or `{"refused": reason}` when
    the layout walk refuses the deployment, `{"no_peak": [dtype, kind]}`
    when the class has no peak for a dtype the deployment computes in
    (`NoPeakForDtype`, said on each card when it is priced), or `{"error":
    …}`. Priced at the default floors; a view's are applied when it is
    read. `item` is (model, cards, HBM fraction, precision profile —
    canonical —, layout). Top-level so a compute worker can run it."""
    model_id, cards, fraction, precision, parallel = item
    cfg = ParallelConfig.from_dict(parallel)
    label = layout_label(cfg)
    requests = {sku: _layout_request(model_id, sku, fraction, precision, parallel)
                for sku in cards}
    book = LawBook()
    try:
        swept = _class_sweeps(requests, seq_lens=list(GALLERY_SEQ_LENS))
        plain = _class_sweep(swept, cards, book)
    except Exception as e:
        # A layout the model does not deploy under is a refused layout, not
        # a failure: the layout walk says which (as `run_layout_cell` asks).
        try:
            layout_verdict(_resolve_one(model_id)[1], cfg)
        except LayoutRefusal as reason:
            return {"refused": str(reason)}
        except Exception as diagnostic:
            logger.warning(
                "gallery frontier: %s under %s failed (%s); layout verdict "
                "also failed (%s)", model_id, label, e, diagnostic)
            return {"error": str(e), "diagnostic": str(diagnostic)}
        if isinstance(e, NoPeakForDtype):
            return {"no_peak": [e.dtype, e.kind]}
        logger.warning("gallery frontier: %s under %s failed: %s", model_id, label, e)
        return {"error": str(e)}
    out: Dict[str, Any] = {"plain": plain}
    # A layout no card of the class draws a point of has no speculative
    # view on any of them (`_restamp_layout`): its draft is not walked.
    if draft_refusal(model_id) is None and any(
            "theory" in priced and priced["theory"]
            for priced in (_restamp_layout(out, sku, cfg) for sku in cards)):
        out["draft"] = _draft_walk(requests, cards, book)
    return out


def _layout_law(item: Tuple[str, List[str], float, dict, dict]) -> dict:
    """One layout's SOL laws, walked once and stored without point payloads."""
    model_id, cards, fraction, precision, parallel = item
    cfg = (parallel if isinstance(parallel, ParallelConfig)
           else ParallelConfig.from_dict(parallel))
    request = _layout_request(model_id, cards[0], fraction, precision,
                              cfg.to_dict())
    try:
        environment = resolve_estimate_environment(request)
        if draft_refusal(model_id) is None:
            capacity = servingLawCapacity(
                environment, gpu_sku=cards[0], parallel=cfg,
                capacityCards=cards, capacityContexts=ROUTINE_SEQ_LENS)
            if all(int(fact["max_decode_batch"] or 0) <= 0
                   and int(fact["max_prefill_batch"] or 0) <= 0
                   for fact in capacity):
                return {"refused": next(
                    fact.get("infeasibility_reason")
                    or "layout admits no request" for fact in capacity)}
            spec = request.model_copy(update={
                "speculative_draft_tokens": 1,
                "speculative_draft": "recursive",
                "acceptance_rate": FRONTIER_ACCEPTANCE_DEFAULT,
                "selection_mode": "theoretical_only",
            })
            record = servingLawRecord(
                resolve_estimate_environment(spec),
                gpu_sku=cards[0], anchorContext=GALLERY_SEQ_LENS[0],
                outputTokens=GALLERY_OUTPUT_TOKENS, parallel=cfg,
                capacityCards=cards, capacityContexts=ROUTINE_SEQ_LENS,
                draft_forwards=draftForwardsLaw)
            if "refused" in record:
                return record
            record["draft_capacity"] = record["capacity"]
            record["capacity"] = capacity
        else:
            record = servingLawRecord(
                environment,
                gpu_sku=cards[0], anchorContext=GALLERY_SEQ_LENS[0],
                outputTokens=GALLERY_OUTPUT_TOKENS, parallel=cfg,
                capacityCards=cards, capacityContexts=ROUTINE_SEQ_LENS)
        return record
    except Exception as e:
        try:
            layout_verdict(_resolve_one(model_id)[1], cfg)
        except LayoutRefusal as reason:
            return {"refused": str(reason)}
        except Exception as diagnostic:
            logger.warning(
                "gallery frontier: law for %s under %s failed (%s); layout "
                "verdict also failed (%s)", model_id, layout_label(cfg), e,
                diagnostic)
            return {"error": str(e), "diagnostic": str(diagnostic)}
        if isinstance(e, NoPeakForDtype):
            return {"no_peak": [e.dtype, e.kind]}
        logger.warning("gallery frontier: law for %s under %s failed: %s",
                       model_id, layout_label(cfg), e)
        return {"error": str(e)}


def draft_refusal(model_id: str) -> Optional[str]:
    """Why the model's frontier has no speculative view (it drafts with
    no MTP stack, `SpeculativeDraft.refusal`), or None when it has one."""
    resolved = _resolve_one(model_id)[1]
    if resolved is None:
        return "no spec resolves for this model"
    return SpeculativeDraft.refusal(Program.of(resolved))


def _draft_walk(requests: Dict[str, InferenceEstimateSweepRequest],
                cards: List[str], book: LawBook) -> dict:
    """A layout's step under speculative decoding with the model's MTP
    draft, at every K of `FRONTIER_DRAFT_TOKENS`, for every card of the
    class: `{"ks": {K: <ClassSweep> or {"error": reason}}}`, or `{"error":
    reason}` when the sweep fails (every K's failure).

    Every K is one sweep: each Gallery context once per K, at that K's
    verify width (`capacity.CapacitySweepPlan`'s `s_verifies`). Every
    speculative width is one walk of the layout, the width a symbol of it,
    so each K's points are that walk's rows read at K + 1 by substitution
    (`capacity._sweep_cell`); what a K adds is reading every point at it.
    The sweep looks no measurement up: a speculative step has no measured
    track here. The acceptance rate is not in a step: it moves only the
    tokens a step keeps, applied when the view is read (`_at_view`)."""
    seq_lens = list(GALLERY_SEQ_LENS)
    ks = list(FRONTIER_DRAFT_TOKENS)
    spec = {sku: req.model_copy(update={
        "speculative_draft_tokens": ks[0], "speculative_draft": "recursive",
        "acceptance_rate": FRONTIER_ACCEPTANCE_DEFAULT,
        "selection_mode": "theoretical_only"}) for sku, req in requests.items()}
    try:
        swept = _class_sweeps(
            spec, seq_lens=seq_lens * len(ks),
            s_verifies=[Speculation(k).s_verify for k in ks for _ in seq_lens])
    except Exception as e:
        logger.warning("gallery frontier: %s on %s, speculative: %s",
                       requests[cards[0]].model_id, cards[0], e)
        return {"error": str(e)}
    n = len(seq_lens)
    out: Dict[str, Any] = {}
    for i, k in enumerate(ks):
        at_k = {sku: card if "error" in card else
                {key: pts[i * n:(i + 1) * n] for key, pts in card.items()}
                for sku, card in swept.items()}
        try:
            out[str(k)] = _class_sweep(at_k, cards, book)
        except Exception as e:
            logger.warning("gallery frontier: %s on %s with K=%d: %s",
                           requests[cards[0]].model_id, cards[0], k, e)
            out[str(k)] = {"error": str(e)}
    return {"ks": out}


def _restamp_layout(walked: dict, gpu_sku: str, cfg: ParallelConfig,
                    walk_key: Optional[ServingWalkKey] = None,
                    view: Optional[FrontierView] = None) -> dict:
    """One layout's walk (`_layout_walk`) on `gpu_sku`, one of the class's
    cards: `{"theory": [point…], "draft_errors": […]}` — each point served
    on the card, its decode step as a function of the floors there, and for
    a model with an MTP draft its step at every K (`_draft_points`) — or
    the walk's refusal or error, said on this card."""
    if walked.get("schema") == walkRecordSchema:
        card = Card.of(gpu_sku)
        view = view if view is not None else FrontierView(gpu_sku)

        theory: List[dict] = []
        draftErrors: List[dict] = []
        byPoint: Dict[Tuple[str, int], dict] = {}
        label, num_gpus = layout_label(cfg), cfg.cluster_size

        def rows(record: dict, *, draft_tokens: int = 0) -> list[list[dict | None]]:
            if walk_key is None:
                return servingLawPoints(
                    record, card=card, contexts=list(GALLERY_SEQ_LENS),
                    decodeBatches=[point["decode_batch_size"]
                                   for point in FRONTIER_BATCH_GRID],
                    outputTokens=GALLERY_OUTPUT_TOKENS, prefillBatches=1,
                    prefixCached=0, verifyWidth=draft_tokens + 1,
                    draftTokens=draft_tokens,
                    acceptanceRate=FRONTIER_ACCEPTANCE_DEFAULT,
                    comm_latency_us=view.comm_latency_us,
                    launch_floor_us=view.launch_floor_us,
                    servedOnly=True)
            request = view.servingLawRequest(
                walk_key, record["parallel"], draft_tokens=draft_tokens)
            return projectServingLawPoints(record, request)

        try:
            points = rows(walked)
        except Exception as e:
            logger.warning("gallery frontier law for %s on %s failed: %s",
                           label, gpu_sku, e)
            return {"error": str(e)}
        for seq, row in zip(GALLERY_SEQ_LENS, points, strict=True):
            for batchPoint, served in zip(FRONTIER_BATCH_GRID, row, strict=True):
                if served is None:
                    continue
                if not point_served(
                        max_batch_per_gpu=served["max_batch_per_gpu"],
                        beyond_max_context=served["beyond_max_context"],
                        requests=batchPoint["decode_batch_size"] or 1):
                    continue
                item = _theory_item(
                    str(seq), label, parallel_mode(cfg), num_gpus,
                    _batch_key(served["phases"]["decode"]["batch"]), served,
                    {}, None)
                theory.append(item)
                byPoint[(str(seq), served["phases"]["decode"]["batch"])] = item
        if walked.get("speculative"):
            draftRecords = {str(k): walked for k in FRONTIER_DRAFT_TOKENS}
        else:
            draftRecords = {}
        for textK, draftRecord in draftRecords.items():
            k = int(textK)
            try:
                draftPoints = rows(draftRecord, draft_tokens=k)
            except Exception as exc:
                draftErrors.append({
                    "layout": label, "draft_tokens": k, "error": str(exc),
                })
                continue
            for seq, row in zip(GALLERY_SEQ_LENS, draftPoints, strict=True):
                for batchPoint, served in zip(FRONTIER_BATCH_GRID, row,
                                              strict=True):
                    if served is None:
                        continue
                    if not point_served(
                            max_batch_per_gpu=served["max_decode_batch_per_gpu"],
                            beyond_max_context=served["beyond_max_context"],
                            requests=batchPoint["decode_batch_size"] or 1):
                        continue
                    batch = served["phases"]["decode"]["batch"]
                    item = byPoint.get((str(seq), batch))
                    if item is None:
                        item = _theory_item(
                            str(seq), label, parallel_mode(cfg), num_gpus,
                            _batch_key(batch), served, {}, None)
                        theory.append(item)
                        byPoint[(str(seq), batch)] = item
                    item.setdefault("draftTimes", {})[textK] = served[
                        "phases"]["decode"]["t_sol_s"]
        return {
            "theory": theory,
            "draft_errors": draftErrors,
        }
    if "refused" in walked or "error" in walked:
        return {k: walked[k] for k in ("refused", "error") if k in walked}
    if "no_peak" in walked:
        return {"error": str(NoPeakForDtype(gpu_sku, *walked["no_peak"]))}
    label, num_gpus = layout_label(cfg), cfg.cluster_size
    floors_s = FrontierView(gpu_sku).floors_s()
    seq_lens = list(GALLERY_SEQ_LENS)
    try:
        swept = ClassSweep.from_json(walked["plain"]).on_card(
            gpu_sku, batch_grid=list(FRONTIER_BATCH_GRID), floors_s=floors_s)
    except Exception as e:
        logger.warning("gallery frontier: %s on %s: %s", label, gpu_sku, e)
        return {"error": str(e)}
    theory = [_theory_item(
                  seq, label, parallel_mode(cfg), num_gpus,
                          _batch_key(per["phases"]["decode"].get("batch")), per,
                          pt.get("measures") or {}, pt["forms"]["decode"])
              for seq, per, pt in served_points(swept, seq_lens,
                                                list(FRONTIER_BATCH_GRID))]
    draft_errors: List[dict] = []
    if theory and walked.get("draft") is not None:
        draft_errors = _draft_points(walked["draft"], theory, gpu_sku)
    return {"theory": theory, "draft_errors": draft_errors}


def _draft_points(draft: dict, theory: List[dict], gpu_sku: str) -> List[dict]:
    """Each theory point's decode step under speculative decoding at every
    K of the layout's draft walk (`_draft_walk`) on `gpu_sku`, as a function
    of the floors (the verify over K + 1 positions plus K one-token draft
    forwards), on the point as `draft[str(K)]`.

    A point whose step at a K does not serve its batch has no form at that
    K; a batch served only at a K (the `max` one, which the verify and the
    draft's cache move) is a point of its own, carrying that K's step.
    Returns the K's that failed, each `{"layout", "draft_tokens", "error"}`
    — the sweep's failure is every K's — published with the view
    (`draft_errors`), never dropped."""
    label, mode, num_gpus = (theory[0]["label"], theory[0]["mode"],
                             theory[0]["num_gpus"])
    ks = list(FRONTIER_DRAFT_TOKENS)
    if "error" in draft:
        return [{"layout": label, "draft_tokens": k, "error": draft["error"]}
                for k in ks]
    seq_lens = list(GALLERY_SEQ_LENS)
    floors_s = FrontierView(gpu_sku).floors_s()
    at = {(t["seq"], t["sol"]["phases"]["decode"]["batch"]): t for t in theory}
    errors: List[dict] = []
    for k in ks:
        walked = draft["ks"][str(k)]
        try:
            if "error" in walked:
                raise ValueError(walked["error"])
            swept = ClassSweep.from_json(walked).on_card(
                gpu_sku, batch_grid=list(FRONTIER_BATCH_GRID), floors_s=floors_s)
            forms = list(served_points(swept, seq_lens, list(FRONTIER_BATCH_GRID)))
        except Exception as e:
            logger.warning("gallery frontier: %s on %s with K=%d: %s",
                           label, gpu_sku, k, e)
            errors.append({"layout": label, "draft_tokens": k, "error": str(e)})
            continue
        for seq, per, pt in forms:
            batch = per["phases"]["decode"]["batch"]
            item = at.get((seq, batch))
            if item is None:
                item = at[(seq, batch)] = _theory_item(
                    seq, label, mode, num_gpus, _batch_key(batch), per, {},
                    None)
                theory.append(item)
            item.setdefault("draft", {})[str(k)] = pt["forms"]["decode"]
    return errors


def _restamp(model_id: str, walk: dict, gpu_sku: str,
             view: Optional[FrontierView] = None) -> dict:
    """A model's frontier entry on `gpu_sku` off its class walk
    (`_layout_walk` of every layout any card of the class prices): the
    layouts a frontier on this card prices (`frontier_layouts`), each
    priced on it (`_restamp_layout`) and folded (`_fold`). No walk."""
    walked = {json.dumps(w["parallel"], sort_keys=True): w["walk"]
              for w in walk["layouts"]}
    layouts = [ParallelConfig.from_dict(p)
               for p in walk["layoutsByCard"][gpu_sku]]
    walk_key = (
        _Stored(model_id, _resolve_one(model_id)[2]).serving_key(view)
        if view is not None and view.keeps_walk() else None)
    results = [_restamp_layout(
                   walked[json.dumps(cfg.to_dict(), sort_keys=True)],
                   gpu_sku, cfg, walk_key, view) for cfg in layouts]
    entry = _fold(model_id, layouts, results)
    # Why a speculative view draws nothing for this model (no MTP draft).
    entry["draft_unavailable"] = draft_refusal(model_id)
    return entry


def _fold(model_id: str, layouts: List[ParallelConfig], results: List[dict]) -> dict:
    """A model's frontier entry from its layouts' points: the frontier at
    each context, what each layout answered, and every layout's failed
    speculative K (`draft_errors`)."""
    summary: List[dict] = []
    theory: List[dict] = []
    draft_errors: List[dict] = []
    for cfg, res in zip(layouts, results, strict=True):
        row: Dict[str, Any] = {"layout": layout_label(cfg), "parallel": cfg.to_dict()}
        if "theory" in res:
            row["points"] = len(res["theory"])
            theory.extend(res["theory"])
            draft_errors.extend(res.get("draft_errors") or [])
        else:
            row.update({k: res[k] for k in ("refused", "error") if k in res})
        summary.append(row)
    failed = [r for r in summary if "error" in r]
    return {
        "model_id": model_id,
        "layouts": summary,
        "_theory": theory,
        "_draft_errors": draft_errors,
        # Every layout failing (not refused) is the model's failure.
        "error": (failed[0]["error"] if failed and len(failed) == len(summary)
                  else None),
    }


def _frontier_scope() -> Tuple[Any, str]:
    """The shared rows' scope, or `none` / `unavailable` when there is
    nothing to fold."""
    try:
        scope = _measure_lookup.resolve_scope(reads=True, owner=None)
    except MeasurementsUnavailable:
        return NO_MEASUREMENTS, "unavailable"
    return scope, ("reads" if scope.reads else "none")


def _public_entry(entry: dict) -> dict:
    """The entry without what each view reads its own part of (`_theory`,
    `_draft_errors`)."""
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _at_view(item: dict, view: FrontierView,
             ) -> Optional[dict]:
    """A theory point with its decode step under the view: the step's
    time already projected at the entry's floors — the plain step's, or with
    K drafted tokens the speculative step's at that K — the
    tokens a step keeps at the view's acceptance (`phases.step_tokens`), and
    its throughput off those (`phases.phase_throughput`, the estimate's own
    reading). None when the point has no step under the view: no form at
    that K, or under plain decode a point whose own step verifies drafts (a
    batch served only speculatively, `_draft_points`). A plain point that
    publishes no time is left as it is."""
    k = int(view.draft_tokens)
    sol = item["sol"]
    dec = dict(sol["phases"]["decode"])
    if not k and dec.get("speculative_draft"):
        return None
    draftTime = (item.get("draftTimes") or {}).get(str(k)) if k else None
    if draftTime is not None:
        dec["t_sol_s"] = draftTime
    else:
        form = (item.get("draft") or {}).get(str(k)) if k else item.get("floors")
        if form is None:
            return None if k else item
        dec["t_sol_s"] = FloorForm.from_json(form).at(*view.floors_s())
    if k:
        binding = dict(dec["binding"])
        tokens, formula = step_tokens(
            "decode", int(dec["batch"]), binding["tokens_per_step"],
            Speculation(k, float(view.acceptance_rate)))
        binding.update(effective_tokens_per_step=tokens,
                       effective_tokens_formula=formula)
        dec["binding"] = binding
    # The groups and GPUs the step's own rates were taken over, as the
    # measured fold re-reads them (`batch_sweep.apply_measured`).
    dec["throughput"] = phase_throughput(
        dec, unique_tokens_multiplier=dec["throughput"]["groups"],
        num_gpus=dec["throughput"]["num_gpus"])
    return {**item, "sol": {**sol, "phases": {**sol["phases"], "decode": dec}}}


def _view_scope(view: FrontierView) -> Tuple[Any, str]:
    """The measurement scope a view's measured track reads, and its state:
    none under a draft — the measured rows are keyed on the plain step's
    kernels, so a speculative step has no measured track here — else the
    shared rows' (`_frontier_scope`). A request reads it once and draws
    every model under it (`build_frontier`)."""
    if view.draft_tokens:
        return NO_MEASUREMENTS, "none"
    return _frontier_scope()


#: `(scope, state)` of `_view_scope`, as one request read it.
Track = Tuple[Any, str]


def _labelled(drawing: dict, state: str) -> dict:
    """A drawing of no measured rates under the track state it is served
    with: "unavailable" draws exactly what "none" does (`_draw` reads no
    rates for either), so one drawing serves both."""
    if drawing["measured_track"]["state"] == state:
        return drawing
    return {**drawing, "measured_track": {"state": state}}


def _entry_parts(entry: dict) -> Tuple[dict, dict]:
    """A built entry as it is stored, in two parts: the entry without its
    points' measures — what every view reads — and the measures, in point
    order, which only the measured fold reads (once per measurement scope,
    `_measured_rates`). The measures are most of an entry's bytes."""
    theory = entry.get("_theory") or []
    core = {**entry, "_theory": [{k: v for k, v in t.items() if k != "measures"}
                                 for t in theory]}
    return core, {"measures": [t.get("measures") or {} for t in theory]}


def _measured_rates(core: dict, measures: dict, scope: Any) -> dict:
    """What the measured fold (`batch_sweep.measure_points`) says of each
    plain point under `scope`, by the point's index in the entry: its
    measured throughput (`rates`; absent for a point the fold publishes no
    measured time for), and its decode coverage (`coverage`,
    `_PointCoverage`). Floors, K and acceptance move neither, so every
    plain view under `scope` reads these; which points are on its frontier
    is the view's (`_draw`)."""
    index: List[int] = []
    items: List[dict] = []
    for i, (t, m) in enumerate(zip(core.get("_theory") or [],
                                   measures["measures"], strict=True)):
        if m and not t["sol"]["phases"]["decode"].get("speculative_draft"):
            index.append(i)
            items.append({"sol": copy.deepcopy(t["sol"]), "measures": m})
    if items:
        measure_points(items, selection_mode=_GALLERY_SELECTION,
                       measurement_scope=scope)
    rates: Dict[str, dict] = {}
    coverage = _PointCoverage()
    for i, item in zip(index, items, strict=True):
        dec = item["sol"]["phases"]["decode"]
        meas = (dec.get("throughput") or {}).get("measured")
        if meas and meas.get("time_per_output_token_s"):
            rates[str(i)] = {"time_per_output_token_s": meas["time_per_output_token_s"],
                             "per_gpu": meas["per_gpu"]}
        if "decode" in item["measures"]:
            coverage.add(i, dec["coverage"])
    return {"rates": rates, "coverage": coverage.to_json()}


class _PointCoverage:
    """Each folded point's decode coverage (`MeasuredTrack.coverage`) as
    the measured rates keep it: `[measured, total, key indices]` per point,
    each non-measured key named once for the entry (`keys`: `[state, key]`,
    or `[state, key, reason]` when declined). A model's points share most of
    their keys, so each is kept once rather than once per point: DeepSeek-V3
    on H100 has 2,545 of them (~231 KB) and the per-point index lists
    ~69 KB. Kept exact rather than grouped by op signature (~30 KB), because
    what the panel lists is the shape to measure next."""

    def __init__(self) -> None:
        self.keys: List[list] = []
        self.index: Dict[tuple, int] = {}
        self.points: Dict[str, list] = {}

    def add(self, i: int, cov: dict) -> None:
        held: List[int] = []
        for name, items in (cov.get("states") or {}).items():
            for it in items:
                k = ((name, it["key"], it["reason"]) if name == DECLINED
                     else (name, it))
                if k not in self.index:
                    self.index[k] = len(self.keys)
                    self.keys.append(list(k))
                held.append(self.index[k])
        self.points[str(i)] = [cov["measured"], cov["total"], held]

    def to_json(self) -> dict:
        return {"keys": self.keys, "points": self.points}


def frontier_coverage(points: List[list], keys: List[list]) -> dict:
    """How much of one context's SOL frontier Kernel Bench has measured,
    from each frontier point's folded coverage (`_PointCoverage`:
    `[measured, total, key indices]` into `keys`): how many points, how
    many are fully measured (those draw a measured point), the best point's
    `{measured, total}` and the median share of rows measured, and per
    state of a row not measured (`auto_sweep.STATES`) how many distinct
    keys are in it and the `COVERAGE_KEYS_LISTED` most needed, each with
    how many frontier points need it — what to measure next. A key names
    one physical shape, so a model's frontier misses hundreds (DeepSeek-V3
    on H100: 142–279 per context): the list is cut, the count is not."""
    shares = [m / t for m, t, _ in points if t]
    best = max(points, key=lambda p: p[0] / p[1] if p[1] else 0.0, default=None)
    needed: Dict[str, Dict[tuple, int]] = {}
    for _m, _t, held in points:
        for k in held:
            state, *rest = keys[k]
            by_key = needed.setdefault(state, {})
            by_key[tuple(rest)] = by_key.get(tuple(rest), 0) + 1
    states: Dict[str, dict] = {}
    for name in STATES:
        if name not in needed:
            continue
        rows = sorted(needed[name].items(), key=lambda kv: (-kv[1], kv[0]))
        states[name] = {"keys": len(rows), "top": [
            {"key": k[0], **({"reason": k[1]} if name == DECLINED else {}), "points": n}
            for k, n in rows[:COVERAGE_KEYS_LISTED]]}
    return {
        "points": len(points),
        "complete": sum(1 for m, t, _ in points if t and m == t),
        "best": None if best is None else {"measured": best[0], "total": best[1]},
        "median": statistics.median(shares) if shares else None,
        **({"states": states} if states else {}),
    }


def _draw(core: dict, view: FrontierView, state: str,
          rates: Optional[dict]) -> dict:
    """A model's entry drawn under `view`: every theory point at the view's
    floors and draft (`_at_view`), the SOL frontier through them, and the
    measured one through the points' measured rates (`_measured_rates`,
    None when the track reads none), and how much of each context's SOL
    frontier Kernel Bench covers (`measured_track.coverage`,
    `frontier_coverage`; only when the track reads). Under a draft, the
    layouts whose step at that K failed, with why (`draft_errors`; none
    under plain decode). The entry is not mutated."""
    out = _public_entry(core)
    out["draft_errors"] = [e for e in core.get("_draft_errors") or []
                           if view.draft_tokens
                           and e["draft_tokens"] == int(view.draft_tokens)]
    out["measured_track"] = {"state": state}
    records: list[CloudRecord] = []
    # The entry index each drawn point was read off, for its coverage.
    for i, point in enumerate(core.get("_theory") or []):
        coordinates = point_coordinates(
            mode=point.get("mode", "custom"), num_gpus=point["num_gpus"],
            gpu_sku=view.gpu_sku, context=int(point["seq"]),
            batch_key=point.get("batch_key", "max"),
            draft_tokens=int(view.draft_tokens),
            acceptance=float(view.acceptance_rate), ep_load_factor=1.0)
        item = _at_view(point, view)
        if item is None:
            records.append(CloudRecord.unanswered(
                coordinates, "layout",
                "this layout has no step at the view's draft width",
                token=i))
            continue
        dec = item["sol"]["phases"]["decode"]
        meas = (rates or {}).get("rates", {}).get(str(i))
        if meas is not None:
            dec = {**dec, "throughput": {**dec["throughput"], "measured": meas}}
        records.append(CloudRecord(
            coordinates=coordinates,
            estimate={"phases": {"decode": dec}},
            provenance={"layout": item["label"]}, token=i))

    cloud = produce_cloud(records)

    def fronts(track: str) -> Dict[str, List[dict]]:
        """The one `frontier_view`, one line per context, as the Gallery
        always drew them (the front across every layout and batch)."""
        drawn = frontier_view(cloud.points, dims=POINT_DIMS, fix={},
                              group_by=["context"], track=track)
        answer = {str(seq): [] for seq in GALLERY_SEQ_LENS}
        for series in drawn["series"]:
            answer[str(series["group"]["context"])] = series["frontier"]
        return answer

    out["frontier"] = fronts("sol")
    out["frontier_measured"] = fronts("measured")
    if state == "reads":
        cov = rates["coverage"]
        out["measured_track"]["coverage"] = {
            sk: frontier_coverage([_point_coverage(cov, cloud.tokens[id(p)],
                                                   core)
                                   for p in pts], cov["keys"])
            for sk, pts in out["frontier"].items()}
    return out


class MeasurementFoldMismatch(ValueError):
    """A kept SOL entry and the current measurement fold disagree on points.

    Measurement overlays are live state, while a pre-materialized SOL entry
    can outlive one lake snapshot. The SOL drawing remains valid; the
    measured overlay must be dropped rather than failing the whole page.
    """


def _point_coverage(cov: dict, i: int, core: dict) -> list:
    """Frontier point `i`'s folded coverage. A point the fold had no
    measure for would carry the walk's unread coverage — no row in any
    state — which is not a coverage of zero: refused rather than counted
    as one."""
    got = cov["points"].get(str(i))
    if got is None:
        t = core["_theory"][i]
        raise MeasurementFoldMismatch(
            f"frontier point {t['label']} at {t['seq']} "
            "has no decode measure to fold")
    return got


def _rates_cover(core: dict, rates: dict) -> bool:
    """Whether a kept rates entry has a fold for every point in this entry."""
    try:
        coverage = rates.get("coverage") or {}
        for i in range(len(core.get("_theory") or [])):
            _point_coverage(coverage, i, core)
    except (MeasurementFoldMismatch, KeyError, TypeError):
        return False
    return True


class _Stored(NamedTuple):
    """Where one model's frontier is kept under one walk (`FrontierView.
    walk_parts`) — its entry's two parts (`_entry_parts`) in `_FRONTIER`,
    and what views read off them in `_FRONTIER_VIEWS` — or nowhere, for a
    model with no `result_version`."""
    model_id: str
    version: Optional[str]

    def _key(self, kind: str, parts: List[Any], scope: Any,
             schema: str = _FRONTIER_SCHEMA) -> str:
        return of_render(kind, self.version, [schema, self.model_id,
                                              *parts], scope).key

    def serving_key(self, view: FrontierView) -> ServingWalkKey:
        """The key of this model's serving-walk bundle for the view's card's
        walk class (`ServingWalkKey.of_card` — the one place a card enters
        a projection key)."""
        return ServingWalkKey.of_card(
            self.version, model=self.model_id,
            card=Card.of(view.gpu_sku),
            facts=ServingWalkFacts.producer())

    def walk_identity(self, view: FrontierView):
        """The shared identity of this model's serving-walk bundle for the
        walk class of the view's card (`ServingWalkKey`): one bundle per
        class, never another class's."""
        return self.serving_key(view).identity

    def keep_walk(self, view: FrontierView, walk: dict,
                  traces: Optional[List[Any]] = None) -> None:
        """Keep a model's class walk, when the view's is the kept one; each
        layout's record with the trace of the job that walked it (`traces`,
        in layout order)."""
        if not (self.version and view.keeps_walk()):
            return False
        walk_key = self.serving_key(view)
        names = [StepLawKey.of_layout(walk_key, item["parallel"]).record_name
                 for item in walk["layouts"]]
        return sharedSolPool.write(
            walk_key.identity,
            zip(names, walk["layouts"], strict=True),
            traces=(dict(zip(names, traces, strict=True))
                    if traces is not None else None),
            metadata={
                "cards": list(walk["cards"]),
                "layoutsByCard": walk["layoutsByCard"],
                "contexts": list(GALLERY_SEQ_LENS),
                "facts": ServingWalkFacts.producer().to_json(),
                "recordSchema": walkRecordSchema,
            })

    def shared_walk(self, view: FrontierView) -> Optional[dict]:
        """The class walk from the local SOL pool, or None on any miss.

        The pool's manifest carries the non-record walk plan. Records are
        read one layout at a time; only the model's walk being drawn enters
        objects, not every model in the pool.
        """
        walk_key = self.serving_key(view)
        bundle_key = walk_key.identity.key
        manifest = sharedSolPool.manifest(bundle_key)
        if not walkManifestIsValid(manifest):
            return None
        metadata = manifest.get("metadata") or {}
        # A canonical bundle holds every public layout (Gallery's frontier ∪
        # Scaling's ladder). The Gallery draws only its own subset of them.
        galleryLayouts = metadata.get("galleryLayoutsByCard",
                                      metadata.get("layoutsByCard", {}))
        if view.gpu_sku not in galleryLayouts:
            return None
        wantedNames = {StepLawKey.of_layout(walk_key, layout).record_name
                       for layout in galleryLayouts[view.gpu_sku]}
        recordIds = sharedSolPool.recordIds(bundle_key)
        if recordIds is None:
            return None
        layouts = []
        for record_name in recordIds:
            if record_name not in wantedNames:
                continue
            item = sharedSolPool.record(bundle_key, record_name)
            if item is None:
                return None
            layouts.append(item)
        return {**metadata, "layouts": layouts}

    def entry_key(self, view: FrontierView) -> str:
        """The entry's: under the constant scope `none` (SOL: no
        measurement moves it), in the form the view reads
        (`FrontierView.entry_parts`) — not the K or acceptance, and for a law
        entry not the floors (each point carries its step as a function of
        them)."""
        return self._key("gallery-frontier", view.entry_parts(),
                         NO_MEASUREMENTS)

    def measures_key(self, view: FrontierView) -> str:
        """The measures part, in its entry's point order: keyed as the entry
        is, so a part is only ever read beside the entry it was cut from."""
        return self._key("gallery-frontier-measures", view.entry_parts(),
                         NO_MEASUREMENTS)

    def rates_key(self, view: FrontierView, scope: Any) -> str:
        """The measured rates': the selection that picks each key's row
        (`_GALLERY_SELECTION`) is the fold's, so it keys them."""
        return self._key("gallery-frontier-measured",
                         [*view.walk_parts(), _GALLERY_SELECTION], scope,
                         _DRAWING_SCHEMA)

    def view_key(self, view: FrontierView, scope: Any, state: str) -> str:
        """The drawing's. A track the lake could not give ("unavailable")
        is the SOL drawing (`_labelled`), kept once under scope `none`
        whatever the outage: a lake that flaps between answering and not
        reads both drawings back, and draws neither again off the entry."""
        if state == "unavailable":
            scope, state = NO_MEASUREMENTS, "none"
        return self._key("gallery-frontier-view",
                         [*view.key_parts(), state, _GALLERY_SELECTION],
                         scope, _DRAWING_SCHEMA)

    def kept(self, view: FrontierView, track: Track) -> Optional[dict]:
        """The drawing kept for `view` under `track`, or None."""
        hit = _FRONTIER_VIEWS.get(self.view_key(view, *track))
        return None if hit is None else _labelled(hit, track[1])

    def drawn(self, view: FrontierView, core: dict, measures: Any,
              track: Optional[Track] = None) -> Optional[dict]:
        """`core` drawn under `view` (`_draw`) and `track` (the request's
        lake answer; read here when not given), its measured rates read off
        the cache or folded off `measures()` (the entry's measures part, or
        None when it is not kept) and kept; None when the view reads
        measures that are not kept. The drawing is kept under the view.

        A fold the lake fails mid-request — its version answered (the view
        reads a measured track) but the rows' lookup did not — draws the
        SOL drawing (`measured_track.state` "unavailable") rather than
        failing the whole frontier. It is kept where every unavailable
        track's is (`view_key`), never under the measured track's: a retry
        the lake answers folds the rates."""
        scope, state = track if track is not None else _view_scope(view)
        rates = None
        if state == "reads":
            key = self.rates_key(view, scope) if self.version else None
            rates = _FRONTIER_VIEWS.get(key) if key else None
            if rates is not None and not _rates_cover(core, rates):
                rates = None
            if rates is None:
                part = measures()
                if part is None:
                    return None
                try:
                    rates = _measured_rates(core, part, scope)
                    if rates is None or not _rates_cover(core, rates):
                        raise MeasurementFoldMismatch(
                            "the fold did not cover every frontier point")
                except (MeasurementsUnavailable, MeasurementFoldMismatch) as e:
                    logger.warning("gallery frontier: %s: measured track "
                                   "dropped: %s", self.model_id, e)
                    scope, state = NO_MEASUREMENTS, "unavailable"
                else:
                    if key:
                        _FRONTIER_VIEWS.put(key, rates)
        out = _draw(core, view, "none" if state == "unavailable" else state,
                    rates)
        if self.version:
            _FRONTIER_VIEWS.put(self.view_key(view, scope, state), out)
        return _labelled(out, state)

    def store(self, view: FrontierView, entry: dict) -> Tuple[dict, dict]:
        """Keep a built entry's two parts, and return them."""
        core, measures = _entry_parts(entry)
        if self.version:
            # Shareable (no user calibration rows) unless the entry is an
            # error: one box's transient failure must not fan out globally.
            shared = not entry.get("error")
            for key, part in ((self.entry_key(view), core),
                              (self.measures_key(view), measures)):
                _FRONTIER.put(key, part, remote_ok=True,
                              remote_if=lambda _part: shared)
        return core, measures


def _read_entry(entry: dict, view: FrontierView) -> dict:
    """A built entry, in hand, drawn under `view` (`_draw`), its measured
    rates folded off its own points' measures. Nothing is kept."""
    core, measures = _entry_parts(entry)
    return _Stored(entry.get("model_id", ""), None).drawn(
        view, core, lambda: measures)


def frontier_request_key(view: FrontierView) -> str:
    """The identity of one frontier request: concurrent requests for one
    view on the same spec generation join one build
    (`offload.within_deadline`)."""
    return of_render("gallery-frontier-request", spec_generation(),
                     view.key_parts(), NO_MEASUREMENTS).key


def _frontier_read(model_id: str, view: FrontierView,
                   track: Optional[Track] = None) -> dict:
    """One model under `view` and `track` (the request's lake answer; read
    here when not given), as far as the caches answer it: `{"drawn":
    …}` — the view's drawing kept, else drawn off the kept entry (its
    measures part read only when the view's measured rates are not kept) —
    else priced off the model's class walk kept for the view's card
    (`_restamp`) — or `{"walk": plan}`: the cards and layouts a class walk
    walks (`FrontierView.walk_cards`, `class_frontier_layouts`, and per
    card the layouts its frontier prices). A whole entry is never read for
    a view already drawn, and its measures never for floors, K or
    acceptance.
    Runs where the compute runs: what crosses back is the drawing."""
    _config, resolved, version = _resolve_one(model_id)
    stored = _Stored(model_id, version)
    if resolved is None:
        return {"drawn": _draw({"model_id": model_id, "layouts": [],
                                "error": "no spec resolves for this model"},
                               view, "none", None)}
    if version is not None:
        track = track if track is not None else _view_scope(view)
        hit = stored.kept(view, track)
        if hit is not None:
            return {"drawn": hit}
        core = _FRONTIER.get(stored.entry_key(view), remote_ok=True)
        if core is not None:
            drawn = stored.drawn(view, core, lambda: _FRONTIER.get(
                stored.measures_key(view), remote_ok=True), track)
            # None: the measures part is not kept — the entry is priced
            # again, both parts with it.
            if drawn is not None:
                return {"drawn": drawn}
        if view.keeps_walk():
            walk = stored.shared_walk(view)
            if walk is not None and view.gpu_sku in walk["layoutsByCard"]:
                return {"drawn": _keep_and_draw(
                    stored, view, _restamp(model_id, walk, view.gpu_sku, view),
                    track)}
    try:
        plan = _walk_plan(resolved, view)
    except Exception as e:
        return {"drawn": _draw({"model_id": model_id, "layouts": [],
                                "error": str(e)}, view, "none", None)}
    return {"walk": plan}


def _scalingLayouts(caps: Any, cards: List[str]) -> List[ParallelConfig]:
    """Every layout Scaling's public ladder prices, each once.

    The Gallery's frontier ends at 64 GPUs; the Scaling ladder continues to
    the engine bound in its four canonical modes. Those higher rungs belong to
    a Scaling warm's bundle. They are not produced for every Gallery model:
    only the panel's featured cohort reaches past the frontier's 64-GPU bound.
    """
    out = list(class_frontier_layouts(caps, cards))
    for mode in ALL_MODES:
        out += [cfg for _x, cfg in parallel_configs_for_mode(
            mode, caps, x_max=SWEEP_MAX_GPUS) if cfg not in out]
    return out


def _walk_plan(resolved: Any, view: FrontierView, *,
               scaling: bool = False) -> dict:
    """What a model's canonical walk for `view` covers: its cards — the view
    card's walk class (`FrontierView.walk_cards`, the walk card first): a
    law's rows are one class's, so a bundle never spans two —, per card the
    layouts a Gallery frontier on it renders (`frontier_layouts`), and every
    layout any producer reads — with `scaling`, the public ladder's extra
    layouts (`_scalingLayouts`)."""
    cards = view.walk_cards()
    caps = sweep_caps({}, resolved=resolved)
    frontierByCard = {
        sku: [cfg.to_dict() for cfg in frontier_layouts(caps, sku)]
        for sku in cards
    }
    layouts = (_scalingLayouts(caps, cards) if scaling
               else class_frontier_layouts(caps, cards))
    return {"cards": cards,
            "layoutsByCard": {sku: [cfg.to_dict() for cfg in layouts]
                              for sku in cards},
            "galleryLayoutsByCard": frontierByCard,
            "layouts": [cfg.to_dict() for cfg in layouts]}


def _keep_and_draw(stored: _Stored, view: FrontierView, entry: dict,
                   track: Optional[Track]) -> dict:
    """A model's entry on the view's card kept (`_Stored.store`) and drawn
    under `view` and `track` (the request's one lake answer)."""
    core, measures = stored.store(view, entry)
    return stored.drawn(view, core, lambda: measures, track)


def _finish_frontier(model_id: str, view: FrontierView, plan: dict,
                     results: List[dict],
                     track: Optional[Track] = None,
                     traces: Optional[List[Any]] = None) -> dict:
    """A model's layout laws (`_layout_law`, one per layout of `plan`) as its
    class bundle, kept (`_Stored.keep_walk`), projected on the view's card
    (`_restamp`), kept and drawn under `view` and `track`. One job, so the
    entry is built, kept and drawn in one process and only the drawing
    crosses back.

    The walk is kept before the card's entry is priced, and per model: a
    build no request waits on any more is cancelled (`offload.UNWATCHED_S`)
    and what it had kept stays kept, so the models it got through are a
    walk every card of the class reads — including the card the user
    switched to, which is the request that cancelled this one."""
    walk = {**plan, "layouts": [{"parallel": p, "walk": r} for p, r in
                                zip(plan["layouts"], results, strict=True)]}
    stored = _Stored(model_id, _resolve_one(model_id)[2])
    stored.keep_walk(view, walk, traces)
    return _keep_and_draw(
        stored, view, _restamp(model_id, walk, view.gpu_sku, view), track)


async def keepSharedWalk(model_id: str, gpu_sku: str) -> bool:
    """`_keptSharedWalk` as one recorded computation
    (`code_trace.provider.recording`)."""
    with recording():
        return await _keptSharedWalk(model_id, gpu_sku)


async def _keptSharedWalk(model_id: str, gpu_sku: str) -> bool:
    """Ensure one card class's serving-walk records exist in the shared pool.

    A page need not be rendered for a class to contribute canonical facts:
    the Scaling series includes RTX-4090 while the Gallery picker does not.
    Existing complete bundles are left alone; immutable records are never
    rewritten or migrated.
    """
    _config, resolved, version = _resolve_one(model_id)
    if resolved is None or version is None:
        return False
    view = FrontierView(gpu_sku)
    stored = _Stored(model_id, version)
    walk_key = stored.serving_key(view)
    bundle_key = walk_key.identity.key
    recordIds = sharedSolPool.recordIds(bundle_key)
    plan = _walk_plan(resolved, view, scaling=True)
    stepKeys = [StepLawKey.of_layout(walk_key, layout)
                for layout in plan["layouts"]]
    wanted = {key.record_name for key in stepKeys}
    existing = {}
    traces: Dict[str, Any] = {}
    if recordIds is not None:
        if walkManifestIsValid(sharedSolPool.manifest(bundle_key)):
            for name in wanted:
                # A record whose code changed reads as a miss
                # (`SolPool.record`): only it is walked again.
                record = sharedSolPool.record(bundle_key, name)
                if (record is not None
                    and record.get("walk", {}).get("schema") == walkRecordSchema):
                    existing[name] = record
                    traces[name] = sharedSolPool.recordTrace(bundle_key, name)
        published = {
            StepLawKey.of_layout(walk_key, layout).record_name
            for layouts in ((sharedSolPool.manifest(bundle_key).get(
                "metadata") or {}).get("layoutsByCard") or {}).values()
            for layout in layouts
        }
        if len(existing) == len(wanted) and wanted <= published:
            return True
    layouts = [ParallelConfig.from_dict(layout) for layout in plan["layouts"]
               if StepLawKey.of_layout(walk_key, layout).record_name
               not in existing]
    results = await map_off_loop(
        functools.partial(run_recorded, None, _layout_law),
        [(model_id, plan["cards"], float(view.gpu_memory_fraction),
          _GALLERY_PRECISION.canonical(), layout)
         for layout in layouts])
    computed = {}
    for layout, (result, trace) in zip(layouts, results, strict=True):
        name = StepLawKey.of_layout(walk_key, layout).record_name
        computed[name] = {"parallel": layout.to_dict(), "walk": result}
        traces[name] = trace
    return sharedSolPool.write(
        walk_key.identity,
        ((name, existing.get(name) or computed[name])
         for name in (key.record_name for key in stepKeys)),
        traces=traces,
        metadata={
            "cards": list(plan["cards"]),
            "layoutsByCard": plan["layoutsByCard"],
            "contexts": list(GALLERY_SEQ_LENS),
            "facts": ServingWalkFacts.producer().to_json(),
            "recordSchema": walkRecordSchema,
        })


async def frontier_model(model_id: str, view: FrontierView,
                         track: Optional[Track] = None, *,
                         compute: Optional[PrioritySlots] = None) -> dict:
    """`_frontier_model` as one recorded computation
    (`code_trace.provider.recording`): the read, the layout jobs and the fold
    (each a worker job whose trace comes back to it), so every part it
    keeps is stored with the code that made it."""
    with recording():
        return await _frontier_model(model_id, view, track, compute=compute)


async def _frontier_model(model_id: str, view: FrontierView,
                          track: Optional[Track] = None, *,
                          compute: Optional[PrioritySlots] = None) -> dict:
    """One model's frontier under `view` and `track`: read off the caches —
    the view's drawing, its entry, or its class walk priced on its card —
    or each layout of the class walk its own compute job and the fold
    (`_finish_frontier`). Kernel Bench rows fold on after the SOL entry is
    in hand, so the lake's answer reaches the pricing and never the walk.

    The read is one quick job and queues for nothing, so a model already
    kept lands at once whatever is computing. Only a model that computes
    holds one of `compute` (the frontier's model slots), and its steps queue
    at its urgency (`offload.urgency`): the model a client picked, or has on
    screen, before the rest."""
    got = await run_off_loop(_frontier_read, model_id, view, track)
    readsMeasurements = track is not None and track[1] == "reads"
    if "drawn" in got and not readsMeasurements:
        return got["drawn"]
    if compute is None:
        return await _compute_frontier(model_id, view, track, got)
    async with compute.hold(urgency(model_id)):
        return await _compute_frontier(model_id, view, track, got)


async def _compute_frontier(model_id: str, view: FrontierView,
                            track: Optional[Track], got: dict) -> dict:
    """The frontier of a model its read did not answer: each layout a job,
    then the fold."""
    readsMeasurements = track is not None and track[1] == "reads"
    if readsMeasurements:
        plan = _walk_plan(_resolve_one(model_id)[1], view)
    else:
        plan = got["walk"]
    # Each layout is its own recorded job, so the record it becomes names
    # only the code that walked it (`SolPool.write`).
    recorded = await map_off_loop(
        functools.partial(run_recorded, None,
                          _layout_walk if readsMeasurements else _layout_law),
        [(model_id, plan["cards"], float(view.gpu_memory_fraction),
          _GALLERY_PRECISION.canonical(), p)
                      for p in plan["layouts"]], name=model_id)
    results = [value for value, _trace in recorded]
    traces = [trace for _value, trace in recorded]
    return await run_ranked(_finish_frontier, model_id, view, plan, results,
                            track, traces, name=model_id)


async def build_frontier(view: FrontierView) -> dict:
    """Every model's decode frontier under `view`, `_model_concurrency`
    models computing at a time, reporting how many have finished and
    publishing each as it lands (`report_partial`), so a page draws it
    without waiting for the slowest. Every model is read first, at once: a
    kept one lands without waiting behind one that computes. Which model
    computes next is the client's order (`offload.urgency`: picked, then on
    screen). The lake's version is read
    once (`_view_scope`) and every model is drawn under that one answer: one
    response reads one lake, and an unanswering lake costs one timeout
    rather than one per model. A spec refined meanwhile serves nothing
    (`ensure_spec_generation`)."""
    generation = spec_generation()
    model_ids = sorted(iter_snapshot_ids())
    head = _frontier_head(view)
    report_partial(head=head)
    track = await run_off_loop(_view_scope, view)
    slots = PrioritySlots(_model_concurrency())
    done = 0
    report_progress(done, len(model_ids))

    async def one(mid: str) -> dict:
        nonlocal done
        entry = await frontier_model(mid, view, track, compute=slots)
        done += 1
        print(f"frontier {view.gpu_sku}: {done}/{len(model_ids)} · {mid}",
              flush=True)
        report_progress(done, len(model_ids))
        report_partial(entry)
        return entry

    models = list(await asyncio.gather(*[one(m) for m in model_ids]))
    ensure_spec_generation(generation)
    return {
        **head,
        "models": models,
        "errors_count": sum(1 for m in models if m.get("error")),
    }


def _frontier_head(view: FrontierView) -> dict:
    """What every model of a frontier response is drawn under: the view's
    config and the contexts. The same head frames a partial answer
    (`report_partial`) and the whole one."""
    return {
        "config": {
            "gpu_sku": view.gpu_sku,
            "comm_latency_us": view.comm_latency_us,
            "launch_floor_us": view.launch_floor_us,
            "draft_tokens": view.draft_tokens,
            "acceptance_rate": (view.acceptance_rate if view.draft_tokens
                                else None),
            "max_gpus": FRONTIER_MAX_GPUS,
            "output_tokens": GALLERY_OUTPUT_TOKENS,
            "precision": _GALLERY_PRECISION.to_json(),
        },
        "seq_lens": list(GALLERY_SEQ_LENS),
    }
