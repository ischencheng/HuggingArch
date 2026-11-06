"""Canonical SOL laws: one walked formula, every serving point substituted.

This layer is deliberately smaller than an estimate: it carries a phase's
SOL account and time as SymPy laws over workload and card symbols, with the
common subexpressions named once. Per-op rows, module tables, shapes and
display formulas belong to an explicit breakdown request and are not stored
here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from backend.arch.law_text import read_walk_law
from backend.measure.auto_sweep import _ExactPrinter, exact_div

from .capacity import (
    CapacityKnobs,
    capacity_ceilings,
    _restamp_gpu,
    _capacity_pending,
    _capacity_preparation,
    point_served,
    without_step,
)
from .card import Card
from .comm_cost import CollectiveRung, collective_ladder
from .gpu_specs import peak_flops_for_dtype, peak_hbm_bw
from .phases import Speculation, phase_binding, phase_throughput, step_tokens
from .parallel_config import ParallelConfig
from .prefill_geometry import PrefillChunkSchedule, PrefillEndpointGeometry
from .row_roofline import StepSum, law_symbol, rung_symbols
from .sol_data import walkRecordSchema as servingLawSchema
from .serving_view import servingCapacityView
from backend.code_trace.memo import memoize


_lawOutputs = {
    "t_sol_s": "t_op",
}

# One draft law serves every speculative K: the request says which draft stack
# exists, while this symbol says how many forwards the point runs.
draftForwardsLaw = sp.Symbol("s_verify") - 1


def _csePreservingExtrema(values: Sequence[sp.Expr]):
    """CSE without asking SymPy to compare the arms of a `Max`/`Min`.

    A roofline extremum is written as-written: its arms can hold `Cases`, so
    SymPy cannot order them until every axis is numeric. CSE rebuilds function
    applications by calling their constructors again, and ``Max(...)``'s
    constructor tries that comparison. An inert application therefore stands in
    for the extremum's *constructor* while CSE runs; its arguments stay visible,
    so the shape and byte subterms repeated by many extrema are still named
    once. The restored graph still prints the law language's as-written
    ``Max`` / ``Min``.
    """
    extrema: dict[sp.Expr, sp.Expr] = {}

    def mask(expr: sp.Expr) -> sp.Expr:
        if not isinstance(expr, sp.Basic) or not expr.args:
            return expr
        args = tuple(mask(arg) for arg in expr.args)
        if isinstance(expr, sp.Max):
            inert = _InertMaximum(*args)
        elif isinstance(expr, sp.Min):
            inert = _InertMinimum(*args)
        else:
            if not args:
                return expr
            with sp.evaluate(False):
                return expr.func(*args)
        extrema[inert] = expr
        return inert

    masked = [mask(value) for value in values]
    definitions, reduced = sp.cse(
        masked, symbols=sp.numbered_symbols("u"), order="canonical", list=True)

    def restore(expr: sp.Expr) -> sp.Expr:
        if expr in extrema:
            return extrema[expr]
        if isinstance(expr, _InertMaximum):
            args = tuple(restore(arg) for arg in expr.args)
            with sp.evaluate(False):
                return sp.Max(*args)
        if isinstance(expr, _InertMinimum):
            args = tuple(restore(arg) for arg in expr.args)
            with sp.evaluate(False):
                return sp.Min(*args)
        if not isinstance(expr, sp.Basic) or not expr.args:
            return expr
        args = tuple(restore(arg) for arg in expr.args)
        if all(new is old for new, old in zip(args, expr.args, strict=True)):
            return expr
        return expr.func(*args)

    return ([(symbol, restore(value)) for symbol, value in definitions],
            [restore(value) for value in reduced])


class _InertMaximum(sp.Function):
    """`Max` during CSE: an application, never a comparison."""

    @classmethod
    def eval(cls, *_args):
        return None


class _InertMinimum(sp.Function):
    """`Min` during CSE: an application, never a comparison."""

    @classmethod
    def eval(cls, *_args):
        return None


def _withoutITE(expr: sp.Expr) -> sp.Expr:
    """Rewrite SymPy's printed ITE as the law language's Piecewise."""
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    old_args = expr.args
    args = [_withoutITE(arg) for arg in old_args]
    if expr.func == sp.ITE:
        return sp.Piecewise((args[1], args[0]), (args[2], sp.true))
    if any(new is not old for new, old in zip(args, old_args, strict=True)):
        return expr.func(*args)
    return expr


def _replaceKeepingShape(expr: sp.Expr, rule: Mapping[sp.Expr, sp.Expr]):
    """Substitute without rebuilding an unevaluated parent eagerly.

    A graph is as-written: `Max` may hold `Cases` that are not orderable until
    a point is numeric. Plain `xreplace` rebuilds changed parents with global
    evaluation on; this walks the tree and constructs changed parents under
    `evaluate(False)`.
    """
    if expr in rule:
        return rule[expr]
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    args = tuple(_replaceKeepingShape(arg, rule) for arg in expr.args)
    if all(new is old for new, old in zip(args, expr.args, strict=True)):
        return expr
    with sp.evaluate(False):
        return expr.func(*args)


@memoize(maxsize=32768)
def _read(text: str, local: frozenset[str]) -> sp.Expr:
    def symbol(name: str) -> sp.Symbol:
        if name in local:
            return sp.Symbol(name)
        return law_symbol(name)

    return read_walk_law(text, symbol=symbol)


@memoize(maxsize=1024)
def _compiledGraph(inputs: tuple[str, ...],
                   definitions: tuple[tuple[str, str], ...],
                   outputs: tuple[tuple[str, str], ...]):
    """One CSE graph as one exact program, compiled once.

    Each definition becomes an assignment to the variable every later
    expression reads. A point no longer interprets the graph node by node;
    it substitutes into the same immutable program.
    """
    localNames: set[str] = set()
    byDefinition: dict[sp.Expr, sp.Expr] = {}
    exprs: list[sp.Expr] = []
    internalNames: set[str] = set()
    for index, (name, text) in enumerate(definitions):
        expr = _replaceKeepingShape(
            _read(text, frozenset(localNames)), byDefinition)
        variable = sp.Symbol(f"_law{index}")
        byDefinition[sp.Symbol(name)] = variable
        internalNames.add(variable.name)
        localNames.add(name)
        exprs.append(expr)
    for _name, text in outputs:
        exprs.append(_replaceKeepingShape(
            _read(text, frozenset(localNames)), byDefinition))
    by_name = {symbol.name: symbol for expr in exprs
               for symbol in expr.free_symbols}
    stray = sorted(set(by_name) - set(inputs) - internalNames)
    if stray:
        raise ValueError(f"laws read {stray}, which the point does not bind "
                         f"(it binds {sorted(inputs)})")
    return sp.lambdify(
        [by_name.get(name, sp.Symbol(name)) for name in inputs],
        list(exprs),
        modules=[{"math": math, "Fraction": Fraction,
                  "_law_div": exact_div}],
        printer=_ExactPrinter,
        cse=_eachLawInOrder)


def _eachLawInOrder(exprs: Sequence[Any]):
    """Assign every graph expression in order, even an atom or a repeat.

    A CSE graph's later expressions read the variables this gives the earlier
    ones; unlike a list of independent laws, an expression cannot be deduped
    away because its variable is itself a fact of the program.
    """
    assigned = []
    out = []
    for index, expr in enumerate(exprs):
        variable = sp.Symbol(f"_law{index}")
        assigned.append((variable, expr))
        out.append(variable)
    return assigned, out


@dataclass(frozen=True)
class LawGraph:
    """A CSE graph: external inputs, local definitions, named outputs."""

    inputs: tuple[str, ...]
    definitions: tuple[tuple[str, str], ...]
    outputs: dict[str, str]

    @classmethod
    def of(cls, exprs: Mapping[str, sp.Expr]) -> "LawGraph":
        values = [_withoutITE(exprs[name]) for name in _lawOutputs]
        replacements, reduced = _csePreservingExtrema(values)
        outputs = {
            name: sp.sstr(value)
            for name, value in zip(_lawOutputs, reduced, strict=True)
        }
        inputs = tuple(sorted(
            {symbol.name for expr in values for symbol in expr.free_symbols}))
        definitions = tuple((symbol.name, sp.sstr(expr))
                            for symbol, expr in replacements)
        return cls(inputs, definitions, outputs)

    def to_json(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs),
                "definitions": [[name, text] for name, text in self.definitions],
                "outputs": self.outputs,
                }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LawGraph":
        if "program" in value:
            raise ValueError("a law graph carries law text only")
        return cls(tuple(value["inputs"]),
                   tuple((name, text) for name, text in value["definitions"]),
                   dict(value["outputs"]))

    def evaluate(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        program = _compiledGraph(
            self.inputs, self.definitions, tuple(self.outputs.items()))
        values = program(*(binding[name] for name in self.inputs))
        firstOutput = len(self.definitions)
        if len(values) != firstOutput + len(self.outputs):
            raise ValueError("a law program does not return its graph outputs")
        return dict(zip(
            self.outputs,
            values[firstOutput:firstOutput + len(self.outputs)],
            strict=True))


def _collectives(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.get("is_comm") or int(row.get("comm_world") or 1) <= 1:
            continue
        signature = json.dumps({
            "op": row["comm_op"], "world": int(row["comm_world"]),
            "count": int(row["count_per_step"]),
            "stride": int(row["comm_base_stride"]),
            "rungs": [dict(rung) for rung in row.get("comm_rungs") or ()],
        }, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            out.append(signature)
    return tuple(out)


def _phase(rows: list[Mapping[str, Any]],
           *, mla_absorbed: bool = False,
           draft_rows: Iterable[Mapping[str, Any]] = (),
           draft_forwards: int = 0) -> dict[str, Any]:
    """One phase's law and every collective its summed terms name.

    Draft rows are terms of this same step (`StepSum.of`): their collectives
    enter the phase law at their own count — commonly 3, while the backbone
    rows name 1, 8 or 39 — so the sidecar must collect them here too. The law
    and its binding signatures are one fact of `_phase`, never of a reader.
    """
    draft_rows = tuple(draft_rows)
    step = StepSum.of(rows, draft_rows=draft_rows,
                      draft_forwards=draft_forwards)
    graph = LawGraph.of({
        wire: step.law(name) for wire, name in _lawOutputs.items()
    })
    return {"law": graph.to_json(),
            "collectives": list(_collectives((*rows, *draft_rows))),
            "terms": len(rows) + len(draft_rows),
            "mla_absorbed": bool(mla_absorbed)}


def servingLawRecord(environment: Mapping[str, Any], *, gpu_sku: str,
                     anchorContext: int, outputTokens: int,
                     parallel: Any,
                     capacityCards: Sequence[str] = (),
                     capacityContexts: Sequence[int] = (),
                     draft_forwards: Any = None) -> dict[str, Any]:
    """Walk one layout once and keep its SOL laws, no expanded points."""
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=gpu_sku, knobs=knobs)
    if preparation.refusal_reason is not None:
        return {"refused": preparation.refusal_reason}

    def preparationOn(cardSku: str) -> Any:
        return _restamp_gpu(preparation, Card.of(cardSku))

    capacity = []
    for cardSku in capacityCards or (gpu_sku,):
        onCard = preparationOn(cardSku)
        for context in capacityContexts or (anchorContext,):
            ceilings = capacity_ceilings(
                onCard, avg_input_tokens=int(context),
                avg_output_tokens=int(outputTokens),
                prefill_chunk_size=environment.get("prefill_chunk_size"),
                prefill_capacity=True)
            capacity.append({
                "gpu_sku": cardSku, "context": int(context),
                "max_decode_batch": int(ceilings.max_batch),
                "max_prefill_batch": int(ceilings.max_prefill_batch),
                "beyond_max_context": bool(ceilings.beyond_max_context),
                "infeasibility_reason": ceilings.infeasibility_reason,
                **servingCapacityView(onCard, ceilings),
            })
    if all(int(fact["max_decode_batch"] or 0) <= 0
           and int(fact["max_prefill_batch"] or 0) <= 0
           for fact in capacity):
        return {"refused": next(
            fact.get("infeasibility_reason") or "layout admits no request"
            for fact in capacity)}

    pending = _capacity_pending(
        preparation,
        avg_input_tokens=anchorContext,
        avg_output_tokens=outputTokens,
        prefill_batch_override=1,
        decode_batch_override=1,
        breakdown=False,
        fold_prefill_chunks=False,
        prefill_capacity=False,
    )
    prefillWalk = pending.walk_of_rows["per_op_timing"]
    decodeWalk = pending.walk_of_rows["per_op_timing_decode"]
    prefillRows = prefillWalk.analysis.per_op_timing
    decodeRows = decodeWalk.analysis.per_op_timing
    draftRows = (
        (pending.walk_of_rows.get("per_op_timing_draft")
         and pending.walk_of_rows["per_op_timing_draft"].analysis.per_op_timing)
        or ())

    draftForwards = (0 if preparation.draft is None
                     else preparation.draft.forwards)
    if draft_forwards is not None:
        draftForwards = draft_forwards
    return {
        "schema": servingLawSchema,
        # The card the phase laws' rows were walked on: every binding of a
        # card into them asks whether that card walks alike (`cardBinding`).
        "walked_on": gpu_sku,
        "parallel": preparation.parallel.to_dict(),
        "speculative": preparation.draft is not None,
        "gpu_memory_fraction": knobs.gpu_memory_fraction,
        "max_context_tokens": preparation.max_context,
        "parallel_warnings": preparation.wb.get("parallel_warnings", []),
        "phases": {
            "prefill": _phase(
                prefillRows,
                mla_absorbed=bool(prefillWalk.step_eval.mla_absorbed)),
            "decode": _phase(
                decodeRows,
                mla_absorbed=bool(decodeWalk.step_eval.mla_absorbed),
                draft_rows=draftRows,
                             draft_forwards=draftForwards),
        },
        "capacity": capacity,
    }


def servingLawCapacity(environment: Mapping[str, Any], *, gpu_sku: str,
                       parallel: Any,
                       capacityCards: Sequence[str] = (),
                       capacityContexts: Sequence[int] = ()) -> list[Any]:
    """The plain (non-speculative) capacity facts without a phase law.

    A speculative record's decode law already covers K=0 through its
    `s_verify` symbol, but its preparation also loads the draft stack. Capacity
    keeps both answers: this plain one and the drafted one
    (`servingLawRecord.capacity`)."""
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=gpu_sku, knobs=knobs)
    if preparation.refusal_reason is not None:
        def refusedOn(cardSku: str) -> Any:
            return _restamp_gpu(preparation, Card.of(cardSku))

        return [{
            "gpu_sku": cardSku, "context": int(context),
            "max_decode_batch": 0, "max_prefill_batch": 0,
            "beyond_max_context": int(context) > int(
                preparation.max_context or 0),
            "infeasibility_reason": preparation.refusal_reason,
            **servingCapacityView(refusedOn(cardSku), capacity_ceilings(
                refusedOn(cardSku), avg_input_tokens=int(context),
                avg_output_tokens=int(environment["avg_output_tokens"]),
                prefill_chunk_size=environment.get("prefill_chunk_size"))),
        } for cardSku in (capacityCards or (gpu_sku,))
        for context in (capacityContexts or (1024,))]
    # A record is one walk class's: its capacity facts are for the cards
    # the walk may be re-priced on, and `_restamp_gpu` refuses any other
    # (`Card.refuse_unless_one_class`). Another class walks its own record.
    def preparationOn(cardSku: str) -> Any:
        return _restamp_gpu(preparation, Card.of(cardSku))

    out = []
    for cardSku in capacityCards or (gpu_sku,):
        onCard = preparationOn(cardSku)
        for context in capacityContexts or (1024,):
            ceilings = capacity_ceilings(
                onCard, avg_input_tokens=int(context),
                avg_output_tokens=int(environment["avg_output_tokens"]),
                prefill_chunk_size=environment.get("prefill_chunk_size"),
                prefill_capacity=True)
            out.append({
                "gpu_sku": cardSku, "context": int(context),
                "max_decode_batch": int(ceilings.max_batch),
                "max_prefill_batch": int(ceilings.max_prefill_batch),
                "beyond_max_context": bool(ceilings.beyond_max_context),
                "infeasibility_reason": ceilings.infeasibility_reason,
                **servingCapacityView(onCard, ceilings),
            })
    return out


def _capacityAt(record: Mapping[str, Any], card: Card, context: int):
    for item in record.get("capacity") or ():
        if item["gpu_sku"] == card.sku and item["context"] == context:
            return item
    raise KeyError(
        f"the serving law has no capacity fact for {card.sku} at {context}")


def servingLawPhaseView(record: Mapping[str, Any], name: str, *, batch: int,
                        time: float, context: int, outputTokens: int,
                        verifyWidth: int, prefixCached: int, draftTokens: int,
                        acceptanceRate: float, groups: int,
                        num_gpus: int) -> dict[str, Any]:
    phaseBinding = phase_binding(
        phase=name, avg_input_tokens=context,
        avg_output_tokens=outputTokens, s_prefix_cached=prefixCached,
        s_verify=verifyWidth)
    tokensPerStep = float(phaseBinding["tokens_per_step"])
    if name == "decode" and draftTokens:
        effectiveTokens, effectiveFormula = step_tokens(
            "decode", batch, tokensPerStep,
            Speculation(draftTokens, acceptanceRate))
    else:
        effectiveTokens = tokensPerStep * batch
        effectiveFormula = {
            "symbolic": (
                "B × s_query"
                if name == "prefill" or draftTokens else "B × 1"),
            "substituted": f"{batch} × {tokensPerStep:g}",
        }
    phase = {
        "phase": name,
        "batch": batch,
        "binding": {
            **{key: value for key, value in phaseBinding.items()
               if key != "phase"},
            "effective_tokens_per_step": effectiveTokens,
            "effective_tokens_formula": effectiveFormula,
            "speculative_draft_tokens": (
                draftTokens if name == "decode" else 0),
            "acceptance_rate": (
                acceptanceRate if name == "decode" else 1.0),
            "mla_absorbed": bool(
                record["phases"][name].get("mla_absorbed")),
        },
        "t_sol_s": time,
        "t_measured_s": None,
        "coverage": None,
    }
    phase["throughput"] = phase_throughput(
        phase, unique_tokens_multiplier=groups, num_gpus=num_gpus)
    return phase


def _servingLawPoint(record: Mapping[str, Any], *, facts: Mapping[str, Any],
                     cardValues: Mapping[str, float],
                     graphs: Mapping[str, LawGraph], context: int,
                     outputTokens: int, decodeBatch: int | None,
                     prefillBatch: int | None, prefixCached: int,
                     verifyWidth: int,
                     draftTokens: int, acceptanceRate: float,
                     prefillTimes: dict[tuple[int, int, int, int], float]) -> dict:
    decodeB = (int(decodeBatch) if decodeBatch is not None
               else int(facts["max_decode_batch"]))
    prefillB = (int(prefillBatch) if prefillBatch is not None
               else int(facts["max_prefill_batch"]))
    phases = {}
    parallel = ParallelConfig.from_dict(record["parallel"])
    groups = parallel.dp_attention * parallel.dp_replicas
    num_gpus = parallel.cluster_size
    for name, batch in (("prefill", prefillB), ("decode", decodeB)):
        graph = graphs[name]
        point = {
            **cardValues, "B": max(1, batch), "s_input": context,
            "s_output": outputTokens, "s_prefix": int(prefixCached),
            "s_verify": verifyWidth,
            "rho": 1,
        }
        if name == "prefill":
            key = (int(context), int(prefixCached), prefillB,
                   int(verifyWidth))
            if key not in prefillTimes:
                schedule = PrefillChunkSchedule.materialize(
                    PrefillEndpointGeometry(
                        int(context), int(prefixCached), 1, max(1, batch)),
                    prefill_chunk_size=8192)
                t = 0.0
                for chunk in schedule.chunks:
                    repeat = (schedule.endpoint_batch // chunk.launch_batch
                              if len(schedule.chunks) > 1 else 1)
                    chunkTime = float(graph.evaluate({
                        **point, "B": chunk.launch_batch,
                        "s_input": chunk.end, "s_prefix": chunk.start,
                    })["t_sol_s"])
                    t += chunkTime * repeat
                prefillTimes[key] = t
            t = prefillTimes[key]
        else:
            t = float(graph.evaluate(point)["t_sol_s"])
        phases[name] = servingLawPhaseView(
            record, name, batch=batch, time=t, context=context,
            outputTokens=outputTokens, verifyWidth=verifyWidth,
            prefixCached=int(prefixCached),
            draftTokens=draftTokens, acceptanceRate=acceptanceRate,
            groups=groups, num_gpus=num_gpus)
    return {
        "avg_input_tokens": context,
        "avg_output_tokens": outputTokens,
        "max_batch_per_gpu": facts["max_decode_batch"],
        "max_decode_batch_per_gpu": facts["max_decode_batch"],
        "max_prefill_batch_per_gpu": facts["max_prefill_batch"],
        "max_decode_batch_total": facts.get("max_decode_batch_total"),
        "max_prefill_batch_total": facts.get("max_prefill_batch_total"),
        "max_tokens_per_gpu": facts.get("max_tokens_per_gpu"),
        "max_tokens_total": facts.get("max_tokens_total"),
        "num_gpus": num_gpus,
        "cluster_size": parallel.cluster_size,
        "parallel": record["parallel"],
        "gpu_memory_fraction": record.get("gpu_memory_fraction"),
        "beyond_max_context": facts["beyond_max_context"],
        "infeasibility_reason": facts["infeasibility_reason"],
        "max_context_tokens": record.get("max_context_tokens"),
        "parallel_warnings": list(record.get("parallel_warnings") or []),
        "memory": facts.get("memory"),
        "kv_cache": facts.get("kv_cache"),
        "phases": phases,
    }


def servingLawPoint(record: Mapping[str, Any], *, card: Card,
                    context: int, outputTokens: int,
                    decodeBatch: int | None = None,
                    prefillBatch: int | None = 1,
                    prefixCached: int = 0,
                    verifyWidth: int = 1,
                    draftTokens: int = 0,
                    acceptanceRate: float = 1.0,
                    comm_latency_us: float, launch_floor_us: float) -> dict:
    """One served point on `card`, derived from the law and the card's
    capacity facts. `card` must walk as the record's walk card does
    (`cardBinding`, the one place a card enters a law)."""
    draftCapacity = (
        {"capacity": record["draft_capacity"]} if draftTokens
        and "draft_capacity" in record else record)
    facts = _capacityAt(draftCapacity, card, context)
    return _servingLawPoint(
        record, facts=facts,
        cardValues=cardBinding(
            record, card, comm_latency_us=comm_latency_us,
            launch_floor_us=launch_floor_us),
        graphs={name: LawGraph.from_json(record["phases"][name]["law"])
                for name in ("prefill", "decode")},
        context=context, outputTokens=outputTokens, decodeBatch=decodeBatch,
        prefillBatch=prefillBatch, prefixCached=prefixCached,
        verifyWidth=verifyWidth,
        draftTokens=draftTokens, acceptanceRate=acceptanceRate,
        prefillTimes={})


def servingLawCapacityFacts(record: Mapping[str, Any], *, card: Card,
                            contexts: Sequence[int],
                            draftTokens: int = 0) -> dict[int, dict[str, Any]]:
    """The capacity facts a serving projection needs, by context.

    The law is not read here: this is the cheap existence half of a
    projection, asked before any point is priced.
    """
    capacityRecord = (
        {"capacity": record["draft_capacity"]} if draftTokens
        and "draft_capacity" in record else record)
    return {int(context): dict(_capacityAt(capacityRecord, card, int(context)))
            for context in contexts}


def servingLawPoints(record: Mapping[str, Any], *, card: Card,
                     contexts: Sequence[int], decodeBatches: Sequence[int | None],
                     outputTokens: int,
                     prefillBatches: int | Sequence[int | None] = 1,
                     prefixCached: int | Sequence[int] = 0,
                     verifyWidth: int | Sequence[int] = 1,
                     draftTokens: int = 0,
                     acceptanceRate: float = 1.0,
                     comm_latency_us: float, launch_floor_us: float,
                     servedOnly: bool = False
                     ) -> list[list[dict | None]]:
    """A context-by-batch projection of one serving-law record.

    This is not a second walk: it reads the same two graphs as
    ``servingLawPoint``. The grid only changes what is shared — the card
    binding, each context's capacity fact and each context's prefill chunk
    schedule — and binds each context's prefix and verify width. With
    ``servedOnly``, capacity is asked before the law is evaluated; a missing
    capacity fact is an absent point, as a reader of one point already
    treats its exception as absence.
    """
    cardValues = cardBinding(
        record, card, comm_latency_us=comm_latency_us,
        launch_floor_us=launch_floor_us)
    if (not isinstance(prefillBatches, int)
            and len(prefillBatches) != len(decodeBatches)):
        raise ValueError("prefillBatches must name one batch per decode batch")
    if isinstance(prefixCached, int):
        prefixes = [prefixCached] * len(contexts)
    elif len(prefixCached) == len(contexts):
        prefixes = list(prefixCached)
    else:
        raise ValueError("prefixCached must name one prefix per context")
    if isinstance(verifyWidth, int):
        widths = [verifyWidth] * len(contexts)
    elif len(verifyWidth) == len(contexts):
        widths = list(verifyWidth)
    else:
        raise ValueError("verifyWidth must name one width per context")
    graphs = {name: LawGraph.from_json(record["phases"][name]["law"])
              for name in ("prefill", "decode")}
    prefillTimes: dict[tuple[int, int, int, int], float] = {}
    out: list[list[dict | None]] = []
    for context_index, context in enumerate(contexts):
        capacityRecord = (
            {"capacity": record["draft_capacity"]} if draftTokens
            and "draft_capacity" in record else record)
        prefixCached = int(prefixes[context_index])
        verifyWidth = int(widths[context_index])
        try:
            facts = _capacityAt(capacityRecord, card, int(context))
        except KeyError:
            if servedOnly:
                out.append([None] * len(decodeBatches))
                continue
            raise
        row: list[dict | None] = []
        for i, decodeBatch in enumerate(decodeBatches):
            requests = int(decodeBatch) if decodeBatch is not None else 1
            if servedOnly and not point_served(
                    max_batch_per_gpu=facts["max_decode_batch"],
                    beyond_max_context=bool(facts["beyond_max_context"]),
                    requests=requests):
                row.append(None)
                continue
            prefillBatch = (
                prefillBatches if isinstance(prefillBatches, int)
                else prefillBatches[i])
            point = _servingLawPoint(
                record, facts=facts, cardValues=cardValues,
                graphs=graphs, context=int(context), outputTokens=outputTokens,
                decodeBatch=decodeBatch, prefillBatch=prefillBatch,
                prefixCached=prefixCached, verifyWidth=verifyWidth,
                draftTokens=draftTokens,
                acceptanceRate=acceptanceRate, prefillTimes=prefillTimes)
            row.append(without_step(point, ())
                       if facts.get("infeasibility_reason") else point)
        out.append(row)
    return out


def cardBinding(record: Mapping[str, Any], card: Card, *,
                comm_latency_us: float, launch_floor_us: float,
                dtypes: Sequence[str] = ()) -> dict[str, float]:
    """The card half of a law graph's inputs — `card`'s peaks, bandwidth and
    fabric — for a card of the class the record's rows were walked on
    (`walked_on`), refused for any other (`Card.refuse_unless_one_class`).

    The one door a card enters a law through, so the rule is asked here and
    not by each projection: a law's terms are one walk's rows, and a card of
    another class runs other rows (another kernel, another activation
    format). Its symbols would bind and the number would look right."""
    Card.of(record["walked_on"]).refuse_unless_one_class(card)
    gpu = card.spec
    fabric = card.fabric(comm_latency_us * 1e-6)
    out: dict[str, float] = {
        "PEAK_BW": float(peak_hbm_bw(gpu) or math.inf),
        "COMM_S_PER_BYTE": 0.0,
        "COMM_LATENCY_S": 0.0,
        "LAUNCH_S": launch_floor_us * 1e-6,
    }
    wanted = set(dtypes)
    for phase in record["phases"].values():
        graph = LawGraph.from_json(phase["law"])
        for name in graph.inputs:
            if name == "PEAK_FLOPS":
                out[name] = float(peak_flops_for_dtype(gpu, "") or math.inf)
            elif name.startswith("PEAK_FLOPS_"):
                dtype = name[len("PEAK_FLOPS_"):]
                wanted.add(dtype)
                out[name] = float(
                    peak_flops_for_dtype(gpu, dtype) or math.inf)

    def collectiveValues(signature: str) -> None:
        facts = json.loads(signature)
        rungs = tuple(CollectiveRung.from_dict(rung)
                      for rung in facts.get("rungs") or ())
        ladder = collective_ladder(
            op=facts["op"], world=int(facts["world"]),
            count_per_step=int(facts["count"]),
            base_stride=int(facts["stride"]), fabric=fabric, rungs=rungs)
        base = (f"_{facts['op']}_{int(facts['world'])}_"
                f"{int(facts['count'])}_{int(facts['stride'])}")
        names = (f"COMM_S_PER_BYTE{base}", f"COMM_LATENCY_S{base}")
        values = (ladder[-1].s_per_byte, ladder[-1].latency_s)
        for symbol, value in zip(names, values, strict=True):
            out[symbol] = float(value)
        for index, segment in enumerate(ladder[:-1]):
            names = tuple(
                f"{symbol.name[:-len(f'_r{index}')] if symbol.name.endswith(f'_r{index}') else symbol.name}{base}_r{index}"
                for symbol in rung_symbols(index))
            values = (segment.max_bytes, segment.s_per_byte,
                      segment.latency_s)
            for symbol, value in zip(names, values, strict=True):
                out[symbol] = float(value)

    seen: set[str] = set()
    for phase in record["phases"].values():
        for signature in phase["collectives"]:
            if signature not in seen:
                seen.add(signature)
                collectiveValues(signature)
    return out
