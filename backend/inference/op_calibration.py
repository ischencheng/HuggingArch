"""Op-level measurement calibration overlay.

Indexes validated benchmark rows — those a Kernel Bench lookup answers for
an estimate's keys (`registry_from_lookup`, each under the key that selected
it), and rows a request posts (`merge_validated_rows`) — into an in-memory
registry keyed by the 12-column CSV identity tuple, and indexed by the
physical op (`PhysicalKey`). An estimate reads the physical index:
`latest_time_for` — the exact provider implementation a serving-framework
binding dispatches to, at the latest provider release the corpus measured it
on — or `physical_time_for` — the fastest kernel across libraries (the
envelope).
Measurements are versioned and framework bindings are not, so the version is
never part of what an estimate asks for; it is part of what a row answers
with. The returned latency is mode-aware peak-drift compensated; an anomaly
badge is attached when the measurement is below
`ANOMALY_THRESHOLD × roofline_at_test`.

Design contract:

- Every row this module sees passed the CLI's validator (`huggingarch bench
  validate`): posted rows through `benchmarks.validate_text` here, looked-up
  rows by the Kernel Bench service under the same gate.
- Peak-drift compensation is mode-aware: classify the original measurement
  as compute-bound iff `achieved_tflops / peak_tflops_at_test ≥
  achieved_bw_gbps / peak_bw_at_test`; apply the matching ratio. Mixed-mode
  rows pick the slower scaled latency (worst-case, never under-predict).
- Transparency over safety: when a measurement scores below
  `ANOMALY_THRESHOLD × roofline_at_test_us`, the returned `t_op_s` STILL
  reflects the measured value (with peak-drift compensation applied);
  the `MeasuredTiming.anomaly_reason` field carries the warning text.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from packaging.version import InvalidVersion, Version

from . import benchmarks as _benchmarks
from backend.measure import lookup as _measure_lookup
from backend.measure.auto_sweep import AMBIGUOUS, MEASURED, MISSING
from backend.measure.scope import MeasurementScope
from backend.arch.spec_id_compose import split_aggregation_id

from .benchmark_identity import (
    hardware_hash,
    op_kernel_hash,
    op_signature,
    project_shape_for_op,
    workload_hash,
)
from backend.code_trace.memo import memoize

_LOG = logging.getLogger(__name__)


ANOMALY_THRESHOLD: float = 0.95


@dataclass(frozen=True)
class BenchmarkKey:
    """The 12-column CSV identity tuple, used verbatim as a lookup key.

    Field order matches `benchmarks.IDENTITY_COLUMNS`. String equality on
    every field is the match rule — including `shape_json` and
    `dtype_env_json`, which stay as raw JSON strings so the same
    byte-identity rule the validator enforces also governs calibration.

    NOTE: this wide key is retained for backward compatibility (existing
    callers, CSV authoring tools, exact-byte lookups). New code should
    prefer `SlimBenchmarkKey` — same identity content, four hashed
    dimensions, much harder to typo.
    """
    model: str
    spec_hash: str
    phase: str
    parallel: str
    framework: str
    framework_version: str
    gpu: str
    op: str
    node_spec_id: str
    shape_json: str
    dtype_env_json: str
    kernel_impl: str

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> "BenchmarkKey":
        return cls(**{c: row[c] for c in _benchmarks.IDENTITY_COLUMNS})

    def to_physical(self) -> "PhysicalKey":
        """Project this wide key onto the library-agnostic physical key
        for the envelope lookup. The op collapses to its SIGNATURE (so
        differently-named-but-same-math fused ops across frameworks match);
        shape is projected to the op-relevant dims; framework /
        framework_version / kernel_impl / model / spec_hash / node_spec_id
        are dropped (provenance, not physics)."""
        signature, physical_shape = _physical_parts(self.op, self.shape_json)
        return PhysicalKey(
            signature=signature,
            shape_json=physical_shape,
            dtype_env_json=self.dtype_env_json,
            gpu=self.gpu,
            phase=self.phase,
        )

    def to_slim(self) -> "SlimBenchmarkKey":
        """Project the 12-field wide key onto the 4-field canonical
        identity. The wide-key string fields collapse into three
        hashes; spec_hash passes through. Any wide-key whose hashes
        match another wide-key's hashes maps to the same row."""
        try:
            workload = _workload_hash_of(self.shape_json, self.dtype_env_json)
        except _benchmarks.BenchmarkValidationError as exc:
            # Raised again naming the row: the memo is keyed by the cells alone.
            raise _benchmarks.BenchmarkValidationError(
                f"{self.model} [{self.node_spec_id}]: {exc}") from exc
        return SlimBenchmarkKey(
            spec_hash=self.spec_hash,
            workload_hash=workload,
            hardware_hash=_hardware_hash_of(
                self.gpu, self.framework, self.framework_version,
                self.phase, self.parallel),
            op_kernel_hash=_op_kernel_hash_of(
                self.node_spec_id, self.op, self.kernel_impl),
        )


@dataclass(frozen=True)
class SlimBenchmarkKey:
    """Four-dimension canonical identity for a calibration row.

    Replaces the 12-string wide `BenchmarkKey` for lookup purposes.
    Each field is an opaque sha256-prefix hash that folds together
    several human-readable strings:

      * `spec_hash`        — which spec topology (content fingerprint,
                              identical to wide-key's `spec_hash`)
      * `workload_hash`    — (B, s_ctx, dim_values, all dtype slots)
      * `hardware_hash`    — (gpu, framework, framework_version, phase,
                              parallel)
      * `op_kernel_hash`   — (node_spec_id, op, kernel_impl)

    Two rows with the same SlimBenchmarkKey are interchangeable for
    calibration purposes — collapsing UI-aliased dtype variants
    (`fp8` vs `fp8_e4m3`) and string-cased framework names
    (`vllm` vs `VLLM`) into one canonical hash means a contributor
    can't silently miss a row by capitalizing differently or by
    forgetting one of the five dtype slots.
    """
    spec_hash: str
    workload_hash: str
    hardware_hash: str
    op_kernel_hash: str


@dataclass(frozen=True)
class PhysicalKey:
    """Physical (library-agnostic) identity for the ENVELOPE lookup.

    Keys ONLY on what physically determines a kernel's runtime — the op
    it computes, the shape that feeds it (projected to the op-relevant
    dims so model-only dims like d_model/d_ffn/n_layers/n_vocab don't
    fragment the match), the
    dtype env, the card and the phase. It
    deliberately DROPS everything that is provenance rather than physics:

      * framework / framework_version / kernel_impl  — WHICH implementation
        produced a number, not what was computed. Dropping them lets FA3,
        FlashInfer, vLLM, … compete for the same physical op.
      * model / spec_hash / node_spec_id             — WHICH model/graph the
        measurement came from. Dropping them lets a kernel measured once be
        reused by any model that hits the same shape.
      * parallel                                     — WHICH deployment the
        per-rank shape was cut from. One card runs a kernel at a shape the
        same way whether tp=8 or ep=64 produced that shape; the deployment
        is workload provenance, kept on the row. Keyed on it, one DeepSeek-V3
        sweep planned 14,207 cells for 5,969 distinct kernels, and a shape
        measured under one deployment was invisible to every other.

    Many `_Row`s share one PhysicalKey — that's the point: the envelope
    picks the fastest among them.

    `phase` is retained (not folded away) because only ops whose shape
    carries `s_query`/`s_kv` (attention, via `identity_shape_from_shape_in`)
    encode prefill-vs-decode IN the shape; ops still on the uniform
    `{B,s_ctx,dims}` fallback need `phase` to stay distinct. For attention
    it's redundant-but-harmless (prefill s_query=s_ctx and decode s_query=1
    already differ).
    """
    signature: str           # library-agnostic op signature (not the fused op name)
    shape_json: str          # projected to op-relevant dims (s_query/s_kv-aware)
    dtype_env_json: str
    gpu: str
    phase: str


def physical_identity(op: str, shape_json: str) -> Dict[str, str]:
    """The two derived parts of a measured row's envelope identity: the
    op's measurement `signature` and its shape projected to the signature's
    identifying axes (`physical_shape`, canonical JSON text). The rest of a
    `PhysicalKey` — dtype env, gpu, phase — is the row's own.
    The one derivation: `BenchmarkKey.to_physical` and the stored columns a
    row is written with both come from here."""
    return {"signature": op_signature(op),
            "physical_shape": project_shape_for_op(op, shape_json)}


# A corpus repeats a few shapes many times (the largest owner: 71k rows,
# 16k (op, shape) pairs), and these projections are pure functions of their
# strings under this process's catalog — so a registry load computes each
# once.
_IDENTITY_MEMO = 1 << 16


@memoize(maxsize=_IDENTITY_MEMO)
def _physical_parts(op: str, shape_json: str) -> Tuple[str, str]:
    """`physical_identity` as a (signature, physical_shape) pair."""
    ident = physical_identity(op, shape_json)
    return ident["signature"], ident["physical_shape"]


@memoize(maxsize=_IDENTITY_MEMO)
def _workload_hash_of(shape_json: str, dtype_env_json: str) -> str:
    """`workload_hash` of a row's two JSON identity cells. The dtype cell is
    parsed so `workload_hash` computes its own canonical form (two
    equivalent JSONs with different whitespace hash alike).

    Every dtype slot the schema declares (`benchmarks.DTYPE_SLOTS`) is the
    row's to state, except the ones that do not apply to it — `weight`,
    `kv`, `metadata` (`benchmarks.check_dtype_slots`). A cell that does not parse, or
    lacks a required slot, is refused: filling the gap with BF16 hashed a
    row nobody described into a key that looked clean."""
    try:
        env = json.loads(dtype_env_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise _benchmarks.BenchmarkValidationError(
            f"dtype_env_json {dtype_env_json!r} is not JSON: {exc}") from exc
    if not isinstance(env, dict):
        raise _benchmarks.BenchmarkValidationError(
            f"dtype_env_json {dtype_env_json!r} is not a JSON object")
    _benchmarks.check_dtype_slots(env, column=f"dtype_env_json {dtype_env_json!r}")
    return workload_hash(
        shape_json=shape_json,
        weight_dtype=env.get("weight"),
        activation_dtype=env["activation"],
        kv_dtype=env.get("kv"),
        output_dtype=env["output"],
        metadata_dtype=env.get("metadata"),
        weight_granularity=env.get("weight_granularity"),
        activation_granularity=env.get("activation_granularity"),
        output_format=env.get("output_format"),
        kv_format=env.get("kv_format"),
    )


@memoize(maxsize=_IDENTITY_MEMO)
def _hardware_hash_of(gpu: str, framework: str, framework_version: str,
                      phase: str, parallel: str) -> str:
    return hardware_hash(gpu=gpu, framework=framework,
                         framework_version=framework_version,
                         phase=phase, parallel=parallel)


@memoize(maxsize=_IDENTITY_MEMO)
def _op_kernel_hash_of(node_spec_id: str, op: str, kernel_impl: str) -> str:
    return op_kernel_hash(node_spec_id=node_spec_id, op=op, kernel_impl=kernel_impl)


@dataclass(frozen=True)
class MeasuredTiming:
    """One calibrated measurement returned from the registry.

    `t_op_s` already includes mode-aware peak-drift compensation. `source`
    is one of `"measured"`, `"measured_anomalous"`. `source_badge` carries
    the per-row metadata the UI surfaces in tooltips. `anomaly_reason` is
    populated iff `source == "measured_anomalous"`. `utilization` is the
    measured row against the card it ran on (`measured_utilization`: `mfu`,
    `mbu`, …), the same reading the Kernel Bench tables show.
    """
    t_op_s: float
    source: str
    source_badge: Dict[str, str]
    anomaly_reason: Optional[str] = None
    utilization: Dict[str, Optional[float]] = field(default_factory=dict)


#: The columns of a measured row the registry reads (`_index_row`): its
#: 12-column identity (`BenchmarkKey`) and the measurement `_Row` holds. A
#: Kernel Bench lookup sends a row as exactly these
#: (`service/bench/lookup.py`); a test indexes a row cut to them.
ROW_COLUMNS: Tuple[str, ...] = (
    *_benchmarks.IDENTITY_COLUMNS, "latency_us", "achieved_tflops", "achieved_bw_gbps",
    "peak_tflops_at_test", "peak_bw_at_test", "contributor", "date", "samples",
    # `outcome` rides along so the reader can tell a measurement from a
    # recorded refusal (an OOM row has no latency at all) rather than
    # discovering it as a ValueError on an empty string.
    "cross_check", "outcome",
    # Which driver ran the row, for the measured badge. Both, because the
    # pair is what says whether a user-mode driver was substituted — sending
    # one alone would leave the reader unable to tell.
    *_benchmarks.DRIVER_COLUMNS)


@dataclass
class _Row:
    """Internal — one parsed + interpreted CSV row.

    Stores the typed measurement / peak fields so `measured_time_for`
    doesn't re-parse them on every lookup.
    """
    key: BenchmarkKey
    latency_us: float
    achieved_tflops: float
    achieved_bw_gbps: float
    peak_tflops_at_test: float
    peak_bw_at_test: float
    contributor: str
    date: str
    samples: int = 0          # measurement sample count — dedup tiebreaker
    # Where this row came from, for UI attribution: "shared" (Kernel Bench's
    # shared rows) or "mine" (the requesting user's own rows, or rows the
    # request posted). Never enters any timing/roofline math.
    origin: str = "shared"
    # The corpus's numerical cross-check verdict for this row (`passed`,
    # `no_reference`, `not_comparable`, `disabled`, `unknown`). Shown on the
    # measured badge so a reader sees which measurements were checked
    # against a reference kernel; most of today's corpus predates the check.
    cross_check: str = "unknown"
    # Which driver ran this row — the box's kernel-mode driver and the
    # user-mode driver the measuring process loaded. Carried so the measured
    # badge can say when the driver that ran a number was not the box's own
    # (`benchmarks.is_driver_substituted`). That and no more: whether such a
    # pairing is supported depends on the pairing, and a row carries no
    # verdict and no `box` to look one up by. Never enters any timing or
    # roofline math, and not part of any key: the pair is provenance, exactly
    # as `box` beside it in the corpus already is.
    # Empty on every row written before schema 14, which reads as unknown.
    kmd_version: str = ""
    umd_version: str = ""

    def utilization(self) -> Dict[str, Optional[float]]:
        return measured_utilization(self.__dict__)

    @property
    def activation_dtype(self) -> str:
        """The dtype the kernel ran in — the MMA family its recorded
        `peak_tflops_at_test` is the card's peak for."""
        return json.loads(self.key.dtype_env_json)["activation"]


#: The current card's peak compute (TFLOPs) at a dtype, None when the card
#: has no column for it. A compute peak is one number per (card, MMA
#: family), not per card: a row measured in FP8 recorded the FP8 peak, so
#: its drift is read against the FP8 peak of the card it is applied to.
PeakAt = Callable[[str], Optional[float]]


def measured_utilization(row: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """A measured row against the GPU it ran on, from its achieved rates and
    the peaks it recorded at test time (`achieved_tflops`,
    `achieved_bw_gbps`, `peak_tflops_at_test`, `peak_bw_at_test` — a corpus
    CSV row and a stored upload row name them alike):

    - `mfu`: achieved FLOP/s over the peak FLOP/s;
    - `mbu`: achieved bytes/s over the peak HBM bytes/s;
    - `arith_intensity`: FLOPs per byte moved (the roofline's x axis);
    - `roof_status`: whether the row beats a peak it recorded — `ok`, or the
      ratio(s) over 1 (`mfu`, `mbu`, `mfu+mbu`): faster than the hardware
      can run, so the measurement is suspect. None with no peak to check.

    Each ratio is None when the number it divides by is missing or zero."""
    def num(key: str) -> float:
        try:
            return float(row.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0
    tflops, bw = num("achieved_tflops"), num("achieved_bw_gbps")
    peak_tflops, peak_bw = num("peak_tflops_at_test"), num("peak_bw_at_test")
    mfu = tflops / peak_tflops if peak_tflops > 0 else None
    mbu = bw / peak_bw if peak_bw > 0 else None
    over = [name for name, ratio in (("mfu", mfu), ("mbu", mbu))
            if ratio is not None and ratio > 1]
    return {
        "mfu": mfu,
        "mbu": mbu,
        "arith_intensity": 1000 * tflops / bw if bw > 0 else None,
        "roof_status": "+".join(over) if over else
                       ("ok" if mfu is not None or mbu is not None else None),
    }


@dataclass
class CalibrationRegistry:
    """In-memory lookup table built from validated CSV rows.

    Two parallel indices on the same `_Row` objects:
      * `rows_by_key`      — exact 12-field BenchmarkKey lookup
      * `rows_by_slim_key` — canonical 4-field SlimBenchmarkKey lookup

    The slim index canonicalizes dtype aliases / case / JSON whitespace
    so contributors can't typo a row into oblivion. The wide index
    serves the row-level reads (`measured_time_for`, the registry dump, and
    kernel_bench's CLI); estimates read the physical index
    (`latest_time_for`, `physical_time_for`).
    """
    rows_by_key: Dict[BenchmarkKey, _Row] = field(default_factory=dict)
    rows_by_slim_key: Dict[SlimBenchmarkKey, _Row] = field(default_factory=dict)
    # Physical (library-agnostic) index — MANY rows per key (the envelope
    # candidates: FA3 / FlashInfer / … / different versions of each).
    rows_by_physical_key: Dict[PhysicalKey, List[_Row]] = field(default_factory=dict)

    def get(self, key: BenchmarkKey) -> Optional[_Row]:
        return self.rows_by_key.get(key)

    def get_slim(self, key: SlimBenchmarkKey) -> Optional[_Row]:
        return self.rows_by_slim_key.get(key)

    def get_physical(self, key: PhysicalKey) -> List[_Row]:
        return self.rows_by_physical_key.get(key, [])

    def covers(
        self, key: PhysicalKey, *, framework: str, framework_version: str,
        kernel_impl: Optional[str] = None,
    ) -> bool:
        """Whether the corpus already has a measurement for this physical
        op from this (framework, version, kernel_impl). The harness consults
        this to skip re-measuring a shape another model already covered — the
        measure-once-reuse payoff. `gpu`/`phase`/dtype are already
        baked into `key`.

        `kernel_impl` is part of the question, not a detail within it: a
        framework ships SEVERAL kernels per signature (DeepGEMM's contiguous
        and masked grouped-GEMM; a paged decode at one block size and another),
        and the envelope exists precisely to hold them side by side. Matching on
        (framework, version) alone let the first implementation measured at a key
        suppress every sibling — so the envelope had one candidate where it
        should have had several, and a re-measurement under corrected kernel
        parameters would be skipped as already covered.

        Callers that genuinely mean "any implementation from this framework"
        pass `kernel_impl=None`.
        """
        return any(
            r.key.framework == framework
            and r.key.framework_version == framework_version
            and (kernel_impl is None or r.key.kernel_impl == kernel_impl)
            for r in self.rows_by_physical_key.get(key, [])
        )

    def __len__(self) -> int:
        return len(self.rows_by_key)

    def copy(self) -> "CalibrationRegistry":
        """A registry over the same rows whose indices are its own: rows
        merged into the copy (`merge_validated_rows`) leave this one as it
        was."""
        return CalibrationRegistry(
            rows_by_key=dict(self.rows_by_key),
            rows_by_slim_key=dict(self.rows_by_slim_key),
            rows_by_physical_key={k: list(v) for k, v
                                  in self.rows_by_physical_key.items()},
        )


def node_label(node_spec_id: str) -> str:
    """The name a measured row's node goes by: its aggregation id's display
    path, without the block template it was measured in."""
    return split_aggregation_id(node_spec_id or "")[1]


def _index_row(
    registry: CalibrationRegistry,
    row: Dict[str, str],
    *,
    source: str,
    on_collision: str = "raise",
    origin: str = "shared",
    divergences: Optional["_Divergences"] = None,
    physical: Optional[PhysicalKey] = None,
) -> BenchmarkKey:
    """Index one validated CSV row into `registry`'s two parallel indices
    (`rows_by_key` exact + `rows_by_slim_key` canonical) and its physical
    index — under `physical` when given (the key that selected the row in a
    lookup), else under the row's own projection (`BenchmarkKey.to_physical`).

    `on_collision` governs what happens when the row's identity is already
    present:
      * "raise"    — corpus integrity: a cross-row slim-key collision is a
        contributor error. Used for the shared rows of a lookup.
      * "override" — last-write-wins: the new row silently replaces any
        existing one. Used when merging a user's own request rows on top
        of the shared corpus, where the user's measurement is meant to
        win over the public baseline.
    """
    # A row that records an OUTCOME rather than a measurement has no latency
    # to calibrate with. Asked HERE, at the one entry every caller goes
    # through (`registry_from_lookup`, `merge_validated_rows`, a test
    # indexing a cut row), because asking it per caller is how the second
    # caller came to read `float("")` and answer a request with a 500.
    if not _benchmarks.is_measurement(row):
        return
    key = BenchmarkKey.from_csv_row(row)
    row_obj = _Row(
        key=key,
        latency_us=float(row["latency_us"]),
        achieved_tflops=float(row["achieved_tflops"]),
        achieved_bw_gbps=float(row["achieved_bw_gbps"]),
        peak_tflops_at_test=float(row["peak_tflops_at_test"]),
        peak_bw_at_test=float(row["peak_bw_at_test"]),
        contributor=row["contributor"],
        date=row["date"],
        samples=int(row.get("samples", 0) or 0),
        origin=origin,
        cross_check=(row.get("cross_check") or "unknown").strip() or "unknown",
        kmd_version=(row.get("kmd_version") or "").strip(),
        umd_version=(row.get("umd_version") or "").strip(),
    )
    registry.rows_by_key[key] = row_obj
    # Parallel slim-key index for canonical lookups. Slim-key collision =
    # two wide keys that canonicalize to the same 4-hash identity (e.g.
    # `fp8` vs `fp8_e4m3` for the same row). Same-row collision is fine
    # (last-write wins gives idempotent behavior); cross-row collision is
    # a real ambiguity the contributor must resolve — but under "override"
    # the caller has opted into last-write-wins.
    slim = key.to_slim()
    existing = registry.rows_by_slim_key.get(slim)
    if existing is not None and existing.key != key and on_collision == "raise":
        raise _benchmarks.BenchmarkValidationError(
            f"slim-key collision in {source!r}: two distinct CSV rows "
            f"canonicalize to the same 4-hash identity {slim!r}. "
            f"Either rows describe the same measurement (drop one) "
            f"or one of the wide-key fields differs only by "
            f"canonicalization (e.g. `fp8` vs `fp8_e4m3` — pick "
            f"the validator-strict form)."
        )
    registry.rows_by_slim_key[slim] = row_obj

    # Physical index — the envelope candidate set. Rows sharing a physical
    # key but a DIFFERENT implementation (framework, version, kernel_impl)
    # are legitimate envelope candidates (FA3 vs FlashInfer vs …) → keep
    # all. Rows sharing physical key AND implementation are the SAME
    # measurement re-observed (e.g. two models that both hit this shape, or
    # a re-upload) → dedup to one, so the corpus stays physically-indexed
    # not model-indexed. Model / spec_hash / node_spec_id are NOT part of
    # the physical key, so cross-model duplicates collapse here.
    physical = physical or key.to_physical()
    candidates = registry.rows_by_physical_key.setdefault(physical, [])
    impl = (key.framework, key.framework_version, key.kernel_impl)
    existing_same_impl = next(
        (r for r in candidates
         if (r.key.framework, r.key.framework_version, r.key.kernel_impl) == impl),
        None,
    )
    if existing_same_impl is None:
        candidates.append(row_obj)
    else:
        own = divergences is None
        if own:
            divergences = _Divergences()
        _resolve_physical_duplicate(candidates, existing_same_impl, row_obj, divergences)
        if own:
            divergences.report(source)
    return key


# A re-measurement of the SAME physical op + SAME implementation + SAME
# hardware should reproduce within measurement noise. A larger gap means
# either environment noise or — more usefully — that the physical key is
# missing a dimension that actually drives runtime (causal flag, mask
# pattern, alignment, page size…). We keep one row and log the divergence
# so the incompleteness surfaces rather than silently averaging away.
_PHYSICAL_DUP_DIVERGENCE: float = 0.10


class _Divergences:
    """The divergent physical-key duplicates of one load, reported as one
    line (a corpus with many re-measurements logged thousands otherwise)."""

    _EXAMPLES = 3

    def __init__(self) -> None:
        self.count = 0
        self.examples: List[str] = []

    def add(self, existing: "_Row", incoming: "_Row", ratio: float) -> None:
        self.count += 1
        if len(self.examples) < self._EXAMPLES:
            k = existing.key
            self.examples.append(
                f"{k.op} @ {k.framework}/{k.framework_version}/{k.kernel_impl}: "
                f"{existing.latency_us:.3f}us vs {incoming.latency_us:.3f}us "
                f"({ratio * 100:.0f}%)")

    def report(self, source: str) -> None:
        if self.count:
            _LOG.warning(
                "physical-key duplicate diverges: %d re-measurement(s) in %s differ "
                "by over %.0f%% (e.g. %s). Either env noise or the physical key is "
                "missing a runtime-determining dimension.",
                self.count, source, _PHYSICAL_DUP_DIVERGENCE * 100,
                "; ".join(self.examples))


def _resolve_physical_duplicate(
    candidates: List["_Row"], existing: "_Row", incoming: "_Row",
    divergences: _Divergences,
) -> None:
    """Collapse two rows that share physical key + implementation. Record a
    divergence (`divergences`, reported once per load); keep the
    higher-confidence row (more samples → later date; else keep existing for
    determinism under sorted load order)."""
    a, b = existing.latency_us, incoming.latency_us
    lo = min(a, b)
    if lo > 0 and abs(a - b) / lo > _PHYSICAL_DUP_DIVERGENCE:
        divergences.add(existing, incoming, abs(a - b) / lo)
    keep_incoming = (
        incoming.samples > existing.samples
        or (incoming.samples == existing.samples and incoming.date > existing.date)
    )
    if keep_incoming:
        candidates[candidates.index(existing)] = incoming


#: The identity columns whose value is a JSON blob — derived from the CSV
#: schema (`benchmarks.IDENTITY_COLUMNS`) by their `_json` spelling, not a
#: second hand-kept list of the same two names.
_JSON_CELL_COLUMNS = tuple(
    c for c in _benchmarks.IDENTITY_COLUMNS if c.endswith("_json"))


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce request rows into the CSV string form `emit_csv` expects.

    Rows arriving from the estimate request may carry `shape_json` /
    `dtype_env_json` as parsed JSON objects (that is how they come back
    from Supabase's JSONB columns) or as canonical strings (a freshly
    parsed CSV). `emit_csv` writes each cell via `csv.writer`, which would
    render a dict as a Python repr (single quotes) that the validator's
    `json.loads` rejects. Serialize those two cells to compact JSON
    strings; leave every other cell (including numerics) untouched — the
    writer stringifies those correctly.

    SORTED, because these two are IDENTITY columns matched byte-for-byte. Both
    producers (`compute_per_row_shape_json`, `compute_per_row_dtype_env_json`)
    sort; this did not, so a JSONB row was keyed in whatever order the object
    came back in, and every order but the sorted one is a key no estimate-side
    lookup will ever build. A privately-uploaded row could therefore be stored,
    listed, and never matched — the row is right, its identity is unreachable.

    And a column an older CSV may omit (`benchmarks.OPTIONAL_COLUMNS`) is
    filled EMPTY when absent — what each one's documentation says absence
    means, and what `validate_header` already lets a file do. Without it a
    row from a schema-14 file (no `outcome_detail`) raised `KeyError` in the
    emit below instead of validating.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        r = {**dict.fromkeys(_benchmarks.OPTIONAL_COLUMNS, ""), **dict(row)}
        for col in _JSON_CELL_COLUMNS:
            v = r.get(col)
            if v is not None and not isinstance(v, str):
                r[col] = json.dumps(v, separators=(",", ":"), sort_keys=True)
        out.append(r)
    return out


def merge_validated_rows(
    registry: CalibrationRegistry,
    rows: List[Dict[str, Any]],
    *,
    source: str = "request",
    origin: str = "mine",
) -> CalibrationRegistry:
    """Overlay `rows` onto `registry` in place and return it.

    Rows are re-validated through the exact CLI/CI gate
    (`benchmarks.validate_text`) before indexing — the caller (e.g. the
    estimate route receiving client-supplied `calibration_rows`) must not
    be trusted to have pre-validated them. Round-tripping through
    `emit_csv` coerces JSON-numeric values back to the canonical string
    form the validator expects.

    Collisions resolve last-write-wins ("override"): a user's own row
    replaces any shared-corpus row with the same identity, so their
    measurement calibrates their own estimate.
    """
    if not rows:
        return registry
    text = _benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), _normalize_rows(rows))
    _header, validated = _benchmarks.validate_text(text, source=source)
    divergences = _Divergences()
    for row in validated:
        _index_row(registry, row, source=source, on_collision="override",
                   origin=origin, divergences=divergences)
    divergences.report(source)
    return registry


def registry_from_lookup(keys: Sequence[PhysicalKey],
                         rows: Sequence[Sequence[Mapping[str, str]]]) -> CalibrationRegistry:
    """The registry of a Kernel Bench lookup (`measure.lookup.lookup`): every
    row of `rows[i]` indexed under `keys[i]`, the key that selected it —
    never re-projected from its stored shape (one projection decides
    membership: the service's current annotations). Within a key the shared
    rows come first and the owner's over them (`origin`), as the shared
    corpus and the owner's corpus were loaded: a shared identity repeated is
    an integrity error (`BenchmarkValidationError`), the owner's row wins an
    identity collision. The service validated every row it sent under the
    current validator."""
    registry = CalibrationRegistry()
    divergences = _Divergences()
    shared: set = set()
    for key, per_key in zip(keys, rows, strict=True):
        for row in per_key:
            origin = row["origin"]
            if origin == "shared":
                ident = tuple(row[c] for c in _benchmarks.IDENTITY_COLUMNS)
                if ident in shared:
                    raise _benchmarks.BenchmarkValidationError(
                        f"the Kernel Bench lookup repeats a shared identity: {ident!r}")
                shared.add(ident)
            _index_row(registry, dict(row), source="Kernel Bench",
                       on_collision="raise" if origin == "shared" else "override",
                       origin=origin, divergences=divergences, physical=key)
    divergences.report("the Kernel Bench lookup")
    return registry


def registry_for(scope: MeasurementScope, keys: Sequence[PhysicalKey],
                 calibration_rows: Optional[list] = None) -> CalibrationRegistry:
    """The registry an estimate reads: the rows Kernel Bench answers for
    `keys` under `scope` (one lookup, `measure.lookup`), each indexed under
    the key that selected it (`registry_from_lookup`), and over those the
    request's own `calibration_rows` (`merge_validated_rows`)."""
    found = _measure_lookup.lookup(scope, keys)
    registry = registry_from_lookup(found.keys, found.rows)
    if calibration_rows:
        registry = merge_validated_rows(registry, calibration_rows, source="request")
    return registry


_MIXED_BOUND_UTIL_THRESHOLD: float = 0.5


def _classify_bound(row: _Row) -> str:
    """Return `"compute"`, `"memory"`, or `"mixed"` based on which roofline
    ratio the measurement saturates.

    - compute: compute-utilization dominates AND >= MIXED threshold
    - memory: bandwidth-utilization dominates AND >= MIXED threshold
    - mixed: neither dominates above the threshold (rare; plan Update 4
      says use the slower of the two scaled latencies — worst-case
      extrapolation, never underpredict)
    """
    util = row.utilization()
    compute_util = util["mfu"] or 0.0
    memory_util = util["mbu"] or 0.0
    if compute_util < _MIXED_BOUND_UTIL_THRESHOLD and memory_util < _MIXED_BOUND_UTIL_THRESHOLD:
        return "mixed"
    return "compute" if compute_util >= memory_util else "memory"


def _peak_drift_scaled_latency_us(
    row: _Row,
    *,
    current_peak_tflops_at: PeakAt,
    current_peak_bw_gbps: float,
) -> Tuple[float, str]:
    """Apply mode-aware peak-drift compensation. Compute-bound rows
    scale by the TFLOPs ratio; memory-bound rows by the bandwidth
    ratio. Mixed rows take the slower of the two scaled latencies per
    plan Update 4 (worst-case extrapolation; never underpredict).

    Ratio direction: faster current hardware should make the op run
    FASTER, so latency scales by `test_peak / current_peak` — if
    current peak doubles, the op takes half as long. The pre-fix
    formula `latency × current_peak / test_peak` had it backwards
    (a 2× faster GPU produced a 2× slower calibrated time). The
    plan's literal wording was also inverted; this corrected
    direction matches the physics.

    When current peaks equal test-time peaks (the common case) the
    returned latency equals `row.latency_us` exactly — pinned for
    transparency regardless of bound classification.
    """
    bound = _classify_bound(row)
    compute_scaled = row.latency_us * _compute_drift(row, current_peak_tflops_at)
    memory_scaled = row.latency_us * row.peak_bw_at_test / current_peak_bw_gbps
    if bound == "compute":
        return compute_scaled, bound
    if bound == "memory":
        return memory_scaled, bound
    # mixed: the slower of the two scaled latencies (max — slower means
    # bigger μs). Equivalent to using the slower-progressing peak ratio.
    return max(compute_scaled, memory_scaled), bound


def _compute_drift(row: _Row, current_peak_tflops_at: PeakAt) -> float:
    """`test_peak / current_peak` at the row's own MMA family; 1 when either
    side is unstated — no evidence of drift is not a drift."""
    current = current_peak_tflops_at(row.activation_dtype) or 0.0
    if row.peak_tflops_at_test <= 0 or current <= 0:
        return 1.0
    return row.peak_tflops_at_test / current


def _roofline_at_test_us(row: _Row, *, flops: float, access_bytes: float) -> float:
    """Compute the theoretical roofline latency in microseconds using
    the measurement's snapshotted peaks, for anomaly comparison."""
    if row.peak_tflops_at_test <= 0 or row.peak_bw_at_test <= 0:
        return 0.0
    peak_flops_per_sec = row.peak_tflops_at_test * 1e12
    peak_bw_bytes_per_sec = row.peak_bw_at_test * 1e9
    t_compute_s = flops / peak_flops_per_sec if peak_flops_per_sec > 0 else 0.0
    t_memory_s = access_bytes / peak_bw_bytes_per_sec if peak_bw_bytes_per_sec > 0 else 0.0
    return max(t_compute_s, t_memory_s) * 1e6


def measured_time_for(
    registry: CalibrationRegistry,
    key: BenchmarkKey,
    *,
    current_peak_tflops_at: PeakAt,
    current_peak_bw_gbps: float,
    flops_for_anomaly: Optional[float] = None,
    access_bytes_for_anomaly: Optional[float] = None,
    source_anchor: Optional[Dict[str, Any]] = None,
) -> Optional[MeasuredTiming]:
    """Exact-match lookup. Returns a `MeasuredTiming` on hit, `None` on miss.

    Peak-drift compensation is mode-aware. When the current peaks equal
    the test-time peaks, the returned `t_op_s` equals `row.latency_us *
    1e-6` exactly (regression-locked).

    Anomaly detection only fires when the caller supplies both
    `flops_for_anomaly` and `access_bytes_for_anomaly` so the function
    can compute a comparable roofline-at-test. The returned latency is
    never clamped: anomalous values are flagged in `source` +
    `anomaly_reason`, then returned as-is per the transparency contract.

    `source_anchor`, when provided, is added to the badge so the UI can
    link back to the upstream kernel definition (framework-binding fact,
    as `AuditedAt.cite` gives it).
    """
    row = registry.get(key)
    if row is None:
        return None
    return _build_measured_timing(
        row,
        current_peak_tflops_at=current_peak_tflops_at,
        current_peak_bw_gbps=current_peak_bw_gbps,
        flops_for_anomaly=flops_for_anomaly,
        access_bytes_for_anomaly=access_bytes_for_anomaly,
        source_anchor=source_anchor,
    )


def _build_measured_timing(
    row: "_Row",
    *,
    current_peak_tflops_at: PeakAt,
    current_peak_bw_gbps: float,
    flops_for_anomaly: Optional[float] = None,
    access_bytes_for_anomaly: Optional[float] = None,
    source_anchor: Optional[Dict[str, Any]] = None,
) -> MeasuredTiming:
    """Shared body of every lookup that returns one row's time. Does
    peak-drift compensation, builds the badge dict, runs anomaly detection."""
    scaled_us, bound = _peak_drift_scaled_latency_us(
        row,
        current_peak_tflops_at=current_peak_tflops_at,
        current_peak_bw_gbps=current_peak_bw_gbps,
    )
    t_op_s = scaled_us * 1e-6

    peak_drift_pct = 0.0
    if bound == "compute":
        peak_drift_pct = (1.0 / _compute_drift(row, current_peak_tflops_at) - 1.0) * 100.0
    elif bound == "memory" and row.peak_bw_at_test > 0:
        peak_drift_pct = (current_peak_bw_gbps / row.peak_bw_at_test - 1.0) * 100.0

    badge: Dict[str, str] = {
        "framework": row.key.framework,
        "framework_version": row.key.framework_version,
        "kernel_impl": row.key.kernel_impl,
        "gpu": row.key.gpu,
        "phase": row.key.phase,
        "contributor": row.contributor,
        "date": row.date,
        "origin": row.origin,
        "cross_check": row.cross_check,
        "bound": bound,
        "peak_drift_pct": f"{peak_drift_pct:+.2f}",
        "measured_us": f"{row.latency_us:.3f}",
        "peak_tflops_at_test": f"{row.peak_tflops_at_test:.3f}",
        "peak_bw_at_test": f"{row.peak_bw_at_test:.3f}",
    }
    # Which driver produced this number. `driver` is the USER-mode driver —
    # the one that actually executed the kernel — and is empty on every row
    # measured before schema 14.
    #
    # `driver_substituted_for` is present ONLY when the row's two drivers
    # disagree, and its value is the driver it was NOT: the box's own
    # kernel-mode driver. Presence is the flag, so absent means native OR
    # unknown, which `driver` itself distinguishes.
    #
    # It says that and nothing more. It does NOT say the substitution was
    # "forward" (the pair carries no direction) and it does NOT say the
    # configuration is unsupported — that is a fact about a specific pairing,
    # and this row carries neither a support verdict nor a `box` to look one
    # up by (`ROW_COLUMNS` drops it on purpose). The comparison is made in
    # ONE place (`benchmarks.is_driver_substituted`), never re-derived here
    # or downstream.
    if row.umd_version:
        badge["driver"] = row.umd_version
    if _benchmarks.is_driver_substituted(
            {"kmd_version": row.kmd_version, "umd_version": row.umd_version}):
        badge["driver_substituted_for"] = row.kmd_version
    if source_anchor:
        badge["source_anchor"] = source_anchor

    source = "measured"
    anomaly_reason: Optional[str] = None
    if flops_for_anomaly is not None and access_bytes_for_anomaly is not None:
        roofline_us = _roofline_at_test_us(
            row, flops=flops_for_anomaly, access_bytes=access_bytes_for_anomaly,
        )
        if roofline_us > 0 and row.latency_us < roofline_us * ANOMALY_THRESHOLD:
            ratio = row.latency_us / roofline_us
            anomaly_reason = (
                f"measured {row.latency_us:.3f} μs vs roofline-at-test "
                f"{roofline_us:.3f} μs (ratio {ratio:.3f} < "
                f"ANOMALY_THRESHOLD={ANOMALY_THRESHOLD})"
            )
            source = "measured_anomalous"
            badge["anomaly_reason"] = anomaly_reason

    return MeasuredTiming(
        t_op_s=t_op_s,
        source=source,
        source_badge=badge,
        anomaly_reason=anomaly_reason,
        utilization=row.utilization(),
    )


def _version_order(version: str) -> tuple:
    """Sort key for a measured row's `framework_version`: release order when
    it parses as a package version (`0.30.0` > `0.27.1`, `2.13.0+cu130` >
    `2.11.0+cu130`), else the string itself after every parseable one."""
    try:
        return (1, Version(version), "")
    except InvalidVersion:
        return (0, None, version)


def _parse_kernel_pin(pin: str) -> Tuple[Optional[str], str]:
    """A user pin as ``(library, impl_id)``.

    A bare implementation id remains accepted for the existing debug payload,
    but it can name only one library among the rows at this key. The explicit
    form is ``library:impl_id`` and compares the same pair identity a dispatch
    declares.
    """
    parts = pin.split(":")
    if len(parts) not in (1, 2) or not all(parts):
        raise ValueError(
            f"kernel pin {pin!r} is an implementation id or "
            "'library:implementation_id'")
    if len(parts) == 1:
        return None, parts[0]
    return parts[0], parts[1]


def latest_time_for(
    registry: CalibrationRegistry,
    key: PhysicalKey,
    *,
    framework: str,
    kernel_impl: Optional[str] = None,
    kernel_dispatch: Optional[Mapping[str, Any]] = None,
    current_peak_tflops_at: PeakAt,
    current_peak_bw_gbps: float,
    flops_for_anomaly: Optional[float] = None,
    access_bytes_for_anomaly: Optional[float] = None,
    source_anchor: Optional[Dict[str, Any]] = None,
) -> Optional[MeasuredTiming]:
    """The implementation a serving framework explicitly dispatches to, at
    the newest release the corpus measured that exact identity.

    `framework` names the DECLARATION holder. It is not a corpus-row filter:
    rows carry their provider library in `key.framework`. A declared dispatch
    names that provider library and the complete `Provider.impl_id` (including
    parameter suffixes); selection compares both by equality, never by base or
    prefix. Without an explicit user pin, an undeclared, no-active, or
    ambiguous dispatch returns no measurement. A missing exact row is a corpus
    gap for `select_time` to classify; this function only selects.

    The provider version is NOT part of the dispatch identity. The newest
    provider `framework_version` wins (then the newest `date`), and the
    badge says which release it was measured on (`framework_version`,
    `selection_mode: latest`) — an older measurement is never presented as a
    newer one.
    """
    state = None if kernel_dispatch is None else kernel_dispatch.get("state")
    valid_states = {None, "declared", "undeclared", "no_active", "ambiguous"}
    if state not in valid_states:
        raise ValueError(
            f"unknown kernel dispatch state {state!r}; one of "
            f"{sorted(s for s in valid_states if s is not None)}")
    declared_library = kernel_dispatch.get("library") if kernel_dispatch else None
    declared_impl = kernel_dispatch.get("impl_id") if kernel_dispatch else None
    if (state == "declared"
            and (not isinstance(declared_library, str) or not declared_library
                 or not isinstance(declared_impl, str) or not declared_impl)):
        raise ValueError(
            "a declared kernel dispatch names both its provider library and "
            "complete implementation id")
    if kernel_impl is None and state != "declared":
        return None
    # A user pin wins over the framework declaration. A qualified pin names
    # the same (library, complete impl) pair identity a declaration names;
    # when it names the active pair, its badge may carry the binding anchor.
    pinned_library: Optional[str] = None
    pinned_impl: Optional[str] = None
    if kernel_impl is not None:
        pinned_library, pinned_impl = _parse_kernel_pin(kernel_impl)
    pinned_to_declaration = (
        kernel_impl is not None and state == "declared"
        and declared_library is not None
        and pinned_impl == declared_impl
        and (pinned_library is None or pinned_library == declared_library))
    wanted_library = (
        declared_library if kernel_impl is None or pinned_to_declaration
        else pinned_library)
    wanted_impl = pinned_impl if pinned_impl is not None else declared_impl
    rows = [r for r in registry.get_physical(key)
            if r.key.kernel_impl == wanted_impl
            and (wanted_library is None
                 or r.key.framework == wanted_library)]
    if (kernel_impl is not None and pinned_library is None
            and len({r.key.framework for r in rows}) > 1):
        raise ValueError(
            f"kernel pin {kernel_impl!r} names implementations in more than "
            "one provider library; write 'library:implementation_id'")
    if not rows:
        return None
    row = max(rows, key=lambda r: (_version_order(r.key.framework_version),
                                   r.date))
    t = _build_measured_timing(
        row,
        current_peak_tflops_at=current_peak_tflops_at,
        current_peak_bw_gbps=current_peak_bw_gbps,
        flops_for_anomaly=flops_for_anomaly,
        access_bytes_for_anomaly=access_bytes_for_anomaly,
        source_anchor=source_anchor,
    )
    badge = dict(t.source_badge)
    badge["selection_mode"] = "latest"
    badge["serving_framework"] = framework
    if pinned_to_declaration or kernel_impl is None:
        badge["declared_kernel_impl"] = declared_impl
    else:
        badge["kernel_pin"] = kernel_impl
    versions = sorted({r.key.framework_version for r in rows},
                      key=_version_order, reverse=True)
    if len(versions) > 1:
        badge["framework_versions_measured"] = ",".join(versions)
    return MeasuredTiming(t_op_s=t.t_op_s, source=t.source,
                          source_badge=badge, anomaly_reason=t.anomaly_reason,
                          utilization=t.utilization)


def physical_time_for(
    registry: CalibrationRegistry,
    key: PhysicalKey,
    *,
    current_peak_tflops_at: PeakAt,
    current_peak_bw_gbps: float,
    flops_for_anomaly: Optional[float] = None,
    access_bytes_for_anomaly: Optional[float] = None,
    source_anchor: Optional[Dict[str, Any]] = None,
) -> Optional[MeasuredTiming]:
    """Envelope lookup: the fastest measured impl for a PHYSICAL op.

    Scans every row sharing the physical key (all libraries, all versions,
    all source models that hit this op+shape+dtype+gpu+phase),
    peak-drift-compensates each, and returns the MINIMUM `t_op_s` — the
    best a well-tuned serving stack could achieve on this hardware. The
    winner's library / version / kernel_impl are surfaced on the badge
    (provenance), along with how many implementations competed. `None`
    when no measurement exists for the physical key.
    """
    rows = registry.get_physical(key)
    if not rows:
        return None
    best: Optional[MeasuredTiming] = None
    best_row: Optional[_Row] = None
    for row in rows:
        t = _build_measured_timing(
            row,
            current_peak_tflops_at=current_peak_tflops_at,
            current_peak_bw_gbps=current_peak_bw_gbps,
            flops_for_anomaly=flops_for_anomaly,
            access_bytes_for_anomaly=access_bytes_for_anomaly,
            source_anchor=source_anchor,
        )
        if best is None or t.t_op_s < best.t_op_s:
            best = t
            best_row = row
    if best is None or best_row is None:
        return None
    badge = dict(best.source_badge)
    badge["selection_mode"] = "envelope"
    badge["envelope_candidates"] = str(len(rows))
    badge["envelope_winner"] = (
        f"{best_row.key.framework}@{best_row.key.framework_version}"
        f"/{best_row.key.kernel_impl}"
    )
    badge["envelope_winner_model"] = best_row.key.model
    return MeasuredTiming(
        t_op_s=best.t_op_s,
        source=best.source,
        source_badge=badge,
        anomaly_reason=best.anomaly_reason,
        utilization=best.utilization,
    )


def select_time(
    registry: CalibrationRegistry,
    key: PhysicalKey,
    *,
    strategy: str,
    framework: str,
    kernel_impl: Optional[str] = None,
    kernel_dispatch: Optional[Mapping[str, Any]] = None,
    current_peak_tflops_at: PeakAt,
    current_peak_bw_gbps: float,
    flops_for_anomaly: Optional[float] = None,
    access_bytes_for_anomaly: Optional[float] = None,
    source_anchor: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[MeasuredTiming], str]:
    """The measurement one row reads at `key`, by `strategy`, and its
    coverage state (`auto_sweep.STATES`): `latest` — the declared provider
    implementation at its newest measured release (`latest_time_for`,
    with an optional explicit `kernel_impl` pin) — or `envelope` — the fastest
    kernel across libraries (`physical_time_for`). The one selection an
    estimate's overlay and a
    sweep's points both make.

    Under latest without a pin, an undeclared, no-active, or multi-active
    dispatch is `ambiguous` — a declaration gap, not something a new corpus
    row can fix. A unique declared identity with no matching row is `missing`.
    Envelope ignores declarations entirely.
    """
    common = dict(current_peak_tflops_at=current_peak_tflops_at,
                  current_peak_bw_gbps=current_peak_bw_gbps,
                  flops_for_anomaly=flops_for_anomaly,
                  access_bytes_for_anomaly=access_bytes_for_anomaly,
                  source_anchor=source_anchor)
    if strategy == "envelope":
        got = physical_time_for(registry, key, **common)
        return got, (MEASURED if got is not None else MISSING)
    if strategy != "latest":
        raise ValueError(f"strategy must be 'latest' or 'envelope'; got {strategy!r}")
    got = latest_time_for(registry, key, framework=framework,
                          kernel_impl=kernel_impl,
                          kernel_dispatch=kernel_dispatch, **common)
    if got is not None:
        return got, MEASURED
    if kernel_impl is not None:
        return None, MISSING
    state = None if kernel_dispatch is None else kernel_dispatch.get("state")
    if state == "declared":
        return None, MISSING
    return None, AMBIGUOUS
