"""A precision profile read against one program (`ResolvedPrecision`).

The profile (`arch.precision_profile`) is model-free; here each object of
one program — a weight tensor, an edge, a matmul, a cached tensor — gets
its conclusion: the most specific rule that selects it, else what the model
ships. The evaluator resolves `Deployment.precision` against the program it
walks and leaves the result on `Analysis.precision`; every consumer asks it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

from backend.arch import cache_families, roles as roles_mod
from backend.arch.activation_formats import (ACTIVATION_FORMATS,
                                              ActivationFormat)
from backend.arch.cache_formats import CacheFormat
from backend.arch.dtypes import (canonical, display_name, dtype_bytes,
                                 peak_kind, served_dtype)
from backend.arch.precision_profile import (  # noqa: F401  (re-exported)
    AS_SHIPPED, BACKBONE, OUTSIDE_ROLES, PRESETS, PROFILE_SCHEMA_VERSION,
    PrecisionError, PrecisionProfile, Rule, presets, scheme_entry,
    vocabulary)
from .formats import Formats, TensorFormat, _bits
from backend.code_trace.memo import memo_property


class ResolvedPrecision:
    """A profile read against one program: each object's conclusion.

    The walk's input (`Deployment.precision` resolved by the evaluator) and
    the one place every consumer asks what precision an object runs at —
    edge widths, matmul dtypes, cache formats, weight intents.
    """

    def __init__(self, profile: PrecisionProfile, program: Any):
        self.profile = profile
        self.program = program
        #: Launch declarations the deployed framework selected for this walk
        #: (`FrameworkBinding.precision_launches_for`). Empty for a walk with
        #: no described deployment; those walks answer operand *semantics*,
        #: and never infer an instruction family from those semantics.
        self.launches: Sequence[Any] = ()
        #: The card's MMA peak families, or `None` when no capability table
        #: participates in this (typically cardless unit-test) resolution.
        self.launch_peak_families: Optional[Tuple[str, ...]] = None
        #: Same-scheme candidates whose required deployment facts did not
        #: hold. They are selector audit gaps, never fallback launches.
        self.unavailable_launches: Tuple[Any, ...] = ()
        #: `(roles, dtype)` pairs the kernels a deployed framework
        #: dispatches to REQUIRE a declared role's output at
        #: (`FrameworkBinding.edge_dtypes_for`), as plain data — this package
        #: holds no binding. Empty for a walk with no described deployment,
        #: and for every role no entry names: those carry no requirement and
        #: keep the width the checkpoint implies (`edge_dtype`).
        self.required_edge_dtypes: Sequence[Tuple[Tuple[str, ...], str]] = ()
        resolved = program.resolved
        dtype, source = resolved.shipped_dtype()
        if dtype is None:
            # No header, no config dtype, no declaration: the shipped
            # precision is unknown, and it is the checkpoint's priced
            # default for weights, edges and cache alike (`FormatPolicy.of`),
            # which refuses rather than assume a width.
            dtype = Formats.of(program).saved.default_dtype
            source = "quant_context"
        #: What the checkpoint STORES the tensors no scheme covers at.
        self.shipped_model_dtype = canonical(dtype)
        self.model_dtype_source = source
        #: What a default deployment RUNS them at (`dtypes.served_dtype`):
        #: the stored dtype for a 16-bit float, bfloat16 for anything wider,
        #: because that is what both frameworks load such a checkpoint as.
        #: As shipped means as served — the walk prices the deployment that
        #: exists, and no framework runs a float32 checkpoint at float32.
        self.model_dtype = served_dtype(self.shipped_model_dtype)
        self._stacks = {s.name: s for s in program.stacks}

    # Both `with_*` below carry a fact a DEPLOYED FRAMEWORK owns onto a walk
    # that has already run. They move together: when launches are resolved
    # with the profile inside the walk instead (so every consumer of
    # `Analysis.precision` — the panel included — reads one answer), this
    # one goes the same way, and both methods disappear with the
    # `dataclasses.replace` at `serving.build_serving_layers`.

    def with_launches(self, launches: Sequence[Any], *,
                      peak_families: Optional[Tuple[str, ...]] = None,
                      unavailable_launches: Sequence[Any] = ()
                      ) -> "ResolvedPrecision":
        """The same resolution under the launches a deployed framework takes.

        The caller owns the facts: they come from a framework binding that
        has already been answered for a deployment. A launch may dequantise a
        narrow weight in software (Humming's Hopper WGMMA path) or feed a
        narrower hardware MMA (DeepGEMM's Blackwell FP8×FP4 path); operand
        widths cannot distinguish the two. `peak_families` is the card's
        `gpu_specs.peak_families`; when supplied, the first ordered candidate
        supported under its declared family semantics is the launch taken.
        """
        out = object.__new__(ResolvedPrecision)
        out.__dict__.update(self.__dict__)
        out.launches = tuple(launches)
        out.launch_peak_families = (None if peak_families is None
                                    else tuple(peak_families))
        out.unavailable_launches = tuple(unavailable_launches)
        return out

    def with_required_edge_dtypes(
            self, declared: Sequence[Tuple[Sequence[str], str]]
    ) -> "ResolvedPrecision":
        """The same resolution under the output widths the kernels a deployed
        framework dispatches to require (`FrameworkBinding.edge_dtypes_for`).

        The caller owns the facts, exactly as `with_launches`' caller does:
        they come from a binding that has already answered each entry's
        `when:` against this deployment and this walk. Materialized once,
        before anything reads an edge, so the edge's bytes, the row's
        `act_dtype`, the cell's key and the oracle's inputs are one answer
        and not four.
        """
        out = object.__new__(ResolvedPrecision)
        out.__dict__.update(self.__dict__)
        out.required_edge_dtypes = tuple(
            (tuple(roles), canonical(dtype)) for roles, dtype in declared)
        return out

    # -- the object coordinates ---------------------------------------------

    def stack_value(self, stack_name: str) -> str:
        """`backbone` / `aux:<kind>` for an actual stack name — read off the
        stack's declaration, never its name."""
        stack = self._stacks.get(stack_name)
        if stack is None:
            if stack_name == "":
                return BACKBONE
            raise ValueError(f"{self.program.spec.name}: no stack "
                             f"{stack_name!r}")
        if not stack.is_aux:
            return BACKBONE
        if not stack.aux_kind:
            raise ValueError(
                f"{self.program.spec.name}: aux stack {stack_name!r} declares "
                f"no `aux_kind`, so a precision rule cannot select it")
        return f"aux:{stack.aux_kind}"

    def role_value(self, role: Optional[str]) -> str:
        """The registered role a node's precision is chosen by: its own,
        else the role its slot's component declares, else the broader roles
        it refines (`roles.selection_order`). No role is `outside_roles`."""
        if not role:
            return OUTSIDE_ROLES
        for candidate in roles_mod.selection_order(
                role, self.program.role_aliases):
            if roles_mod.is_registered(candidate):
                return candidate
        return role

    def _conclusion(self, table: str, point: Mapping[str, str]
                    ) -> Optional[Rule]:
        hits = [r for r in self.profile.tables()[table] if r.selects(point)]
        if not hits:
            return None
        best = min(hits, key=lambda r: sum(len(v) for _d, v in r.sigma))
        bd = best.dims()
        for r in hits:
            if not all(bd[d] <= r.dims()[d] for d in bd):
                raise AssertionError(
                    f"precision rules {best.describe()} and {r.describe()} "
                    f"are not ordered on {dict(point)}; the canonical set "
                    f"is not closed")
        return best

    # -- edges ----------------------------------------------------------------

    def edge_dtype(self, role: Optional[str], stack_name: str = "") -> str:
        """The dtype of a tensor produced at `role` in `stack_name`.

        Three sources, in order. The deployed framework's declaration
        (`required_edge_dtypes`) first, because the kernel downstream exists
        only at that width and a narrower edge would be a different kernel;
        it wins over a profile rule for the same reason a launch wins over a
        compute pin in `compute_dtype`. Then the profile's activation table.
        Then — the ASSUMPTION every undeclared edge rests on — the width the
        checkpoint implies (`model_dtype`), on which no kernel we serve
        imposes a requirement of its own.

        WHEN the declaration arrives matters: it is materialized onto the
        analysis after the walk (`serving.build_serving_layers`), so a reader
        during the walk, and every reader of the framework-free analysis —
        `cell.physical()`, `Analysis.precision_effective()` and the precision
        panel behind it, `layout.materializer` — sees the checkpoint width
        whether or not a binding declares one. That gap is named in
        `docs/inference_cost_model.md`; it is not an absence of declarations.
        """
        required = self._required_edge_dtype(role)
        if required is not None:
            return required
        if not self.profile.activations:
            return self.model_dtype
        rule = self._conclusion("activations", {
            "roles": self.role_value(role),
            "stacks": self.stack_value(stack_name)})
        if rule is None:
            return self.model_dtype
        return json.loads(rule.format)["dtype"]

    def _required_edge_dtype(self, role: Optional[str]) -> Optional[str]:
        """The width the kernel a deployed framework dispatches to requires
        this role's output at, None where none is declared.

        Matched on the role coordinate `role_value` resolves, which for a
        registered role is that role itself: an entry reaches the roles it
        names and no others. (`role_value` does climb `refines:` for an
        alias or an unregistered role; the schema refuses those before a
        walk, so no declaration rides that climb today.) A role's output
        width is a separate fact from its parent's (DeepSeek-V4's `router`
        writes logits while `router_hash_table` holds a looked-up table),
        so inheriting one would state a requirement nobody audited.
        """
        if not self.required_edge_dtypes:
            return None
        asked = self.role_value(role)
        for roles, dtype in self.required_edge_dtypes:
            if asked in roles:
                return dtype
        return None

    def edge_bytes(self, role: Optional[str], stack_name: str = "") -> float:
        return float(dtype_bytes(self.edge_dtype(role, stack_name)))

    @property
    def default_edge_bytes(self) -> float:
        """The width of an edge no leaf produces — a model input, a block's
        incoming residual — and of a collective moving the residual
        stream: a model-level edge of the backbone, the model's own dtype
        unless an activation rule selects `outside_roles`."""
        return self.edge_bytes(None, "")

    # -- matmuls ----------------------------------------------------------------

    def _declared_compute_dtype(self, role: Optional[str], stack_name: str,
                                formats: Sequence[Any],
                                operand: Optional[str] = None) -> str:
        """The dtype a leaf's kernel computes in, whose peak column it runs
        at: the dtype of the operand it reads, unless a resident weight
        format quantises the activation operand (`activation:` of the
        checkpoint's own scheme, or the `activation_options` a deployment
        chose), in which case the wider of that activation format's and the
        weight's element dtype. A compute pin (§3.7) answers instead of
        both.

        `operand` is the dtype the leaf's operand edge carries as traffic
        billed it (`access._operand_dtype`) — fp32 indexer logits are read
        as fp32 by the in-place scale over them, whatever the role's
        activation dtype. None (no data operand) is the role's edge dtype."""
        pin = self._compute_pin(role, stack_name)
        if pin is not None:
            # A pin does not touch the weights (§3.7): a matmul cannot run
            # narrower than the operand it streams. Narrower weights are
            # legal — the kernel dequantises them (W4A16).
            wider = sorted({rep.resident.logical_dtype for rep in formats
                            if dtype_bytes(rep.resident.logical_dtype)
                            > dtype_bytes(pin[0])})
            if wider:
                raise PrecisionError([{
                    "path": f"compute[{self.role_value(role)}]",
                    "message": (
                        f"the compute rule pins {self.role_value(role)} to "
                        f"{display_name(pin[0])}, and this model stores those "
                        f"weights at {', '.join(display_name(w) for w in wider)}"
                        f" — a pin does not convert weights"),
                    "fix": ("add a weight rule putting these tensors in a "
                            "scheme of the pinned width, or pin a width the "
                            "weights already have")}])
            return pin[0]
        edge = operand or self.edge_dtype(role, stack_name)
        chosen: Optional[str] = None
        for rep in formats:
            act = rep.resident.activation_format
            if act is None:
                continue
            data = ACTIVATION_FORMATS[act].dtype
            weight = rep.resident.logical_dtype
            wide = max((data, weight), key=dtype_bytes)
            if chosen is None or dtype_bytes(wide) > dtype_bytes(chosen):
                chosen = wide
        return chosen or edge

    def compute_dtype(self, role: Optional[str], stack_name: str,
                      formats: Sequence[Any],
                      operand: Optional[str] = None) -> str:
        launch = self._launch_for(role, stack_name, formats)
        if launch is not None and launch.compute_dtype is not None:
            return launch.compute_dtype
        return self._declared_compute_dtype(
            role, stack_name, formats, operand)

    def _declared_required_format(self, role: Optional[str], stack_name: str,
                                  formats: Sequence[Any]
                                  ) -> Optional[ActivationFormat]:
        """The activation format a leaf's matmul actually eats (R, §3.7):
        the compute pin that selects it, else the one its resident weight
        format declares (`activation:` for a checkpoint's own scheme, the
        `activation_options` choice for a deployment's). None — "as it
        arrives" — for every weight-only kernel and every leaf with no
        quantised weight at all."""
        pin = self._compute_pin(role, stack_name)
        if pin is not None:
            return pin[1]
        best: Optional[ActivationFormat] = None
        for rep in formats:
            name = rep.resident.activation_format
            if name is None:
                continue
            fmt = ACTIVATION_FORMATS[name]
            if best is None or dtype_bytes(fmt.dtype) > dtype_bytes(best.dtype):
                best = fmt
        return best

    def required_format(self, role: Optional[str], stack_name: str,
                        formats: Sequence[Any]) -> Optional[ActivationFormat]:
        launch = self._launch_for(role, stack_name, formats)
        if launch is not None:
            if launch.activation_format is None:
                return None
            if launch.activation_format != "requested":
                return ACTIVATION_FORMATS[launch.activation_format]
        return self._declared_required_format(role, stack_name, formats)

    def _launch_for(self, role: Optional[str], stack_name: str,
                    formats: Sequence[Any]) -> Optional[Any]:
        """The framework launch that selected this matmul, if one is declared.

        A launch selector names the checkpoint scheme and requested activation
        treatment. It is answered only from declarations supplied by the
        deployed binding; no width comparison can synthesize one.
        """
        schemes = {rep.resident.scheme.name for rep in formats
                   if rep.resident.scheme is not None}
        requested = self._declared_required_format(role, stack_name, formats)
        requested_name = requested.name if requested is not None else None
        for launch in self._matching_launches(schemes, requested_name):
            if self._launch_is_supported(launch):
                return launch
        return None

    def _matching_launches(self, schemes: set,
                           requested_name: Optional[str]) -> Tuple[Any, ...]:
        """Ordered candidates for one scheme/request coordinate.

        The coordinate is deliberately lexical: stack and requested activation
        semantics choose the candidate set, then framework file order supplies
        its preference. No condition here inspects operand widths.
        """
        return tuple(launch for launch in self.launches
                     if schemes.intersection(launch.schemes)
                     and (not launch.requested_activation_formats
                          or requested_name in
                          launch.requested_activation_formats))

    def _launch_is_supported(self, launch: Any) -> bool:
        """Whether the card supports the launch's declared MMA semantics.

        A cardless resolution (`peak_families is None`) keeps unit fixtures
        backward-compatible. A described card is asked only through the
        capability table, never through a compute-capability family guess.
        A normal launch needs every declared family; a promotion ladder,
        explicitly audited on the launch, needs a non-empty intersection and
        lets pricing name the widest supported rung.
        """
        if self.launch_peak_families is None:
            return True
        families = self.launch_peak_families
        available = (peak_kind(dtype) in families
                     for dtype in launch.mma_dtypes)
        return any(available) if launch.promotion_ladder else all(available)

    def _launch_missing_mma_dtypes(self, launch: Any) -> Tuple[str, ...]:
        """Declared MMA families without a card column, for a skipped launch."""
        if self.launch_peak_families is None:
            return ()
        return tuple(dtype for dtype in launch.mma_dtypes
                     if peak_kind(dtype) not in self.launch_peak_families)

    def launch_gaps(self, role: Optional[str], stack_name: str,
                    formats: Sequence[Any]) -> Tuple[str, ...]:
        """Why ordered launch candidates were skipped, most specific first.

        A skipped explicit-backend candidate is recorded even when a later
        audited fallback wins: it is an alternative selector gap, not a
        request to borrow that alternative's kernel. When no candidate is
        viable, the first reason is the row's launch gap.
        """
        schemes = {rep.resident.scheme.name for rep in formats
                   if rep.resident.scheme is not None}
        requested = self._declared_required_format(role, stack_name, formats)
        requested_name = requested.name if requested is not None else None
        matches = self._matching_launches(schemes, requested_name)
        unavailable = tuple(
            launch for launch in self.unavailable_launches
            if schemes.intersection(launch.schemes)
            and (not launch.requested_activation_formats
                 or requested_name in launch.requested_activation_formats))
        out = list(
            f"missing_card_mma:{launch.name or index}:"
            + "+".join(self._launch_missing_mma_dtypes(launch))
            for index, launch in enumerate(
                (l for l in matches if not self._launch_is_supported(l)), 1))
        out.extend(
            f"missing_deployment_fact:{launch.name or index}"
            for index, launch in enumerate(unavailable, 1))
        return tuple(out)

    def launch_gap(self, role: Optional[str], stack_name: str,
                   formats: Sequence[Any]) -> Optional[str]:
        """The explicit gap when no launch candidate is viable, else None."""
        if self._launch_for(role, stack_name, formats) is not None:
            return None
        gaps = self.launch_gaps(role, stack_name, formats)
        return gaps[0] if gaps else None

    def mma_dtypes(self, role: Optional[str], stack_name: str,
                   formats: Sequence[Any],
                   operand: Optional[str] = None) -> Tuple[str, ...]:
        """The selected launch's declared MMA-family vocabulary.

        This is a *hardware* fact about the launch, not a restatement of
        operand widths. A block-FP8 activation may feed Humming's Hopper
        WGMMA after software dequantisation of an MXFP4 weight, while the
        same two logical operands feed DeepGEMM's FP4 MMA on Blackwell.
        Consequently a narrow resident weight is added only when a deployed
        framework launch explicitly says its family belongs to the launch.
        For a normal launch every member is required simultaneously; for an
        explicitly declared promotion ladder, the members are its audited
        rungs and downstream pricing chooses the widest card-supported one.

        No launch declaration means only the compute family is asserted —
        never that a narrower weight joined the hardware MMA. Such a walk
        has no deployed framework answer and must not be mistaken for a box
        plan (`kernel_bench.plan`).
        """
        compute = self.compute_dtype(role, stack_name, formats, operand)
        launch = self._launch_for(role, stack_name, formats)
        if launch is not None:
            return launch.mma_dtypes
        return (compute,)

    def launch_promotion_ladder(self, role: Optional[str], stack_name: str,
                                formats: Sequence[Any]) -> bool:
        """Whether the selected launch audited a promotion ladder.

        This carries the launch's family semantics to access intact. It is
        never inferred from a name, an operand width, or a missing card
        column: only the launch declaration can say that a supported subset
        runs the same audited framework path.
        """
        launch = self._launch_for(role, stack_name, formats)
        return launch is not None and launch.promotion_ladder

    def _compute_pin(self, role: Optional[str], stack_name: str
                     ) -> Optional[Tuple[str, Optional[ActivationFormat]]]:
        """`(dtype, required format)` the profile's compute table pins this
        leaf's matmul to, None when no rule selects it. A pin of 2 bytes or
        more needs no format: the operand arrives at that dtype with no
        scale (§3.7)."""
        if not self.profile.compute:
            return None
        rule = self._conclusion("compute", {
            "roles": self.role_value(role),
            "stacks": self.stack_value(stack_name)})
        if rule is None:
            return None
        pinned = json.loads(rule.format)
        name = pinned.get("activation_format")
        return (canonical(pinned["dtype"]),
                None if name is None else ACTIVATION_FORMATS[name])

    def conversion(self, role: Optional[str], stack_name: str,
                   formats: Sequence[Any],
                   edge: Optional[ActivationFormat] = None
                   ) -> Optional["Conversion"]:
        """The format change a leaf's matmul pays for on its input (§3.7,
        invariant 1's consumer half): None when the edge arrives in the
        format the kernel eats, else the one conversion it runs.

        `edge` is the format the producer wrote, when it wrote a quantised
        one (a framework's fused quantise, `traffic`'s absorption); None
        means a plain edge at `edge_dtype`.
        """
        return self.conversion_to(
            self.required_format(role, stack_name, formats), role,
            stack_name, edge)

    def conversion_to(self, required: Optional[ActivationFormat],
                      role: Optional[str], stack_name: str,
                      edge: Optional[ActivationFormat] = None
                      ) -> Optional["Conversion"]:
        """The conversion an operand pays for to arrive in `required` (None:
        as it arrives) from the format `edge` its producer wrote: the law
        behind `conversion`, for a requirement read off the leaf's weights
        or off the cached operand it multiplies (`cache.paired_records`)."""
        if edge is not None and required is not None and edge == required:
            return None                      # the same format, byte for byte
        if required is None:
            if edge is None:
                return None                  # plain edge, kernel takes it
            # A quantised edge into a kernel that reads tensors as they are:
            # it is dequantised to the model's own dtype first.
            return Conversion(kind="dequantise", target=None,
                              target_dtype=self.edge_dtype(role, stack_name))
        kind = "quantise" if edge is None else "requantise"
        return Conversion(kind=kind, target=required,
                          target_dtype=required.dtype)

    def compute_dtype_is_chosen(self, role: Optional[str], stack_name: str,
                                formats: Sequence[Any]) -> bool:
        """Whether the deployment CHOSE this leaf's matmul dtype — a compute
        pin, or a weight rule that put the tensor in a scheme of its own —
        rather than reading it off the checkpoint. A card with no column for
        a chosen dtype refuses the estimate (§3.8); one with no column for
        what a checkpoint ships is priced at the card's own widest, because
        that is what a framework does with it."""
        if self._compute_pin(role, stack_name) is not None:
            return True
        return any(rep.resident.activation_format is not None
                   and rep.resident.chosen_by_profile
                   for rep in formats)

    def conversion_label(self, role: Optional[str], stack_name: str,
                         formats: Sequence[Any]) -> str:
        """What a leaf's input conversion is, named for the row that shows
        it; empty when its kernel eats the edge as it arrives."""
        conversion = self.conversion(role, stack_name, formats)
        return "" if conversion is None else conversion.label()

    # -- weights ----------------------------------------------------------------

    @property
    def served_as_stored(self) -> bool:
        """Whether the dtype the checkpoint stores its own tensors at is the
        one a framework runs them at. False for a checkpoint wider than 16
        bits, whose weights are loaded narrower than they are saved."""
        return self.model_dtype == self.shipped_model_dtype

    @memo_property
    def weight_intent(self) -> Optional["WeightIntent"]:
        """The weight table asked per tensor, and — for a checkpoint no
        framework runs at its stored width — the serving downcast, which is
        an intent for the same reason a profile rule is: it is what the
        deployment runs, against a `saved` that stays the checkpoint's."""
        return (WeightIntent(self)
                if self.profile.weights or not self.served_as_stored
                else None)

    # -- cache ------------------------------------------------------------------

    def cache_format(self, kind: Optional[str], stack_name: str,
                     declared: Optional[CacheFormat]) -> CacheFormat:
        """A cached tensor's format: the profile's rule for its slot kind,
        else the format it ships at (`shipped_cache_format`)."""
        if kind is not None and self.profile.cache:
            rule = self._conclusion("cache", {
                "slots": kind, "stacks": self.stack_value(stack_name)})
            if rule is not None and kind in cache_families.formattable_kinds():
                return CacheFormat.parse(json.loads(rule.format), "profile")
        return self.shipped_cache_format(kind, declared)

    def shipped_cache_format(self, kind: Optional[str],
                             declared: Optional[CacheFormat]) -> CacheFormat:
        """The format the model ships a cached tensor at: the model's own
        declaration for its slot kind (`quant_context.cache_formats`), else
        the component entry's (`declared`), else the model's dtype with no
        scale."""
        if kind is not None and kind in self._model_cache_formats:
            return self._model_cache_formats[kind]
        return declared if declared is not None else CacheFormat.plain(
            self.model_dtype)

    @memo_property
    def _model_cache_formats(self) -> Mapping[str, CacheFormat]:
        quant_context = self.program.spec.quant_context
        if quant_context is None:
            return {}
        return {kind: CacheFormat.parse(
                    fmt, f"quant_context.cache_formats.{kind}")
                for kind, fmt in quant_context.cache_formats.items()}

@dataclass(frozen=True)
class Conversion:
    """One format change on a matmul's input operand, billed to the kernel
    that needs it (§2.3 invariant 1).

    The bytes are the tensor the conversion writes and the matmul then reads
    back: a standalone quantise kernel writes the narrow tensor and its
    scales, and the GEMM streams them in — priced where the launch is
    recorded (`traffic._settle_conversions`), by the one width law of an
    activation record (`traffic._format_width`). The wide read the
    conversion does is the leaf's own activation read, already priced on
    the edge, so it is not counted again here.

    The arithmetic — one absmax reduction and one multiply per element — is
    not in the FLOPs tree: at two elementwise operations over a tensor
    written and read once the intensity is below one FLOP per byte on every
    card in `gpu_specs.csv`, so the roofline reads this as the memory it is.
    """

    #: `quantise` (a plain edge into a quantised operand), `requantise` (one
    #: quantised format into another) or `dequantise` (a quantised edge into
    #: a kernel that eats tensors as they arrive).
    kind: str
    #: The activation format written, None when the conversion lands on a
    #: plain dtype (dequantise).
    target: Optional[ActivationFormat]
    #: The element dtype written.
    target_dtype: str

    def label(self) -> str:
        target = (self.target.label() if self.target is not None
                  else display_name(self.target_dtype))
        return f"{self.kind} -> {target}"


class WeightIntent:
    """What the deployment runs one stored tensor at, or None — run it as
    the checkpoint saved it.

    Two sources, in that order. The profile's weight table, asked per
    tensor: only tensors a scheme can describe are objects of a weight rule
    — a matrix, a head bank, a lookup table's rows
    (`StoredTensor.as_dims`); a vector (norm gain, bias) or buffer is never
    selected. Then the serving downcast (`_served`), which applies to every
    tensor stored at the model's own dtype, matrix or not."""

    def __init__(self, resolved: ResolvedPrecision):
        self.resolved = resolved

    def for_node(self, saved_policy: Any, stored: Any, role: Optional[str],
                 stack: Optional[str], node_dims: Optional[Mapping[str, Any]],
                 saved: Any) -> Optional[Any]:
        """`saved` is the format this segment is stored at, and it is
        required: the one caller (`Formats.resolve`) has it in hand, and the
        serving downcast is a statement ABOUT it."""
        rule = self._from_rule(saved_policy, stored, role, stack, node_dims)
        return rule if rule is not None else self._served(saved)

    def _served(self, saved: Any) -> Optional[Any]:
        """The serving downcast as this tensor's intent: a tensor the
        checkpoint stores at the model's own dtype is loaded at the dtype a
        framework serves that dtype at (`ResolvedPrecision.model_dtype`).
        None when the two are the same, which is every 16-bit checkpoint.

        Asked of every stored tensor and not only of the matrices a scheme
        describes, because the framework creates every unquantised parameter
        at the model dtype — a norm's gain is loaded narrower too. Asked of
        none a scheme covers, and that needs no test: a scheme's storage
        dtype is not the model's, so the comparison below excludes it, as it
        excludes a tensor an `overrides` row puts at a dtype of its own.
        Those are the per-tensor stored-vs-loaded widths the cost model
        leaves unmodelled and says so; this is the MODEL's dtype, which
        every framework decides once from the config."""
        if self.resolved.served_as_stored:
            return None
        if canonical(saved.logical_dtype) != self.resolved.shipped_model_dtype:
            return None
        served = self.resolved.model_dtype
        # Not `chosen_by_profile`: nothing was chosen. This is what the
        # checkpoint ships as, once the framework that serves it is the one
        # reading it, so an estimate is not refused for want of a column the
        # deployment asked for (`compute_dtype_is_chosen`).
        return TensorFormat(storage_dtype=served, num_bits=_bits(served),
                            selected_by="served")

    def _from_rule(self, saved_policy: Any, stored: Any, role: Optional[str],
                   stack: Optional[str], node_dims: Optional[Mapping[str, Any]]
                   ) -> Optional[Any]:
        if stored is None or not stored.as_dims():
            return None
        if stored.kind.value in ("metadata", "buffer"):
            return None
        rule = self.resolved._conclusion("weights", {
            "roles": self.resolved.role_value(role),
            "stacks": self.resolved.stack_value(stack or "")})
        if rule is None:
            return None
        fmt = json.loads(rule.format)
        why = f"precision:{self.resolved.profile.key}"
        if "dtype" in fmt:
            return TensorFormat(storage_dtype=fmt["dtype"],
                                num_bits=_bits(fmt["dtype"]),
                                selected_by=why, chosen_by_profile=True)
        entry = scheme_entry(fmt)
        out = saved_policy._from_scheme(fmt["scheme"], entry,
                                        stored.as_dims(), why, stored=stored,
                                        node_dims=dict(node_dims or {}))
        if out.scheme is None:
            # The scheme does not describe this tensor (not-a-matrix).
            return None
        out = replace(out, chosen_by_profile=True)
        options = dict(out.scheme.activation_options)
        if options:
            # The deployment chose how this scheme treats the activation
            # operand (`w4a4` / `w4a16`); the rule's `activation` is one of
            # the scheme's own options (validated when the profile was
            # canonicalised).
            return replace(out, activation_format=options.get(
                fmt.get("activation")))
        # A scheme with no deployment choice keeps the treatment its own
        # kernel does (`activation:`, stamped by `_from_scheme` against these
        # fields) — running a BF16 tensor as FP8 block-128 runs the FP8
        # kernel, activation quantisation included.
        return out
