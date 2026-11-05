"""Claim checks — the A-class anchors in `claims/components/`, judged at
every call of the component in this model's own analysis.

`claims/components/<identity>.yaml` declares KV facts read from the modeling
source, and the loader attaches them to the component. Validate analyses the
model once and reads, at each call of a component that carries claims, the
cache that call grew — from the inputs the model hands it, in the dims the
model binds there — and judges the claims against it. A mismatch means the
SPEC is wrong: the axiom is the judge, never an override. A model that binds
dims per layer calls a component once per binding, and each call is judged at
its own numbers; a component the model never calls is not judged.

Layer-2 discipline: returns `{kind, template, detail}` warnings, never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

import backend.arch.cache_families as cache_families
import backend.arch.roles as roles
from backend.arch.eval.cache import CachedTensor, by_call
from backend.arch.eval.values import CACHE_LEN
from backend.arch.eval.dim import algebra, as_extent, eval_dim_expr
from backend.arch.eval.program import BlockStage, Program
from backend.arch.eval.transforms import Deployment, phase_form
from backend.arch.ops import is_dsa_op
from backend.arch.schema.claims import Axiom, AxiomSet
from backend.arch.structure import iter_reachable_components

from ._finding import Finding


def _at_infinite_context(expr) -> Any:
    """A stream's storage-positions expr with the sequence grown without end:
    `s_kv` substituted by ∞. `Min(W, s_kv)` → W, `Min(n_topk, ceiling(s_kv/4))`
    → n_topk, `s_kv` → ∞, a recurrent state's constant → itself."""
    import sympy as sp
    return algebra(expr).subs(CACHE_LEN, sp.oo)


def _storage_is_bounded(expr) -> Optional[bool]:
    """Does a KV stream's storage-positions expr stay BOUNDED as the sequence
    grows? True = caps (windowed), False = grows (per_token), None = no cache.

    Boundedness, not expr-shape, is the discriminator — so it survives any
    composition (Min(s_window,s_kv), ceiling(s_kv/4),
    Min(n_topk_keys,ceiling(s_kv/4)), …) without a
    bucket-matching rule. This is the only thing that holds up when a single
    attention layer stacks SWA + DSA + linear streams with different laws.

    An expr whose value at ∞ SymPy cannot call finite or infinite raises
    ValueError; the caller reports the claim unjudged."""
    if expr is None:
        return None
    finite = _at_infinite_context(expr).is_finite
    if finite is None:
        raise ValueError(
            f"the storage expression {expr} cannot be judged bounded or "
            f"growing as s_kv→∞. Write the window in closed form, such as "
            f"Min(s_kv, <window dim>), or split it into its own stream in "
            f"cache.entries")
    return finite


def _is_window(tensor: CachedTensor) -> bool:
    """A cached tensor that grows with the sequence up to a cap."""
    return tensor.is_growing and _storage_is_bounded(tensor.stored) is True


def _expr_depends_on_s_kv(expr) -> bool:
    """True if a storage-positions expr still references `s_kv` — i.e. it grows
    or caps with context, so it is NOT a constant recurrent state. A constant
    (e.g. `sp.Integer(1)` for a linear/SSM matrix state) has no free `s_kv`."""
    return CACHE_LEN in algebra(expr).free_symbols


@dataclass(frozen=True)
class _Call:
    """One call of a component, as the model's analysis grew it: the cache
    it keeps and the dims it was bound in."""

    #: The component's identity — what its claims name as their subject. A
    #: template-bound copy (`gqa_qknorm_attention__rope-mrope`) reports the
    #: component it was bound from; a decode form reports its own name.
    template: str
    path: str
    role: Optional[str]
    label: Optional[str]
    axioms: Optional[AxiomSet]
    tensors: Tuple[CachedTensor, ...]
    dims: Mapping[str, Any]
    #: Whether the cache outlives the forward pass. An encoder's does not.
    persists: bool = True
    #: Why this call's cache could not be read, if it could not.
    unread: Optional[str] = None

    @classmethod
    def all_in(cls, resolved, analysis, *, forms=None,
               only_forms: bool = False) -> List["_Call"]:
        """Every component invocation in `analysis` whose cache the claims
        are about — the language model's (`Program.language_nodes`): each
        call node, and each block a stack runs directly as its `template:`.
        An encoder keeps no cache, so no cache claim is judged on it.

        `forms` maps a component to the form `analysis`'s phase runs in its
        place (`phase_form`): there a call keeps its authored name and runs
        the form `Phase.substitute` bound in, so the form's name is reported.
        `only_forms` returns those calls alone — a decode phase's, whose
        other calls were judged in prefill."""
        forms = forms or {}
        spec = analysis.program.spec
        caches = by_call(analysis)
        out: List[_Call] = []
        for node in analysis.program.language_nodes():
            if not node.is_component:
                continue
            if only_forms and node.op not in forms:
                continue
            bg = spec.get_block(node.op)
            cache = caches.get(node)
            out.append(cls(
                template=(forms.get(node.op) or bg.synth_template
                          or node.op),
                path=node.display_path,
                role=bg.role,
                label=node.kv_label,
                axioms=bg.axioms,
                tensors=cache.tensors if cache is not None else (),
                dims=analysis.node_dims.get(node) or {},
                persists=cache.persists if cache is not None else True,
            ))
        if only_forms:
            return out
        for stack in analysis.program.walk_stacks:
            if stack.is_encoder:
                continue
            out.extend(cls.of_stage(analysis, spec, stage)
                       for stage in stack.stages
                       if isinstance(stage, BlockStage))
        return out

    @classmethod
    def of_stage(cls, analysis, spec, stage) -> "_Call":
        """A block a stack runs directly. A cache belongs to the component
        calls inside it, so the block itself keeps nothing; a block that
        declares one anyway is a cache no call grew, which nothing here can
        judge — that is a finding whether or not the block has claims."""
        template = stage.template
        bg = spec.get_block(template.name)
        # The frame the walk ran the block in (`Analysis.block_frame`), as
        # a call's `dims` is the frame the walk ran the call in.
        dims = dict(analysis.block_frame(template))
        return cls(
            template=bg.synth_template or template.name,
            path=stage.name,
            role=bg.role,
            label=bg.kv_label,
            axioms=bg.axioms,
            tensors=(),
            dims=dims,
            unread=(f"block {template.name!r} declares `cache:` but is run by "
                    f"its stack directly, not called; only a component call "
                    f"keeps a cache, so neither its claims nor its cache "
                    f"family's window contract can be judged — move the "
                    f"cached nodes into a component the block calls"
                    if bg.cache is not None else None),
        )

    @property
    def keeps_a_window(self) -> bool:
        """A cached tensor that grows with the sequence up to a cap. A
        recurrent state is bounded without being a window."""
        return any(_is_window(t) for t in self.tensors)

    def unjudged(self, why: Any) -> Finding:
        """This call's cache could not be judged, and why: its cache was not
        read, or a stream's boundedness is undecidable (`_storage_is_bounded`
        raises). Every judgement of the call leaves through here."""
        return Finding(kind="axiom_unjudged", template=self.template,
                       node=self.path,
                       detail=f"{self.template} at {self.path}: {why}")

    def judge(self) -> List[Finding]:
        """Every claim against this call's cache, plus the window contract
        its cache family states."""
        if self.unread is not None:
            return [self.unjudged(self.unread)]
        out: List[Finding] = []
        try:
            for ax in (self.axioms.checks if self.axioms is not None else ()):
                out.extend(self._kv_growth(ax))
            out.extend(self._window_contract())
        except ValueError as exc:
            return [self.unjudged(exc)]
        return out

    def _window_contract(self) -> List[Finding]:
        """A sliding-window cache family keeps windows: every tensor that
        grows with the sequence caps, and at least one does."""
        if self.label not in cache_families.of_kind("windowed"):
            return []
        growing = [t for t in self.tensors if t.is_growing]
        if growing and all(_is_window(t) for t in growing):
            return []
        return [Finding(
            kind="swa_cache_unbounded", template=self.template,
            node=self.path,
            detail=(
                f"component {self.template!r} is labeled {self.label!r} "
                f"(sliding-window KV storage) but at {self.path} its cache "
                f"does not cap as the sequence grows: "
                f"{[(t.name, str(t.stored)) for t in self.tensors]}. "
                f"Window every cached stream in the graph — a "
                f"`sliding_window_kv` over each tensor `cache.entries` "
                f"names, with `dims: {{ s_window: <the model's window> }}`."),
        )]

    def _resolve(self, expr: str) -> Any:
        """A claim's dim expression, as a number in this call's dims. An
        expression the dims leave unbound raises: it has no number."""
        return as_extent(eval_dim_expr(str(expr), self.dims), rounding="floor")

    def _kv_growth(self, ax: Axiom) -> List[Finding]:
        """Verify a KV GROWTH LAW — PER STREAM. A modern attention layer is a
        composition of KV streams (V4 CSA: a windowed SWA cache + a growing
        compressed cache + the indexer's cache), so the law lives on each
        cached tensor, not the component. The discriminator is whether THAT
        stream's storage caps (windowed) or grows (per_token).

        `node` set → check that one stream. `node` omitted → a UNIFORM claim
        over every stream (fires if the component is secretly mixed — telling
        you to split it per-stream). constant_state/none → assert there is no
        growing stream at all (linear/SSM)."""
        out: List[Finding] = []
        name = self.template
        storage = ax.storage
        if storage is None:
            out.append(Finding(kind="axiom_kv_malformed", template=name,
                               node=self.path,
                               detail=f"kv_growth axiom has no `storage` class  [src: {ax.src}]"))
            return out

        expected = None
        if ax.expect is not None:
            try:
                expected = self._resolve(ax.expect)
            except Exception as exc:
                out.append(Finding(kind="axiom_unresolvable", template=name,
                                   node=self.path,
                                   detail=f"axiom expect {ax.expect!r} for node {ax.node!r} "
                                          f"didn't resolve in the dims at {self.path}: {exc}"))
                return out

        cached = list(self.tensors)
        by_name = {cn.name: cn for cn in cached}

        def _fire(kind, msg):
            out.append(Finding(kind=kind, template=name, node=self.path,
                               detail=f"{name} at {self.path}: {msg}  [src: {ax.src}]"))

        if storage == "none":
            # `none` = literally no KV state (a no-op / identity attention).
            if cached:
                _fire("axiom_kv_growth_violation",
                      f"declared none but the engine caches {[cn.name for cn in cached]} "
                      "— the linear/SSM overcount")
            return out
        if storage == "constant_state":
            # A fixed-size recurrent state (linear attention / SSM) IS allowed to
            # cache — that state is real per-request HBM. What it must NOT do is
            # grow: every cached stream's storage has to be s_kv-INDEPENDENT. A
            # stream that grows (s_kv) or merely caps (Min(W, s_kv) — still s_kv-dependent
            # below the window) is the linear/SSM overcount the axiom guards against.
            for cn in cached:
                se = cn.stored
                if se is None or _expr_depends_on_s_kv(se):
                    _fire("axiom_kv_growth_violation",
                          f"stream {cn.name!r} storage={se} depends on s_kv (or is unset) but "
                          "declared constant_state — a growing/windowed cache is the "
                          "linear/SSM overcount")
            return out

        # per_token / windowed — select the target stream(s).
        if ax.node is not None:
            cn = by_name.get(ax.node)
            if cn is None:
                _fire("axiom_kv_uncounted",
                      f"stream {ax.node!r} not in cached set {sorted(by_name)} — declared in cache.entries?")
                return out
            targets = [cn]
        else:
            if not cached:
                _fire("axiom_kv_growth_violation",
                      f"declared {storage} but nothing is cached (no growing KV)")
                return out
            targets = cached

        for cn in targets:
            bounded = _storage_is_bounded(cn.stored)
            se = cn.stored
            if storage == "per_token" and bounded is True:
                _fire("axiom_kv_growth_violation",
                      f"stream {cn.name!r} CAPS (storage={se}) but declared per_token — "
                      "windowed, or a mixed component that must be split per-stream")
            elif storage == "windowed":
                if bounded is False:
                    _fire("axiom_kv_growth_violation",
                          f"stream {cn.name!r} GROWS (storage={se}) but declared windowed — the SWA "
                          "overcount, or a mixed component that must be split per-stream")
                elif ax.bound is not None and bounded is True:
                    try:
                        want = self._resolve(ax.bound)
                        cap = _at_infinite_context(se)
                        if cap != want:
                            _fire("axiom_kv_growth_violation",
                                  f"stream {cn.name!r} caps at {cap} as s_kv→∞, "
                                  f"not at {want} (={ax.bound}); storage={se}")
                    except Exception as exc:
                        _fire("axiom_kv_growth_unverified",
                              f"stream {cn.name!r}: bound {ax.bound!r} could not be "
                              f"checked against storage={se}: {type(exc).__name__}: {exc}")

        # Optional per-position element count (the target stream, or total if uniform).
        if expected is not None:
            total = (as_extent(targets[0].elements, rounding="floor")
                     if ax.node is not None
                     else sum(as_extent(cn.elements, rounding="floor")
                              for cn in targets))
            if total != expected:
                _fire("axiom_kv_violation",
                      f"per-position KV = {total} elements but axiom expects {ax.expect} = {expected}")
        return out


def _check_config_windowing(resolved, calls: List[_Call]) -> List[dict]:
    """Config-vs-spec window contract (model-level, not axiom-declaration-driven).

    If config.json declares an ACTIVE sliding window (`sliding_window` > 0 and
    not disabled via `use_sliding_window: false`), the spec must model at least
    one BOUNDED KV stream — some cached tensor whose storage caps as S→∞.
    Otherwise the spec models every layer as full attention and its KV /
    attention read grow ∝ S where config caps them at the window. That
    "windowed model built as dense" gap moves neither bytes nor tensor shapes,
    and the window contract above fires only once a component is ALREADY
    labeled SWA, so nothing else catches it.

    Boundedness via `_storage_is_bounded` is composition-proof — DSA/CSA streams
    that cap (`Min(s_kv,W)`) count as bounded, dense `s_kv` does not — so this stays
    FP-free across SWA hybrids (gemma-2, MiMo, gpt-oss), DSA (DSv4), and
    window-disabled models (Qwen2.5 `use_sliding_window=false`). Escape hatch:
    explain the exception under `validation_exemptions.sliding_window`."""
    config = resolved.config or {}
    sw = config.get("sliding_window")
    if not (isinstance(sw, int) and sw > 0 and config.get("use_sliding_window") is not False):
        return []
    exemptions = resolved.spec.validation_exemptions
    if exemptions and exemptions.sliding_window:
        return []
    attention = [call for call in calls
                 if roles.kind_of(call.role) == "attention" and call.persists]
    unjudged: List[Finding] = []
    for call in attention:
        try:
            if call.keeps_a_window:
                return []
        except ValueError as exc:
            unjudged.append(call.unjudged(exc))
    if unjudged or not attention:
        return unjudged
    return [Finding(
        kind="windowing_declared_but_unmodelled",
        template="<model>",
        detail=(
            f"config declares an active sliding window (sliding_window={sw}) but no "
            "attention call caches a BOUNDED KV stream — the spec models every "
            "layer as full attention, so KV storage and attention read grow ∝ S where "
            "config caps them at the window. Label the windowed layers "
            "(`cache.family: SWA`) and window their cached streams in the graph "
            "(`sliding_window_kv`), or declare `sliding_window` in "
            "`validation_exemptions.sliding_window` with a reason if the "
            "model truly doesn't use it."
        ),
    )]


def _config_declares_kda(config) -> bool:
    """The config literally names Kimi Delta Attention layers via a `kda_layers`
    key (Kimi-K3, GLM-5-next). Absent from generic GatedDeltaNet configs
    (Qwen3-Next), so it never flags real GatedDeltaNet as KDA."""
    def walk(v) -> bool:
        if isinstance(v, dict):
            return "kda_layers" in v or any(walk(x) for x in v.values())
        if isinstance(v, list):
            return any(walk(x) for x in v)
        return False
    return walk(config or {})


def _check_attn_variant_label(resolved) -> List[dict]:
    """The DISPLAYED attention label is `attn_display = attn_variant or
    cache.family` (schema/graph.py). `cache.family` is the coarse KV-cache
    STORAGE class; when the COMPUTE flavor differs — DSA over an MLA-shaped
    cache, KDA over a GatedDeltaNet-shaped state — the spec must set
    `attn_variant` or the model shows the wrong mechanism name (e.g. a DSA
    layer displayed as plain "MLA", a KDA layer as "GatedDeltaNet"). This moves
    no bytes/shapes, so no other contract catches it, and the fallback is a real
    attention name — a plausible-but-imprecise label, the hardest to notice.

    Fires only on UNAMBIGUOUS flavor evidence: the DSA ops in the component's
    graph, or a `kda_layers` config key. Genuine plain-MLA and real
    GatedDeltaNet (which carry no such signal) are never flagged. Repairable —
    the fix is a one-line `attn_variant:`."""
    out: List[dict] = []
    config_kda = _config_declares_kda(resolved.config)
    for name, bg in iter_reachable_components(resolved):
        if roles.kind_of(bg.role) != "attention" or bg.attn_variant is not None:
            continue
        dsa_ops = sorted({node.op for node in bg.graph.values()
                          if is_dsa_op(node.op)})
        if dsa_ops:
            out.append(Finding(
                kind="attn_variant_missing", template=name,
                detail=(
                    f"component {name!r} runs DSA sparse-selection ops "
                    f"({', '.join(dsa_ops)}) but declares no `attn_variant`, so it "
                    f"displays as its cache family {bg.kv_label!r}. Set "
                    f"`attn_variant: DSA` — the compute flavor over an "
                    f"{bg.kv_label}-shaped cache."
                )))
        elif config_kda and bg.kv_label == "GatedDeltaNet":
            out.append(Finding(
                kind="attn_variant_missing", template=name,
                detail=(
                    f"config declares `kda_layers` (Kimi Delta Attention) but "
                    f"component {name!r} declares no `attn_variant`, so it displays "
                    f"as {bg.kv_label!r}. KDA is a distinct mechanism from generic "
                    f"GatedDeltaNet; set `attn_variant: KDA`."
                )))
    return out


def _calls(resolved, phase: str = "prefill",
           only_forms: bool = False) -> List[_Call]:
    """The model's component invocations, from one analysis of the whole
    model at one token (stored lengths stay symbolic in the cache length)."""
    analysis = Deployment(Program.of(resolved)).during(phase).at(
        batch=1, input_tokens=1).analyze()
    forms = {name: form
             for name, bg in iter_reachable_components(resolved)
             if (form := phase_form(bg, phase))}
    return _Call.all_in(resolved, analysis, forms=forms,
                        only_forms=only_forms)


def _check_decode_forms(resolved) -> List[dict]:
    """The same judgement over the components a DECODE phase runs instead.

    A dense component's `decode_variant:` form is not called by the prefill
    graph — the decode phase binds it in (`Phase.substitute`; a sparse
    attention's prefill runs it too, `phase_form`) — so the prefill analysis
    never reaches it and its claims would be checked by nobody.
    `mla_attention_absorbed` shipped with an `expect` that could have been
    wrong by any margin and this validator would still have said PASSED,
    which is worse than having no sidecar: it reads as a verified fact. The
    decode phase's own analysis is the honest context: it is the graph and
    the workload a decode step evaluates.
    """
    if not any(bg.decode_variant
               for _, bg in iter_reachable_components(resolved)):
        return []
    out: List[dict] = []
    for call in _calls(resolved, "decode", only_forms=True):
        out.extend(call.judge())
    return out


def _distinct(findings: List[Finding]) -> List[Finding]:
    """One finding per statement: two templates calling a component in the
    same dims grow the same cache, and say the same thing about it."""
    seen = set()
    out: List[Finding] = []
    for f in findings:
        key = (f.kind, f.template, f.detail.replace(f.node or "", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def run_axiom_checks(model_id: str, resolved) -> List[dict]:
    """Judge every component call's claims and window contract in this
    model's analysis — plus the decode forms it swaps in, plus the
    model-level config-vs-spec window contract and the attention label.
    A model that cannot be analysed has no cache to judge: that is one
    finding, carrying the analysis's own refusal."""
    out: List[Finding] = []
    try:
        calls = _calls(resolved)
    except Exception as exc:
        calls = None
        out.append(Finding(
            kind="axiom_unjudged", template="<model>",
            detail=(f"the model's analysis raised {type(exc).__name__}: {exc} "
                    f"— claims and window contracts are judged on it, so "
                    f"none were judged")))
    if calls is not None:
        for call in calls:
            out.extend(call.judge())
        out.extend(_check_config_windowing(resolved, calls))
    try:
        out.extend(_check_decode_forms(resolved))
    except Exception as exc:
        out.append(Finding(kind="axiom_unjudged", template="<decode>",
                           detail=f"the decode forms' analysis raised "
                                  f"{type(exc).__name__}: {exc} — their "
                                  f"claims were not judged"))
    try:
        out.extend(_check_attn_variant_label(resolved))
    except Exception as exc:
        out.append(Finding(kind="axiom_check_error", template="<model>",
                           detail=f"attn-variant-label check raised {type(exc).__name__}: {exc}"))
    return _distinct(out)
