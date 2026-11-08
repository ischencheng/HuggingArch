"""Type contracts — declared semantic attributes ⟹ structural consequences.

The right-bottom cell of the proposal's matrix: a component's declared
attention/KV semantics must be *internally consistent* with the fields the
inference engine reads to compute KV storage / read cardinality. These are
exactly the non-weight-producing attributes the bytes-vs-bytes validator is
blind to — and the class of gap the 128TB bug slipped through.

Why declaration contracts, not a symbolic `flops_expr.has(Min)` predicate
-------------------------------------------------------------------------
The inference-rules doc (D-class) sketches "declared window ⟹ the SymPy
FLOPs expression must contain a `Min` clamp". Empirically that does NOT
hold in this engine: the sliding window is applied to the *cached tokens*
(the attention op's storage cardinality resolves to `Min(s_kv, s_window)`), NOT to
the attention *score* FLOPs — a windowed SWA component and a global GQA
component emit structurally identical `attention_flops`. And DSA's sparsity
lives in its top-k indexer read. Asserting `has(Min)` on FLOPs would
false-positive on every windowed component and mis-handle DSA — a
confident-but-wrong check. So the contracts here are grounded in the
component's lowered cache attributes (`kv_label`, `kv_cached`), verified
FP-free against gemma2 / qwen3 / qwen3_moe / deepseek_v3_2 / deepseek_v4.

Whether a sliding-window component's cache actually caps is not a declaration
question: the model's analysis grows that cache, and `_axioms` judges it there
(`swa_cache_unbounded`) at every call.

Deliberately NOT here: "does the declared attribute appear in the source?"
That is a *semantic-equivalence* judgment (is this component faithful to the
source module it models), and judging it by regex-grepping the source for
keywords is a lossy approximation with endless corner cases. Per the
proposal, equivalence is the inspector AGENT's job — it reads the source
(the ground truth) and reflects; the tool only does deterministic checks on
spec fields it can know for sure.

Every emitted contract is deterministic after transparent adapter wrappers are
removed. The orchestrator promotes these findings to validation errors.
"""
from __future__ import annotations

from typing import List

import backend.arch.cache_families as cache_families
import backend.arch.roles as roles
from backend.arch.eval.program import Program
from backend.arch.spec_parsing import RETIRED_KV_REPLACEMENTS
from backend.arch.structure import iter_reachable_components

from ._finding import Finding

_CACHE_FAMILY = RETIRED_KV_REPLACEMENTS["kv_label"]
_CACHE_ENTRIES = RETIRED_KV_REPLACEMENTS["kv_cached"]



def check_field_contracts(resolved) -> List[dict]:
    """The robust, FP-free declaration-consistency contracts.

    Pure over `resolved` — no source / engine needed, directly testable.
    """
    out: List[dict] = []
    # Components the language model runs (`Program.walked_components`). A
    # component only an encoder reaches runs in the encode phase and keeps
    # no cache to declare.
    decode_path = Program.of(resolved).walked_components
    for name, bg in iter_reachable_components(resolved):
        role = bg.role
        if roles.kind_of(role) != "attention":
            continue

        # An alias adapter delegates its storage semantics to the attention
        # component it wraps. That component is visited and checked on its
        # own, so checking the wrapper would report a duplicate missing-cache
        # error (chatglm_attention -> gqa_fused_qkv_attention).
        def _component(op: str):
            try:
                return resolved.spec.get_block(op)
            except KeyError:
                return None
        target = bg.alias_target(_component)
        if target is not None and roles.kind_of(target.role) == "attention":
            continue

        kv_label = bg.kv_label
        kv_cached = bg.kv_cached or []

        # (1) attention ⟹ cache.family declared. Without it the inference KV
        # classifier falls back to the dim heuristic the explicit topology
        # was meant to replace.
        if not kv_label:
            out.append(Finding(
                kind="attention_missing_kv_label",
                template=name,
                detail=f"attention component {name!r} declares no `{_CACHE_FAMILY}` "
                       "(MLA / GQA / MHA / SWA …) — inference falls back to a "
                       "dim heuristic to classify its KV cache.",
            ))

        # (2) attention ⟹ a declared KV state: either cache.entries of graph
        # nodes (per-token growing streams) or a `state:` entry under
        # cache.entries (constant-size recurrent state — linear attention /
        # SSM). Empty of BOTH means the engine guesses the KV shape from dims.
        # Asked of decode-path attention only: an encoder's attention caches
        # nothing by construction.
        # A family of kind `none` caches nothing by declaration (a
        # diffusion transformer's joint attention, recomputed every step).
        caches_nothing = (cache_families.is_registered(kv_label or "")
                          and cache_families.kind_of(kv_label) == "none")
        if (name in decode_path and not kv_cached and not bg.kv_states
                and not caches_nothing):
            out.append(Finding(
                kind="attention_missing_kv_cached",
                template=name,
                detail=f"attention component {name!r} declares neither "
                       f"`{_CACHE_ENTRIES}` (per-token KV nodes) nor a "
                       f"recurrent `state:` under `{_CACHE_ENTRIES}` — "
                       "inference can't tell what enters the KV cache and "
                       "falls back to a dim heuristic.",
            ))

    return out


def run_type_contract_checks(model_id: str, resolved) -> List[dict]:
    """Declaration-consistency contracts on spec KV/attention fields.
    Best-effort collection; the orchestrator gates every emitted contract.
    (`model_id` accepted for signature parity
    with the other check entry points; not needed here.)"""
    return check_field_contracts(resolved)
