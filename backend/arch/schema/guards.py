"""Quant-suffix lockdown + weight-dim geometry guards shared by the spec models."""
from __future__ import annotations

import re
from typing import (Optional,

                    Tuple)

from backend.arch.quant_schemes import QUANT_SCHEMES
from backend.code_trace.memo import memoize





# ---------------------------------------------------------------------------
# Weight-tensor dim geometry guard.
#
# `linear: y = W @ x` with W of shape [Out, In] has EXACTLY Out*In
# parameters by matrix-multiplication geometry — no fractional packing
# can change that count at the logical-shape layer. Literal-integer
# divisors (`/2`, `/8`, `/16`, …) in a `parameter_vector` dim are a
# tell-tale sign of storage-layout packing leaking into the spec. The
# rule holds for every `parameter_vector`: a legitimate integer division
# (a group or block size) is a dim of the model, and once it is declared
# by name under `dims:` it has a source and no literal is left to trip.
#
# Such leaks usually accompany a missing entry under
# backend/arch/quant_schemes/: the spec was hand-tuned to match
# an HF accounting pipeline that didn't recognize the on-disk packing,
# producing a self-fulfilling "exact match" on doubly-undercounted
# totals. The validator is a trip-wire — when it fires, both the spec
# AND the quant-scheme yaml should be reviewed together.
# ---------------------------------------------------------------------------

# Match any literal `/<integer>` inside a dim expression. Whitespace
# between `/` and the digits is tolerated.
_LITERAL_INT_DIVISION = re.compile(r"/\s*\d+")



# ---------------------------------------------------------------------------
# Quant-metadata-suffix lockdown for spec components.
#
# The spec language is quant-agnostic. Quantization metadata (scales /
# zero_points / packed-weight sidecars / activation scales / KV-cache
# scales / column-permute indices) doesn't belong in component yamls —
# it lives in `backend/arch/quant_schemes/<method>.yaml` and is
# materialized by the engine at compute time when a model declares
# `quant_context.per_role`.
#
# The suffix list below enumerates known quant-metadata tensor names.
# A `parameter_vector` whose name ends with any of these is rejected
# at load time — the spec was hard-coding what should come from the
# scheme registry.
# ---------------------------------------------------------------------------

# Cached union of every registered scheme's metadata-sibling suffix
# AND non-trivial weight-storage suffix. Computed once on first
# access so the schemes registry has a chance to load.
@memoize(maxsize=1)
def _quant_suffix_blacklist() -> Tuple[str, ...]:
    """Suffixes that mark a `parameter_vector` as quant metadata,
    derived from the registered schemes' `metadata_siblings` +
    `format_variants[*].weight_storage_suffix`. Anything in this list
    must come from the engine via materialize, not be hand-coded in
    spec components.
    """
    return tuple(sorted(set(_quant_metadata_sibling_suffixes())
                        | set(_quant_storage_suffixes())))




def _quant_metadata_sibling_suffixes() -> Tuple[str, ...]:
    """Suffixes naming a tensor that is NOT weight: scales, zero points,
    group indices."""
    return tuple(sorted({sib.suffix for scheme in QUANT_SCHEMES.values()
                         for sib in scheme.metadata_siblings}))



def _quant_storage_suffixes() -> Tuple[str, ...]:
    """Suffixes naming the WEIGHT, in a scheme's packed on-disk form
    (`.qweight`, `.weight_packed`, `_blocks`). These stay subject to the
    In × Out geometry rule — a packing factor in one of their dims is the
    storage leak `_check_weight_dim_geometry` exists to reject, and the
    engine recovers the real packing from the scheme registry."""
    return tuple(sorted({
        fv.weight_storage_suffix
        for scheme in QUANT_SCHEMES.values()
        for fv in scheme.format_variants.values()
        if fv.weight_storage_suffix and fv.weight_storage_suffix != ".weight"
    }))


def _is_quant_metadata_suffix(name: str) -> bool:
    """Does `name` end with a registered quant-metadata suffix?

    The blacklist is derived purely from the quant_schemes registry's
    `metadata_siblings.suffix` + non-trivial `weight_storage_suffix`
    values, so it only contains tensor-name tails that some scheme
    actually emits on disk. Architectural names like `hc_attn_scale`
    (hyper-connections statistic) or `vision_tower.std_scale` (image
    normalization) end in `_scale` — single underscore — which is
    NOT in the blacklist (no scheme uses that as a metadata suffix),
    so they pass without needing a whitelist exception.
    """
    return any(name.endswith(suf) for suf in _quant_suffix_blacklist())



def _check_quant_suffix_in_arch_spec(
    container_kind: str,
    container_label: str,
    node_name: str,
    op: str,
    quant_metadata: Optional[bool] = None,
) -> None:
    """Reject `parameter_vector` whose name carries a quant-metadata
    suffix in spec components / models. The correct home for
    these tensors is the scheme yaml's `metadata_siblings:` block —
    engine emits them at compute time when the model declares a
    matching per_role.

    Author opt-out: when an architectural tensor's checkpoint name
    legitimately collides with a registered quant-metadata suffix
    (e.g. Hyper-Connections per-head `<W>.scale` vs FP4's `.scale`),
    set `quant_metadata: false` on the node. The check then trusts
    the author's claim that this `parameter_vector` is real arch
    state, not engine-materialized metadata.
    """
    if op != "parameter_vector":
        return
    if quant_metadata is False:
        return
    if not _is_quant_metadata_suffix(node_name):
        return
    raise ValueError(
        f"{container_kind} {container_label!r}: parameter_vector "
        f"{node_name!r} has a quantization-metadata suffix. The spec "
        f"is quant-agnostic — these tensors must come from the engine "
        f"via `quant_context.per_role` + the matching "
        f"`backend/arch/quant_schemes/<method>.yaml`'s "
        f"`metadata_siblings:` declaration, not be hand-coded here.\n"
        f"  • Remove this parameter_vector node and add\n"
        f"    `quant_context.per_role.<role>: {{method: <m>, …}}` to\n"
        f"    every model spec that consumes this component.\n"
        f"  • If this is an architectural tensor whose checkpoint\n"
        f"    name happens to collide with a registered quant suffix,\n"
        f"    declare `quant_metadata: false` on the node to opt out."
    )



def _check_weight_dim_geometry(
    container_kind: str,        # "block_graph" | "model_layer"
    container_label: str,       # e.g. block name, layer name (for the error)
    node_name: str,             # the parameter_vector node's name
    op: str,
    dim: Optional[str],
) -> None:
    """Reject a `parameter_vector` whose dim contains a literal integer
    divisor (storage-packing leak). See module-level comment for rationale.
    """
    if op != "parameter_vector" or dim is None:
        return
    if not _LITERAL_INT_DIVISION.search(dim):
        return
    raise ValueError(
        f"{container_kind} {container_label!r}: parameter_vector "
        f"{node_name!r} declares dim={dim!r}, which contains a literal "
        f"integer divisor. A weight's element count is In × Out; if this "
        f"divisor is a group or block size, declare it as a named dim under "
        f"`dims:` and divide by that name instead. Weight tensors must have dim equal to the "
        f"matrix-multiplication geometric truth (In × Out) — there is "
        f"no fractional weight count under any quantization scheme. "
        f"Encoding storage-packing factors (e.g. `/2` for FP4 packed "
        f"into I8, `/8` for INT4 packed into I32) into a logical-shape "
        f"dim is what produces self-fulfilling 'exact match' against "
        f"an HF accounting pipeline that ALSO didn't recognize the "
        f"packing — both sides undercount by the same factor and the "
        f"diff badge lights up on a wrong number.\n\n"
        f"This trip means a quantization method is not yet supported "
        f"end-to-end. Resolve it by:\n"
        f"  1. Setting dim back to logical `In × Out`.\n"
        f"  2. Reviewing / adding the matching entry under "
        f"backend/arch/quant_schemes/<method>.yaml so the engine "
        f"recovers the on-disk packing factor and metadata-sibling "
        f"cardinality from the scheme registry, not from the spec dim."
    )
