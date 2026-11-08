"""Architecture-view badge derivation.

Inputs:
  • `resolved_spec` — drives attention-flavor badges (read from each
    attention component's `kv_label`, NO substring matching) and aux-
    module badges (read from each `group: true` layer's `aux_kind:`).
  • `summary` — the weight bytes per quant method (`weight_methods`),
    which pick the quant badge of a mixed checkpoint.
  • `config` — unused for the quant badge. Quantization is read from
    the spec's `quant_context` (CLAUDE.md §2.4: runtime does not
    re-parse `config.quantization_config`).

Output: an ordered list of badge strings for the architecture-view
header (`["MoE", "MLA", "FP8"]`, `["Hybrid", "GQA", "SWA"]`, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, get_args

from backend.arch.eval.program import Program
from backend.arch.schema.model import ModelLayer
from backend.arch.spec_tree import iter_blocks
from backend.arch.quant_schemes import QUANT_SCHEMES


def collect_attention_labels(resolved_spec, fallback: str) -> List[str]:
    """The backbone's attention / sequence mixers (`Program.attention_mix`).
    Fallback is the analyzer headline when the backbone has no attention
    cache."""
    if not resolved_spec:
        return [fallback] if fallback else []
    labels = list(Program.of(resolved_spec).attention_mix)
    if labels:
        return labels
    return [fallback] if fallback else []


def _aux_kind_badges() -> Dict[str, str]:
    """Badge text per aux-stack kind.

    The KINDS are the schema's — `ModelLayer.aux_kind`'s Literal, which the
    loader refuses anything outside — so they are read off it rather than
    listed again, and a kind added there with no badge is refused here at
    import. Only the text is this module's: `MTP` is an acronym and
    `Vision` a word, which no declaration says and no spelling rule
    recovers.
    """
    text = {"mtp": "MTP", "vision": "Vision", "audio": "Audio",
            "speculative": "Speculative", "text": "Text encoder",
            "latent": "Latent input", "decoder": "Decoder"}
    declared = {k for k in get_args(get_args(
        ModelLayer.model_fields["aux_kind"].annotation)[0]) if isinstance(k, str)}
    missing = sorted(declared - set(text))
    if missing or not declared:
        raise ValueError(
            f"aux stack kind(s) {missing or 'none'} are declared by "
            f"`ModelLayer.aux_kind` with no architecture-view badge here; "
            f"a stack the walk instantiates would show none")
    return text


# Spec-declared aux_kind value → badge text (`_aux_kind_badges`).
_AUX_KIND_BADGE = _aux_kind_badges()


def _aux_module_badges(resolved_spec) -> Iterable[str]:
    """Badge each aux stack the walk actually instantiated.

    Reads `iter_blocks` (same enumeration rendering uses for templates),
    not the raw `spec.model` list: a group with `repeat: 0` is absent
    from this checkpoint and must not show an MTP/Vision badge.
    Order follows the walk, which follows spec declaration order.
    """
    if not resolved_spec:
        return
    seen: set = set()
    for meta in iter_blocks(resolved_spec):
        kind = meta.aux_kind
        if kind and kind in _AUX_KIND_BADGE and kind not in seen:
            seen.add(kind)
            yield _AUX_KIND_BADGE[kind]


def _quant_badge(resolved_spec, summary: Dict[str, Any]) -> Optional[str]:
    """Pick the badge string for the model's quantization scheme: the
    scheme holding the most weight bytes (`summary["weight_methods"]`, the
    analysis' join), so a mixed checkpoint is badged by what it mostly is —
    DeepSeek-V4's FP4 experts, not its FP8 attention, which `per_role`
    happens to list first.

    Reads `spec.quant_context.per_role`, not `config.quantization_config`.
    `method: none` is unquantized and skipped. Scheme YAML `badge:` is
    the label; an unregistered method surfaces uppercased so a missing
    yaml is visible.
    """
    spec = getattr(resolved_spec, "spec", None)
    qc = getattr(spec, "quant_context", None) if spec is not None else None
    if qc is None:
        return None
    labels: Dict[str, str] = {}
    for method, num_bits, _extra in qc.scheme_entries():
        scheme = QUANT_SCHEMES.get(method)
        if scheme is None:
            label = method.upper()
        else:
            label = scheme.format_badge(num_bits, method) or method.upper()
        if label:
            labels.setdefault(method, label)
    for row in summary.get("weight_methods") or ():
        if row.get("method") in labels:
            return labels[row["method"]]
    return None


def collect_badges(
    resolved_spec,
    summary: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    fallback_attn_type: str,
) -> tuple:
    """Build the full badges list + the attention-labels list.

    Returns `(badges, attn_labels)`. `attn_labels` is also returned so
    callers (e.g. layout cards) can read the same set without recomputing.
    """
    attn_labels = collect_attention_labels(resolved_spec, fallback_attn_type)

    badges: List[str] = (
        ["Hybrid", *attn_labels] if len(attn_labels) > 1 else list(attn_labels)
    )
    if resolved_spec and Program.of(resolved_spec).has_dispatch_zone:
        badges.insert(0, "MoE")

    badges.extend(_aux_module_badges(resolved_spec))

    qb = _quant_badge(resolved_spec, summary)
    if qb is not None:
        badges.append(qb)

    return badges, attn_labels
