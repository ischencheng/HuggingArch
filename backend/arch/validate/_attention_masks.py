"""The mask an attention declares, checked where a wrong default is silent.

A score / softmax / apply node prices its matrix through the attention's own
mask declaration (`is_causal`, `is_chunked`; `ops_base._pairs`). An unstated
`is_causal` means causal, which is right for a decoder and wrong for every
encoder tower the corpus has: vLLM runs vision and audio encoders through
`MMEncoderAttention` with the mask off (`v1/attention/ops/
vit_attn_wrappers.py:71, 185, 368`). Nothing numeric catches the difference —
the matrix just costs half — so the declaration is asked for where the
default is not evidence:

  * a mask-reading node inside an encoder stack (`Program.encoder_uids`) must
    state `is_causal`. The check does not assume encoders are bidirectional
    (vLLM's Voxtral-realtime encoder is causal, `whisper_causal.py`); it asks
    the author to say which;
  * the nodes of one attention — a score, the softmax reading it, the apply
    reading that — state the same mask;
  * a bidirectional sliding band is symmetric (vLLM `flash_attn.py`
    `_maybe_symmetrize_window`; MiMo `sliding_window_q = sliding_window_k`),
    so its `s_window` = 2w + 1 is odd;
  * a chunked mask (`is_chunked`) is the windowed ops' bidirectional one:
    three lawful shapes — causal band, bidirectional band, bidirectional
    chunks — and the fourth combination is refused.

Each finding names the node and the fix.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.ops import OP_REGISTRY
from backend.arch.ops_base import mask_flags

from ._finding import Finding

def _stated(node: Any) -> dict:
    """The mask flags stated for the node: its own `dims:`, or the frame it
    sits in — a call site or a block assignment that states the mask for a
    library attention whose nodes leave it to the caller (a pipeline's text
    encoder runs an LM attention causally inside an encoder stack)."""
    raw = dict(node.dims_raw or {})
    for key in ("is_causal", "is_chunked"):
        if key not in raw and node.frame and node.frame.get(key) is not None:
            raw[key] = value_of(node.frame.get(key))
    return {k: raw[k] for k in ("is_causal", "is_chunked") if k in raw}


def _mask(node: Any) -> Tuple[int, int]:
    """The node's `(is_causal, is_chunked)`, defaults filled as the cost
    reads them (`ops_base.mask_flags`)."""
    return tuple(int(str(v).strip()) for v in mask_flags(_stated(node)))


def run_attention_mask_check(resolved: Any) -> List[Finding]:
    program = Program.of(resolved)
    encoders = program.encoder_uids
    out: List[Finding] = []
    seen = set()

    def add(kind: str, node: Any, detail: str) -> None:
        where = node.display_path or node.name
        if (kind, where) in seen:
            return
        seen.add((kind, where))
        out.append(Finding(kind=kind, detail=detail, node=where, op=node.op))

    for node in program.nodes():
        spec = OP_REGISTRY.get(node.op or "")
        if node.children or spec is None or not spec.reads_mask():
            continue
        mask = _mask(node)
        where = node.display_path or node.name
        if node.uid in encoders and "is_causal" not in _stated(node):
            add("encoder_mask_undeclared", node,
                f"{where} (`op: {node.op}`) prices an attention matrix inside "
                f"an encoder stack and does not state its mask. Encoder "
                f"towers usually run without a causal mask (`is_causal=False` "
                f"in the tower's modeling code): write "
                f"`is_causal: \"0\"` in the dims of its score, softmax and "
                f"apply nodes, or `is_causal: \"1\"` if the source masks it "
                f"causally.")
        for ref in tuple(node.inputs_obj or ()):
            src = ref.src
            if src is None:
                continue
            src_spec = OP_REGISTRY.get(src.op or "")
            if src_spec is None or not src_spec.reads_mask():
                continue
            if _mask(src) != mask:
                add("attention_mask_disagrees", node,
                    f"{where} (`op: {node.op}`) states "
                    f"is_causal={mask[0]}, is_chunked={mask[1]} but the "
                    f"{src.display_path or src.name} it reads states "
                    f"is_causal={_mask(src)[0]}, "
                    f"is_chunked={_mask(src)[1]}: the score, "
                    f"softmax and apply of one attention are one mask — "
                    f"write the same `is_causal` / `is_chunked` on all three.")
        causal, chunked = mask
        if chunked == 1 and (causal == 1 or not spec.windowed):
            add("chunked_mask_unsupported", node,
                f"{where} (`op: {node.op}`) states is_chunked=1 with "
                f"is_causal={causal}. A chunked mask is priced on the "
                f"windowed `swa_*` ops, bidirectional within each chunk "
                f"(`is_causal: \"0\"`): the step's K / V read is a band "
                f"union, which a causal chunk does not have.")
        causal, chunked = mask
        if spec.windowed and causal == 0 and chunked == 0:
            window = value_of((node.dims or {}).get("s_window"))
            if isinstance(window, int) and window % 2 == 0:
                add("bidirectional_window_even", node,
                    f"{where} (`op: {node.op}`) is a bidirectional sliding "
                    f"window of s_window={window}. A bidirectional band is "
                    f"symmetric — w keys each side plus the query, "
                    f"s_window = 2w + 1 — so the width is odd. Write the "
                    f"source's reach as `2 * w + 1`, or `is_chunked: \"1\"` "
                    f"if the keys are fixed groups of {window}.")
    return out
