"""The rotation a spec declares against the position streams its config sets.

A rotary position embedding reads one position per token, or — a sectioned
rotation — several, each channel section rotated by its own stream. The
arithmetic is the same on text tokens, so no numeric anchor can see a spec
that writes one for the other. The constraint runs one way. Rope parameters
that carry a section list declare a sectioned rotation: rope type `xdrope`
with `xdrope_section` (HunyuanOCR, four streams) or `mrope_section` (Qwen3-VL,
three streams), and the text rotation must be the op built from that key
(`OpSpec.rotary_sections_key`) with that many streams
(`OpSpec.rotary_position_axes`). Rope parameters without a section list do
not decide it: the model's HF rotary class does, and it may section by its
own default (`Qwen3_5MoeTextRotaryEmbedding` defaults `mrope_section` to
[11, 11, 10] and rotates by three streams for a text-only config), so the
spec declares the rotation its forward runs and cites it.

Only the text path is checked against the config: an encoder tower's 2-D
rotation (`Program.encoder_uids`) rotates by its own image grid, which no
text rope parameter describes. An encoder rotation is held to its own two
forms instead: `rope_2d` (an image grid's h and w; vLLM `ApplyRotaryEmb`)
or a one-position `rope` (MiMo's audio transformer). A sectioned rotation is
the text decoder's rotation of mixed text and image tokens, never an
encoder's.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from backend.arch.dim_resolution import language_config
from backend.arch.eval.program import Program
from backend.arch.ops import OP_REGISTRY

from ._finding import Finding

def _rotation_of(params: Any) -> Tuple[str, int]:
    """The rotation one rope-parameter mapping declares."""
    kind = params.get("rope_type") or params.get("type")
    xd = params.get("xdrope_section")
    if kind == "xdrope" or xd is not None:
        section = xd if xd is not None else params.get("mrope_section")
        return "xdrope_section", len(section or ()) or 1
    section = params.get("mrope_section")
    if isinstance(section, list) and section:
        return "mrope_section", len(section)
    return "", 1


def config_rotation(fields: Any) -> Tuple[str, int]:
    """The rotation the language model's rope parameters declare, read from
    its own config fields (`dim_resolution.language_config`):
    `(sections key, position streams)`. `("xdrope_section", n)` for rope
    type xdrope (its sections under `xdrope_section`, or `mrope_section`
    once HF has normalised the config), `("mrope_section", n)` for a
    `mrope_section` list, `("", 1)` for one position per token. HF names
    the mapping `rope_parameters` (transformers v5) or `rope_scaling`; a
    config stating both with different rotations contradicts itself and is
    refused rather than read in either order."""
    stated = {_rotation_of(fields[key]) for key in ("rope_parameters", "rope_scaling")
              if isinstance(fields.get(key), dict)}
    if len(stated) > 1:
        raise ValueError(
            f"the config's `rope_parameters` and `rope_scaling` declare "
            f"different rotations {sorted(stated)}")
    return stated.pop() if stated else ("", 1)


def config_position_axes(fields: Any) -> int:
    """The position streams the language model's rotation reads
    (`config_rotation`)."""
    return config_rotation(fields)[1]


def _ops_for(rotation: Tuple[str, int]) -> List[str]:
    """The registered ops that rotate as `rotation` asks, by declaration."""
    key, axes = rotation
    return sorted(name for name, spec in OP_REGISTRY.items()
                  if spec.rotary_position_axes == axes
                  and spec.rotary_sections_key == key)


def _rotation(op: str) -> Optional[Tuple[str, int]]:
    spec = OP_REGISTRY.get(op)
    if spec is None or not spec.rotary_position_axes:
        return None
    return spec.rotary_sections_key, spec.rotary_position_axes


def _write(ops: List[str]) -> str:
    return " or ".join(f"`op: {op}`" for op in ops)


def run_rotary_position_check(resolved: Any, config: Any) -> List[Finding]:
    """One finding per text-path rotation that is not the sectioned
    rotation the config's rope parameters declare, naming the fix. A config
    without a section list asks nothing of the text path."""
    want = config_rotation(language_config(resolved.spec.dims, config)[1])
    program = Program.of(resolved)
    # The most position streams an encoder rotation reads: one per axis of
    # an item of what it encodes — a video's (t, h, w), an image's (h, w),
    # one position per token of a clip or a prompt.
    grid_axes: dict = {}
    for entry in program.model_inputs.values():
        if entry.stack is not None:
            grid_axes[entry.stack] = max(grid_axes.get(entry.stack, 1),
                                         len(entry.axes) - 1)
    out: List[Finding] = []
    seen = set()
    for node in program.nodes():
        if node.children:
            continue
        got = _rotation(node.op)
        encoder = node.uid in program.encoder_uids
        most = grid_axes.get(program.stack_of(node), 2) if encoder else None
        if got is None or (
                (got[1] <= most and not got[0]) if encoder
                else (got == want or not want[0])):
            continue
        where = node.display_path or node.name
        if where in seen:
            continue
        seen.add(where)
        if encoder:
            detail = (
                f"{where} is an encoder rotation written `op: {node.op}` "
                f"({got[1]} position streams). An encoder rotates by its own "
                f"input's positions: write `op: rope_3d` for a video grid's "
                f"(t, h, w) rotation, `op: rope_2d` for an image grid's "
                f"(h, w), or `op: rope` for one position per token.")
        else:
            ops = _ops_for(want)
            detail = (
                f"the config's rope parameters carry `{want[0]}` "
                f"({want[1]} position streams), but {where} is `op: {node.op}`, "
                f"which reads {got[1]}"
                + (f" by `{got[0]}`" if got[0] else "") + ". "
                + (f"Write {_write(ops)}; a library attention takes it as "
                   f"`params: {{ rope: {ops[0]} }}`." if ops else
                   "No registered rotation reads that many streams by "
                   "that key."))
        out.append(Finding(kind="rotary_position_mismatch", detail=detail,
                           node=where, op=node.op))
    return out
