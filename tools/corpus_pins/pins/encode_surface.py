"""P11: the encode phase, for every snapshot with an encoder, at TP=1 and
TP=8 and two reference media.

The encode phase runs each encoder stack over its own media inputs
(`Deployment.during("encode", stack=...)`); nothing of it is in the language
model's prefill or decode. Per (snapshot x TP x reference x encoder stack):
the stack's FLOPs and parameters, its leaves' activation read and write bytes
(`traffic.derive`, weights excluded), its resident weight bytes on one rank,
its activation peak on one rank (the encode step's liveness, BF16
activations, no framework fusion), and the rows each of its outputs hands the
language model per request (`Deployment.handed_over`). Per (snapshot x TP x
reference): the prompt positions each media input fills
(`Deployment.media_tokens`), an encoder-side fact no language-model number
reads, and the bytes one cached encoder-output token holds
(`EncoderOutputCache`, BF16).

The references are the workload's, written here and nowhere in the
production path: `one` is one item per media input, `two` two items. An
image side is a whole number of the processor's resize factor the spec
declares on its cut (`align`): `one` is `7·align` square, `two`
`14·align` high and `7·align` wide; a side the config fixes stays fixed.
An audio clip is 125 frames in `one` and 750 in `two`; a video 4 sampled
frames `7·align` square in `one` and 5 frames (an odd count: the last frame
repeated to a whole temporal patch) `7·align` high and `5·align` wide in
`two` (within gemma-4's 70 soft tokens a frame). The items of each
item of another input (Step-3.7's crops) number 0 in `one` and 2 in `two`;
an input the spec declares and does not walk is stated with no item.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from tools.corpus_pins.corpus import CorpusModel, shared_loader
from tools.corpus_pins.pin import JsonMapPin, Row, Scope

TP_DEGREES = (1, 8)
#: Per reference: items per media input, image sides in units of the
#: processor's resize factor (height, width), audio frames per clip.
REFERENCES = {
    "one": {"items": 1, "sides": (7, 7), "frames": 125, "sub_items": 0,
            "video_frames": 4, "video_sides": (7, 7)},
    "two": {"items": 2, "sides": (14, 7), "frames": 750, "sub_items": 2,
            "video_frames": 5, "video_sides": (7, 5)},
}


def tp_config(degree: int) -> Any:
    from backend.inference.parallel_config import ParallelConfig
    return ParallelConfig(
        tp_attention=degree, tp_dense_ffn=degree, tp_moe_expert=degree)


def reference_media(program: Any, reference: str) -> Dict[str, Any]:
    """The reference's media for every media input `program` declares,
    each image side a multiple of the `align` of the processor's cut that
    reads it (`ModelInput.least`)."""
    ref = REFERENCES[reference]
    media: Dict[str, Any] = {}
    for name, entry in program.model_inputs.items():
        if entry.kind == "tokens":
            continue
        if entry.unmodeled is not None:
            # Stated with every request; any item is refused.
            media[name] = {entry.axes[0]: 0}
            continue
        fixed = dict(entry.fixed or {})
        axes = {entry.axes[0]: (ref["items"] if entry.per_item_of is None
                                else ref["sub_items"])}
        if entry.kind in ("image", "video"):
            sides = entry.axes[-2:]
            units_of = (ref["video_sides"] if entry.kind == "video"
                        else ref["sides"])
            for axis, units in zip(sides, units_of, strict=True):
                if axis not in fixed:
                    axes[axis] = units * int(entry.least[axis])
            if entry.kind == "video":
                axes[entry.axes[1]] = ref["video_frames"]
        else:
            axes[entry.axes[1]] = ref["frames"]
        media[name] = axes
    return media


def measure_case(resolved: Any, degree: int, reference: str) -> Dict[str, Any]:
    from backend.arch.eval import traffic as traffic_mod
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.layout.persistent import (
        physical_artifacts, resident_weights)
    from backend.inference.activation_peak import activation_peak_at
    from backend.inference.encode_phase import EncoderOutputCache
    from backend.inference.step_eval import evaluate_encode
    program = Program.of(resolved)
    media = reference_media(program, reference)
    parallel = tp_config(degree)
    deployment = Deployment(program, config=parallel).at(
        batch=1, media=media)
    handed = deployment.handed_over()
    stacks: Dict[str, Any] = {}
    for encode in deployment.encoders():
        stack = encode.phase.stack
        analysis = encode.analyze()
        # The stack's own: an encoder that runs another's weights walks
        # beside it (`Stack.reruns`), at no item of that one's.
        total = analysis.by_stack()[stack]
        leaves = traffic_mod.derive(analysis).leaves()
        rows = [leaves[n] for n in leaves
                if analysis.program.stack_of(analysis.program.by_uid(n))
                == stack]
        stacks[stack] = {
            "flops": int(value_of(total.flops)),
            "params": int(value_of(total.params)),
            "act_read_bytes": int(sum(value_of(t.act_reads) for t in rows)),
            "act_write_bytes": int(sum(value_of(t.act_writes) for t in rows)),
            "weight_bytes_per_rank": int(
                resident_weights(physical_artifacts(analysis)).by_stack.get(
                    stack, 0)),
            # The encode step's own activation peak, read where the serving
            # path reads it (`activation_peak_at` of the step
            # `encode_phase.plan` bills this encoder by), with no framework
            # fusion and no framework overhead.
            "act_peak_bytes_per_rank": int(activation_peak_at(
                evaluate_encode(resolved, parallel, stack=stack, batch=1,
                                media=deployment.media),
                overhead_bytes=0)["peak_activation_bytes_per_rank"]),
            "handoff_rows": {
                h.name: [t.rows for t in handed[h.name]]
                for h in program.handoffs if h.stack == stack},
        }
    cache = EncoderOutputCache.of(deployment, scheduler_tokens=None)
    return {"media": media,
            "media_tokens": deployment.media_tokens(),
            "output_cache_bytes_per_token": int(cache.bytes_per_token),
            "max_tokens_per_item": cache.max_tokens_per_item,
            "stacks": stacks}


def encoder_stacks(resolved: Any) -> set:
    from backend.arch.eval.program import Program
    return {stack.name for stack in Program.of(resolved).stacks
            if stack.is_encoder}


def measure_snapshot(model: CorpusModel,
                     loader: Optional[Any] = None) -> Dict[str, Any]:
    """{tp: {reference: case}} for one snapshot; empty without encoders."""
    resolved = model.resolved(loader or shared_loader())
    if not encoder_stacks(resolved):
        return {}
    return {str(degree): {reference: measure_case(resolved, degree, reference)
                          for reference in REFERENCES}
            for degree in TP_DEGREES}


@dataclass(frozen=True)
class EncodeSurfacePin(JsonMapPin):
    def applies(self, model: CorpusModel) -> bool:
        return bool(encoder_stacks(model.resolved()))

    def measure(self, model: CorpusModel) -> Row:
        row = measure_snapshot(model)
        return {model.stem: row} if row else {}


ENCODE_SURFACE = EncodeSurfacePin(
    id="P11",
    title="encode surface",
    scope=Scope.CONDITIONAL,
    artifact="tests/golden/encode_surface.json",
    owner_tests=(
        "tests/backend/arch/eval/test_encode_surface_golden.py::"
        "TestEncodeSurfaceGolden::test_every_encoder_holds_its_pinned_surface",
    ),
    waits_on=("G4",),
)
