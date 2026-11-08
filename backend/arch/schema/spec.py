"""ArchSpec — the top-level architecture spec, composed from the sub-models."""
from __future__ import annotations

from backend.arch.law_text import read_law

import re
from dataclasses import dataclass
from typing import (Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Set,
                    Tuple)

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

import backend.arch.dims as dim_registry
from backend.arch.axis_propagation import call_site_dims_error
from backend.arch.ops_base import ProcessorStep, ShapeKind
from backend.arch.quant_schemes import QUANT_SCHEMES
from .custom_ops import CustomOpSpec
from .guards import (_check_quant_suffix_in_arch_spec,
                     _check_weight_dim_geometry)
from .graph import BlockGraph, resolve_block_input_ports
from .model import (CheckpointAddress, HISTORY_AXIS, LayerMap, ModelInputSpec,
                     ModelLayer, StreamLayerRef, StreamSpec,
                     DenoiseLoop, UnreadCheckpointModule)
from .quant import _QUANT_METHOD_NONE, QuantContext



def _all_graphs(spec: "ArchSpec") -> List[Tuple[str, "BlockGraph"]]:
    """Every graph this spec holds, named: its inline `blocks:` then the
    library `components:` the loader pulled in for it. The distinction is
    where the text was written, not what it is, and a spec-level check that
    looked at only one of them would pass a model for the reason that it
    authored its recurrence in the other place."""
    out: List[Tuple[str, "BlockGraph"]] = []
    if spec.blocks:
        out.extend(spec.blocks.items())
    if spec.components:
        out.extend(spec.components.items())
    return out



@dataclass(frozen=True)
class StreamNames:
    """Every stream the model declares, by the scope its name lives in.

    A layer is named by its path from the top of `model:` (`("mtp",
    "mtp.layers")`), because a name alone does not identify one: an MTP
    head's `layers` and the backbone's `layers` are two stacks.

      carried  — stack → the stream it carries across its block boundary.
                 Local: the name means something only in that stack, so
                 several stacks may use the same one.
      produced — appended stream name → the one stack whose `streams:`
                 declares it, and the declaration. The name is model-wide:
                 `ArchSpec.stream_names` refuses a second declaration of it
                 rather than keep either.
      inner    — `(stack, name)` → the declaration of an inner stream. Local
                 to its stack like a carried one: the readers of a stack
                 name the layer they read by its number in that stack
                 (`reads_from`), so an MTP head and the backbone may each
                 declare a stream of one name.
    """

    carried: Mapping[Tuple[str, ...], str]
    produced: Mapping[str, Tuple[Tuple[str, ...], "StreamSpec"]]
    inner: Mapping[Tuple[Tuple[str, ...], str], "StreamSpec"]

    def inner_names(self) -> FrozenSet[str]:
        """Every name some stack declares an inner stream under."""
        return frozenset(name for _stack, name in self.inner)


#: A dim-expression identifier, and whether a call's `(` follows it — a
#: called name is a function (`Min`, `ceiling`), not a dim.
_EXPR_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\()?")


def _dim_names(text: Any) -> set:
    """The dim identifiers a dim expression reads, calls excluded."""
    return {m.group(1) for m in _EXPR_NAME.finditer(str(text))
            if not m.group(2)}


def _streams_recurrent_state(
        op: str, custom_ops: Optional[Dict[str, "CustomOpSpec"]]) -> bool:
    """Whether the named op declares that it carries a state across decode
    steps. Both kinds answer under the same field name, so this asks the
    declaration rather than matching `mamba2_scan` and its neighbours by
    name — the set of recurrences is open, and a name list would be a
    second, staler copy of it."""
    from backend.arch.ops import OP_REGISTRY

    declaration = (custom_ops or {}).get(op) or OP_REGISTRY.get(op)
    return bool(getattr(declaration, "carries_recurrent_state", False))



def _validate_quant_context_methods(qc: "QuantContext") -> List[str]:
    """Cross-check that every `method` referenced by quant_context is
    registered under `backend/arch/quant_schemes/`. Returns a list
    of error strings (empty when OK).
    """
    if not qc.per_role and not qc.overrides:
        return []
    registered = set(QUANT_SCHEMES.keys()) | {_QUANT_METHOD_NONE}
    used = qc.known_methods()
    unknown = sorted(used - registered)
    if not unknown:
        return []
    return [
        f"quant_context references unregistered method(s) {unknown!r}; "
        f"known methods: {sorted(registered)}. Either register the "
        f"missing scheme under backend/arch/quant_schemes/<name>.yaml "
        f"or correct the spelling in quant_context."
    ]



def _layers_under(layer: ModelLayer) -> Iterator[Tuple[Tuple[str, ...], ModelLayer]]:
    """`layer` and every layer nested in it, with its path from `layer`."""
    def walk(entry: ModelLayer, prefix: Tuple[str, ...]):
        path = (*prefix, entry.name)
        yield path, entry
        for sub in entry.layers or ():
            yield from walk(sub, path)
    return walk(layer, ())


class ValidationExemptions(BaseModel):
    """Narrow, reason-bearing opt-outs from named validation contracts."""

    model_config = ConfigDict(extra="forbid")

    sliding_window: Optional[str] = None

    @field_validator("sliding_window")
    @classmethod
    def _reason_is_nonempty(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError(
                "validation_exemptions.sliding_window must explain why the "
                "checkpoint config's window is intentionally not modelled")
        return value




#: The `view:` vocabulary — facts the architecture view displays that are not
#: dims (a dim counts something; these do not): RoPE base and scaling, the FFN
#: gate nonlinearity, the norm epsilon, and the SWA-variant head widths and
#: RoPE base a few hybrid models publish (MiMo `swa_rope_theta`). A spec binds each key to the config path(s) its
#: models publish (`config.rope_theta ?? config.text_config.rope_theta`); the
#: loader keeps each value's native type. A key outside this tuple is a load
#: error naming it, so there is no second alias table anywhere downstream.
VIEW_KEYS: Tuple[str, ...] = (
    "rope_theta", "rope_scaling", "hidden_act", "norm.eps",
    "swa.num_heads", "swa.head_dim", "swa.v_head_dim", "swa.rope_theta",
)


class ArchSpec(BaseModel):
    """Top-level architecture specification."""
    schema_version: str = "1.0"
    name: str
    #: Spec this one is derived from — its lineage. Copied from the
    #: authored document.
    derived_from: Optional[str] = None

    # Dimension bindings: symbol name → config path or expression
    # e.g. {"d_model": "config.hidden_size",
    #       "d_head": "config.head_dim ?? 'd_model // n_heads'"}
    dims: Dict[str, str]
    # Symbols this spec declares for itself (`SpecDocument.dim_symbols`):
    # symbol → its dims.yaml row. Kept on the assembly product so the spec
    # fingerprint covers the classification; left out of a dump while empty,
    # so a spec that declares none fingerprints as it did before the field.
    # Not `LayerTree.dim_symbols`.
    dim_symbols: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, exclude_if=lambda v: not v)
    # Display bindings: VIEW_KEYS member → config path(s). See VIEW_KEYS.
    view: Dict[str, str] = {}

    # Block computation graphs, by template name. The name is the author's:
    # `blocks["block"]` is what a `template: block` layer resolves to, and it
    # is a graph like any other. There used to be a second field holding that
    # one graph, which every consumer here had to remember to prepend to
    # `blocks.items()` — six places, each free to forget.
    blocks: Optional[Dict[str, BlockGraph]] = None

    # Full model structure
    model: List[ModelLayer]

    # The tensors the model receives from outside its graph — token ids, an
    # encoder's image patches or audio features — by the name a layer's
    # `inputs:` reads each by. See `ModelInputSpec`.
    model_inputs: Dict[str, ModelInputSpec] = Field(default_factory=dict)

    # The checkpoint path the model's entries hang off (`root`); absent is
    # the file root. Not serialized when absent.
    checkpoint: Optional[CheckpointAddress] = Field(
        default=None, exclude_if=lambda v: v is None)

    # Reusable sub-DAG components (same schema as block graph).
    # A node whose op matches a component name is recursively expanded.
    components: Dict[str, BlockGraph] = {}

    # Reverse map: component_name → role (e.g. "gqa_attention" → "attention")
    # Populated during loading from the YAML components section.
    component_roles: Dict[str, str] = {}

    # Spec-scoped custom primitive ops — for novel compute primitives that
    # can't be expressed by composing existing OP_REGISTRY entries. Each entry
    # is a full CustomOpSpec; node.op names matching a key here are evaluated
    # via the custom definition. See CustomOpSpec for fields and usage.
    custom_ops: Dict[str, CustomOpSpec] = {}

    # Populated by the loader from `# UNRESOLVED:` comment blocks at the
    # top of the spec YAML — the agent's escape-hatch marker when it gave
    # up before reaching diff=0. The UI surfaces this as a warning banner
    # so users know the spec is partial and needs human review.
    unresolved_note: Optional[str] = None
    # Source path of the spec YAML (relative or absolute). Used for cache
    # management and UI "open in editor" links. Populated by loader.
    source_path: Optional[str] = None

    validation_exemptions: ValidationExemptions = Field(
        default_factory=ValidationExemptions)

    #: Modules the checkpoint stores and this path never runs
    #: (`UnreadCheckpointModule`).
    checkpoint_unread: List[UnreadCheckpointModule] = Field(
        default_factory=list, exclude_if=lambda v: not v)

    #: The denoise loop that runs the language model's step per request
    #: (`DenoiseLoop`); None for a model whose step is not one.
    denoise_loop: Optional[DenoiseLoop] = Field(
        default=None, exclude_if=lambda v: v is None)

    # Quantization context.
    # Optional; absent means the model is treated as unquantized
    # (HF safetensors header dtype is canonical at load time).
    quant_context: Optional[QuantContext] = None

    @model_validator(mode="after")
    def validate_block_definitions(self) -> "ArchSpec":
        """Ensure either block or blocks is defined, and model references are valid."""
        dim_registry.refuse_written_derived(
            self.dims, "the model's `dims:`", local=self.dim_symbols)
        if self.blocks is None and not self.components:
            raise ValueError("Must define either 'blocks' or 'components'")
        if self.checkpoint is not None:
            self.checkpoint.only("root", where="spec")
        self._check_denoise_loop()
        self._check_view()
        self._check_model_layer_templates()
        self._check_quant_override_scopes()
        self._check_layer_wiring()
        self._check_model_inputs()
        self._check_media_layers()
        self._check_calls()
        self._check_prompt_slots()
        self._check_required_inputs()
        self._check_custom_op_callsites()
        self.check_call_bindings()
        self._check_call_site_dims()
        self._check_carried_streams()
        self._check_stream_scopes()
        self._check_stream_seeds()
        self._check_appended_streams()
        self._check_recurrent_state_carriers()
        self._check_delegated_cache_kinds()
        self._check_quant_context()
        self._check_custom_op_collisions()
        self._resolve_all_input_ports()
        return self

    @property
    def checkpoint_root(self) -> str:
        """The checkpoint path the model's entries hang off, empty when the
        spec declares none."""
        return (self.checkpoint.root or "") if self.checkpoint is not None else ""

    def _check_view(self) -> None:
        """`view:` keys come from VIEW_KEYS and every binding is a config
        path (or a `??` chain of them) — no expressions, no literals, no
        `?? absent`: a value a config does not publish is simply not shown."""
        unknown = [k for k in self.view if k not in VIEW_KEYS]
        if unknown:
            raise ValueError(
                f"view: unknown key(s) {unknown}; the vocabulary is {list(VIEW_KEYS)}")
        for key, binding in self.view.items():
            parts = [p.strip() for p in str(binding).split("??")]
            bad = [p for p in parts if not p.startswith("config.") or p == "config."]
            if bad:
                raise ValueError(
                    f"view.{key}: every alternative must be a config path "
                    f"(`config.<field>`), got {bad!r} in {binding!r}")

    def _weight_shape(self, layer: "ModelLayer") -> Optional[Tuple[str, ...]]:
        from backend.arch.ops import OP_REGISTRY
        if layer.op == "parameter_tensor" and layer.shape:
            return tuple(str(axis) for axis in layer.shape)
        declaration = (self.custom_ops or {}).get(layer.op or "") \
            or OP_REGISTRY.get(layer.op or "")
        template = tuple(
            getattr(declaration, "weight_shape", None) or ())
        if not template:
            return None
        bindings = {key: str(value) for key, value in layer.dims.items()}
        if layer.dim is not None:
            bindings["dim"] = str(layer.dim)

        def _bind(axis: Any) -> str:
            text = str(axis)
            if text in bindings:
                return bindings[text]
            for key in sorted(bindings, key=len, reverse=True):
                text = re.sub(
                    rf"\b{re.escape(key)}\b", f"({bindings[key]})", text)
            return text

        return tuple(_bind(axis) for axis in template)

    @staticmethod
    def _same_axis(left: str, right: str) -> bool:
        if re.sub(r"\s+", "", left) == re.sub(r"\s+", "", right):
            return True
        try:
            import sympy as sp
            return bool(sp.simplify(read_law(left) - read_law(right)) == 0)
        except (TypeError, ValueError, SyntaxError):
            return False

    def _validate_tie(self, layer: "ModelLayer", target: "ModelLayer") -> None:
        # A target whose own tie is conditional owns its weight whenever the
        # condition is off, and hands this reader on to its owner when it is
        # on (`resolve_tied_weights` follows the chain) — an MTP head tied to
        # an lm_head that is tied to the embedding only under
        # `tie_word_embeddings`. An unconditional alias is never an owner.
        if target.tied_weight is not None and target.tied_weight.when is None:
            raise ValueError(
                f"Model layer {layer.name!r} ties to {target.name!r}, which "
                "is itself an alias. Tie directly to the canonical owner.")
        own_shape = self._weight_shape(layer)
        target_shape = self._weight_shape(target)
        if own_shape is None or target_shape is None:
            raise ValueError(
                f"Model layer {layer.name!r} ties to {target.name!r}, but "
                "both sides must be direct weight-bearing ops with declared "
                "weight geometry")
        compatible = (
            len(own_shape) == len(target_shape)
            and all(self._same_axis(left, right) for left, right in
                    zip(own_shape, target_shape, strict=True))
        )
        if not compatible:
            raise ValueError(
                f"Model layer {layer.name!r} weight shape {own_shape} cannot "
                f"tie to {target.name!r} shape {target_shape}")

    def _check_model_layer_templates(self) -> None:
        """Every `template:`/`assignments:` reference names a known block or component; model-layer weight/quant guards run here too."""
        # Collect available template names. Components are equally valid as
        # block templates — model layers can reference any compound op.
        def _walk(layers: List["ModelLayer"]) -> List["ModelLayer"]:
            out = []
            for ml in layers:
                out.append(ml)
                if ml.layers:
                    out.extend(_walk(ml.layers))
            return out

        templates = set()
        if self.blocks is not None:
            templates.update(self.blocks.keys())
        if self.components:
            templates.update(self.components.keys())

        for layer in _walk(self.model):
            if layer.template and layer.assignments:
                # Both paths are layer-0 metadata consumers keyed on ONE of
                # these fields (`spec_tree.enumerate_blocks` takes template
                # first); declaring both lets two consumers pick differently
                # and silently skip or double-count the block. `per_layer:`
                # lowers to `assignments`, so this is reachable from the
                # authored surface too — reject it here, where the answer is
                # still free.
                raise ValueError(
                    f"Model layer {layer.name!r} declares both `template:` "
                    f"and `assignments:` — one layer, one block-binding path; "
                    f"split it or drop one."
                )
            if layer.template and layer.template not in templates:
                raise ValueError(
                    f"Model layer {layer.name!r} references unknown template "
                    f"{layer.template!r}. Available: {sorted(templates)}"
                )
            if layer.assignments:
                for assign in layer.assignments:
                    if assign.template not in templates:
                        raise ValueError(
                            f"Assignment references unknown template "
                            f"{assign.template!r}. Available: {sorted(templates)}"
                        )
            # Quant-suffix lockdown at the model-layer level.
            _check_quant_suffix_in_arch_spec(
                "model_layer", layer.name, layer.name, layer.op or "",
                layer.quant_metadata,
            )
            # Geometric weight-dim check at model-layer level (top-level
            # parameter_vector layers like hc_head_*, pos_emb_*, etc.).
            _check_weight_dim_geometry(
                "model_layer", layer.name, layer.name,
                layer.op or "", layer.dim,
            )

    def _check_quant_override_scopes(self) -> None:
        """A scoped quant override must name an aux stack / a component that
        exists."""
        # Otherwise the scope never matches anything and the entry is dead
        # config that fails to say so.
        if self.quant_context and self.quant_context.overrides:
            aux_stacks = {layer.name for layer in self.model
                          if layer.group and layer.aux_kind}
            graphs = {*(self.components or {}), *(self.blocks or {})}
            for entry in self.quant_context.overrides:
                selector = entry.select
                if selector is None:
                    continue
                if selector.stack and selector.stack not in aux_stacks:
                    raise ValueError(
                        f"quant_context override scopes itself to stack "
                        f"{selector.stack!r}, but the model declares no aux "
                        f"group of that name. Available: "
                        f"{sorted(aux_stacks)}"
                    )
                if selector.component and selector.component not in graphs:
                    raise ValueError(
                        f"quant_context override scopes itself to component "
                        f"{selector.component!r}, but the spec declares no "
                        f"component of that name. Available: "
                        f"{sorted(graphs)}"
                    )

    def _check_layer_wiring(self) -> None:
        """`inputs:`/`tied_weight` references resolve to EARLIER layers
        (top-level, plus earlier siblings inside a group); names don't repeat."""
        # `inputs:` references must resolve to an EARLIER top-level layer
        # in the same `model:` list. We restrict to forward references
        # (no out-of-order, no cycles) so the model-level diagram stays
        # acyclic and downstream renderers can rely on positional order.
        # Group-nested references use the same earlier-only rule, extended with
        # earlier siblings in that group.
        seen_names: set = set()
        seen_layers: Dict[str, ModelLayer] = {}
        # A model input is produced before any layer runs.
        model_inputs = set(self.model_inputs)
        # An appended stream is readable once the stack that appends to it
        # is declared, wherever a later layer sits.
        seen_streams: set = set()
        seen_inner: set = set()
        # A stack that carries no stream has no single value to read by its
        # name, at any depth of the model tree; what it produces leaves it
        # as an appended stream.
        every = [owner for _path, owner in self.layer_paths()]
        parallel = {owner.name for owner in every if owner.is_parallel_stack}
        for owner in every:
            unreadable = [ref for ref in owner.inputs or () if ref in parallel]
            if unreadable:
                raise ValueError(
                    f"Model layer {owner.name!r} reads {unreadable[0]!r} by "
                    f"name, a repeated stack that carries no stream: its "
                    f"iterations hand nothing on, so its name reads no one "
                    f"value. Append what it produces to a stream "
                    f"(`from: out`) and read that stream")
            both = sorted(set(owner.inputs or ())
                          & {*owner.read_streams, *owner.concat_streams})
            if both:
                raise ValueError(
                    f"Model layer {owner.name!r} reads appended stream(s) "
                    f"{both} both at its block boundary (`inputs:`) and at "
                    f"an inner node (`read: each` / `read: concat`); read "
                    f"each one one way")

        for layer in self.model:
            if layer.name in seen_names:
                raise ValueError(
                    f"top-level model layer name {layer.name!r} is duplicated")
            if (layer.tied_weight is not None
                    and layer.tied_weight.to not in seen_names):
                raise ValueError(
                    f"Model layer {layer.name!r} ties its weight to "
                    f"{layer.tied_weight.to!r}, which is not an earlier "
                    f"top-level model layer. Available: {sorted(seen_names)}")
            if layer.tied_weight is not None:
                self._validate_tie(layer, seen_layers[layer.tied_weight.to])
            if layer.inputs:
                for ref in layer.inputs:
                    if (ref not in seen_names and ref not in seen_streams
                            and ref not in model_inputs):
                        raise ValueError(
                            f"Model layer {layer.name!r} `inputs:` references "
                            f"unknown layer {ref!r}. Each entry must match the "
                            f"`name` of an earlier top-level model layer, a "
                            f"stream, or a name in `model_inputs:`. "
                            f"Available so far: "
                            f"{sorted(seen_names | model_inputs)}"
                        )
            self._refuse_inner_stream_read_by_stack(
                layer, seen_inner - seen_names)
            seen_names.add(layer.name)
            seen_layers[layer.name] = layer
            seen_streams.update(layer.appended_streams)
            # An inner stream's final value is readable by name after its
            # stack, as an appended stream's is.
            seen_inner.update(layer.inner_streams)
            seen_streams.update(layer.inner_streams)
            if layer.layers:
                # Group-nested `inputs:` follow the same rule as top-level:
                # every ref must name something already produced — an
                # earlier top-level layer, or an earlier sibling in the
                # same group. (LongCat's MTP block reads the backbone
                # `norm` from inside the `mtp` group; the group's own
                # `inputs:` lands on its first stage, so a mid-group stage
                # that wants the backbone must say so itself.)
                siblings: set = set()
                sibling_layers: Dict[str, ModelLayer] = {}
                for sub in layer.layers:
                    if (sub.tied_weight is not None
                            and sub.tied_weight.to not in seen_names
                            and sub.tied_weight.to not in siblings):
                        raise ValueError(
                            f"Group-nested layer {sub.name!r} ties its weight "
                            f"to {sub.tied_weight.to!r}, which is neither an "
                            f"earlier top-level layer nor an earlier sibling "
                            f"in group {layer.name!r}. Available: "
                            f"{sorted(seen_names | siblings)}")
                    if sub.tied_weight is not None:
                        target = (seen_layers.get(sub.tied_weight.to)
                                  or sibling_layers[sub.tied_weight.to])
                        self._validate_tie(sub, target)
                    if sub.inputs:
                        for ref in sub.inputs:
                            if (ref not in seen_names and ref not in siblings
                                    and ref not in seen_streams
                                    and ref not in model_inputs):
                                raise ValueError(
                                    f"Group-nested layer {sub.name!r} `inputs:` "
                                    f"references {ref!r}, which is neither an "
                                    f"earlier top-level layer, an earlier "
                                    f"sibling in group {layer.name!r}, nor a "
                                    f"name in `model_inputs:`. Available: "
                                    f"{sorted(seen_names | siblings | model_inputs)}"
                                )
                    self._refuse_inner_stream_read_by_stack(
                        sub, seen_inner - seen_names - siblings)
                    siblings.add(sub.name)
                    sibling_layers[sub.name] = sub
                    seen_streams.update(sub.appended_streams)
                    seen_inner.update(sub.inner_streams)
                    seen_streams.update(sub.inner_streams)
                # An encoder's stages are tensors the text model reads by
                # name (Qwen3-VL's decoder reads `visual.merger.linear_fc2`,
                # which replaces the image-token embeddings). A head group's
                # labels (`mtp`, `layers`) are not: they repeat backbone
                # names.
                if layer.name in self.encoder_groups():
                    seen_names.update(sub.name for sub in layer.layers)

    @staticmethod
    def _refuse_inner_stream_read_by_stack(layer: ModelLayer,
                                           inner: set) -> None:
        """An inner stream named in `inputs:` is read whole, its final
        value, by a model-level `op:`; a stack binds its `inputs:` to its
        block's formals and reads an inner stream through `reads_from`."""
        named = sorted(set(layer.inputs or ()) & inner)
        if named and layer.op is None:
            raise ValueError(
                f"Model layer {layer.name!r} names inner stream(s) {named} in "
                f"`inputs:`; only a model-level `op:` reads an inner stream "
                f"by name (its final value). A stack's layers read it "
                f"through `reads_from`")

    def _check_denoise_loop(self) -> None:
        """A denoise loop steps latents the model declares: each starts and
        ends as a model input of one kind and width, and is updated by a
        top-level layer of the language model's step; its step count reads
        the spec's dims."""
        loop = self.denoise_loop
        if loop is None:
            return
        unknown = sorted(str(sym) for sym in read_law(loop.steps).free_symbols
                         if str(sym) not in self.dims)
        if unknown:
            raise ValueError(
                f"denoise_loop.steps {loop.steps!r} names {unknown}, which "
                f"this spec's `dims:` does not bind")
        top = {layer.name for layer in self.model if not layer.group}
        for latent in loop.latents:
            ends = [self.model_inputs.get(latent.start),
                    self.model_inputs.get(latent.end)]
            if None in ends:
                raise ValueError(
                    f"denoise_loop latent {latent.start!r} -> {latent.end!r}: "
                    f"both are model inputs (`model_inputs:`)")
            if (ends[0].kind, ends[0].width) != (ends[1].kind, ends[1].width):
                raise ValueError(
                    f"denoise_loop latent {latent.start!r} -> {latent.end!r}: "
                    f"one latent, so one kind and width; got "
                    f"{(ends[0].kind, ends[0].width)} and "
                    f"{(ends[1].kind, ends[1].width)}")
            if latent.update not in top:
                raise ValueError(
                    f"denoise_loop latent {latent.start!r}: `update: "
                    f"{latent.update!r}` is no top-level layer of the "
                    f"language model's step")

    def _check_model_inputs(self) -> None:
        """Each model input is a tensor some layer reads, under a name no
        layer or stream takes. The token ids are read outside the encoders,
        a media input only inside one, and an image only by `patchify`; an
        audio frame's width is written in the model's own dims, and a media
        input's axes are its own names, bound by the workload."""
        encoders = self.encoder_groups()
        unattached = sorted(
            layer.name for layer in self.model
            if layer.group is True and layer.aux_kind is not None
            and not layer.inputs and layer.name not in encoders)
        if unattached:
            raise ValueError(
                f"group(s) {unattached} name no `inputs:` and read no media "
                f"input: a modality encoder reads the media model input it "
                f"encodes (`model_inputs:` declares it, a `patchify` reads "
                f"an image), and any other group names the backbone tensor "
                f"it reads in `inputs:`")
        for index, layer in enumerate(self.model):
            if layer.name not in encoders:
                continue
            for sub_name, outside in self._encoder_backbone_reads(index):
                raise ValueError(
                    f"encoder group {layer.name!r} reads its media input "
                    f"and names no `inputs:`, and its layer {sub_name!r} "
                    f"reads {outside}, which the language model produces. "
                    f"A modality encoder reads its media and its own layers "
                    f"only; a group that reads the backbone names that "
                    f"tensor in the group's `inputs:` and reads no media "
                    f"input")
        if not self.model_inputs:
            return
        names = set(self.model_inputs)
        taken = sorted(names & {layer.name
                                for _path, layer in self.layer_paths()})
        streams = self.stream_names()
        taken += sorted(names & (set(streams.produced)
                                 | streams.inner_names()))
        if taken:
            raise ValueError(
                f"model_inputs {taken} share a name with a model layer or a "
                f"stream; an `inputs:` entry must name one tensor. Rename "
                f"the model input.")
        readers = self.model_input_readers()
        unmodeled = sorted(name for name, found in readers.items()
                           if found and self.model_inputs[name].unmodeled)
        if unmodeled:
            raise ValueError(
                f"model_inputs {unmodeled} are declared `unmodeled:` and "
                f"read by a layer; an unmodeled input is not walked, so "
                f"no layer reads it. Delete `unmodeled:` or the reader.")
        unread = sorted(name for name, found in readers.items()
                        if not found and not self.model_inputs[name].unmodeled)
        if unread:
            raise ValueError(
                f"model_inputs {unread} are read by no layer. Name each in "
                f"the `inputs:` of the layer that reads it, or delete it.")
        for name, found in sorted(readers.items()):
            kind = self.model_inputs[name].kind
            for reader, group in found:
                encoder = group is not None and group.name in encoders
                if (kind == "tokens") == encoder:
                    where = (f"encoder group {group.name!r}" if encoder
                             else "a stack that is not an encoder")
                    raise ValueError(
                        f"Model layer {reader.name!r} reads model input "
                        f"{name!r} of kind {kind!r} from {where}. The token "
                        f"ids are read by the language model; an image or "
                        f"an audio input only by the encoder group that "
                        f"encodes it.")
                if kind in ("image", "video") and \
                        self._processor_step(reader) is not ProcessorStep.CUT:
                    raise ValueError(
                        f"Model layer {reader.name!r} reads {kind} input "
                        f"{name!r} with op {reader.op or reader.template!r}; "
                        f"an {kind} is read by a `patchify` layer (an op "
                        f"declaring `processor_step: cut`), which "
                        f"declares how the processor cuts it into patch rows "
                        f"(`dims: {{patch, temporal, channels, align}}`, "
                        f"`segment:`, `source_anchor:`)")
        declared = set(self.dims)
        text = {"s_ctx", "s_kv", "s_query", "s_past", "B"}
        seen_axes: Dict[str, str] = {}
        for name, entry in sorted(self.model_inputs.items()):
            for axis in entry.axes:
                if axis in declared or axis in text:
                    raise ValueError(
                        f"model_inputs.{name}.axes names {axis!r}, which "
                        f"{'this spec binds in `dims:`' if axis in declared else 'the language model binds'}; "
                        f"a media input's axes are its own names, given by "
                        f"the workload")
                if axis in seen_axes:
                    raise ValueError(
                        f"model_inputs.{name}.axes names {axis!r}, which "
                        f"model input {seen_axes[axis]!r} also names; each "
                        f"input's axes are its own")
                seen_axes[axis] = name
            if entry.per_item_of is not None:
                parent = self.model_inputs.get(entry.per_item_of)
                if (parent is None or parent.kind == "tokens"
                        or entry.per_item_of == name):
                    raise ValueError(
                        f"model_inputs.{name}.per_item_of names "
                        f"{entry.per_item_of!r}, which is not another media "
                        f"input of this spec")
            expressions = dict(entry.fixed)
            if entry.width is not None:
                expressions["width"] = entry.width
            for field, text_expr in sorted(expressions.items()):
                try:
                    symbols = read_law(text_expr).free_symbols
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise ValueError(
                        f"model_inputs.{name}.{field} {text_expr!r} is not a "
                        f"dim expression: {exc}") from exc
                unknown = sorted(str(sym) for sym in symbols
                                 if str(sym) not in declared)
                if unknown:
                    raise ValueError(
                        f"model_inputs.{name}.{field} {text_expr!r} names "
                        f"{unknown}, which this spec's `dims:` does not "
                        f"bind. Write it in the model's dims.")

    def _check_calls(self) -> None:
        """A call (`ModelLayer.call`) runs a modality encoder's module again
        on other inputs — a model's videos through its image tower, Step's
        crops through its vision tower — so it sits in an encoder group or
        is one; the language model's step runs each module once."""
        encoders = self.encoder_groups()
        for layer in self.model:
            calls = [sub for _path, sub in _layers_under(layer) if sub.call]
            if calls and layer.name not in encoders:
                raise ValueError(
                    f"Model layer {calls[0].name!r} runs {calls[0].call!r} "
                    f"again (`call:`), which only a modality encoder does; "
                    f"{layer.name!r} is no encoder group")

    def _check_media_layers(self) -> None:
        """The declarations of the layers that turn a media input into rows:
        the ops declaring a `processor_step`. Each cites the processor
        (`source_anchor:`) and binds the dims its op requires; a cut
        (`patchify`) reads one image or video and declares its attention
        `segment:`. `segment` and `source_anchor` belong to those layers
        only."""
        for _path, layer in self.layer_paths():
            step = self._processor_step(layer)
            if step is None:
                if layer.segment is not None or layer.source_anchor:
                    raise ValueError(
                        f"Model layer {layer.name!r} declares "
                        f"`segment:`/`source_anchor:`, which a `patchify`, "
                        f"`pad_rows`, `unpad` or `group_items` layer "
                        f"declares (an op with a `processor_step`); "
                        f"{layer.op or layer.template!r} reads neither")
                continue
            required = self._op_declaration(layer.op).required_dims
            missing = [key for key in required if key not in layer.dims]
            if missing or not (layer.source_anchor or "").strip():
                raise ValueError(
                    f"Model layer {layer.name!r} ({layer.op}) declares "
                    f"`dims:` {sorted(layer.dims)} and "
                    f"`source_anchor: {layer.source_anchor!r}`; it needs "
                    f"`dims:` {list(required)} and a "
                    f"`source_anchor` citing the processor or framework "
                    f"source each value is read from")
            if (step is ProcessorStep.CUT) != (layer.segment is not None):
                raise ValueError(
                    f"Model layer {layer.name!r}: `segment:` "
                    f"(`item` | `temporal_patch`, the processor's "
                    f"`cu_seqlens` unit) is declared on a `patchify` layer "
                    f"(`processor_step: cut`) and nowhere else")
            if step is ProcessorStep.CUT:
                refs = list(layer.inputs or ())
                if len(refs) != 1 or refs[0] not in self.model_inputs or \
                        self.model_inputs[refs[0]].kind not in ("image",
                                                                "video"):
                    raise ValueError(
                        f"Model layer {layer.name!r} ({layer.op}) reads "
                        f"{refs}; it cuts exactly one image or video model "
                        f"input into patches")

    def _op_declaration(self, op: Optional[str]) -> Any:
        """The declaration of op `op`: the spec's custom op of that name,
        else the registered primitive; None for no op or an unknown one."""
        from backend.arch.ops import OP_REGISTRY
        if not op:
            return None
        return (self.custom_ops or {}).get(op) or OP_REGISTRY.get(op)

    def _processor_step(self, layer: ModelLayer) -> Optional[str]:
        """The step of a media processor's work `layer`'s op declares
        (`OpSpec.processor_step`); None for any other layer."""
        declaration = self._op_declaration(layer.op)
        return declaration.processor_step if declaration is not None else None

    def encoder_edges(self) -> List[Tuple[ModelLayer, str, Optional[str],
                                          ModelLayer]]:
        """Every edge from an encoder group into the rest of the model: the
        reading layer, the name it reads, the stream entry it reads it
        through (None for an `inputs:` read), and the encoder group that
        produces it. A layer reads an encoder's stage by name, or a stream
        an encoder appends to, either in `inputs:` or through a `read:`
        entry of its `streams:`."""
        produced: Dict[str, ModelLayer] = {}
        encoders = self.encoder_groups()
        for layer in self.model:
            if layer.name not in encoders:
                continue
            for _path, sub in _layers_under(layer):
                produced[sub.name] = layer
                for stream in sub.appended_streams:
                    produced[stream] = layer
        out: List[Tuple[ModelLayer, str, Optional[str], ModelLayer]] = []
        for path, layer in self.layer_paths():
            top = next(l for l in self.model if l.name == path[0])
            if top.name in encoders:
                continue
            out.extend((layer, ref, None, produced[ref])
                       for ref in layer.inputs or () if ref in produced)
            out.extend((layer, stream, stream, produced[stream])
                       for stream, decl in layer.streams.items()
                       if decl.read and stream in produced)
        return out

    def _check_prompt_slots(self) -> None:
        """Every edge from an encoder into the language model says which
        prompt positions it fills (`prompt_slots:`), and nothing else does.
        A domain is a media input the producing encoder reads, not one whose
        items belong to another's; every such media input has exactly one
        defining edge, and every encoder defines at least one domain."""
        edges = self.encoder_edges()
        on_edge = set()
        readers = self.model_input_readers() if self.model_inputs else {}
        defining: Dict[str, List[str]] = {}
        encoders_defining: set = set()
        by_layer: Dict[int, List[str]] = {}
        for layer, ref, stream, encoder in edges:
            if stream is None:
                by_layer.setdefault(id(layer), []).append(ref)
                if (len(by_layer[id(layer)]) > 1
                        and not isinstance(layer.prompt_slots, dict)):
                    raise ValueError(
                        f"Model layer {layer.name!r} reads encoder outputs "
                        f"{by_layer[id(layer)]}; give each the prompt "
                        f"positions it fills, `prompt_slots: {{<the name it "
                        f"reads>: {{domain, defines, source_anchor}}, ...}}`")
            slots = (layer.streams[stream].prompt_slots if stream is not None
                     else layer.slots_for(ref))
            where = (f"stream entry {stream!r} of {layer.name!r}"
                     if stream is not None else f"model layer {layer.name!r}")
            if slots is None:
                raise ValueError(
                    f"{where} reads {ref!r} from encoder group "
                    f"{encoder.name!r} and declares no `prompt_slots`. Write "
                    f"`prompt_slots: {{domain: <media input>, defines: "
                    f"true|false, source_anchor: ...}}`: `defines: true` "
                    f"where the framework writes the encoder's rows into the "
                    f"media's prompt positions, `false` where it adds onto "
                    f"positions another edge defines")
            on_edge.add(id(slots))
            entry = self.model_inputs.get(slots.domain)
            if entry is None or entry.kind == "tokens":
                raise ValueError(
                    f"{where}: `prompt_slots.domain: {slots.domain!r}` is "
                    f"not a media input in `model_inputs:` (media inputs: "
                    f"{sorted(n for n, e in self.model_inputs.items() if e.kind != 'tokens')})")
            if entry.per_item_of is not None:
                raise ValueError(
                    f"{where}: `prompt_slots.domain: {slots.domain!r}` holds "
                    f"items of {entry.per_item_of!r}, whose prompt positions "
                    f"they fill; name {entry.per_item_of!r}")
            if not any(group is encoder for _r, group in readers[slots.domain]):
                raise ValueError(
                    f"{where}: `prompt_slots.domain: {slots.domain!r}` is "
                    f"not read by encoder group {encoder.name!r}, which "
                    f"produces {ref!r}")
            if slots.defines:
                defining.setdefault(slots.domain, []).append(where)
                encoders_defining.add(encoder.name)
        for _path, layer in self.layer_paths():
            declared = (list(layer.prompt_slots.values())
                        if isinstance(layer.prompt_slots, dict)
                        else [layer.prompt_slots])
            stray = [f"model layer {layer.name!r}"
                     for slots in declared
                     if slots is not None and id(slots) not in on_edge]
            stray += [f"stream entry {name!r} of {layer.name!r}"
                      for name, decl in layer.streams.items()
                      if decl.prompt_slots is not None
                      and id(decl.prompt_slots) not in on_edge]
            if stray:
                raise ValueError(
                    f"{stray[0]} declares `prompt_slots` but reads no "
                    f"encoder output; `prompt_slots` belongs on the edge "
                    f"that carries an encoder's rows into the language model")
        twice = {d: w for d, w in defining.items() if len(w) > 1}
        if twice:
            domain, where = sorted(twice.items())[0]
            raise ValueError(
                f"domain {domain!r} has {len(where)} defining edges "
                f"({where}); the prompt positions of a media input are "
                f"defined once, and every other edge onto them declares "
                f"`defines: false`")
        encoders = self.encoder_groups()
        # A decoder (a diffusion pipeline's autoencoder decoder) is the
        # pipeline's sink: it reads the final latents and hands the language
        # model nothing.
        silent = sorted(layer.name for layer in self.model
                        if layer.name in encoders
                        and layer.aux_kind != "decoder"
                        and layer.name not in encoders_defining)
        if silent:
            raise ValueError(
                f"encoder group(s) {silent} define no prompt positions: no "
                f"edge reads their output with `prompt_slots: {{defines: "
                f"true}}`. Write the layer that merges the encoder's rows "
                f"into the language model's embeddings (HF "
                f"`masked_scatter`, vLLM `_merge_multimodal_embeddings`: "
                f"`scatter_rows_replace`)")
        # What only a decoder reads defines no prompt positions either.
        decoders = {sub.name for layer in self.model
                    if layer.group is True and layer.aux_kind == "decoder"
                    for sub in (layer, *(s for _p, s in _layers_under(layer)))}
        readers = self.model_input_readers()
        undefined = sorted(
            name for name, entry in self.model_inputs.items()
            if entry.kind != "tokens" and entry.per_item_of is None
            and name not in defining
            and not (readers[name] and all(
                group is not None and group.name in decoders
                for _reader, group in readers[name])))
        if undefined:
            raise ValueError(
                f"media input(s) {undefined} have no defining edge: no edge "
                f"from the encoder that reads them into the language model "
                f"declares `prompt_slots: {{domain: {undefined[0]}, defines: "
                f"true}}`. Each media input's prompt positions are defined "
                f"once — on the layer that merges its encoder rows into the "
                f"token embeddings (HF `masked_scatter`, vLLM "
                f"`_merge_multimodal_embeddings`)")

    def _encoder_backbone_reads(self, index: int
                                ) -> Iterator[Tuple[str, str]]:
        """Each read of a layer nested in top-level group `index` that
        resolves outside the group — `(layer name, the read)` — by the
        resolution the walk uses: a name in `inputs:` to the layers its
        scope admits (`layers_named`), an appended stream to the layer that
        appends to it, a `reads_from` `{stack, layer}` to that layer's path
        (`layer_position`)."""
        streams = self.stream_names().produced
        group = self.model[index].name
        inputs = self.model_inputs or {}

        def walk(layers: Sequence[ModelLayer], path: Tuple[int, ...]):
            for k, sub in enumerate(layers or ()):
                yield (*path, k), sub
                yield from walk(sub.layers, (*path, k))

        for at, sub in walk(self.model[index].layers, (index,)):
            for ref in sub.inputs or ():
                found = self.layers_named(at, ref)
                if found:
                    if any(p[0] != index for p in found):
                        yield sub.name, f"{ref!r}"
                elif ref in streams:
                    if streams[ref][0][0] != group:
                        yield sub.name, f"stream {ref!r}"
                elif ref not in inputs:
                    yield sub.name, f"{ref!r}"
            for name, decl in sub.streams.items():
                if decl.read and name in streams \
                        and streams[name][0][0] != group:
                    yield sub.name, f"stream {name!r}"
            declared = [*(sub.reads_from or {}).items(),
                        *(item for entry in sub.assignments or ()
                          for item in (entry.reads_from or {}).items())]
            for name, value in declared:
                refs = ([value] if isinstance(value, StreamLayerRef)
                        else [v for v in value.map.values()
                              if isinstance(v, StreamLayerRef)]
                        if isinstance(value, LayerMap)
                        and isinstance(value.map, dict) else [])
                for ref in refs:
                    source = self.layer_position(ref.stack.split("/"))
                    if source is None or source[0] != index:
                        yield sub.name, (f"stream {name!r} from layer "
                                         f"{ref.layer} of {ref.stack!r} "
                                         f"(`reads_from`)")

    def encoder_groups(self) -> FrozenSet[str]:
        """The groups of the model's modality encoders: each top-level
        auxiliary group (`group: true` with an `aux_kind`) that names no
        `inputs:` and reads a media model input (`kind: image | audio`), with
        every group nested in it. The declaration decides, never a name: a
        module of the language model's own step (DeepSeek-V4's compressors
        and indexers, a draft head reading the backbone) is no encoder,
        whatever it is called, and a group that reads only the token ids is
        not one either."""
        media = {name for name, entry in (self.model_inputs or {}).items()
                 if entry.kind != "tokens"}
        out: Set[str] = set()
        for layer in self.model:
            if not (layer.group is True and layer.aux_kind is not None
                    and not layer.inputs):
                continue
            under = [sub for _path, sub in _layers_under(layer)]
            if any(ref in media for sub in under for ref in sub.inputs or ()):
                out.add(layer.name)
                out.update(sub.name for sub in under if sub.group)
        return frozenset(out)

    def model_input_readers(self) -> Dict[
            str, List[Tuple[ModelLayer, Optional[ModelLayer]]]]:
        """Each model input's readers: every layer naming it in `inputs:`,
        with the group it sits in (None at the top level)."""
        readers: Dict[str, List[Tuple[ModelLayer, Optional[ModelLayer]]]] = {
            name: [] for name in self.model_inputs}

        def walk(layers: Sequence[ModelLayer],
                 group: Optional[ModelLayer]) -> None:
            for layer in layers:
                for ref in layer.inputs or ():
                    if ref in readers:
                        readers[ref].append((layer, group))
                if layer.layers:
                    walk(layer.layers, layer)

        walk(self.model, None)
        return readers

    def _is_source_layer(self, layer: ModelLayer) -> bool:
        """A layer that reads no tensor: a zero-input op."""
        if not layer.op:
            return False
        from backend.arch.ops import OP_REGISTRY
        spec = OP_REGISTRY.get(layer.op) or (self.custom_ops or {}).get(layer.op)
        return spec is not None and spec.arity() == 0

    def _op_port_signatures(self, op_name: str,
                            port: int) -> Tuple[Tuple[str, ...], ...]:
        """The input shapes a registered op declares for input `port`, one
        per signature it declares (`shape_in`, then each `alt_shape_in`);
        empty when it declares none there. A single-tensor template is input
        0's."""
        from backend.arch.ops import OP_REGISTRY
        spec = OP_REGISTRY.get(op_name)
        if spec is None or not spec.shape_in:
            return ()
        out = []
        for template in (spec.shape_in, *(spec.alt_shape_in or ())):
            per_port = (tuple(template)
                        if isinstance(template[0], (list, tuple))
                        else (tuple(template),))
            if port < len(per_port):
                out.append(tuple(per_port[port]))
        return tuple(out)

    def _op_takes_axis(self, op_name: str) -> bool:
        """Whether a model-level op reads `axis:` itself: a broadcast, a
        narrowing, a selection or a join along the axis the layer names,
        unless the op fixes that axis in its own declaration."""
        from backend.arch.ops import OP_REGISTRY
        spec = OP_REGISTRY.get(op_name) or (self.custom_ops or {}).get(op_name)
        if spec is None:
            return False
        return (ShapeKind(spec.shape_kind) in (
                    ShapeKind.BROADCAST, ShapeKind.AXIS_REPLACE,
                    ShapeKind.AXIS_SELECT, ShapeKind.AXIS_CONCAT)
                and spec.axis_locked is None)

    def _check_required_inputs(self) -> None:
        """Every layer that reads a tensor names it in `inputs:`/`in:`; only
        a zero-input op omits it."""
        def walk(layers: List[ModelLayer]) -> None:
            for layer in layers:
                if layer.group and layer.layers:
                    walk(layer.layers)
                    if layer.inputs:
                        wanted = set(layer.inputs)
                        if not any(wanted & set(sub.inputs or [])
                                   for sub in layer.layers):
                            raise ValueError(
                                f"Group {layer.name!r} declares "
                                f"inputs {list(layer.inputs)} but no child "
                                f"names any of them. A child that reads the "
                                f"group's attachment must say so."
                            )
                    continue
                if self._is_source_layer(layer):
                    continue
                if not layer.inputs:
                    raise ValueError(
                        f"Model layer {layer.name!r} reads a tensor and must "
                        f"name it in `inputs:`/`in:`: an earlier layer, a "
                        f"stream, or a name in `model_inputs:` (token ids are "
                        f"`input_ids: {{kind: tokens}}`; an image is `{{kind: "
                        f"image, axes: [n_img, img_h, img_w]}}` read by a "
                        f"`patchify`, an audio clip `{{kind: audio, axes: "
                        f"[n_aud, s_audio]}}`). Only a zero-input op (a "
                        f"parameter, a buffer) names none."
                    )

        walk(self.model)

    def _check_custom_op_callsites(self) -> None:
        """Custom-op arity + port references at call sites."""
        # Custom-op call-site arity check. Every graph node whose op resolves
        # to a custom_op must pass exactly the number of inputs the op's
        # `signature` declares. Catches bugs like wiring 3 inputs to an op
        # whose formula references 4.
        if self.custom_ops:
            graphs: List[Tuple[str, "BlockGraph"]] = []
            if self.blocks:
                graphs.extend(self.blocks.items())
            if self.components:
                graphs.extend(self.components.items())
            for graph_name, bg in graphs:
                for node_name, node in bg.graph.items():
                    cop = self.custom_ops.get(node.op)
                    if cop is None:
                        continue
                    expected = cop.arity()
                    actual = len(node.inputs)
                    if expected != actual:
                        raise ValueError(
                            f"Node {node_name!r} in {graph_name!r} calls "
                            f"custom_op {node.op!r} with {actual} input(s), "
                            f"but the op declares {expected} input(s) "
                            f"({list(cop.signature.keys())}). Either fix the "
                            f"node's `in:` list or update the op's `signature`."
                        )

                    # Custom-op port reference check. After arity passes,
                    # walk node.inputs for any `<head>.<port>` form and
                    # confirm head's op (if a custom_op) declares the
                    # port in `secondary_outputs`. Catches typos like
                    # `splitter.podt` against custom-op `splitter`
                    # whose ports are `pre / post / comb`.
                    # see post-arity port walk below

            # Cross-graph port-reference walk: every `<head>.<port>` ref
            # whose head node uses a custom_op must name a real port in
            # that op's `secondary_outputs`.
            for graph_name, bg in graphs:
                for node_name, node in bg.graph.items():
                    for src in node.inputs:
                        if "." not in src:
                            continue
                        head, _, port = src.rpartition(".")
                        head_node = bg.graph.get(head)
                        if head_node is None:
                            continue
                        head_cop = self.custom_ops.get(head_node.op)
                        if head_cop is None:
                            continue   # not a custom_op — primitive port check ran in BlockGraph
                        if port not in (head_cop.secondary_outputs or {}):
                            raise ValueError(
                                f"Node {node_name!r} in {graph_name!r} "
                                f"references port {src!r}, but custom_op "
                                f"{head_node.op!r} declares "
                                f"secondary_outputs="
                                f"{sorted((head_cop.secondary_outputs or {}).keys())}. "
                                f"Use `<node>` (= main `out` port) or one of "
                                f"the declared port names."
                            )


    def check_call_bindings(self) -> None:
        """Every call binds what its callee takes: as many tensors as it
        declares formals — a component call node, a stack, a model-level
        `op:` layer — and a component for each slot only a caller can fill
        (`BlockGraph.open_slots`). Run at load, and by an edit that rebinds
        a call to another component (the Playground's slot swap) on the
        spec it builds."""
        self._check_component_call_arity()
        self._check_stack_call_arity()
        self._check_model_op_arity()
        self._check_component_call_slots()

    def _check_component_call_slots(self) -> None:
        """Every call of a component binds each of its open slots in
        `params:` — a graph node's, a stack's `template:` or assignment, a
        model-level `op:` layer; left unbound, the slot's node runs no op."""
        callees: Dict[str, BlockGraph] = {**(self.components or {}),
                                          **(self.blocks or {})}

        def check(where: str, called: Optional[str],
                  params: Optional[Dict[str, Any]]) -> None:
            callee = callees.get(called) if called else None
            if callee is None:
                return
            unbound = [slot for slot in callee.open_slots
                       if slot not in (params or {})]
            if unbound:
                raise ValueError(
                    f"{where} calls component {called!r}, which leaves its "
                    f"slots {unbound} to its caller, and binds none of them. "
                    f"Bind each in the call's `params:`.")

        for graph_name, bg in _all_graphs(self):
            for node_name, node in bg.graph.items():
                check(f"Node {node_name!r} in {graph_name!r}", node.op,
                      node.params)

        def walk(layers: List[ModelLayer]) -> None:
            for layer in layers:
                if layer.layers:
                    walk(layer.layers)
                where = f"Model layer {layer.name!r}"
                check(where, layer.template, layer.params)
                check(where, layer.op, layer.params)
                for assignment in layer.assignments or ():
                    check(where, assignment.template, assignment.params)

        walk(self.model)

    def _check_component_call_arity(self) -> None:
        """Caller `in:` arity must equal the callee component's declared
        `inputs:` slots — zip-binding silently drops extra refs."""
        # Component-invocation arity check. When a node's op resolves to
        # a component (BlockGraph) name, the caller's `in:` list is
        # zip-bound positionally to the callee's `inputs:` declaration
        # by `engine._eval_graph`. Python's `zip` silently drops the
        # longer side, so an over-wired call like
        #   expert: { op: swiglu_ffn, in: [gate, permute] }   # 2 refs
        # against `swiglu_ffn` (declares `inputs: [x]`, 1 slot) binds
        # `x → gate` and **silently discards** `permute`. Inner
        # consumers then resolve `x` to the wrong upstream, which only
        # surfaces downstream as a wrong edge in the rendered DAG or a
        # shape mismatch deep inside a primitive op.
        #
        # This check raises at load time on any caller-vs-callee arity
        # mismatch (strict equality, mirrors the custom_ops check
        # above). Model-level stacks bind their `inputs:` to the body's
        # formals on a path of their own; `_check_stack_call_arity` holds
        # them to the same equality.
        all_components: Dict[str, BlockGraph] = {}
        if self.components:
            all_components.update(self.components)
        if self.blocks:
            all_components.update(self.blocks)
        if all_components:
            for graph_name, bg in _all_graphs(self):
                for node_name, node in bg.graph.items():
                    comp = all_components.get(node.op)
                    if comp is None:
                        continue
                    expected = len(comp.inputs)
                    actual = len(node.inputs)
                    if expected != actual:
                        raise ValueError(
                            f"Node {node_name!r} in {graph_name!r} calls "
                            f"component {node.op!r} with {actual} input(s), "
                            f"but it declares {expected} input slot(s) "
                            f"(inputs={comp.inputs!r}). Fix the node's `in:` "
                            f"list to match the component's signature — "
                            f"engine binds positionally via `zip` and "
                            f"silently drops extra refs."
                        )

    def _check_stack_call_arity(self) -> None:
        """A stack's `inputs:` bind every variant's formals one to one.

        A repeated stack names its upstream tensors on the model layer and
        each block variant its assignments select declares its own formals;
        the walk binds the first to the second in order. One more name than
        formals is a tensor the stack claims to read and never binds — Kimi's
        decoder once named its vision tower beside the embedding, and the
        tower's output was dropped. Checked at load, so against every
        variant the assignments name (after shortcut slots resolve to their
        template): a selector that matches no layer under one config may
        match under the next.
        """
        def walk(layers: List[ModelLayer]) -> None:
            for layer in layers:
                if layer.layers:
                    walk(layer.layers)
                if layer.template:
                    variants = [layer.template]
                elif layer.assignments:
                    variants = [a.template for a in layer.assignments]
                else:
                    continue
                named = list(layer.inputs or ())
                for template in dict.fromkeys(variants):
                    formals = list(self.get_block(template).inputs or ())
                    if len(named) != len(formals):
                        raise ValueError(
                            f"Model layer {layer.name!r} names "
                            f"{len(named)} input(s) "
                            f"{named} for variant {template!r}, "
                            f"which declares {len(formals)} formal(s) "
                            f"{formals}. A stack binds its `inputs:` to the "
                            f"body's formals one to one; merge extra tensors "
                            f"into the stream with a model-level op first.")

        walk(self.model)

    def _check_model_op_arity(self) -> None:
        """A model-level `op:` layer binds its `inputs:` to the op's inputs
        one to one.

        A component's formals, a custom op's signature, a registered
        primitive's arity: the walk binds the layer's `inputs:` to them in
        order and nothing else supplies one, so any other count is a tensor
        the layer names and never reads, or an input nothing feeds.
        """
        from backend.arch.ops import OP_REGISTRY

        components: Dict[str, BlockGraph] = {
            **(self.components or {}), **(self.blocks or {})}

        def walk(layers: List[ModelLayer]) -> None:
            for layer in layers:
                if layer.layers:
                    walk(layer.layers)
                if layer.op is None:
                    continue
                if layer.op in components:
                    expected = len(components[layer.op].inputs)
                    what = f"component {layer.op!r}'s formals"
                elif layer.op in (self.custom_ops or {}):
                    expected = self.custom_ops[layer.op].arity()
                    what = f"custom op {layer.op!r}'s signature"
                elif layer.op in OP_REGISTRY:
                    expected = OP_REGISTRY[layer.op].arity()
                    what = f"op {layer.op!r}'s inputs"
                    if (OP_REGISTRY[layer.op].repeats_last_input
                            and len(layer.inputs or ()) >= expected):
                        continue
                else:
                    continue
                named = list(layer.inputs or ())
                if len(named) != expected:
                    raise ValueError(
                        f"Model layer {layer.name!r} names "
                        f"{len(named)} input(s) {named}, "
                        f"but {what} number {expected}. A model-level `op:` "
                        f"binds its `inputs:` one to one, in order.")

        walk(self.model)

    def _check_call_site_dims(self) -> None:
        """A model-level op's `dims:` say what its DimMap reads, and nothing
        it derives (`axis_propagation.call_site_dims_error`) — the check
        every graph node gets."""
        for _path, layer in self.layer_paths():
            error = call_site_dims_error(layer.op, layer.dims or {})
            if error is not None:
                raise ValueError(f"Model layer {layer.name!r}: {error}")

    def _check_carried_streams(self) -> None:
        """Where a `carries: true` stream may be declared, and who may name
        it.

        It goes on the layer that repeats: a `template:` / `per_layer:`
        stack, or a repeated group whose block does not repeat itself
        (DeepSeek's `mtp`). Around a stack that repeats itself the
        declaration belongs to that inner stack, so a group declaring one
        over a repeated child would be a second carried stream for the same
        chain. It starts from the stack's `inputs[0]`, so the stack names
        one. Who may name it is `_check_stream_scopes`'s question.
        """
        def walk(layers: List[ModelLayer]) -> None:
            for layer in layers:
                stream = layer.carried_stream
                if stream is not None:
                    if layer.group:
                        blocks = [sub for sub in layer.layers or ()
                                  if sub.template or sub.assignments]
                        nested = [sub.name for sub in blocks
                                  if sub.repeat is not None]
                        if nested:
                            raise ValueError(
                                f"Model group {layer.name!r} carries stream "
                                f"{stream!r}, but its stack(s) {nested} "
                                f"repeat themselves; declare the stream on "
                                f"the innermost repeated stack")
                        starts = [sub for sub in blocks if not sub.inputs]
                    else:
                        starts = [] if layer.inputs else [layer]
                    if starts:
                        raise ValueError(
                            f"Model layer {starts[0].name!r} carries stream "
                            f"{stream!r} but names no `inputs:`; the "
                            f"stream's initial value is `inputs[0]`")
                if layer.layers:
                    walk(layer.layers)

        walk(self.model)

    def stream_names(self) -> StreamNames:
        """The model's streams by scope (see `StreamNames`).

        An appended name has exactly one producing declaration: its `from:`
        entry. Readers name it through a stack's `read: each` / `read:
        concat` or a component's `reads:`, which is a reference, not a
        second declaration; a second stack appending to it too would leave
        two catalogs for one name, and which one a reader meant would be
        decided by walk order. Refused.

        An inner name is its stack's: two stacks may declare one name, and
        each reader's `reads_from` says which stack's layer it reads. It is
        not also an appended name, which a reader reaches by name alone.
        """
        carried: Dict[Tuple[str, ...], str] = {}
        produced: Dict[str, Tuple[Tuple[str, ...], StreamSpec]] = {}
        inner: Dict[Tuple[Tuple[str, ...], str], StreamSpec] = {}
        for path, layer in self.layer_paths():
            if layer.carried_stream is not None:
                carried[path] = layer.carried_stream
            for name, decl in layer.inner_streams.items():
                inner[(path, name)] = decl
            for name, decl in layer.appended_streams.items():
                if name in produced:
                    raise ValueError(
                        f"stream {name!r} is declared by model layer "
                        f"{'/'.join(produced[name][0])!r} and again by "
                        f"{'/'.join(path)!r}; an appended stream has one "
                        f"declaring stack, and the others read it through "
                        f"`read: each` / `read: concat` or their components' "
                        f"`reads:`")
                produced[name] = (path, decl)
        clash = sorted({name for _path, name in inner} & set(produced))
        if clash:
            raise ValueError(
                f"stream(s) {clash} are declared both as an appended stream "
                f"and as an inner stream; a component's `reads:` of the name "
                f"would read two values. Rename one of them.")
        return StreamNames(carried=carried, produced=produced, inner=inner)

    def layer_paths(self) -> Iterator[Tuple[Tuple[str, ...], ModelLayer]]:
        """Every model layer at any depth, with its path from the top."""
        def walk(layers: Sequence[ModelLayer], prefix: Tuple[str, ...]):
            for layer in layers:
                path = (*prefix, layer.name)
                yield path, layer
                yield from walk(layer.layers or (), path)
        return walk(self.model, ())

    def layer_position(self, names: Sequence[str]
                       ) -> Optional[Tuple[int, ...]]:
        """The model layer a path of names from the top locates
        (`layers`, `mtp/layers`), by position; None when no layer is there.
        Names are unique among siblings (`_check_layer_wiring`,
        `ModelLayer`), so a path names one layer."""
        out: List[int] = []
        layers: Sequence[ModelLayer] = self.model
        for name in names:
            found = [i for i, layer in enumerate(layers) if layer.name == name]
            if len(found) != 1:
                return None
            out.append(found[0])
            layers = layers[found[0]].layers or ()
        return tuple(out)

    def layers_named(self, at: Tuple[int, ...], ref: str
                     ) -> Tuple[Tuple[int, ...], ...]:
        """The model layers, by position, that a name in the `inputs:` of
        the layer at `at` reads, in the scope `_check_layer_wiring` admits:
        the earlier siblings in each enclosing group, innermost first, out
        to the earlier top-level layers and the stages of the encoder
        groups among them. The innermost scope that has the name answers.
        Empty for a stream or a model input, which no layer is."""
        for depth in range(len(at) - 1, -1, -1):
            parent = at[:depth]
            siblings: Sequence[ModelLayer] = self.model
            for index in parent:
                siblings = siblings[index].layers or ()
            found: List[Tuple[int, ...]] = []
            for index, layer in enumerate(siblings[:at[depth]]):
                if layer.name == ref:
                    found.append((*parent, index))
                if not depth and layer.name in self.encoder_groups():
                    found.extend((index, k)
                                 for k, sub in enumerate(layer.layers or ())
                                 if sub.name == ref)
            if found:
                return tuple(found)
        return ()

    def inner_stream_named(self, at: Tuple[int, ...], ref: str
                           ) -> Optional[Tuple[int, ...]]:
        """The stack whose inner stream `ref` a name in the `inputs:` of the
        layer at `at` reads (its final value), in the scope `layers_named`
        searches: the earlier siblings in each enclosing group, innermost
        first, out to the earlier top-level layers. None when no earlier
        stack in scope declares it."""
        for depth in range(len(at) - 1, -1, -1):
            parent = at[:depth]
            siblings: Sequence[ModelLayer] = self.model
            for index in parent:
                siblings = siblings[index].layers or ()
            found: List[Tuple[int, ...]] = []

            def walk(layers, prefix):
                for index, layer in enumerate(layers):
                    here = (*prefix, index)
                    if ref in layer.inner_streams:
                        found.append(here)
                    walk(layer.layers or (), here)

            for index, layer in enumerate(siblings[:at[depth]]):
                here = (*parent, index)
                if ref in layer.inner_streams:
                    found.append(here)
                walk(layer.layers or (), here)
            if len(found) > 1:
                raise ValueError(
                    f"`inputs:` name {ref!r} reads the final value of an "
                    f"inner stream, and {len(found)} earlier stacks declare "
                    f"one by that name; rename one of them")
            if found:
                return found[0]
        return None

    def _check_stream_scopes(self) -> None:
        """Clashes between names, each asked in the scope the name lives in.

        * A stream name — appended or inner — is not also a name a stage is
          published by: an `inputs:` or `in:` naming it would point at two
          tensors.
        * A carried stream is local to its stack. No component the stack
          binds may `reads:` / `stream_out:` that name — the block reads the
          carried stream through its formal 0 and writes it with its output
          — but another stack may use the same name for a model-wide stream
          of its own.
        """
        names = self.stream_names()
        published = {layer.name for _path, layer in self.layer_paths()}
        clash = sorted((set(names.produced) | names.inner_names())
                       & published)
        if clash:
            raise ValueError(
                f"stream(s) {clash} share a name with a model layer; a name "
                f"in `inputs:` / `in:` must point at one tensor. Rename the "
                f"stream.")
        layers = dict(self.layer_paths())
        for path, stream in names.carried.items():
            layer = layers[path]
            stacks = [sub for sub in layer.layers or ()] if layer.group \
                else [layer]
            for graph_name, bg in self.reachable_graphs(
                    self.bound_templates(stacks)):
                if stream in bg.reads or stream in bg.stream_out:
                    raise ValueError(
                        f"Component {graph_name!r}, bound in stack "
                        f"{'/'.join(path)!r}, names stream {stream!r} in "
                        f"`reads:` / `stream_out:`, but that stack carries "
                        f"{stream!r} across its block boundary: a block reads "
                        f"it through its formal 0 and writes it with its "
                        f"output, and nothing names it inside")

    def _check_stream_seeds(self) -> None:
        """An inner stream's `seed` names one earlier `op:` layer, in the
        scope the stack's own `inputs:` read in (`layers_named`): the value
        the stream holds before the stack's first layer writes it."""
        def walk(layers: Sequence[ModelLayer], prefix: Tuple[int, ...]):
            for index, layer in enumerate(layers):
                at = (*prefix, index)
                for name, decl in layer.inner_streams.items():
                    if decl.seed is None:
                        continue
                    found = self.layers_named(at, decl.seed)
                    ops = [path for path in found
                           if self._layer_at(path).op is not None]
                    if len(found) != 1 or len(ops) != 1:
                        raise ValueError(
                            f"Model layer {layer.name!r} declares stream "
                            f"{name!r} with `seed: {decl.seed}`, which "
                            f"names no one earlier `op:` layer; `seed` is "
                            f"the output of the model layer that seeds the "
                            f"stream before its first write, named as "
                            f"`inputs:` names one")
                walk(layer.layers or (), at)
        walk(self.model, ())

    def _layer_at(self, path: Tuple[int, ...]) -> ModelLayer:
        layer = self.model[path[0]]
        for index in path[1:]:
            layer = (layer.layers or ())[index]
        return layer

    def _known_graphs(self) -> set:
        return set(self.components or {}) | set(self.blocks or {})

    def _names_in_params(self, params: Any) -> List[str]:
        """Component names a `params:` binding hands a template, at any
        depth (`{attn: <name>}`, `{inner: {template: <name>, params: …}}`)."""
        known = self._known_graphs()
        out: List[str] = []
        for value in (params or {}).values():
            if isinstance(value, str) and value in known:
                out.append(value)
            elif isinstance(value, dict):
                template = value.get("template")
                if isinstance(template, str) and template in known:
                    out.append(template)
                out.extend(self._names_in_params(value.get("params")))
        return out

    def bound_templates(self, layers: Sequence[ModelLayer]) -> List[str]:
        """The components `layers` (and their sub-layers) invoke directly:
        their `template:`, every assignment's template, and the components
        their `params:` bind."""
        out: List[str] = []
        for layer in layers:
            if layer.template:
                out.append(layer.template)
            out.extend(self._names_in_params(layer.params))
            for assign in layer.assignments or ():
                if assign.template:
                    out.append(assign.template)
                out.extend(self._names_in_params(assign.params))
            out.extend(self.bound_templates(layer.layers or ()))
        return out

    def reachable_graphs(self, seeds: Optional[Sequence[str]] = None
                         ) -> Iterator[Tuple[str, BlockGraph]]:
        """`(name, BlockGraph)` for every component reachable from `seeds`
        (default: everything `model:` binds), following `op:` references
        and `params:` component bindings. Deduped by name."""
        known = self._known_graphs()
        stack = list(self.bound_templates(self.model) if seeds is None
                     else seeds)
        seen: set = set()
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            try:
                graph = self.get_block(name)
            except KeyError:
                continue
            yield name, graph
            for node in graph.graph.values():
                op = node.op
                if op and not op.startswith("<") and op in known:
                    stack.append(op)
                stack.extend(self._names_in_params(node.params))

    def _check_appended_streams(self) -> None:
        """Who may read and write an appended stream, from the model's
        stream table (`stream_names`), where its `from:` / `after:` entry
        is its one producing declaration.

        A `read: each` or `read: concat` references an appended stream an
        earlier stack appends to; it is not a declaration of its own. A
        stack reads one at its block boundary only when it hands nothing
        from one iteration to the next, so that iteration j binds element
        j; a model-level `op:` names one in `inputs:` to read all of it — as
        a history, or joined along the layer's `axis:` when it declares one;
        a history goes only to an input the op declares one for (a `Sar`
        axis in its signature). A stream published in a declared order or cropped (`order` /
        `slice`) is read only joined. The appending stack reads its own
        prefix through `read: all` on its declaration. Nothing but the
        appending stack writes one: a component's `stream_out` of that name
        would be a second writer. (A stack's carried and appended streams
        are keys of one `streams:` mapping, so they cannot share a name.)
        """
        names = self.stream_names()
        order = {path: i for i, (path, _layer) in enumerate(self.layer_paths())}
        appended = {name: path for name, (path, decl) in names.produced.items()
                    if decl.appends}
        # A stream published in a declared order or cropped is only ever
        # joined: every other reading follows the iterations, whole.
        joined_only = {name for name, (_path, decl) in names.produced.items()
                       if decl.appends and (decl.order is not None
                                            or decl.slice is not None)}
        for path, layer in self.layer_paths():
            for kind, refs in (("each", layer.read_streams),
                               ("concat", layer.concat_streams)):
                for name in refs:
                    writer = appended.get(name)
                    if writer is None or order[writer] >= order[path]:
                        raise ValueError(
                            f"model layer {layer.name!r} declares `{name}: "
                            f"{{read: {kind}}}`, a reference to an appended "
                            f"stream (`from:` / `after:`) an earlier stack "
                            f"appends to; no earlier stack appends to "
                            f"{name!r}")
            named = [ref for ref in layer.inputs or () if ref in appended]
            elementwise = list(layer.read_streams)
            if layer.op is None or layer.axis is None:
                elementwise.extend(named)
            refused = [name for name in elementwise if name in joined_only]
            if refused:
                raise ValueError(
                    f"model layer {layer.name!r} reads appended stream(s) "
                    f"{refused} element by element or as a history, but "
                    f"{'they declare' if len(refused) > 1 else 'it declares'} "
                    f"`order: declared` or `slice`, which only a concat "
                    f"reads: declare `{refused[0]}: {{read: concat, axis: "
                    f"<axis>}}`, or give a model-level `op:` an `axis:`")
            if named and layer.op is not None and layer.axis is not None \
                    and self._op_takes_axis(layer.op):
                raise ValueError(
                    f"model layer {layer.name!r} names appended stream(s) "
                    f"{named} with `axis: {layer.axis}`, which joins them "
                    f"along that axis before `{layer.op}` runs; but "
                    f"`{layer.op}` takes an `axis:` of its own, and one "
                    f"field cannot say both")
            # Without `axis:` a model-level `op:` reads the stream as one
            # `[B, S, m, D]` history, so the port that binds it must be one
            # the op declares a history for; otherwise the op runs on each
            # element where a join was meant.
            if named and layer.op is not None and layer.axis is None:
                for port, ref in enumerate(layer.inputs or ()):
                    signatures = self._op_port_signatures(layer.op, port)
                    if ref in appended and signatures and not any(
                            HISTORY_AXIS in sig for sig in signatures):
                        raise ValueError(
                            f"model layer {layer.name!r} names appended "
                            f"stream {ref!r} in `inputs:` without `axis:`, "
                            f"so `{layer.op}` input {port} receives all of "
                            f"its elements as one [B, S, {HISTORY_AXIS}, D] "
                            f"history; but `{layer.op}` takes "
                            f"{list(signatures[0])} there, not a history. To "
                            f"join the elements (`torch.cat`), declare "
                            f"`axis:` on {layer.name!r} (`axis: -1` for the "
                            f"feature axis) and give its dims the joined "
                            f"width; see the schema section \"Appended "
                            f"streams: values a stack hands on after chosen "
                            f"iterations\"")
            # A model-level `op:` runs once, after every stack: it reads all
            # of the stream, its final value.
            if named and layer.op is None and not layer.is_parallel_stack:
                raise ValueError(
                    f"model layer {layer.name!r} names appended stream(s) "
                    f"{named} in `inputs:`; a stack reads an appended stream "
                    f"at its block boundary when it repeats once per element "
                    f"and carries no stream, so that iteration j binds "
                    f"element j. A component reads one at an inner node "
                    f"through its stack's `read: each`, or the appending "
                    f"stack's own `read: all`; a model-level `op:` reads all "
                    f"of it")
        for graph_name, bg in _all_graphs(self):
            written = sorted(set(appended) & set(bg.stream_out))
            if written:
                raise ValueError(
                    f"Component {graph_name!r} `stream_out`s appended "
                    f"stream(s) {written}; an appended stream is written "
                    f"by its stack's `from:` declaration alone")

    def _check_delegated_cache_kinds(self) -> None:
        """A `node:` entry naming a sub-component that keeps a cache of its
        own gives one kind to every per-token tensor that sub-cache keeps
        (`eval.cache`: DSv4's indexer pools its compressor's rows as an
        `index_key`). That is one statement about one kind of tensor; a
        sub-cache whose per-token entries hold different kinds — a latent
        and a rope key — cannot be relabelled by it without erasing the
        difference, so the parent names each of them instead."""
        graphs = dict(_all_graphs(self))
        for graph_name, bg in graphs.items():
            if bg.cache is None:
                continue
            for entry in bg.cache.entries:
                if entry.node is None or entry.node not in bg.graph:
                    continue
                sub = graphs.get(bg.graph[entry.node].op)
                if sub is None or sub.cache is None:
                    continue
                kinds = {e.name: e.kind for e in sub.cache.entries
                         if e.node is not None}
                if len(set(kinds.values())) > 1:
                    raise ValueError(
                        f"Component {graph_name!r}: cache entry "
                        f"{entry.node!r} names {bg.graph[entry.node].op!r}, "
                        f"whose own cache keeps per-token tensors of "
                        f"{len(set(kinds.values()))} kinds ({kinds}); "
                        f"`kind: {entry.kind}` would give them all one. Name "
                        f"each slot through the sub-component's port "
                        f"(`{{node: {entry.node}.<slot>, kind: <kind>}}`), "
                        f"one entry per kind.")

    def _check_recurrent_state_carriers(self) -> None:
        """Every recurrent state has a kernel carrying it, and every kernel
        that carries one has its state declared."""
        # A recurrent state has to reach the kernel that streams it.
        #
        # `cache.entries: [{state: ...}]` declares bytes that no graph node
        # outputs — that is what makes it a `state` entry rather than a
        # `node` one — so the byte walk cannot find the reader by following
        # an edge back to its producer, the way it does for every K/V
        # stream. The entry names its `carrier:` instead, and the state's
        # whole per-step read is billed there. `BlockGraph` already checks
        # that the carrier is one of its nodes; this is the level that also
        # holds the op declarations — including the merged `custom_ops`
        # table a component's own recurrence (GLA's `gla_simple_recurrence`)
        # is declared in — so the two directions are checked here:
        #
        #   • a carrier must be a kernel: a sub-component has no single
        #     frame to resolve the state's `shape:` in and no single row
        #     to bill;
        #   • an op declaring `carries_recurrent_state` is a recurrence by
        #     construction, so a node running one with no state naming it
        #     is a state the model keeps and nothing bills — a Mamba
        #     layer's decode, which is that read almost entirely, free.
        from backend.arch.ops import OP_REGISTRY

        custom_ops = self.custom_ops or {}
        # What each name a state axis may read counts — dims.yaml's and this
        # spec's own `dim_symbols` — so a position or request count is
        # refused by its kind, whatever it is called.
        kinds = {name: sym.kind
                 for name, sym in dim_registry.base_symbols().items()}
        kinds.update({name: row.get("kind")
                      for name, row in (self.dim_symbols or {}).items()
                      if isinstance(row, dict)})
        for graph_name, bg in _all_graphs(self):
            carried = {e.carrier: e.state for e in bg.kv_states}
            spelled = (set(self.dims or {})
                       | dim_registry.registered_for(self.dim_symbols))
            for node in bg.graph.values():
                for value in [*node.dims.values(), node.dim,
                              *(node.shape or [])]:
                    if value is not None:
                        spelled |= _dim_names(value)
            for entry in bg.kv_states:
                node = bg.graph.get(entry.carrier)
                if node is None:
                    continue          # BlockGraph reported it already
                for index, axis in enumerate(entry.shape):
                    read = _dim_names(axis)
                    unknown = sorted(read - spelled - set(node.dims))
                    if unknown:
                        raise ValueError(
                            f"Component {graph_name!r}: recurrent state "
                            f"{entry.state!r} axis {index} of `shape: "
                            f"{entry.shape}` is `{axis}`, which reads "
                            f"{unknown} — not a dim of its carrier "
                            f"{entry.carrier!r} (its `dims:` binds "
                            f"{sorted(node.dims)}), of this spec, of this "
                            f"component's graph, or of the dim registry. Fix "
                            f"the spelling, or bind the name in the "
                            f"carrier's `dims:`."
                        )
                    for name in read & set(node.dims):
                        read = read | _dim_names(node.dims[name])
                    per_request = sorted(
                        name for name in read
                        if kinds.get(name) in ("sequence", "batch"))
                    if per_request:
                        raise ValueError(
                            f"Component {graph_name!r}: recurrent state "
                            f"{entry.state!r} axis {index} `{axis}` reads "
                            f"{per_request}, which count positions or "
                            f"requests. A state is one fixed tensor per "
                            f"request whose size does not grow with the "
                            f"context, so its axes are the kernel's own "
                            f"widths (heads, head dims, the conv window). A "
                            f"cache that grows per token is a `node:` entry."
                        )
            for carrier, state in carried.items():
                node = bg.graph.get(carrier)
                if node is None:
                    continue          # BlockGraph reported it already
                if (custom_ops.get(node.op) or OP_REGISTRY.get(node.op)) is None:
                    raise ValueError(
                        f"Component {graph_name!r}: recurrent state "
                        f"{state!r} names carrier {carrier!r}, which runs "
                        f"{node.op!r} — not a primitive or custom op. The "
                        f"carrier is the kernel that streams the state; name "
                        f"the leaf inside that sub-component, or declare the "
                        f"state on the sub-component itself."
                    )
            for entry in bg.kv_states:
                node = bg.graph.get(entry.carrier)
                if node is None:
                    continue          # BlockGraph reported it already
                recurrence = _streams_recurrent_state(node.op, custom_ops)
                if recurrence != (entry.kind == "recurrent_state"):
                    raise ValueError(
                        f"Component {graph_name!r}: state {entry.state!r} "
                        f"declares `kind: {entry.kind}` and its carrier "
                        f"{entry.carrier!r} runs {node.op!r}, which "
                        + ("declares `carries_recurrent_state`: the state a "
                           "recurrence streams is `kind: recurrent_state`."
                           if recurrence else
                           "is not a recurrence (no `carries_recurrent_state`)"
                           ": a `recurrent_state` is the matrix a recurrence "
                           "kernel rewrites every step. Name that kernel as "
                           "the carrier, or give the state the kind of what "
                           "it holds (`conv_state`, `compressor_state`, "
                           "`token_window`)."))
            for name, node in bg.graph.items():
                if (_streams_recurrent_state(node.op, custom_ops)
                        and name not in carried):
                    raise ValueError(
                        f"Component {graph_name!r}: node {name!r} runs "
                        f"{node.op!r}, which declares "
                        f"`carries_recurrent_state`, but no `cache.entries` "
                        f"state names it as its carrier. Add "
                        f"`{{state: <name>, carrier: {name}, shape: "
                        f"[<axis>, ...]}}` under `cache.entries`, listing the "
                        f"state's axes outermost first."
                    )

    def _check_quant_context(self) -> None:
        """quant_context methods cross-checked against the quant_schemes registry."""
        # quant_context: cross-check methods against quant_schemes registry.
        # See `_validate_quant_context_methods` for rationale (lazy import,
        # graceful degradation when registry is unavailable).
        if self.quant_context is not None:
            errs = _validate_quant_context_methods(self.quant_context)
            if errs:
                raise ValueError("; ".join(errs))

    def _check_custom_op_collisions(self) -> None:
        """custom_ops must not shadow registered primitive names."""
        from backend.arch.ops import OP_REGISTRY
        collisions = sorted(set(self.custom_ops).intersection(OP_REGISTRY))
        if collisions:
            raise ValueError(
                f"custom_ops shadows registered primitive(s) {collisions}. "
                "Registered op names are global language keywords and cannot "
                "be overridden by one spec; choose a novel custom-op name.")

    def _resolve_all_input_ports(self) -> None:
        """Pre-resolve every BlockGraph node's input port disambiguation ONCE, with full ArchSpec context."""
        # The engine reads `node.resolved_inputs` directly at eval time
        # instead of calling `parse_input_ref` per input per primitive op —
        # port resolution is a static property of the surrounding graph +
        # op registry, not runtime data.
        custom_ops = self.custom_ops or None
        all_components: Dict[str, BlockGraph] = {}
        if self.components:
            all_components.update(self.components)
        if self.blocks:
            all_components.update(self.blocks)
        port_blocks: List[BlockGraph] = []
        if self.blocks:
            port_blocks.extend(self.blocks.values())
        if self.components:
            port_blocks.extend(self.components.values())
        for bg in port_blocks:
            resolve_block_input_ports(bg, custom_ops, all_components)


    def get_block(self, template_name: str = "block") -> BlockGraph:
        """Get a block graph by template name.

        Looks in `blocks` (named templates) and `components` (library compound
        ops) — no distinction between them from the engine's POV.
        """
        if self.blocks and template_name in self.blocks:
            return self.blocks[template_name]
        if self.components and template_name in self.components:
            return self.components[template_name]
        raise KeyError(f"Block template {template_name!r} not found")

    def all_block_names(self) -> List[str]:
        """Return all available block template names.

        Includes `components` because `get_block()` resolves component
        names too — a model layer can point at a library component
        (e.g. an aux module like `kimi_k25_vision_block`) directly via
        `template:` without first declaring it in `blocks:`. Excluding
        components here would make those layers silently miss the
        block_graph build pass.
        """
        names = []
        if self.blocks:
            names.extend(self.blocks.keys())
        if self.components:
            names.extend(self.components.keys())
        return names
