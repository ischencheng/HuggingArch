"""Rewrite stored spec YAML across known schema evolutions.

Private generation records keep the YAML the agent wrote. Schema later
forbids a renamed field or tightens a rule, and those rows stop loading.
This module is the mechanical rewrite: a pure function over a mapping,
idempotent, never touching the database.

`normalize_bundle` calls it on the flattened identity document and each
draft body before serialize. Overlay write consumes that result; it is
not a second rewrite hook. A stored row is rewritten the same way at
every consumer of the normalized bundle.

What it will rewrite, when the target is unique:

  * the retired dim symbol vocabulary → the current one (see
    `_rename_dim_symbols` and `spec_migrations.yaml`)
  * `tie_weights:` → `tied_weight: {to}` / `{to, when}` (see `_rewrite_tie_weights`)
  * `in:` / `inputs:` on a vision/audio group (encoders name no backbone)
  * missing `boundary_role: input_embedding` on the one top-level embedding
  * missing `boundary_role: output_head` on the one top-level `lm_head`
    linear the engine took as the head before the role existed (see
    `_stamp_output_head`)
  * retired document keys (`ignored_config_fields:`)
  * retired block `name:` (identity is the file stem / `blocks:` key)
  * a missing `axis:` on an op that used to default one (see
    `_stamp_retired_axis`)
  * a call to a component the library renamed, and a dim binding a component
    renamed (see `_rename_renamed_components` / `_rename_component_dims`)
  * an unbound config switch a component's step gained (see
    `_bind_retired_switches`)
  * a missing `checkpoint:` on an indexed aux group (see
    `_stamp_aux_checkpoint`)
  * retired custom-op keys, and the `couples_on:` a custom op's weight
    implies (see `_migrate_custom_ops`)
  * omitted `inputs:` on a model-level consumer → the stage the retired
    sequential default read (see `_thread_omitted_inputs`)
  * a chain stack with no carried stream gets `streams: { hidden: { carries:
    true } }` (see `_carry_hidden`): the hidden state it used to
    thread by default is now a stream the stack declares
  * `inputs: [<embedding>, <encoder group>]` on a stack whose block takes one
    formal becomes the explicit `scatter_rows_replace` it reads (see
    `_merge_encoder_rows`)
  * `keep: {p: ...}` becomes the appended stream `p: {from: <carried>,
    after, index, src}` and its readers' `inputs: [<stack>.<p>]` become
    `inputs: [p]` (see `_keep_to_appended`)
  * a lookup that read the token ids without naming them reads a declared
    model input (`model_inputs:`; see `_declare_model_inputs`)
  * `kind: vision` becomes `kind: image` (see `_migrate_media_inputs`)
  * `out:` on a `view_merge` goes: its width is the product of the axes it
    merges (see `_drop_view_merge_out`)
  * a graph with no `outputs:` declares the outputs the retired parser
    chose (see `_stamp_outputs`)
  * `window:` and `init:` on an inner stream go: they annotated the
    retired latest-writer pairing, and a reader now names the layer it reads
    (see `_drop_inner_stream_annotations`)
  * a cache `state:` entry's `elements:` product whose stored axis order is
    unique across the library becomes `shape:` (see
    `_rewrite_cache_state_elements`)
  * a cache state with no `kind:` whose carrier is a recurrence kernel
    declares `kind: recurrent_state` (see `_stamp_cache_slot_kinds`)

What it will not guess: retired KV / carry fields, `assignments:`, an
embedding that is not unique (zero or more than one declared
`input_embedding`), an omitted `inputs:` whose predecessor is a
parameter-like source, an encoder's media input (its axes, its `patchify`
and the `prompt_slots` of its edges into the language model), a graph
with no `outputs:` and other than one unread node, a component's
`stream_input:`, a chain stack that
already has a stream named `hidden` that is not its carried one, a `keep:`
port whose name is already taken model-wide, read by a stack that carries a
stream, or written or read under a repeating group, and the layer an inner
stream's reader reads (`reads_from`; see `_require_reads_from`), a dim
that lost its registry default (see `_require_retired_defaults`), and the
`kind:` of any other cache entry. Those come back as
blockers with a regenerate hint — a data error, not a 500. A record whose
every blocker names a declaration (`SpecMigrationBlocked`) is repairable:
Agent Fix opens it as saved, with the blockers as its work order. One with a
retired syntax is not, and has to be regenerated.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import yaml

from backend.arch import cache_families
from backend.arch import dims as dim_registry
from backend.arch.authored_yaml import AuthoredYaml
from backend.arch.loader import SpecLoadError, merge_base_documents
from backend.arch.ops import OP_REGISTRY, lookup_of
from backend.arch.ops_base import ShapeKind
from backend.arch.schema.custom_ops import CustomOpSpec
from backend.arch.schema.spec import ValidationExemptions
from backend.arch.spec_parsing import (
    _DEFAULT_INPUTS,
    _RETIRED_CARRY_KEYS,
    _RETIRED_IDENTITY_KEYS,
    _RETIRED_KV_KEYS,
)
from backend.code_trace.memo import memoize


# Same condition the current schema accepts for `tied_weight.when`
# (`backend.arch.schema.model.TiedWeightSpec`).
_TIED_WHEN_RE = re.compile(
    r"config(?:\.[A-Za-z_][A-Za-z_0-9]*)+"
    r"(?:\s*\?\?\s*(?:true|false|1|0))?",
    re.IGNORECASE,
)
# S4 §3j: schema/load failures are regenerated; Fix seed itself runs
# normalize and would refuse the same blocker.
REGENERATE_HINT = "Regenerate it from the Inspector."
#: The Inspector card: the refusals themselves ride in `errors[]` for the
#: spec writer, not in this line.
SCHEMA_ERROR = "Schema error."

#: The rename tables every rewrite below reads, one per schema change.
_TABLES_PATH = pathlib.Path(__file__).with_name("spec_migrations.yaml")


@memoize(maxsize=1)
def _tables() -> dict:
    with _TABLES_PATH.open(encoding="utf-8") as handle:
        return load_yaml(handle) or {}


def symbol_renames() -> dict:
    """Old → new dim symbol, as `spec_migrations.yaml` renamed the corpus."""
    return dict(_tables()["symbol_vocabulary"]["rename"])


def measured_row_migration(signature: str) -> dict:
    """How the shape of a row measured under `signature` before an identity
    change becomes the current one (`measured_row_vocabulary`): `rename`,
    old → new axis — the dim symbol renames, then where the measured
    identity's letter meant something else; `derived`, an axis the identity
    gained → the axes whose product it is for an old row."""
    table = _tables()["measured_row_vocabulary"]
    renames = symbol_renames()
    renames.update(table["rename"])
    renames.update(table["by_signature"].get(signature, {}))
    derived = {axis: tuple(factors)
               for axis, factors in table["derived"].get(signature, {}).items()}
    return {"rename": renames, "derived": derived}


def replaced_keys() -> dict:
    """Old → new key, for every key a rule replaces one to one: the declared
    replacements and the dim renames. A rewrite moves a replaced key's
    comments onto its replacement (`AuthoredYaml`)."""
    renames = dict(_tables()["replaced_keys"])
    renames.update(symbol_renames())
    for dims in _tables()["component_dims"].values():
        renames.update(dims)
    return renames


def _retired_axis_defaults() -> dict:
    return {str(op): int(axis)
            for op, axis in _tables()["retired_axis_defaults"].items()}


def _stamp_retired_axis(node: dict, notes: list[str], *, identity: str) -> None:
    """Write out the axis an op used to default to (be09cb7b).

    `view_split` split the last axis until a window fold needed another one;
    the op declares `axis_required` now, and a stored node that omits it meant
    the axis the table records.
    """
    axis = _retired_axis_defaults().get(str(node.get("op") or ""))
    if axis is None or node.get("axis") is not None:
        return
    node["axis"] = axis
    notes.append(
        f"declared axis: {axis} on the {node['op']} {identity} "
        f"(was the retired default)")


def _refuse_weight_copies(node: dict, blockers: list[str], *,
                          identity: str) -> None:
    """`weight_copies:` is retired: an expert bank's experts are the leading
    axis of its weights (`shape: [n_expert_copies, out, in]`) and the
    checkpoint's storage of that axis is `expert_weight: unrolled |
    leading_axis`. The old record counted the bank by `repeat:` or a
    flattened axis, so the shapes have to be written, not copied over."""
    if "weight_copies" in node:
        blockers.append(
            f"node {identity} still declares removed `weight_copies:`; call "
            f"an expert template (components/library, e.g. gated_experts) "
            f"with `expert_weight: unrolled | leading_axis`, or declare the "
            f"weight's `shape: [n_expert_copies, ...]`, or regenerate")


def retired_path_roots() -> dict:
    """Old → new repository root of a stored model / draft file path."""
    return {tuple(old.split("/")): tuple(new.split("/"))
            for old, new in _tables()["retired_path_roots"].items()}


#: The authoring channels a component identity travels: a top-level opt-in or
#: slot binding, a template call, a component parameter, a graph `op:` call,
#: a decode twin, a per-layer assignment. A rewrite handles these rather than
#: every string in the document, because a component name can coincide with a
#: dim symbol and renaming that symbol would silently change evaluation.
_component_reference_keys = frozenset(
    {"template", "op", "decode_variant", "attention", "ffn"})


def rename_components(doc: dict, mapping) -> dict:
    """Rewrite component identities in a parsed document, never by text
    regex. Returns a new document; the input is untouched."""
    if not mapping:
        return doc

    def rewrite_params(value):
        if isinstance(value, dict):
            return {
                slot: (mapping.get(item, item) if isinstance(item, str)
                       else rewrite_params(item))
                for slot, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite_params(item) for item in value]
        return value

    def rewrite(value, parent_key: Optional[str] = None):
        if isinstance(value, str):
            if parent_key in _component_reference_keys:
                return mapping.get(value, value)
            return value
        if isinstance(value, list):
            return [rewrite(item, parent_key) for item in value]
        if isinstance(value, dict):
            out = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    out[key] = rewrite(child, None)
                    continue
                if key == "drafts" and isinstance(child, list):
                    out[key] = [mapping.get(item, item) if isinstance(item, str) else item
                                for item in child]
                elif key == "map" and isinstance(child, dict):
                    # `per_layer.map` rows are keyed by a layer type, not by a
                    # component slot. Reset the parent so each row's
                    # `template` / `attention` / `ffn` bindings are visible as
                    # reference keys (while row-level `dims` stays put).
                    out[key] = {
                        layer_type: rewrite(entry, None)
                        for layer_type, entry in child.items()
                    }
                elif key in ("components", "params") and isinstance(child, dict):
                    # A slot name is not a component identity; its value (or a
                    # parameter slot's default) is.
                    if key == "components":
                        out[key] = {
                            slot: mapping.get(item, item) if isinstance(item, str)
                            else rewrite(item)
                            for slot, item in child.items()
                        }
                    else:
                        out[key] = rewrite_params(child)
                else:
                    out[key] = rewrite(child, key)
            return out
        return value

    return rewrite(doc)


class Refusals(list):
    """The messages a rewrite refuses with, and which of them the author can
    clear in the document.

    `append` keeps meaning what it meant: the document uses something that no
    longer exists (a retired carry field, `assignments:`, YAML that does not
    parse) and there is nothing to add — it has to be written again.
    `declare` is the other half: the refusal names a declaration the schema
    now requires, or a name that has to be freed, and writing that one line
    is the whole fix. The serving boundary reports the second kind as a
    missing declaration rather than as a spec to regenerate, and the spec
    writer gets it back in `errors[]` either way.
    """

    def __init__(self) -> None:
        super().__init__()
        self.declarations: list[str] = []

    def declare(self, message: str) -> None:
        self.declarations.append(message)
        self.append(message)


class SpecMigrationBlocked(SpecLoadError):
    """A stored spec whose every refusal names a declaration to add.

    A `SpecLoadError`, so every caller that already handles one is unchanged;
    a distinct type, so the serving boundary can tell the author to write that
    declaration instead of telling them to regenerate.
    """

    def __init__(self, message: str, *, errors: Iterable[str] = ()):
        super().__init__(message)
        self.errors = list(errors)


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of one YAML rewrite. `text` is the input when nothing
    changed or when `blockers` is non-empty (do not serve a partial rewrite).
    `needs_declaration` is the subset of `blockers` the author clears by
    declaring something."""

    text: str
    notes: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    needs_declaration: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.blockers

    @property
    def changed(self) -> bool:
        return bool(self.notes) and not self.blockers

    @property
    def only_needs_declarations(self) -> bool:
        """Every refusal names something to write, so nothing is lost by
        editing the spec instead of regenerating it."""
        return bool(self.blockers) and len(self.needs_declaration) == len(
            self.blockers)


def serving_error(model_id: str, blockers: Iterable[str], *,
                  needs_declaration: bool = False) -> str:
    """User-facing refusal when a stored spec cannot be rewritten.

    The blockers stay on the exception (`errors`) for the spec writer. This
    line is the Inspector card: a declaration is repaired in place, a
    retired syntax has to be written again.
    """
    if needs_declaration:
        return SCHEMA_ERROR
    return f"{SCHEMA_ERROR} {REGENERATE_HINT}"


def refuse(model_id: str, result: MigrationResult) -> SpecLoadError:
    """The exception a blocked rewrite raises, of the kind its blockers are."""
    if result.only_needs_declarations:
        return SpecMigrationBlocked(
            serving_error(model_id, result.blockers, needs_declaration=True),
            errors=result.blockers)
    return SpecLoadError(serving_error(model_id, result.blockers))


def migrate_stored_bundle(model_id: str, spec_yaml: str,
                          new_component_yamls=None) -> tuple[str, list]:
    """Rewrite a stored identity + draft bodies. Raises SpecLoadError
    when a document cannot be rewritten; never writes the database.
    """
    result = migrate_yaml_text(spec_yaml)
    if not result.ok:
        raise refuse(model_id, result)
    spec_yaml = result.text
    out = []
    for comp in (new_component_yamls or []):
        if not isinstance(comp, dict):
            out.append(comp)
            continue
        body = comp.get("body")
        if not body:
            out.append(comp)
            continue
        migrated = migrate_yaml_text(body)
        if not migrated.ok:
            raise refuse(model_id, migrated)
        if migrated.changed:
            out.append({**comp, "body": migrated.text})
        else:
            out.append(comp)
    return spec_yaml, out


def migrate_yaml_text(text: str, components=None,
                      models=None) -> MigrationResult:
    """Parse, rewrite, dump. Unparseable input is a blocker, original kept."""
    try:
        doc = load_yaml(text)
    except yaml.YAMLError as exc:
        return MigrationResult(
            text=text, blockers=(f"YAML parse error: {exc}",))
    if not isinstance(doc, dict):
        return MigrationResult(
            text=text, blockers=("document is not a YAML mapping",))
    notes: list[str] = []
    blockers = Refusals()
    changed = _migrate_mapping(doc, notes, blockers, components, models)
    if blockers:
        return MigrationResult(
            text=text, notes=tuple(notes), blockers=tuple(blockers),
            needs_declaration=tuple(blockers.declarations))
    if not changed:
        return MigrationResult(text=text)
    return MigrationResult(
        text=AuthoredYaml(text).rewrite(doc, replaced=replaced_keys()),
        notes=tuple(notes))


def migrate_mapping(doc: dict, components=None,
                    models=None) -> tuple[dict, MigrationResult]:
    """In-place rewrite of an already-parsed mapping.

    `normalize_bundle` already parsed the identity YAML. Apply this, then
    serialize. Returns the (possibly mutated) mapping plus a result whose
    `text` is empty — callers that need YAML should dump themselves or
    use `migrate_yaml_text`.
    """
    notes: list[str] = []
    blockers = Refusals()
    _migrate_mapping(doc, notes, blockers, components, models)
    return doc, MigrationResult(
        text="", notes=tuple(notes), blockers=tuple(blockers),
        needs_declaration=tuple(blockers.declarations))


#: An identifier in an expression. Never one behind a dot: `config.<path>` and
#: `checkpoint.<tensor>` name fields of another namespace.
_EXPR_IDENT = re.compile(r"(?<![A-Za-z0-9_.])[A-Za-z_][A-Za-z_0-9]*")


def _custom_op_names(value: Any) -> set:
    """Every op a document declares itself, wherever it declares it (the
    document root, a component body, one inline `blocks:` entry)."""
    names: set = set()
    if isinstance(value, dict):
        table = value.get("custom_ops")
        if isinstance(table, dict):
            names.update(str(name) for name in table)
        for child in value.values():
            names |= _custom_op_names(child)
    elif isinstance(value, list):
        for child in value:
            names |= _custom_op_names(child)
    return names


def _rewrite_symbols(doc: dict, *, symbol, parameter, expression) -> None:
    """Walk a document's dim positions, rewriting each through its own rule.

    Three positions, three rules: a model's own `dims:` block DEFINES symbols;
    a node's `dims: {…}` key is the callee's formal PARAMETER, bound by its
    value; and every other position holds an EXPRESSION in the enclosing
    scope. Everything else in a document is a name — a layer, a node, a
    component, a checkpoint path, prose — and is left alone.

    A document's own `custom_ops:` are a namespace it defines: their shape
    labels are local names bound at each call site, not symbols of the
    registry, and `dim_resolution` asks the registry only about a model's
    `dims:` keys. So neither those labels nor the call-site keys that bind
    them are rewritten — a stored spec that calls its own op keeps saying
    the same thing.
    """
    own = _custom_op_names(doc)
    scalar_keys = frozenset(_tables()["dim_expression_keys"]["scalar"])
    list_keys = frozenset(_tables()["dim_expression_keys"]["list"])

    def walk(value: Any, *, root: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if root and key == "dims":
                    continue           # the model's own symbol definitions
                if key == "custom_ops":
                    continue           # the document's own label namespace
                if key == "dims" and isinstance(child, dict):
                    local = str(value.get("op") or "") in own
                    value[key] = {
                        (str(name) if local else parameter(str(name))):
                            expression(item) if isinstance(item, str) else item
                        for name, item in child.items()}
                elif key in scalar_keys and isinstance(child, str):
                    value[key] = expression(child)
                elif key in list_keys and isinstance(child, list):
                    value[key] = [
                        expression(entry) if isinstance(entry, str) else entry
                        for entry in child]
                    for entry in child:
                        walk(entry)
                else:
                    walk(child)
        elif isinstance(value, list):
            for entry in value:
                walk(entry)

    root_dims = doc.get("dims")
    if isinstance(root_dims, dict):
        doc["dims"] = {symbol(str(name)):
                       expression(item) if isinstance(item, str) else item
                       for name, item in root_dims.items()}
    walk(doc, root=True)


def _map_symbols(mapping: dict, counter: list[int]):
    """The `_rewrite_symbols` rules for a plain old → new symbol table."""
    def one(name: str) -> str:
        new = mapping.get(name)
        if new is None:
            return name
        counter[0] += 1
        return new

    def expression(text: str) -> str:
        return _EXPR_IDENT.sub(lambda match: one(match.group(0)), text)

    return one, expression


def _rename_dim_symbols(doc: dict, notes: list[str], blockers: list[str],
                        *, label: str) -> None:
    """Rewrite the retired dim vocabulary (59db3827) through `spec_migrations
    .yaml`, in expression positions only.

    `D` is the one symbol whose reading depends on where it sits: the
    family-neutral parameter name on the left of a node's `dims:`, the
    quantity the document's own `D` binding reads everywhere else (see the
    table's `dispatched:` note).
    """
    mapping = symbol_renames()
    dispatched = _tables()["symbol_vocabulary"]["dispatched"]
    root_dims = doc.get("dims") if isinstance(doc.get("dims"), dict) else None
    old, rule = next(iter(dispatched.items()))
    by_key = rule["by_config_key"]
    readings = sorted(set(by_key.values()))
    family: Optional[str] = None
    if root_dims is None:
        # A component body names no family (it binds no dims of its own).
        unknown = (f"{label} spells the retired symbol {old!r}, which named a "
                   f"different quantity in each attention family "
                   f"({' / '.join(readings)}); write the one its callers "
                   f"bind, then migrate")
    else:
        binding = root_dims.get(old)
        keys = {path.rsplit(".", 1)[-1]
                for path in re.findall(r"\bconfig\.([\w.]+)", str(binding or ""))}
        families = {by_key.get(key) for key in keys}
        if keys and len(families) == 1 and None not in families:
            family = families.pop()
        unknown = (f"{label} binds the retired symbol {old!r} to "
                   f"{binding!r}, which does not read exactly one of the "
                   f"config keys that say its quantity ({sorted(by_key)}); "
                   f"rename it to the quantity it is ({' / '.join(readings)}), "
                   f"then migrate")
    hits = 0

    def refuse() -> None:
        if unknown not in blockers:
            blockers.declare(unknown)

    def expression(text: str) -> str:
        nonlocal hits

        def one(match: re.Match) -> str:
            nonlocal hits
            name = match.group(0)
            if name == old:
                if family is None:
                    refuse()
                    return name
                hits += 1
                return family
            new = mapping.get(name)
            if new is None:
                return name
            hits += 1
            return new

        return _EXPR_IDENT.sub(one, text)

    def parameter(name: str) -> str:
        nonlocal hits
        if name == old:
            hits += 1
            return str(rule["as_parameter_key"])
        new = mapping.get(name)
        if new is None:
            return name
        hits += 1
        return new

    def symbol(name: str) -> str:
        nonlocal hits
        if name == old:
            if family is None:
                refuse()
                return name
            hits += 1
            return family
        new = mapping.get(name)
        if new is None:
            return name
        hits += 1
        return new

    _rewrite_symbols(
        doc, symbol=symbol, parameter=parameter, expression=expression)
    if hits and not blockers:
        notes.append(
            f"renamed {hits} retired dim symbol(s) on {label} to the current "
            "vocabulary")


def _spells(text: str, name: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def _component_references(value: Any, parent: Optional[str] = None) -> set:
    """The component identities a parsed document CALLS.

    The channels are the declared ones (`_component_reference_keys`, plus the
    `drafts:` opt-in list and the slot / parameter maps) — the same set
    `rename_components` rewrites. Read any identifier instead, and a
    component named in a neighbour's comment or prose counts as called: that
    is how a first cut of this had `hunyuan_v3` reaching
    `router_softmax_topk`, which it does not call.
    """
    names: set = set()
    if isinstance(value, str):
        return {value} if parent in _component_reference_keys else names
    if isinstance(value, list):
        for item in value:
            names |= _component_references(item, parent)
        return names
    if not isinstance(value, dict):
        return names
    for key, child in value.items():
        if key == "drafts" and isinstance(child, list):
            names.update(item for item in child if isinstance(item, str))
        elif key in ("components", "params") and isinstance(child, dict):
            names.update(item for item in child.values()
                         if isinstance(item, str))
            for item in child.values():
                names |= _component_references(item, None)
        elif key == "map" and isinstance(child, dict):
            # Keyed by layer type, not by a component slot.
            for row in child.values():
                names |= _component_references(row, None)
        else:
            names |= _component_references(
                child, key if isinstance(key, str) else None)
    return names


class _LibraryDocuments:
    """The component or model documents a rewrite reads, by name.

    The texts are the library's and the record's own drafts; each is parsed
    the first time a rewrite asks for it. A text that does not parse is not a
    document with nothing in it: reading it as empty would decide a rename or
    a binding on a document nobody saw, so asking for one refuses the record
    — once, naming it.
    """

    def __init__(self, kind: str, texts, blockers: list[str]) -> None:
        self._kind = kind
        self._texts = dict(texts or {})
        self._blockers = blockers
        self._parsed: dict = {}

    def __bool__(self) -> bool:
        return bool(self._texts)

    def __contains__(self, name: object) -> bool:
        return name in self._texts

    def names(self) -> list:
        return list(self._texts)

    def parsed(self, name: str) -> Any:
        """The parsed document `name`, or None when there is none by that
        name or its text does not parse (which refuses the record)."""
        if name not in self._texts:
            return None
        if name not in self._parsed:
            try:
                self._parsed[name] = load_yaml(self._texts[name] or "")
            except yaml.YAMLError as exc:
                self._parsed[name] = None
                self._blockers.append(
                    f"the {self._kind} {name!r} this spec reads does not "
                    f"parse as YAML: {exc}")
        return self._parsed[name]


def _component_closure(doc: dict, components: _LibraryDocuments) -> dict:
    """Name → parsed body of every component the model RUNS.

    A component's own body reaches further ones (a block binds an attention
    and an FFN, which bind a router), and the dim a component reads is bound
    by the including MODEL, so the question "does anything this spec uses
    read this symbol" is a question about the whole closure.

    The walk starts where the build does: the model's layers and blocks
    (`_walked_seeds`), not the `components:` slot map. The loader loads every
    slot, but a slot runs only where a layer reaches it; LongCat declares
    `ffn_moe: moe_ffn` while its block calls the classifier router, and
    counting that slot demanded dims nothing reads.
    """
    if not components:
        return {}
    queue = [name for name in _walked_seeds(doc) if name in components]
    closure: dict = {}
    while queue:
        name = queue.pop()
        if name in closure:
            continue
        parsed = components.parsed(name)
        closure[name] = parsed
        queue.extend(
            found for found in _component_references(parsed)
            if found in components and found not in closure)
    return closure


def _walked_seeds(doc: dict) -> set:
    """The component names the model's own layers and blocks call, with the
    `components:` slots resolved the two ways the loader consumes them: a
    layer whose template is the bare `block` (the slot map bound to the
    block pattern), and an assignment's `attention:` / `ffn:` shortcut,
    whose unsaid parts come from the slots
    (`SpecLoader._synthesize_referenced_blocks` /
    `_desugar_assignment_slots`). The `drafts:` opt-in list is not a call.
    """
    slots = doc.get("components")
    slots = slots if isinstance(slots, dict) else {}
    names = _component_references(
        {key: value for key, value in doc.items()
         if key not in ("components", "drafts")})
    attention = slots.get("attention")
    ffn = slots.get("ffn", slots.get("ffn_dense"))
    norm = slots.get("norm", "rmsnorm")
    pattern = slots.get("block", "pre_norm")

    def walk(layers: Any) -> None:
        for layer in layers if isinstance(layers, list) else ():
            if not isinstance(layer, dict):
                continue
            if layer.get("template") == "block":
                names.update(
                    name for name in (pattern, attention, ffn, norm) if name)
            for assign in layer.get("assignments") or ():
                if not isinstance(assign, dict) or not (
                        assign.get("attention") or assign.get("ffn")):
                    continue
                names.update(name for name in (
                    assign.get("attention") or attention,
                    assign.get("ffn") or ffn, norm,
                    assign.get("template") or pattern) if name)
            walk(layer.get("layers"))

    walk(doc.get("model"))
    return {str(name) for name in names}


def _symbols_read(parsed: Any) -> set:
    """The dim symbols a parsed component body's EXPRESSIONS name.

    Its comments and prose are not expressions: a component that documents a
    symbol in a `# Required dims:` line does not read it, and a rewrite that
    counted those would fire on a spec whose components never ask.
    """
    if not isinstance(parsed, dict):
        return set()
    names: set = set()

    def collect(name: str) -> str:
        names.add(name)
        return name

    def expression(text: str) -> str:
        for match in _EXPR_IDENT.finditer(text):
            names.add(match.group(0))
        return text

    # The same walk the vocabulary rename uses, reading instead of writing:
    # one answer to "where does a dim expression sit".
    _rewrite_symbols(parsed, symbol=collect, parameter=lambda name: name,
                     expression=expression)
    return names


def _rename_renamed_components(doc: dict, notes: list[str]) -> None:
    """Point a call at the component's current name (1299577b)."""
    table = _tables()["components"]
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    mapping = {old: new for old, new in table.items() if _spells(text, old)}
    if not mapping:
        return
    renamed = rename_components(doc, mapping)
    if renamed == doc:
        return
    doc.clear()
    doc.update(renamed)
    for old, new in mapping.items():
        notes.append(f"renamed the component {old!r} to {new!r}")


def _rename_component_dims(doc: dict, components, notes: list[str],
                           blockers: list[str]) -> None:
    """Bind a component's dim under the name it reads today (7f7f0267).

    A component has no `dims:` block of its own — it reads what the including
    model binds — so renaming the dim renamed the binding in every model that
    calls it. The rewrite is the same, and only while the old name has no
    other reader in the spec's component closure: a model that also runs MLA
    binds `d_v_head` for it, and renaming that one binding would unbind the
    other component. That needs a second binding, which is a value only the
    author knows.
    """
    root = doc.get("dims")
    if not isinstance(root, dict):
        return
    closure = _component_closure(doc, components)
    for component, renames in _tables()["component_dims"].items():
        if component not in closure:
            continue
        for old, new in renames.items():
            if old not in root or new in root:
                continue
            others = sorted(name for name, body in closure.items()
                            if name != component and old in _symbols_read(body))
            if others:
                blockers.declare(
                    f"{component!r} reads the dim {new!r}, renamed from "
                    f"{old!r}, which {others} still read; bind {new!r} for "
                    f"{component!r} beside {old!r}, then migrate")
                continue
            counter = [0]
            one, expression = _map_symbols({old: new}, counter)
            _rewrite_symbols(
                doc, symbol=one, parameter=one, expression=expression)
            notes.append(
                f"bound {new!r} where {old!r} was bound, the dim "
                f"{component!r} reads")


def _family_binding(doc: dict, models: _LibraryDocuments,
                    symbol: str) -> Optional[str]:
    """How the models of this spec's family bind `symbol`, or None.

    The fallback a switch needs is the config CLASS's default, and that fact
    is already declared — by the corpus model of the same family, which was
    hand-migrated when the switch landed (`qwen3_5_moe.yaml` reads
    `config.text_config.norm_topk_prob ?? "1"`, `longcat_flash.yaml`
    `config.norm_topk_prob ?? "0"`: a different default AND a different
    config path). A family is what the document itself declares it is: a
    shared `match.model_type` or `match.model_id` entry, or the corpus model
    it names as its base / `derived_from`. Two families disagreeing is no
    answer at all.
    """
    if not models:
        return None
    match = doc.get("match") if isinstance(doc.get("match"), dict) else {}
    kinds = {str(value) for value in (match.get("model_type") or [])}
    ids = {str(value) for value in (match.get("model_id") or [])}
    named = {str(doc.get(key)) for key in ("base", "derived_from")
             if isinstance(doc.get(key), str)}
    bindings = set()
    for stem in models.names():
        other = models.parsed(stem)
        if not isinstance(other, dict):
            continue
        binding = (other.get("dims") or {}).get(symbol)
        if not binding:
            continue
        other_match = (other.get("match")
                       if isinstance(other.get("match"), dict) else {})
        if (stem in named
                or kinds & {str(v) for v in (other_match.get("model_type") or [])}
                or ids & {str(v) for v in (other_match.get("model_id") or [])}):
            bindings.add(str(binding))
    return bindings.pop() if len(bindings) == 1 else None


def _bind_retired_switches(doc: dict, components, models, notes: list[str],
                           blockers: list[str]) -> None:
    """Bind the config switch a component's step gained (603f85f0, d3ae42d6).

    A step a component runs only when the config says so is a `when:` switch
    on a dim, and a spec stored before the switch existed binds nothing — the
    switch then resolves to its own name, which is not a 0 / 1. What to bind
    is not one global fact: the step is off or on per config CLASS
    (`Qwen3NextConfig` and `Qwen3_5MoeConfig` default `norm_topk_prob` to
    True, and Qwen3.5 publishes it under `text_config`; `LongcatFlashConfig`,
    `Qwen3MoeConfig` and the DeepSeek-V2 configs default it to False), so the
    binding is taken from the corpus model of the spec's own family, which
    declares it. No family, or two families disagreeing, is a refusal:
    writing a fallback of 0 there would silently turn a renormalization off.
    """
    root = doc.get("dims")
    if not isinstance(root, dict):
        return
    closure = _component_closure(doc, components)
    for symbol in _tables()["retired_switches"]:
        if symbol in root:
            continue
        readers = sorted(name for name, body in closure.items()
                         if symbol in _symbols_read(body))
        if not readers:
            continue
        binding = _family_binding(doc, models, symbol)
        if binding is None:
            field = dim_registry.base_symbols()[symbol].config
            blockers.declare(
                f"{readers[0]!r} reads the switch {symbol!r}, which this spec "
                f"does not bind, and no model of its family declares how to "
                f"read it; bind `{symbol}: config.{field} ?? "
                f"\"<your config class's default>\"` — 1 if that class "
                f"renormalizes when the field is absent, 0 if it does not")
            continue
        root[symbol] = binding
        notes.append(
            f"bound the switch {symbol!r} that {readers[0]!r} reads to "
            f"`{binding}`, as this spec's family declares it")


def _as_served(doc: dict, models: _LibraryDocuments) -> dict:
    """The document as the loader serves it: merged down its `base:` chain
    of library models by the one base-merge rule (`merge_base_documents`),
    under which a child that writes `dims:` replaces its base's `dims:`
    whole. `normalize_bundle` has already flattened the record's own bases
    and stops at a library one."""
    merged, seen = doc, set()
    while True:
        base = merged.get("base")
        if not isinstance(base, str) or base in seen or base not in models:
            return merged
        seen.add(base)
        parent = models.parsed(base)
        if not isinstance(parent, dict):
            return merged
        merged = merge_base_documents(parent, merged)


def _call_site_bindings(value: Any) -> set:
    """The dims a `dims:` mapping below a model layer, block node or
    component call gives a value: a block's dim overrides and a call site's
    `dims:` bind for everything under them. `x: x` passes the caller's
    binding through and binds nothing."""
    bound: set = set()
    if isinstance(value, list):
        for item in value:
            bound |= _call_site_bindings(item)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key == "dims" and isinstance(child, dict):
                bound.update(str(name) for name, expr in child.items()
                             if str(expr).strip() != str(name))
            bound |= _call_site_bindings(child)
    return bound


def _require_retired_defaults(doc: dict, components, models,
                              blockers: Refusals) -> None:
    """Send a stored spec back for the dims that lost their registry default
    (`retired_defaults` in spec_migrations.yaml).

    The value was filled in for a spec that bound nothing, which dropped a
    step the forward source runs (DeepSeek-V2's routed scale, its group
    limit) without a word. What to bind is a fact of the model's own forward
    source and config, so the record's author writes it: one refusal per
    unbound dim a called component reads, naming the binding dims.yaml
    documents for it.
    """
    served = _as_served(doc, models)
    closure = _component_closure(served, components)
    if not closure:
        return
    root = served.get("dims") if isinstance(served.get("dims"), dict) else {}
    bound = {str(name) for name in root} | _call_site_bindings(
        [served.get("model"), served.get("blocks"), list(closure.values())])
    rows = dim_registry.base_symbols()
    for symbol in _tables()["retired_defaults"]:
        if symbol in bound:
            continue
        readers = sorted(name for name, body in closure.items()
                         if symbol in _symbols_read(body))
        if not readers:
            continue
        blockers.declare(
            f"{readers[0]!r} reads {symbol!r}, which this spec does not bind; "
            f"bind it in `dims:` from the model's forward source and config "
            f"— {rows[symbol].binding_hint}")


def _derived_coupling(raw: dict) -> Optional[list]:
    """The `couples_on` a custom op's other declarations already imply, or
    None when only its author knows.

    Asked of `CustomOpSpec` itself rather than restated here: an op applying
    an `[out, in]` weight is a linear map, whose in axis contracts by sum
    (`layout_axis_use`), and whose `axis_use` must therefore NOT be declared.
    An op that tags its axes has named them. Anything else is the geometry
    family, where a `couples_on: []` would claim separability rather than
    report it.
    """
    fields = {"params": "0", "weight_shape": None, "out_key": None,
              "signature": {}, "shape_kind": "preserve",
              **{key: value for key, value in raw.items()
                 if key in CustomOpSpec.model_fields}}
    labels = CustomOpSpec.model_construct(**fields).linear_labels
    if labels is not None:
        return [labels[1]]
    if raw.get("axis_use"):
        return sorted(str(name) for name in raw["axis_use"])
    return None


def _migrate_custom_ops(doc: dict, notes: list[str], blockers: list[str],
                        *, identity: str) -> None:
    """Bring a custom op up to the declaration f0c58501 (08-23) requires.

    It dropped `formula_symbols:` and made `couples_on:` required — the shape
    labels the op is NOT separable along, which is what a layout walk needs
    to place it. A linear map's coupling follows from its weight, and an op
    that tags its axes has named them; a geometry op's does not follow from
    anything, and a guessed `couples_on: []` would tell every layout that
    shard is free to cut an axis the op joins.
    """
    table = doc.get("custom_ops")
    if not isinstance(table, dict):
        return
    for name, raw in table.items():
        if not isinstance(raw, dict):
            continue
        for key in _tables()["retired_keys"]["custom_op"]:
            if key in raw:
                raw.pop(key, None)
                notes.append(
                    f"dropped retired {key}: on custom op {name!r} of "
                    f"{identity}")
        if "couples_on" in raw:
            continue
        derived = _derived_coupling(raw)
        if derived is None:
            blockers.declare(
                f"custom op {name!r} of {identity} declares no `couples_on:`, "
                f"required since it became the layout contract: name the "
                f"shape labels the op is NOT separable along, each with an "
                f"`axis_use: {{category, param}}` tag saying how it couples "
                f"(an empty list declares a fully separable op). Only a "
                f"`[out, in]` weight's contraction follows from what is "
                f"already written")
            continue
        raw["couples_on"] = derived
        notes.append(
            f"declared couples_on: {derived} on custom op {name!r} of "
            f"{identity} (its weight's contracted axis)" if derived
            else f"declared couples_on: [] on custom op {name!r}")


def _migrate_mapping(doc: dict, notes: list[str], blockers: list[str],
                     components=None, models=None) -> bool:
    before = len(notes) + len(blockers)
    library = _LibraryDocuments("component", components, blockers)
    family = _LibraryDocuments("model", models, blockers)
    _rename_dim_symbols(
        doc, notes, blockers,
        label="the model" if isinstance(doc.get("model"), list)
        else "this component")
    _rename_renamed_components(doc, notes)
    _migrate_custom_ops(
        doc, notes, blockers,
        identity="the model" if isinstance(doc.get("model"), list)
        else "this component")
    if isinstance(doc.get("model"), list):
        _rename_component_dims(doc, library, notes, blockers)
        _bind_retired_switches(doc, library, family, notes, blockers)
        _require_retired_defaults(doc, library, family, blockers)
        _migrate_model(doc, notes, blockers, library)
    for key in ("blocks", "components"):
        table = doc.get(key)
        if isinstance(table, dict):
            for name, body in table.items():
                if isinstance(body, dict):
                    _migrate_custom_ops(
                        body, notes, blockers, identity=f"block {name}")
                    _migrate_block(body, notes, blockers, identity=str(name))
    # A draft component file is itself a block graph (role/graph/cache).
    if "model" not in doc and ("graph" in doc or "role" in doc or "cache" in doc):
        _migrate_block(doc, notes, blockers, identity="<component>")
    return len(notes) + len(blockers) > before


def _keep_exemptions(doc: dict, retired: Any, notes: list[str]) -> None:
    """Carry an opt-out the retired list actually held into
    `validation_exemptions:`.

    `ignored_config_fields:` was prose for the reviewer with one real reader:
    a `sliding_window` entry turned off the config-window check, which
    f0c58501 replaced with a reason-bearing `validation_exemptions` field.
    Dropping such an entry would turn a declared exemption back into a
    warning, so the entry moves — and the reason says where it came from,
    since the author's own reason was a YAML comment the record no longer
    carries. Which entries can move is read off the exemption model, so this
    is not a second list of field names.
    """
    if not isinstance(retired, list):
        return
    named = {str(entry).split("#")[0].strip().split(".")[-1]
             for entry in retired if isinstance(entry, str)}
    exemptions = doc.get("validation_exemptions")
    if not isinstance(exemptions, dict):
        exemptions = {}
    for field in ValidationExemptions.model_fields:
        if field not in named or exemptions.get(field):
            continue
        exemptions[field] = (
            f"declared unmodelled in the retired `ignored_config_fields:` "
            f"list, whose reason was a comment this record no longer carries "
            f"— restate it")
        notes.append(
            f"kept the {field!r} exemption the retired list held, as "
            f"validation_exemptions.{field}")
    if exemptions:
        doc["validation_exemptions"] = exemptions


def _migrate_model(doc: dict, notes: list[str], blockers: list[str],
                   components: _LibraryDocuments) -> None:
    layers = doc.get("model") or []
    if not isinstance(layers, list):
        return
    for key in _tables()["retired_keys"]["document"]:
        if key in doc:
            _keep_exemptions(doc, doc.pop(key, None), notes)
            notes.append(f"dropped retired {key}: on the model")
    _drop_retired_stream_keys(layers, notes)
    for layer in layers:
        if isinstance(layer, dict):
            _migrate_layer(layer, notes, blockers)
            for child in layer.get("layers") or []:
                if isinstance(child, dict):
                    _migrate_layer(child, notes, blockers)
    _stamp_aux_checkpoint(layers, notes)
    _stamp_input_embedding(layers, notes, blockers)
    _stamp_output_head(layers, notes, blockers)
    _declare_model_inputs(doc, layers, components, notes, blockers)
    _migrate_media_inputs(doc, notes, blockers)
    for layer in _walk_layers(layers):
        _drop_view_merge_out(layer, notes, identity=str(layer.get("name")))
    custom_ops = doc.get("custom_ops")
    _thread_omitted_inputs(
        layers, custom_ops if isinstance(custom_ops, dict) else {},
        notes, blockers)
    _merge_encoder_rows(doc, layers, components, notes, blockers)
    _keep_to_appended(layers, notes, blockers)
    _carry_hidden(layers, notes, blockers)
    _drop_inner_stream_annotations(layers, notes)
    _require_reads_from(doc, layers, components, blockers)


def _inner_stream_names(layers: list) -> set:
    """The inner streams a parsed model declares: neither carried, nor
    appended, nor a reference to an appended one."""
    return {name for layer in _walk_layers(layers)
            if isinstance(layer.get("streams"), dict)
            for name, decl in layer["streams"].items()
            if isinstance(decl, dict) and not (
                decl.get("carries") or "from" in decl or "read" in decl)}


def _drop_inner_stream_annotations(layers: list, notes: list[str]) -> None:
    """`window:` bounded how far back a reader of an inner stream found its
    latest writer, and `init:` said what it read before the first one. A
    reader now names the layer whose write it reads (`reads_from`), so both
    say nothing and go."""
    for layer in _walk_layers(layers):
        streams = layer.get("streams")
        if not isinstance(streams, dict):
            continue
        for name, decl in streams.items():
            if not isinstance(decl, dict) or decl.get("carries") \
                    or "from" in decl or "read" in decl:
                continue
            dropped = [key for key in ("window", "init") if key in decl]
            for key in dropped:
                decl.pop(key)
            if dropped:
                notes.append(
                    f"dropped {' and '.join(dropped)} on inner stream "
                    f"{name!r} of {layer.get('name') or '<layer>'!r}: a "
                    f"reader names the layer it reads in `reads_from`")


def _require_reads_from(doc: dict, layers: list,
                        components: _LibraryDocuments,
                        blockers: Refusals) -> None:
    """A declaration whose components read an inner stream names the layer
    each of its layers reads (`reads_from`). Which one is the reader's to
    say — the layer whose write its forward reads — so a missing one is a
    declaration to write, never one filled in from where the latest writer
    happened to be."""
    inner = _inner_stream_names(layers)
    if not inner:
        return
    inline = doc.get("blocks") if isinstance(doc.get("blocks"), dict) else {}
    slots = doc.get("components") if isinstance(
        doc.get("components"), dict) else {}

    def body(name: str) -> Any:
        found = inline.get(name)
        return found if found is not None else components.parsed(name)

    def reads(names) -> set:
        out: set = set()
        seen: set = set()
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            parsed = body(name)
            if not isinstance(parsed, dict):
                continue
            out.update(r for r in parsed.get("reads") or ()
                       if isinstance(r, str))
            pending.extend(_component_references(parsed))
        return out

    def check(where: str, binding: dict, declared: Any) -> None:
        names = set(_component_references(binding))
        if not binding.get("template"):
            # The slot shortcut fills what it leaves out from `components:`.
            names |= {value for slot, value in slots.items()
                      if isinstance(value, str) and slot not in binding}
        missing = sorted((reads(names) & inner)
                         - set(declared if isinstance(declared, dict) else ()))
        for stream in missing:
            blockers.declare(
                f"{where} runs components that read inner stream "
                f"{stream!r}: declare reads_from: {{ {stream}: <layer> }} — "
                f"the layer whose write these layers read: \"i - 1\" for "
                f"the layer before, or {{ nearest: {{ from: config.<list>, "
                f"is: <value> }} }} for the nearest earlier layer of a set "
                f"the config lists, or {{ map: {{ <layer>: <source>, … }} }} "
                f"per reading layer")

    for layer in _walk_layers(layers):
        name = layer.get("name") or "<layer>"
        if layer.get("template"):
            check(f"layer {name!r}",
                  {"template": layer["template"],
                   "params": layer.get("params") or {}},
                  layer.get("reads_from"))
        per_layer = layer.get("per_layer")
        if not isinstance(per_layer, dict):
            continue
        rows = per_layer.get("map") if isinstance(
            per_layer.get("map"), dict) else {}
        for k, assign in enumerate(per_layer.get("assign") or ()):
            if not isinstance(assign, dict):
                continue
            row = rows.get(assign.get("type"))
            binding = dict(row) if isinstance(row, dict) else {}
            if isinstance(assign.get("params"), dict):
                binding["params"] = {**(binding.get("params") or {}),
                                     **assign["params"]}
            check(f"per_layer.assign[{k}] (type {assign.get('type')}) of "
                  f"{name!r}", binding, assign.get("reads_from"))


#: What a record with an encoder declares for its media, as a blocker states
#: it: every value is read off the framework's processor and merge source.
_MEDIA_DECLARATION = (
    "Declare each media input with its own axes — an image "
    "`{kind: image, axes: [n_img, img_h, img_w]}`, an audio clip "
    "`{kind: audio, axes: [n_aud, s_audio], width: <frame width, or none for "
    "codebook ids>}` — read an image through a `patchify` layer (`dims: "
    "{patch, temporal, channels, align}`, `segment: item | temporal_patch`, "
    "`source_anchor:` the processor's resize factor, patch order and "
    "`cu_seqlens`), and give every edge from an encoder into the language "
    "model `prompt_slots: {domain: <media input>, defines: true | false, "
    "source_anchor: ...}` — one `defines: true` per media input, where the "
    "framework writes the encoder's rows into the prompt")


def _migrate_media_inputs(doc: dict, notes: list[str],
                          blockers: list[str]) -> None:
    """`kind: vision` is `kind: image` now; its rows come from the image's
    axes through a `patchify`, which the record has to declare from the
    processor source, as it has to declare which edge fills the prompt."""
    declared = doc.get("model_inputs")
    if not isinstance(declared, dict):
        return
    for name, entry in declared.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") == "vision":
            entry["kind"] = "image"
            entry.pop("width", None)
            notes.append(f"model input {name!r}: `kind: vision` is "
                         f"`kind: image`")
        if entry.get("kind") in ("image", "audio") and not entry.get("axes"):
            blockers.declare(
                f"model input {name!r} ({entry.get('kind')}) declares no "
                f"`axes:`. " + _MEDIA_DECLARATION)


#: The name a migrated record gives the token ids: the forward's own.
_TOKEN_IDS = "input_ids"


def _lookup(op: Any) -> Any:
    """The row gather `op` makes (`OpSpec.lookup`), or None."""
    return lookup_of(OP_REGISTRY.get(str(op or "")))


def _is_lookup(op: Any) -> bool:
    """Whether `op` looks ids up in a table (`OpSpec.lookup`)."""
    return _lookup(op) is not None


def _is_ids_view(op: Any) -> bool:
    """Whether `op` only lays its one input out again (`OpSpec.is_view_op`
    with one input: a processor's `pad_rows`), so ids stay ids through it."""
    declared = OP_REGISTRY.get(str(op or ""))
    return (declared is not None and declared.is_view_op
            and declared.arity() == 1)


def _declare_model_inputs(doc: dict, layers: list,
                          components: _LibraryDocuments,
                          notes: list[str], blockers: list[str]) -> None:
    """Declare the model inputs a record read before `model_inputs:` existed.

    A lookup outside an encoder group — the input embedding, an MTP head's
    own embedding — read the token ids without naming them; it reads
    `input_ids: {kind: tokens}`. An encoder group's first stage that read
    its media off its op's signature is a declaration to add (its axes, its
    `patchify`, the `prompt_slots` of its edges), read off the processor
    source. A stack calling a library block that gained a trailing
    `input_ids` formal passes the model input to it.
    """
    declared = doc.get("model_inputs")
    declared = dict(declared) if isinstance(declared, dict) else {}
    blocks = doc.get("blocks") if isinstance(doc.get("blocks"), dict) else {}

    def reads(layer: dict, name: str, entry: dict) -> None:
        declared.setdefault(name, entry)
        layer.pop("in", None)
        layer["inputs"] = [name]
        notes.append(f"declared model input {name!r} read by "
                     f"{layer.get('name')!r}")

    def encoder_entry(group: dict, layer: dict) -> None:
        # An encoder's input is the media's own axes cut by its processor,
        # which no width in the record states: the declaration is the
        # writer's, read off the processor source.
        blockers.declare(
            f"encoder group {group.get('name')!r} reads its media at "
            f"{layer.get('name')!r} without naming it. "
            + _MEDIA_DECLARATION)

    id_views: set = set()

    def walk(entries: list, encoder: Optional[dict]) -> None:
        for index, layer in enumerate(entries):
            if not isinstance(layer, dict):
                continue
            children = layer.get("layers")
            if layer.get("group") and isinstance(children, list):
                opens = (layer.get("aux_kind") in ("vision", "audio")
                         and not (layer.get("inputs") or layer.get("in")))
                walk(children, layer if opens else None)
                continue
            if layer.get("inputs") or layer.get("in"):
                stage_inputs = layer.get("inputs") or layer.get("in")
                if isinstance(stage_inputs, str):
                    stage_inputs = [stage_inputs]
                if _is_ids_view(layer.get("op")) and len(stage_inputs) == 1 \
                        and (stage_inputs[0] in declared
                             or stage_inputs[0] in id_views):
                    # The same ids laid out again (a processor's padding).
                    id_views.add(str(layer.get("name")))
                lookup = _lookup(layer.get("op"))
                ids = (stage_inputs[lookup.ids] if lookup is not None
                       and lookup.ids < len(stage_inputs) else None)
                if ids is not None and ids not in declared \
                        and ids not in id_views:
                    _lookup_reads_an_activation(
                        str(layer.get("name")), str(ids), blockers)
                template = layer.get("template")
                body = (blocks.get(template) if template in blocks
                        else components.parsed(template) if template else None)
                formals = (body.get("inputs", list(_DEFAULT_INPUTS))
                           if isinstance(body, dict) else None)
                if (isinstance(formals, list)
                        and len(formals) == len(stage_inputs) + 1
                        and formals[-1] == _TOKEN_IDS):
                    layer.pop("in", None)
                    layer["inputs"] = [*stage_inputs, formals[-1]]
                    declared.setdefault(formals[-1], {"kind": "tokens"})
                    notes.append(
                        f"passed model input {formals[-1]!r} to "
                        f"{layer.get('name')!r}, whose block {template!r} "
                        f"takes it")
                continue
            if encoder is not None:
                if index == 0:
                    encoder_entry(encoder, layer)
            elif _is_lookup(layer.get("op")):
                reads(layer, _TOKEN_IDS,
                      {"kind": "tokens"})

    walk(layers, None)
    if declared:
        doc["model_inputs"] = declared


def _lookup_reads_an_activation(where: str, reads: str,
                                blockers: list[str]) -> None:
    """A table lookup reads ids, a model input with no width. One wired to
    a tensor the graph computes is a wrong edge the record has to fix."""
    blockers.declare(
        f"{where} looks ids up in a table and reads {reads!r}, a tensor the "
        f"graph computes; read a model input of ids instead "
        f"(`model_inputs: {{input_ids: {{kind: tokens}}}}`, passed through "
        f"the component formals that reach the lookup)")


def _rewrite_cache_state_elements(raw: dict, notes: list[str],
                                  blockers, *, identity: str) -> None:
    """A state's size used to be one product; `shape:` is the axes in
    stored order. Rewrite when that order is unique; otherwise name the
    `shape:` to write."""
    cache = raw.get("cache")
    if not isinstance(cache, dict):
        return
    entries = cache.get("entries")
    if not isinstance(entries, list):
        return
    table = _tables().get("cache_state_elements") or {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or "elements" not in entry:
            continue
        expr = str(entry["elements"]).strip()
        name = entry.get("state") or entry.get("node") or f"entries[{index}]"
        if entry.get("shape") is not None:
            blockers.declare(
                f"cache entry {name!r} of {identity} declares both "
                f"`elements:` and `shape:`; a state's size is the product "
                f"of `shape:`. Delete `elements:`.")
            continue
        shape = table.get(expr)
        if shape is None:
            blockers.declare(
                f"cache entry {name!r} of {identity} declares "
                f"`elements: {expr}`. A state's size is the product of its "
                f"`shape:` axes, outermost first in the order the serving "
                f"framework stores it. Replace `elements:` with "
                f"`shape: [...]`.")
            continue
        axes = [str(axis) for axis in shape]
        entry.pop("elements")
        entry["shape"] = axes
        notes.append(
            f"rewrote cache entry {name!r} of {identity}: elements {expr!r} "
            f"→ shape {axes}")


def _stamp_cache_slot_kinds(raw: dict, notes: list[str], blockers, *,
                            identity: str) -> None:
    """Every cache entry declares what it holds (`kind:`). A state carried by
    a kernel that declares `carries_recurrent_state` is that recurrence's
    matrix, `recurrent_state`; any other entry without a kind is named for
    the record to declare — which node is a key, a latent or an index key is
    the author's statement, not something its name settles."""
    cache = raw.get("cache")
    if not isinstance(cache, dict):
        return
    entries = cache.get("entries")
    if not isinstance(entries, list):
        return
    graph = raw.get("graph") if isinstance(raw.get("graph"), dict) else {}
    vocabulary = cache_families.slot_kinds()
    missing = []
    for index, entry in enumerate(entries):
        if isinstance(entry, dict) and entry.get("kind") is not None:
            continue
        if isinstance(entry, dict) and entry.get("state") is not None:
            carrier = graph.get(entry.get("carrier"))
            op = carrier.get("op") if isinstance(carrier, dict) else None
            spec = OP_REGISTRY.get(op)
            if spec is not None and spec.carries_recurrent_state:
                entry["kind"] = "recurrent_state"
                notes.append(
                    f"declared kind: recurrent_state on cache state "
                    f"{entry['state']!r} of {identity} (its carrier runs "
                    f"{op!r}, a recurrence)")
                continue
        name = (entry if isinstance(entry, str) else
                entry.get("node") or entry.get("state")
                if isinstance(entry, dict) else f"entries[{index}]")
        law = ("state" if isinstance(entry, dict) and entry.get("state")
               else "node")
        missing.append((name, law))
    for name, law in missing:
        storage = "state" if law == "state" else "per_token"
        kinds = sorted(k for k, v in vocabulary.items() if v.storage == storage)
        blockers.declare(
            f"cache entry {name!r} of {identity} declares no `kind:`; write "
            f"it as `{{{law}: {name}, kind: <kind>, ...}}` with the kind of "
            f"what it holds, one of {kinds}")


def _drop_view_merge_out(node: dict, notes: list[str], *,
                         identity: str) -> None:
    """A `view_merge`'s width is the product of the two axes it merges; the
    `out:` its call site used to restate goes."""
    dims = node.get("dims")
    if node.get("op") != "view_merge" or not isinstance(dims, dict) \
            or "out" not in dims:
        return
    dims.pop("out")
    if not dims:
        node.pop("dims")
    notes.append(f"dropped `out:` from view_merge {identity}: its width is "
                 f"the product of the axes it merges")


def _stamp_outputs(raw: dict, notes: list[str], *, identity: str) -> None:
    """Declare the outputs a graph left to the parser before `outputs:` was
    required (386c7a7c): every node no other node reads, in graph order —
    the first is what the block hands its caller — or the last node when
    every node is read."""
    graph = raw.get("graph")
    if raw.get("outputs") or not isinstance(graph, dict) or not graph:
        return
    read: set = set()
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        refs = node.get("in", node.get("inputs")) or []
        for ref in [refs] if isinstance(refs, str) else refs:
            read.add(str(ref).partition(".")[0] if str(ref) not in graph
                     else str(ref))
    outputs = [name for name in graph if name not in read] or [list(graph)[-1]]
    raw["outputs"] = outputs
    notes.append(f"declared outputs {outputs} on block {identity}")


def _stack_formals(stack: dict, blocks: dict,
                   components: _LibraryDocuments) -> Optional[list]:
    """How many formals each variant of a stack's block declares, or None
    when neither the document nor the library says.

    A block's `inputs:` is its formal list, defaulting to the one stream
    formal (`spec_parsing._DEFAULT_INPUTS`). A `per_layer:` stack has one
    variant per row of its map, and every one of them is called with the
    stack's `inputs:`.
    """
    templates = [stack["template"]] if stack.get("template") else []
    per_layer = stack.get("per_layer")
    if isinstance(per_layer, dict):
        rows = per_layer.get("map")
        if not isinstance(rows, dict):
            return None
        for row in rows.values():
            if not isinstance(row, dict) or not row.get("template"):
                return None
            templates.append(row["template"])
    counts = []
    for name in templates:
        body = blocks.get(name)
        if body is None:
            body = components.parsed(name)
        if not isinstance(body, dict):
            return None
        formals = body.get("inputs", list(_DEFAULT_INPUTS))
        if not isinstance(formals, list):
            return None
        counts.append(len(formals))
    return counts or None


def _merge_encoder_rows(doc: dict, layers: list, components,
                        notes: list[str], blockers: list[str]) -> None:
    """`inputs: [<embedding>, <encoder group>]` on a stack whose block takes
    ONE formal becomes an explicit `scatter_rows_replace` the stack reads.

    A stack's `inputs:` bind its block's formals one to one since f40c3ad5
    (09-29); before, a decoder whose block declares one formal could name two
    and the second was silently dropped — which is how Kimi-K2.5, Kimi-K3 and
    MiniMax-M3 lost their vision tower. What the two inputs meant is what the
    corpus now writes: the tower's rows overwrite the image-token rows of the
    embedding stream before the decoder runs (sglang `mm_utils.py`
    `masked_scatter_`), so the merge is a model-level node beside the
    embedding and the stack reads only it.

    Two inputs are not by themselves the old form. The same commit KEPT two
    inputs binding two formals (`ArchSpec._check_stack_call_arity`), so a
    block that declares a second formal is a stack saying what it means, and
    a stack whose formals neither the document nor the library spells out
    says nothing either way: both are left alone.
    """
    blocks = doc.get("blocks") if isinstance(doc.get("blocks"), dict) else {}
    top = [layer for layer in layers if isinstance(layer, dict)]
    embedding = next(
        (layer for layer in top
         if layer.get("boundary_role") == "input_embedding"), None)
    if embedding is None or not embedding.get("name"):
        return
    encoders = {
        layer.get("name") for layer in top
        if layer.get("group") and layer.get("aux_kind") in ("vision", "audio")}
    taken = {layer.get("name") for layer in _walk_layers(layers)}
    for layer in _walk_layers(layers):
        taken.update(layer.get("streams") or {})
    # The merge is a model-level node, so the stack that reads it is one too;
    # a stack inside a group has no place to put it and keeps the schema's
    # arity refusal.
    for stack in list(top):
        if not (stack.get("template") or stack.get("per_layer")):
            continue
        inbound = list(stack.get("inputs") or ())
        if len(inbound) != 2 or inbound[0] != embedding["name"]:
            continue
        if inbound[1] not in encoders:
            continue
        formals = _stack_formals(stack, blocks, components)
        if formals is None or any(count != 1 for count in formals):
            continue
        parent, _, _ = str(embedding["name"]).rpartition(".")
        name = f"{parent}.embed_merge" if parent else "embed_merge"
        if name in taken:
            blockers.declare(
                f"stack {stack.get('name')!r} reads {inbound}, which is now "
                f"the merge {name!r} of the encoder's rows into the embedding "
                f"stream, and that name is already taken; rename it, then "
                f"migrate")
            continue
        taken.add(name)
        stack["inputs"] = [name]
        layers.insert(layers.index(stack), {
            "name": name, "op": "scatter_rows_replace", "inputs": inbound})
        notes.append(
            f"declared {name!r} (scatter_rows_replace) for the encoder rows "
            f"stack {stack.get('name')!r} read as a second input")


def _chain_stacks(layers: list) -> Iterable[dict]:
    """The layers that repeat a block and hand its output to the next
    iteration: a `template:` / `per_layer:` layer with `repeat:`, and a
    repeated `group:` whose children do not repeat themselves. Under a
    repeated group whose child repeats, the innermost repeated layer.
    The same rule as `tools/chain_stacks.py`, over a parsed document."""
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        children = layer.get("layers")
        if isinstance(children, list) and children:
            if layer.get("repeat") is not None and not any(
                    isinstance(c, dict) and c.get("repeat") is not None
                    for c in children):
                yield layer
            yield from _chain_stacks(children)
            continue
        if layer.get("repeat") is not None and (
                layer.get("template") or layer.get("per_layer")):
            yield layer


def _carry_hidden(layers: list, notes: list[str], blockers: list[str]) -> None:
    """Declare the carried stream on every chain stack that has none.

    Before it was declared, every stack threaded the hidden state through
    its block's first formal by default. That is now the stack's
    `hidden: { carries: true }`; a stack without it hands
    nothing from one iteration to the next. A stack that already carries a
    stream is left alone, so the rewrite is idempotent.
    """
    appended = _appended_names(layers)
    for stack in _chain_stacks(layers):
        name = stack.get("name") or "<layer>"
        streams = stack.get("streams")
        if streams is None:
            streams = {}
        if not isinstance(streams, dict):
            blockers.append(
                f"layer {name!r} has `streams:` that is not a mapping")
            continue
        if any(isinstance(decl, dict) and decl.get("carries")
               for decl in streams.values()):
            continue
        if "hidden" in streams:
            blockers.declare(
                f"chain stack {name!r} already declares a stream named "
                f"`hidden` that it does not carry; rename it, then the "
                f"carried stream `hidden` can be declared")
            continue
        # A stack that reads an appended stream at its block boundary, or
        # appends its block's output, runs its iterations side by side.
        if (appended & set(stack.get("inputs") or ())) or any(
                isinstance(decl, dict) and decl.get("from") == "out"
                for decl in streams.values()):
            continue
        stack["streams"] = {"hidden": {"carries": True},
                            **streams}
        notes.append(
            f"declared the carried stream `hidden` on chain stack {name!r}")


def _appended_names(layers: list) -> set:
    """The appended streams (`from:`) a parsed model declares."""
    return {name for layer in _walk_layers(layers)
            if isinstance(layer.get("streams"), dict)
            for name, decl in layer["streams"].items()
            if isinstance(decl, dict) and "from" in decl}


def _keep_to_appended(layers: list, notes: list[str],
                      blockers: list[str]) -> None:
    """`keep: {p: {layers, index, src}}` becomes the appended stream
    `p: {from: <carried>, after: <layers>, index, src}`, and a reader's
    `inputs: [<stack>.<p>]` becomes `inputs: [p]`.

    A port published the stack's carried stream after chosen iterations,
    and its readers were stacks whose iteration j read element j — what an
    appended stream and a stack reading it at its block boundary say now.
    `p` becomes a model-wide name, so it must not already name a layer, a
    stream or another port. A reader that carries a stream, and a writer or
    reader under a group that repeats, have no element-wise reading to
    rewrite to; those come back as blockers.
    """
    groups = {id(child): layer for layer in _walk_layers(layers)
              for child in layer.get("layers") or []
              if isinstance(child, dict)}
    writers = [layer for layer in _walk_layers(layers)
               if layer.get("keep") is not None]
    if not writers:
        return
    taken = {layer.get("name") for layer in _walk_layers(layers)}
    for layer in _walk_layers(layers):
        streams = layer.get("streams")
        if isinstance(streams, dict):
            taken.update(streams)
    ports = [(writer, port) for writer in writers
             if isinstance(writer["keep"], dict) for port in writer["keep"]]
    for writer, port in ports:
        taken.add(f"{writer.get('name')}.{port}")

    def repeated_group(layer: dict) -> Optional[str]:
        group = groups.get(id(layer))
        if group is not None and str(group.get("repeat", "1")) != "1":
            return group.get("name")
        return None

    rewrites: list = []
    for writer in writers:
        name = writer.get("name") or "<layer>"
        keep = writer["keep"]
        if not isinstance(keep, dict):
            blockers.declare(f"layer {name!r} has `keep:` that is not a mapping")
            continue
        streams = writer.get("streams") or {}
        carried = [n for n, decl in streams.items()
                   if isinstance(decl, dict) and decl.get("carries")]
        for port, kept in keep.items():
            if not isinstance(kept, dict) or set(kept) != {
                    "layers", "index", "src"}:
                blockers.declare(
                    f"`keep:` port {port!r} on {name!r} is not "
                    f"`{{layers, index, src}}`")
                continue
            if port in taken or sum(p == port for _, p in ports) > 1:
                blockers.declare(
                    f"`keep:` port {name}.{port} becomes the model-wide "
                    f"stream {port!r}, which already names a layer, a "
                    f"stream or another port; rename the port, then migrate")
                continue
            source = f"{name}.{port}"
            readers = [layer for layer in _walk_layers(layers)
                       if source in (layer.get("inputs") or ())]
            nested = [n for n in (repeated_group(writer),
                                  *map(repeated_group, readers)) if n]
            if nested:
                blockers.declare(
                    f"`keep:` port {source!r} is written or read inside "
                    f"group {nested[0]!r}, which repeats; no element-wise "
                    f"pairing is defined there")
                continue
            chained = [layer.get("name") for layer in readers
                       if any(isinstance(decl, dict) and decl.get("carries")
                              for decl in (layer.get("streams") or {}).values())]
            if chained:
                blockers.declare(
                    f"stack {chained[0]!r} reads `keep:` port {source!r} and "
                    f"carries a stream; whether it reads each element in "
                    f"parallel or carries its own cannot be told from the "
                    f"document")
                continue
            rewrites.append((writer, port, kept, carried[0] if carried
                             else "hidden", readers, source))
    if blockers:
        return
    for writer, port, kept, carried, readers, source in rewrites:
        writer.setdefault("streams", {})[port] = {
            "from": carried, "after": kept["layers"], "index": kept["index"],
            "src": kept["src"]}
        for reader in readers:
            reader["inputs"] = [port if ref == source else ref
                                for ref in reader["inputs"]]
        notes.append(
            f"rewrote `keep:` port {source!r} as the appended stream "
            f"{port!r} and its {len(readers)} reader(s)")
    for writer in writers:
        writer.pop("keep", None)


def _migrate_layer(layer: dict, notes: list[str], blockers: list[str]) -> None:
    name = layer.get("name") or "<layer>"
    if "assignments" in layer:
        blockers.append(
            f"layer {name!r} still uses the removed `assignments:` syntax; "
            "rewrite it as `per_layer: {assign, map}` or regenerate")
        return
    if layer.get("group") is True and layer.get("aux_kind") in ("vision", "audio"):
        dropped = [k for k in ("in", "inputs") if k in layer]
        if dropped:
            for k in dropped:
                layer.pop(k, None)
            notes.append(
                f"dropped {', '.join(dropped)} on {layer.get('aux_kind')} "
                f"group {name!r} (encoders name no backbone tensor)")
    if layer.get("op"):
        _stamp_retired_axis(layer, notes, identity=repr(name))
        _refuse_weight_copies(layer, blockers, identity=repr(name))
    if "tie_weights" in layer:
        _rewrite_tie_weights(layer, notes, blockers)


def _rewrite_tie_weights(layer: dict, notes: list[str], blockers: list[str]) -> None:
    """`tie_weights` became `tied_weight: {to}` / `{to, when}` (f0c58501).

    False constants (`"0"` / `"false"`) drop the field. True constants
    (`"1"` / `"true"`) become an unconditional `{to}`. Spellings the old
    resolver treated as false (`"yes"` / `"on"` / `"no"` / `"off"`) also
    drop — they are not `config.*` and must not become an unconditional
    tie. Only a `when` the current schema accepts (`config.<path>` with
    optional `??`) is kept as a condition. Anything else is a blocker.
    """
    name = layer.get("name") or "<layer>"
    old = layer.get("tie_weights")
    if "tied_weight" in layer:
        layer.pop("tie_weights", None)
        notes.append(f"dropped redundant tie_weights on {name!r}")
        return
    if not isinstance(old, str) or not old.strip():
        blockers.declare(
            f"layer {name!r} has tie_weights={old!r}; cannot rewrite to "
            f"tied_weight {{to, when}}")
        return
    value = old.strip()
    folded = value.lower()
    if folded in ("0", "false", "no", "off", "yes", "on"):
        layer.pop("tie_weights", None)
        notes.append(f"dropped tie_weights={old!r} on {name!r} (not tied)")
        return
    if folded in ("1", "true"):
        # `to` is filled once the unique embedding is known.
        layer["_pending_tied_when"] = None
        layer.pop("tie_weights", None)
        return
    if _TIED_WHEN_RE.fullmatch(value):
        layer["_pending_tied_when"] = value
        layer.pop("tie_weights", None)
        return
    blockers.declare(
        f"layer {name!r} has tie_weights={old!r}; not a config.* condition "
        "the current schema accepts, and not a true/false constant")


def _stamp_aux_checkpoint(layers: list, notes: list[str]) -> None:
    """Declare where an indexed aux stack lives (a6020cca, 08-23).

    An aux group that repeats a block must declare `checkpoint: {prefix,
    index}`; before that commit the prefix was derived from the document and
    the index space was always the stack's own. The derivation, from
    `spec_tree.block_metas` as it then read: a child whose name carries the
    structural path (a dotted `model.visual.blocks`) was the prefix, and a
    bare MTP-style label meant the group's own name.

    Only the group whose index space ONE child owns is stamped — the one this
    reads unambiguously. With two indexed children the retired rule gave each
    its own prefix, which one declaration on the group cannot say, and the
    schema's refusal names what to write.
    """
    for group in layers:
        if not isinstance(group, dict):
            continue
        children = group.get("layers")
        if not (group.get("group") and group.get("aux_kind")
                and isinstance(children, list) and children):
            continue
        if group.get("checkpoint") is not None:
            continue
        blocks = [child for child in children if isinstance(child, dict)
                  and (child.get("template") or child.get("per_layer")
                       or child.get("assignments"))]
        indexed = [child for child in blocks
                   if child.get("repeat") is not None
                   or child.get("assignments") or child.get("per_layer")]
        if not indexed and group.get("repeat") is not None:
            # The group carries the count; its one block child is the stack.
            indexed = blocks
        if len(indexed) != 1:
            continue
        name = str(indexed[0].get("name") or "")
        prefix = name if "." in name else str(group.get("name") or "")
        if not prefix:
            continue
        group["checkpoint"] = {"prefix": prefix, "index": "local"}
        notes.append(
            f"declared checkpoint: {{prefix: {prefix}, index: local}} on aux "
            f"group {group.get('name')!r} (was derived from the stack's name)")


def _stamp_input_embedding(layers: list, notes: list[str],
                           blockers: list[str]) -> None:
    top = [layer for layer in layers if isinstance(layer, dict)]
    # The document's rule (`SpecDocument._nonempty_model`): exactly one
    # input embedding, at the top level or — a pipeline whose tokens enter
    # its text encoder — inside a group.
    declared = [layer for layer in _walk_layers(top)
                if isinstance(layer, dict)
                and layer.get("boundary_role") == "input_embedding"]
    embeddings = [layer for layer in top if layer.get("op") == "embedding"]
    target: Optional[dict] = None
    if len(declared) == 1:
        target = declared[0]
    elif not declared and len(embeddings) == 1:
        embeddings[0]["boundary_role"] = "input_embedding"
        notes.append(
            f"stamped boundary_role: input_embedding on {embeddings[0].get('name')!r}")
        target = embeddings[0]
    else:
        blockers.declare(
            "an authored model must declare exactly one top-level "
            "`boundary_role: input_embedding`; found "
            f"{len(declared)} declared and {len(embeddings)} embedding op(s)")

    pending = [layer for layer in _walk_layers(top)
               if isinstance(layer, dict) and "_pending_tied_when" in layer]
    if not pending:
        return
    if target is None or not target.get("name"):
        blockers.declare(
            "tie_weights cannot be rewritten because no unique embedding "
            "layer was found to use as tied_weight.to")
        for layer in pending:
            layer.pop("_pending_tied_when", None)
        return
    for layer in pending:
        when = layer.pop("_pending_tied_when")
        tied = {"to": target["name"]}
        if when is not None:
            tied["when"] = when
        layer["tied_weight"] = tied
        notes.append(
            f"rewrote tie_weights on {layer.get('name')!r} to "
            f"tied_weight.to={target['name']!r}")


def _stamp_output_head(layers: list, notes: list[str],
                       blockers: list[str]) -> None:
    """Declare the output head a6020cca (08-23) started requiring.

    The twin of `_stamp_input_embedding`. Before ebbbd13e (08-13) gave the
    head `boundary_role: output_head`, the engine took as the head the
    top-level `op: linear` stage, not repeated, whose name contains
    `lm_head`; a stored spec without the role meant that stage. Two such
    stages are a blocker (stamping the wrong one would move the tied weight
    and the logits); with none, the model has no exit to name and the
    document check says so.
    """
    top = [layer for layer in layers if isinstance(layer, dict)]
    if any(layer.get("boundary_role") == "output_head" for layer in top):
        return
    heads = [layer for layer in top
             if layer.get("op") == "linear" and not layer.get("repeat")
             and "lm_head" in str(layer.get("name") or "")]
    if len(heads) > 1:
        blockers.declare(
            f"no layer declares `boundary_role: output_head`, and "
            f"{[layer.get('name') for layer in heads]} are all `lm_head` "
            f"linears; declare which one is the head")
        return
    if not heads:
        return
    heads[0]["boundary_role"] = "output_head"
    notes.append(
        f"stamped boundary_role: output_head on {heads[0].get('name')!r}")


def _thread_omitted_inputs(layers: list, custom_ops: dict,
                           notes: list[str], blockers: list[str]) -> None:
    """Omitted `inputs:` meant "the previous stage" until 401d7164 made it
    a load error. Write out what that default read.

    The retired rule, per stack: a stage without `inputs:` read the last
    earlier stage of its stack that advanced the hidden state (a buffer does
    not). An `aux_kind` group is a stack of its own whose `inputs:` land on
    its first stage; a group without one is part of the enclosing stack.
    Sources and an encoder's first stage read nothing and stay empty.

    `parameter_vector` / `parameter_tensor` did not advance either, but the
    flag that said so is gone, so a consumer whose predecessor is a source
    other than the input embedding is a blocker, not a guess. The refusal
    names the two readings and the stage the author would name for each,
    because the migration this record needs is an authoring decision: the
    Qwen3.8 vision towers put a position table before their blocks, and what
    they mean is `position_table_add` over [patch_embed, pos_embed] — a
    threaded predecessor would validate and be wrong.
    """
    def declaration(op: str):
        custom = custom_ops.get(op)
        return custom if isinstance(custom, dict) else OP_REGISTRY.get(op)

    def shape_kind(op: str):
        decl = declaration(op)
        if isinstance(decl, dict):
            return decl.get("shape_kind")
        return None if decl is None else decl.shape_kind

    def is_embedding(layer: dict) -> bool:
        return layer.get("boundary_role") == "input_embedding"

    def is_source(layer: dict) -> bool:
        # A table lookup read the token ids the retired default never
        # threaded, so it read nothing from the stage before it.
        op = layer.get("op")
        return is_embedding(layer) or _is_lookup(op) or (
            bool(op) and shape_kind(str(op)) == ShapeKind.SOURCE)

    def is_buffer(layer: dict) -> bool:
        decl = declaration(str(layer.get("op") or ""))
        return decl is not None and not isinstance(decl, dict) and decl.is_buffer

    def walk(entries: list, stack: dict) -> None:
        for layer in entries:
            if not isinstance(layer, dict):
                continue
            children = layer.get("layers")
            if layer.get("group") and isinstance(children, list) and children:
                if layer.get("aux_kind"):
                    walk(children, {
                        "previous": None, "stage": None, "first": True,
                        "attach": list(layer.get("inputs") or layer.get("in") or []),
                        "encoder": layer.get("aux_kind") in ("vision", "audio"),
                    })
                else:
                    walk(children, stack)
                continue
            first = stack["first"]
            stack["first"] = False
            if not (layer.get("inputs") or layer.get("in")) \
                    and not is_source(layer) and not (first and stack["encoder"]):
                fill(layer, stack, first)
            if not is_buffer(layer):
                stack["previous"] = layer
                if not is_source(layer):
                    stack["stage"] = layer

    def fill(layer: dict, stack: dict, first: bool) -> None:
        name = layer.get("name") or "<layer>"
        previous = stack["previous"]
        if first and stack["attach"]:
            target = stack["attach"]
        elif previous is None or not previous.get("name"):
            blockers.declare(
                f"layer {name!r} omits `inputs:` and no earlier stage of its "
                "stack exists to read; declare what it reads")
            return
        elif is_source(previous) and not is_embedding(previous):
            source = previous.get("name")
            stage = (stack["stage"] or {}).get("name")
            reads = (f"`inputs: [{stage}]` if this layer only carries that "
                     f"stream on, or `inputs: [{stage}, {source}]` with the op "
                     f"that consumes the table (a position table is read by "
                     f"`position_table_add`)"
                     if stage else
                     f"`inputs: [{source}]` if this layer reads the table "
                     f"itself")
            blockers.declare(
                f"layer {name!r} omits `inputs:`, and the layer before it, "
                f"{source!r}, is a `{previous.get('op')}` source: a parameter "
                f"or buffer the retired sequential default did NOT hand on. "
                f"Whether this layer reads that table, the stage before it, "
                f"or both is not in the document — declare it: {reads}")
            return
        else:
            target = [previous["name"]]
        layer.pop("in", None)
        layer["inputs"] = list(target)
        notes.append(
            f"declared inputs {list(target)} on {name!r} "
            "(was the retired sequential default)")

    walk(layers, {"previous": None, "stage": None, "first": True,
                  "attach": [], "encoder": False})


def _drop_retired_stream_keys(layers: list, notes: list[str]) -> None:
    """Drop the stream keys the schema retired (`retired_keys.stream`)."""
    for layer in _walk_layers(layers):
        streams = layer.get("streams")
        if not isinstance(streams, dict):
            continue
        for stream, decl in streams.items():
            if not isinstance(decl, dict):
                continue
            for key in _tables()["retired_keys"]["stream"]:
                if key in decl:
                    decl.pop(key)
                    notes.append(
                        f"dropped retired {key}: on stream {stream!r} of "
                        f"layer {layer.get('name') or '<layer>'!r}")


def _walk_layers(layers: list) -> Iterable[dict]:
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        yield layer
        children = layer.get("layers") or []
        if isinstance(children, list):
            yield from _walk_layers(children)


def _migrate_block(raw: dict, notes: list[str], blockers: list[str],
                   *, identity: str) -> None:
    retired = [k for k in _RETIRED_CARRY_KEYS if k in raw]
    if retired:
        blockers.append(
            f"block {identity} still declares removed carry field(s) "
            f"{retired}; rewrite as streams / stream_out / reads, or regenerate")
        return
    retired_kv = [k for k in _RETIRED_KV_KEYS if k in raw]
    if retired_kv:
        blockers.append(
            f"block {identity} still declares removed KV field(s) "
            f"{retired_kv}; rewrite as cache: {{family, entries}}, or regenerate")
        return
    if "stream_input" in raw:
        blockers.append(
            f"block {identity} declares removed `stream_input: "
            f"{raw['stream_input']}`; a stack's carried stream binds formal "
            f"0 by position, so list that formal first in `inputs:` and "
            f"reorder its call sites, or regenerate")
        return
    for key in _RETIRED_IDENTITY_KEYS:
        if key in raw:
            raw.pop(key, None)
            notes.append(f"dropped retired {key}: on block {identity}")
    graph = raw.get("graph")
    if isinstance(graph, dict):
        for node_name, node in graph.items():
            if isinstance(node, dict):
                _stamp_retired_axis(
                    node, notes, identity=f"{identity}.{node_name}")
                _refuse_weight_copies(
                    node, blockers, identity=f"{identity}.{node_name}")
                _drop_view_merge_out(
                    node, notes, identity=f"{identity}.{node_name}")
                reads = node.get("in", node.get("inputs")) or []
                reads = [reads] if isinstance(reads, str) else list(reads)
                lookup = _lookup(node.get("op"))
                ids = (reads[lookup.ids] if lookup is not None
                       and lookup.ids < len(reads) else None)
                if ids is not None \
                        and str(ids).partition(".")[0] in graph:
                    _lookup_reads_an_activation(
                        f"{identity}.{node_name}", str(ids), blockers)
    _rewrite_cache_state_elements(
        raw, notes, blockers, identity=identity)
    _stamp_cache_slot_kinds(raw, notes, blockers, identity=identity)
    _stamp_outputs(raw, notes, identity=identity)
