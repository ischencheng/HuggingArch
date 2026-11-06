"""The deployment precision profile — one entry for every precision choice.

This module is the profile itself, model-free: its vocabulary, its
canonical form, its key, the presets. `arch.eval.precision` reads a profile
against one program (`ResolvedPrecision`).

A profile is a set of rules, `selector → format`, in four tables: weights,
activations (the dtype of the edges between kernels), compute (matmul
pins) and cache (per slot kind). An object no rule selects keeps the
precision the model ships with. `as_shipped` is the profile with every
table empty.

Selectors are conjunctions of dimensions, each a closed vocabulary of
declarations — never node or tensor names:

    weights / activations / compute   roles  × stacks
    cache                             slots  × stacks

    roles   roles.yaml names, `weight_class` (the TP weight classes),
            `outside_roles` (no role: embedding, lm_head, model-level norm),
            `all`. A role names itself and every role that refines it
            (`roles.yaml` `refines:`): `attention` reaches
            `attention_output` and an `attention_swa` slot, and a rule
            naming the finer role is the more specific one.
    stacks  `backbone`, `aux:<aux_kind>`, `aux` (every aux kind), `all`
    slots   cache_families.yaml slot kinds a format applies to, `all`

Resolution (`canonicalize`, then `ResolvedPrecision`): a rule's selector
expands to the Cartesian product Σ of its dimension sets. Within one table
rules are deduplicated; two rules with equal Σ and different formats are
refused; two rules that intersect without either containing the other
need a rule whose Σ is their intersection — added automatically
(`derived`) when both name the same format, refused otherwise. The
canonical rule set is therefore closed under intersection, so the rules
selecting any object have a unique most specific one, and that rule's
format is the object's. Tables never override each other.

Canonicalisation happens here and nowhere else: the server expands
presets, fills omitted dimensions with `all`, closes the lattice and
sorts; `profile_key` hashes the result with the schema version and the
registries the formats come from.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (Any, ClassVar, Dict, FrozenSet, Iterable, List, Mapping, Optional,
                    Sequence, Tuple, get_args)


from backend.arch import (activation_formats, cache_families, cache_formats,
                          quant_schemes, roles as roles_mod)
from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.cache_formats import CACHE_FORMATS, CacheFormat
from backend.arch.cache_formats import describe as describe_cache
from backend.arch.dtypes import (UnknownDtype, canonical, dtype_bytes,
                                 peak_kind)
from backend.arch.quant_schemes import QUANT_SCHEMES
from backend.arch.schema.model import ModelLayer
from backend.arch.schema.quant import PerRoleEntry
from backend.code_trace.memo import memo_property, memoize

#: Raised when selector semantics, resolution rules or the object
#: definitions change: a key computed under the old meaning must not hit.
PROFILE_SCHEMA_VERSION = 1

OUTSIDE_ROLES = "outside_roles"
BACKBONE = "backbone"


class PrecisionError(ValueError):
    """A profile that cannot be canonicalised. `errors` is the repairable
    list a request returns; the message joins them."""

    def __init__(self, errors: Sequence[Dict[str, Any]]):
        self.errors = list(errors)
        super().__init__("; ".join(e["message"] for e in self.errors))


# -- vocabulary ---------------------------------------------------------------

def _aux_kinds() -> Tuple[str, ...]:
    annotation = ModelLayer.model_fields["aux_kind"].annotation
    literal = next(arg for arg in get_args(annotation) if get_args(arg))
    return tuple(get_args(literal))


@memoize(maxsize=1)
def _vocab() -> Mapping[str, FrozenSet[str]]:
    return {
        "roles": frozenset(roles_mod.all_roles()) | {OUTSIDE_ROLES},
        "stacks": frozenset({BACKBONE} | {f"aux:{k}" for k in _aux_kinds()}),
        "slots": cache_families.formattable_kinds(),
    }


def deployable_schemes() -> Dict[str, Any]:
    return {name: s for name, s in QUANT_SCHEMES.items() if s.deployable}


def vocabulary() -> Dict[str, Any]:
    """Everything a precision editor may name (`GET /precision/vocabulary`)."""
    v = _vocab()
    return {
        "roles": sorted(v["roles"] - {OUTSIDE_ROLES}),
        "role_sets": {"weight_class": sorted(roles_mod.weight_classes()),
                      OUTSIDE_ROLES: [], "all": sorted(v["roles"])},
        "stacks": sorted(v["stacks"]),
        "slots": {name: {"storage": kind.storage, "meaning": kind.meaning,
                         "formattable": kind.storage_format}
                  for name, kind in cache_families.slot_kinds().items()},
        "weight_schemes": {
            name: {"fields": dict(s.deployment_fields),
                   "activation_options": {opt: fmt for opt, fmt
                                          in s.activation_options}}
            for name, s in deployable_schemes().items()},
        "activation_formats": {name: f.to_json()
                               for name, f in ACTIVATION_FORMATS.items()},
        "cache_formats": describe_cache(),
        "layouts": [],
    }


@memoize(maxsize=1)
def registry_fingerprint() -> str:
    """Content hash of every registry a canonical profile's formats and
    selectors are read against."""
    h = hashlib.sha256()
    arch = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(arch, "roles.yaml"),
             os.path.join(arch, "cache_families.yaml"),
             os.path.join(arch, "precision_presets.yaml")]
    for pkg in (quant_schemes, activation_formats, cache_formats):
        here = os.path.dirname(pkg.__file__)
        paths += [os.path.join(here, f) for f in sorted(os.listdir(here))
                  if f.endswith(".yaml")]
    for path in paths:
        h.update(os.path.relpath(path, arch).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


# -- rules --------------------------------------------------------------------

Sigma = Tuple[Tuple[str, Tuple[str, ...]], ...]


@dataclass(frozen=True)
class Rule:
    """One canonical rule: its selector as the expanded Σ, and its format
    as canonical JSON text (hashable, comparable)."""

    table: str
    sigma: Sigma
    format: str
    derived: bool = False

    def dims(self) -> Dict[str, FrozenSet[str]]:
        return {d: frozenset(vals) for d, vals in self.sigma}

    def selects(self, point: Mapping[str, str]) -> bool:
        return all(point[d] in vals for d, vals in self.sigma)

    def to_json(self) -> Dict[str, Any]:
        out = {"select": {d: list(vals) for d, vals in self.sigma},
               "format": json.loads(self.format)}
        if self.derived:
            out["derived"] = True
        return out

    def describe(self) -> str:
        sel = ", ".join(f"{d}: [{', '.join(v)}]" for d, v in self.sigma)
        return f"{self.table} {{{sel}}} → {self.format}"


def _sigma(dims: Mapping[str, Iterable[str]]) -> Sigma:
    return tuple((d, tuple(sorted(dims[d]))) for d in sorted(dims))


def _expand(table: str, dim: str, raw: Any, where: str
            ) -> FrozenSet[str]:
    vocab = _vocab()[dim]
    values = [raw] if isinstance(raw, str) else list(raw or [])
    if not values:
        raise PrecisionError([{"path": where, "message":
                               f"{where}: `{dim}` names nothing; omit it "
                               f"to mean `all`"}])
    out: set = set()
    for value in values:
        value = str(value)
        if value == "all":
            out |= vocab
        elif dim == "roles" and value == "weight_class":
            out |= roles_mod.weight_classes()
        elif dim == "roles" and value in vocab and value != OUTSIDE_ROLES:
            # A role and every role that refines it.
            out |= {r for r in vocab - {OUTSIDE_ROLES}
                    if value in roles_mod.lineage(r)}
        elif dim == "stacks" and value == "aux":
            out |= {s for s in vocab if s.startswith("aux:")}
        elif value in vocab:
            out.add(value)
        elif dim == "slots" and value in cache_families.slot_kinds():
            try:
                cache_families.refuse_storage_format(value)
            except ValueError as exc:
                raise PrecisionError([{"path": where,
                                       "message": f"{where}: {exc}"}]
                                     ) from None
        else:
            extra = {"roles": ["weight_class", "all"],
                     "stacks": ["aux", "all"], "slots": ["all"]}[dim]
            raise PrecisionError([{"path": where, "message":
                                   f"{where}: {dim} value {value!r} is not "
                                   f"in the vocabulary "
                                   f"{sorted(vocab) + extra}"}])
    return frozenset(out)


def _canonical_format(table: str, raw: Any, slots: FrozenSet[str],
                      where: str) -> str:
    def fail(message: str) -> PrecisionError:
        return PrecisionError([{"path": where, "message":
                                f"{where}: {message}"}])

    if table == "weights":
        if not isinstance(raw, Mapping):
            raise fail("a weight format is `{scheme: <name>, ...}` or "
                       "`{dtype: <dtype>}`")
        if "dtype" in raw:
            if set(raw) != {"dtype"}:
                raise fail("a plain weight dtype takes no other fields; "
                           "a quantised weight names its `scheme`")
            dtype = _dtype(raw["dtype"], fail)
            if dtype_bytes(dtype) < 2:
                raise fail(f"{dtype} is a quantised width; name the scheme "
                           f"that stores it ({sorted(deployable_schemes())}) "
                           f"so its scales are counted")
            return json.dumps({"dtype": dtype}, sort_keys=True)
        scheme_name = raw.get("scheme")
        schemes = deployable_schemes()
        if scheme_name not in schemes:
            raise fail(f"weight scheme {scheme_name!r} is not one a "
                       f"deployment may use; one of {sorted(schemes)}")
        scheme = schemes[scheme_name]
        fields = dict(scheme.deployment_fields)
        allowed = (set(fields) | set(scheme.required_fields)
                   | set(scheme.optional_fields))
        options = dict(scheme.activation_options)
        body = {k: v for k, v in raw.items()
                if k not in ("scheme", "activation")}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise fail(f"scheme {scheme_name!r} has no field(s) {unknown}; "
                       f"its fields are {sorted(allowed)}")
        fields.update(body)
        out = {"scheme": scheme_name, **fields}
        activation = raw.get("activation")
        if options:
            if activation not in options:
                raise fail(f"scheme {scheme_name!r} takes `activation:` one "
                           f"of {sorted(options)}")
            out["activation"] = activation
        elif activation is not None:
            raise fail(f"scheme {scheme_name!r} offers no activation "
                       f"options; drop `activation`")
        return json.dumps(out, sort_keys=True)
    if table == "activations":
        if isinstance(raw, Mapping) and set(raw) == {"dtype"}:
            dtype = _dtype(raw["dtype"], fail)
            if dtype_bytes(dtype) < 2:
                raise fail(f"{dtype} edges carry scales; name an activation "
                           f"format {sorted(ACTIVATION_FORMATS)}")
            return json.dumps({"dtype": dtype}, sort_keys=True)
        name = raw if isinstance(raw, str) else (
            raw.get("format") if isinstance(raw, Mapping) else None)
        if name in ACTIVATION_FORMATS:
            raise fail(f"an edge below 2 bytes ({name!r}) is written by its "
                       f"producer, and what writes one is a framework's "
                       f"fused quantise; a rule here states the dtype the "
                       f"edges carry")
        raise fail("an activation format is `{dtype: <dtype>}`")
    if table == "compute":
        # A pin states the dtype the matmul runs at, and — below two bytes —
        # the activation format its kernel then eats, because that is what
        # the conversion is priced from. Everything else in §3.7's table is
        # a model fact the pin cannot state, and is checked against the
        # program when the profile is resolved (`ResolvedPrecision`).
        if not isinstance(raw, Mapping) or not set(raw) <= {
                "dtype", "activation_format"}:
            raise fail("a compute pin is `{dtype: <dtype>}`, plus "
                       "`activation_format: <name>` below 2 bytes")
        dtype = _dtype(raw.get("dtype"), fail)
        try:
            peak_kind(dtype)
        except (KeyError, ValueError) as exc:
            raise fail(f"{dtype} has no GPU peak column, so no matmul can "
                       f"run at it ({exc})") from None
        name = raw.get("activation_format")
        if dtype_bytes(dtype) < 2:
            if name not in ACTIVATION_FORMATS:
                raise fail(f"a pin at {dtype} needs the `activation_format:` "
                           f"its kernel eats, one of "
                           f"{sorted(ACTIVATION_FORMATS)}")
            if canonical(ACTIVATION_FORMATS[name].dtype) != dtype:
                raise fail(f"activation format {name!r} is "
                           f"{ACTIVATION_FORMATS[name].dtype}, not the "
                           f"pinned {dtype}")
            return json.dumps({"dtype": dtype, "activation_format": name},
                              sort_keys=True)
        if name is not None:
            raise fail(f"a pin at {dtype} carries no scales; drop "
                       f"`activation_format`")
        return json.dumps({"dtype": dtype}, sort_keys=True)
    # cache
    try:
        fmt = CacheFormat.parse(raw, where)
    except ValueError as exc:
        raise PrecisionError([{"path": where, "message": str(exc)}]
                             ) from None
    if fmt.scale is None and fmt.element_bytes < 2:
        raise fail(f"{fmt.dtype} needs scales; name a format "
                   f"{sorted(CACHE_FORMATS)} or spell out its `scale`")
    if fmt.scale is not None and fmt.scale.rides_positions:
        states = sorted(s for s in slots if cache_families.slot_kinds()[s]
                        .storage == "state")
        if states:
            raise fail(f"{fmt.scale.granularity} scales ride stored "
                       f"positions, and slot kind(s) {states} keep a state "
                       f"with none; narrow `slots`")
    return json.dumps(fmt.to_json(), sort_keys=True)


def _dtype(raw: Any, fail) -> str:
    try:
        return canonical(str(raw))
    except UnknownDtype:
        raise fail(f"{raw!r} is not a dtype spelling") from None


def _close(table: str, rules: List[Rule]) -> List[Rule]:
    """§3.6 step 3: dedupe, refuse equal-Σ conflicts, add the meets."""
    errors: List[Dict[str, Any]] = []
    by_sigma: Dict[Sigma, Rule] = {}
    for rule in rules:
        held = by_sigma.get(rule.sigma)
        if held is None:
            by_sigma[rule.sigma] = rule
        elif held.format != rule.format:
            errors.append({"path": table, "message":
                           f"{held.describe()} and {rule.describe()} select "
                           f"the same objects with different formats; "
                           f"delete one"})
    if errors:
        raise PrecisionError(errors)
    changed = True
    while changed:
        changed = False
        current = list(by_sigma.values())
        for i, a in enumerate(current):
            for b in current[i + 1:]:
                da, db = a.dims(), b.dims()
                meet = {d: da[d] & db[d] for d in da}
                if any(not v for v in meet.values()):
                    continue
                if all(da[d] <= db[d] for d in da) or all(
                        db[d] <= da[d] for d in da):
                    continue
                sigma = _sigma(meet)
                if sigma in by_sigma:
                    continue
                if a.format == b.format:
                    by_sigma[sigma] = Rule(table, sigma, a.format,
                                           derived=True)
                    changed = True
                else:
                    errors.append({
                        "path": table,
                        "intersection": {d: sorted(v) for d, v in meet.items()},
                        "message": (
                            f"{a.describe()} and {b.describe()} both select "
                            f"{ {d: sorted(v) for d, v in meet.items()} } "
                            f"and neither is more specific; add a rule with "
                            f"exactly that selector saying which format it "
                            f"takes")})
        if errors:
            raise PrecisionError(errors)
    # `derived` is a property of the closed set, not of how a rule arrived:
    # a rule is derived exactly when it is the meet of two incomparable
    # rules of its own format. Read off the result, so a canonical profile
    # sent back canonicalizes to itself, flags and key included.
    closed = list(by_sigma.values())
    marked = []
    for rule in closed:
        dims = rule.dims()
        derived = any(
            a.format == b.format == rule.format
            and not _ordered(a, b)
            and all(a.dims()[d] & b.dims()[d] == dims[d] for d in dims)
            for i, a in enumerate(closed) for b in closed[i + 1:]
            if rule is not a and rule is not b)
        marked.append(replace(rule, derived=derived))
    return sorted(marked, key=lambda r: (r.sigma, r.format))


def _ordered(a: Rule, b: Rule) -> bool:
    """Whether one of two rules selects a subset of what the other does."""
    da, db = a.dims(), b.dims()
    return (all(da[d] <= db[d] for d in da)
            or all(db[d] <= da[d] for d in da))


# -- the profile --------------------------------------------------------------

@dataclass(frozen=True)
class PrecisionProfile:
    """A canonical profile: closed rule tables, a label for display."""

    #: Each rule table, and the dimensions its selectors range over.
    DIMENSIONS: ClassVar[Mapping[str, Tuple[str, ...]]] = MappingProxyType({
        "weights": ("roles", "stacks"),
        "activations": ("roles", "stacks"),
        "compute": ("roles", "stacks"),
        "cache": ("slots", "stacks"),
    })

    weights: Tuple[Rule, ...] = ()
    activations: Tuple[Rule, ...] = ()
    compute: Tuple[Rule, ...] = ()
    cache: Tuple[Rule, ...] = ()
    layouts: Tuple[str, ...] = ()
    label: str = field(default="As shipped", compare=False)

    @classmethod
    def canonicalize(cls, raw: Mapping[str, Any]) -> "PrecisionProfile":
        """Expand, validate and close a profile body. Raises
        `PrecisionError` with every repairable error found."""
        if not isinstance(raw, Mapping):
            raise PrecisionError([{"path": "", "message":
                                   "a precision profile is a mapping of "
                                   f"{[*cls.DIMENSIONS, 'layouts', 'label']}"}])
        unknown = sorted(set(raw) - set(cls.DIMENSIONS) - {"layouts", "label"})
        errors: List[Dict[str, Any]] = []
        if unknown:
            errors.append({"path": "", "message":
                           f"unknown precision field(s) {unknown}; the "
                           f"tables are {list(cls.DIMENSIONS)} and `layouts`"})
        tables: Dict[str, Tuple[Rule, ...]] = {}
        for table in cls.DIMENSIONS:
            rules: List[Rule] = []
            for i, entry in enumerate(raw.get(table) or ()):
                where = f"{table}[{i}]"
                try:
                    if not isinstance(entry, Mapping) or "format" not in entry:
                        raise PrecisionError([{"path": where, "message":
                                               f"{where}: a rule is "
                                               f"`{{select, format}}`"}])
                    extra = sorted(set(entry) - {"select", "format", "derived"})
                    if extra:
                        raise PrecisionError([{"path": where, "message":
                                               f"{where}: unknown field(s) "
                                               f"{extra}"}])
                    select = entry.get("select") or {}
                    bad = sorted(set(select) - set(cls.DIMENSIONS[table]))
                    if bad:
                        raise PrecisionError([{"path": where, "message":
                                               f"{where}: {table} rules "
                                               f"select by "
                                               f"{list(cls.DIMENSIONS[table])}, "
                                               f"not {bad}"}])
                    dims = {d: _expand(table, d, select.get(d, "all"), where)
                            for d in cls.DIMENSIONS[table]}
                    fmt = _canonical_format(
                        table, entry["format"], dims.get("slots", frozenset()),
                        where)
                    rules.append(Rule(table, _sigma(dims), fmt))
                except PrecisionError as exc:
                    errors.extend(exc.errors)
            if not errors:
                try:
                    tables[table] = tuple(_close(table, rules))
                except PrecisionError as exc:
                    errors.extend(exc.errors)
        layouts = sorted(set(raw.get("layouts") or ()))
        errors.extend({"path": "layouts", "message":
                       f"layout {name!r} is not declared by any component "
                       f"(layouts: {vocabulary()['layouts']})"}
                      for name in layouts)
        if errors:
            raise PrecisionError(errors)
        label = str(raw.get("label") or "")
        profile = cls(layouts=tuple(layouts), **tables)
        if not label:
            label = PRESET_LABELS.get(profile.preset_id() or "", "Custom")
        return cls(layouts=tuple(layouts), label=label, **tables)

    @classmethod
    def from_request(cls, body: Optional[Mapping[str, Any]]
                     ) -> "PrecisionProfile":
        """`{"preset": id}` or `{"canonical": {...}}`; None is as shipped."""
        if body is None:
            return AS_SHIPPED
        if not isinstance(body, Mapping) or len(body) != 1 or not (
                set(body) <= {"preset", "canonical"}):
            raise PrecisionError([{"path": "precision", "message":
                                   "precision is {\"preset\": <id>} or "
                                   "{\"canonical\": <profile>}"}])
        if "preset" in body:
            preset = PRESETS.get(str(body["preset"]))
            if preset is None:
                raise PrecisionError([{"path": "precision.preset", "message":
                                       f"unknown preset {body['preset']!r}; "
                                       f"one of {sorted(PRESETS)}"}])
            return preset
        return cls.canonicalize(body["canonical"])

    def tables(self) -> Dict[str, Tuple[Rule, ...]]:
        return {"weights": self.weights, "activations": self.activations,
                "compute": self.compute, "cache": self.cache}

    def canonical(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {t: [r.to_json() for r in rules]
                               for t, rules in self.tables().items()}
        out["layouts"] = list(self.layouts)
        return out

    @memo_property
    def key(self) -> str:
        """`sha256(version ‖ registry fingerprint ‖ canonical JSON)[:12]`."""
        text = (f"{PROFILE_SCHEMA_VERSION}|{registry_fingerprint()}|"
                + json.dumps(self.canonical(), sort_keys=True))
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    @property
    def is_as_shipped(self) -> bool:
        return not any(self.tables().values()) and not self.layouts

    def preset_id(self) -> Optional[str]:
        for pid, preset in PRESETS.items():
            if preset == self:
                return pid
        return None

    def to_json(self) -> Dict[str, Any]:
        return {"profile_key": self.key, "label": self.label,
                "preset": self.preset_id(), "canonical": self.canonical()}



def _load_presets() -> Dict[str, Dict[str, Any]]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "precision_presets.yaml")
    with open(path, encoding="utf-8") as f:
        raw = (load_yaml(f) or {}).get("presets") or {}
    for pid, entry in raw.items():
        if set(entry) != {"label", "description", "profile"}:
            raise ValueError(f"precision_presets.yaml: preset {pid!r} is "
                             f"exactly {{label, description, profile}}")
    return raw


#: The presets as the yaml declares them (`precision_presets.yaml`).
_PRESET_DECLS = _load_presets()
PRESET_LABELS: Dict[str, str] = {
    pid: d["label"] for pid, d in _PRESET_DECLS.items()}
PRESETS: Dict[str, PrecisionProfile] = {
    pid: PrecisionProfile.canonicalize({**d["profile"], "label": d["label"]})
    for pid, d in _PRESET_DECLS.items()}
AS_SHIPPED = PRESETS["as_shipped"]


def presets() -> List[Dict[str, Any]]:
    """`GET /precision/presets`."""
    return [{"id": pid, "label": _PRESET_DECLS[pid]["label"],
             "description": _PRESET_DECLS[pid]["description"],
             "profile_key": p.key, "canonical": p.canonical()}
            for pid, p in PRESETS.items()]


def scheme_entry(fmt: Mapping[str, Any]) -> PerRoleEntry:
    """A canonical scheme weight format as the scheme registry reads a
    checkpoint's per-role entry: its method and its fields."""
    fields = {k: v for k, v in fmt.items()
              if k not in ("scheme", "activation")}
    return PerRoleEntry(method=fmt["scheme"], **fields)
