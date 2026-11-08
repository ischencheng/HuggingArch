"""The KV-cache family vocabulary, read from `cache_families.yaml`.

`CacheSpec.family` was free text. This is the registry the loader checks it
against, and the one place a family's `kind` (full / windowed / recurrent /
vision / none) is declared; the validator's sliding-window contract asks
`of_kind("windowed")` instead of keeping its own `{"SWA"}`.

The same file holds the cache SLOT kinds (`slot_kinds:`): what each
`cache.entries` item holds (`kv`, `latent`, `recurrent_state`, ...), and the
storage law (`per_token` / `state`) the kind lives under. `CacheEntry` checks
its `kind:` against them at load.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional

from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.dtypes import canonical
from backend.code_trace.memo import memoize


_REGISTRY_PATH = Path(__file__).parent / "cache_families.yaml"

KINDS: FrozenSet[str] = frozenset({"full", "windowed", "recurrent", "vision", "none"})


@dataclass(frozen=True)
class CacheFamily:
    name: str
    kind: str
    meaning: str


@dataclass(frozen=True)
class SlotKind:
    name: str
    storage: str
    meaning: str
    #: Whether a deployment may choose how the slot is stored. False for a
    #: kind whose values no storage format applies to (`token_window`'s
    #: integer ids).
    storage_format: bool = True
    #: The attention kernel family that reads keys and values from this
    #: kind of slot, or None for a kind no attention span reads them from.
    kernel: Optional[str] = None
    #: What a kernel reading this kind of slot quantises the operand it
    #: multiplies the stored rows with to, by the element dtype the rows are
    #: stored at: an activation format name (`activation_formats/`). Empty
    #: for a kind whose readers take that operand as it arrives. A stored
    #: dtype with no entry here, read by such a kernel, is refused
    #: (`cache.paired_records`).
    paired_operand: Mapping[str, str] = field(default_factory=dict)


@memoize(maxsize=1)
def _raw() -> dict:
    return load_yaml(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}


@memoize(maxsize=1)
def _load() -> Dict[str, CacheFamily]:
    raw = _raw()
    out: Dict[str, CacheFamily] = {}
    for name, entry in (raw.get("families") or {}).items():
        fam = CacheFamily(name=name, **entry)
        if fam.kind not in KINDS:
            raise ValueError(
                f"cache_families.yaml: family {name!r} has kind {fam.kind!r}, "
                f"not one of {sorted(KINDS)}.")
        if not fam.meaning:
            raise ValueError(f"cache_families.yaml: family {name!r} has no meaning.")
        out[name] = fam
    return out


def all_families() -> Mapping[str, CacheFamily]:
    return _load()


def get(name: str) -> Optional[CacheFamily]:
    return _load().get(name)


def is_registered(name: str) -> bool:
    return name in _load()


def kind_of(name: str) -> str:
    fam = _load().get(name)
    if fam is None:
        raise KeyError(f"cache family {name!r} is not registered in cache_families.yaml")
    return fam.kind


def of_kind(kind: str) -> FrozenSet[str]:
    if kind not in KINDS:
        raise KeyError(f"cache family kind {kind!r} is not one of {sorted(KINDS)}")
    return frozenset(n for n, f in _load().items() if f.kind == kind)


@memoize(maxsize=1)
def slot_kinds() -> Mapping[str, SlotKind]:
    """The cache slot kind vocabulary, `slot_kinds:` in cache_families.yaml."""
    out: Dict[str, SlotKind] = {}
    for name, entry in (_raw().get("slot_kinds") or {}).items():
        kind = SlotKind(name=name, **entry)
        # The two storage laws a cache entry obeys: `node:` / `state:`.
        if kind.storage not in ("per_token", "state"):
            raise ValueError(
                f"cache_families.yaml: slot kind {name!r} has storage "
                f"{kind.storage!r}, not `per_token` or `state`.")
        if not kind.meaning:
            raise ValueError(
                f"cache_families.yaml: slot kind {name!r} has no meaning.")
        _check_paired_operand(name, kind.paired_operand)
        if not isinstance(kind.storage_format, bool):
            raise ValueError(
                f"cache_families.yaml: slot kind {name!r} has storage_format "
                f"{kind.storage_format!r}, not true or false.")
        out[name] = kind
    return out


def _check_paired_operand(name: str, paired: Mapping[str, str]) -> None:
    """Each key a stored element dtype, each value a registered activation
    format of that element dtype: the operand is quantised to the stored
    rows' element type, its scales are the reader's."""
    if not isinstance(paired, Mapping):
        raise ValueError(
            f"cache_families.yaml: slot kind {name!r}: `paired_operand` maps "
            f"a stored dtype to an activation format, not {paired!r}.")
    for dtype, fmt in paired.items():
        if fmt not in ACTIVATION_FORMATS:
            raise ValueError(
                f"cache_families.yaml: slot kind {name!r}: paired_operand "
                f"{fmt!r} is not an activation format "
                f"({sorted(ACTIVATION_FORMATS)}).")
        if ACTIVATION_FORMATS[fmt].dtype != canonical(str(dtype)):
            raise ValueError(
                f"cache_families.yaml: slot kind {name!r}: paired_operand "
                f"{fmt!r} is {ACTIVATION_FORMATS[fmt].dtype}, not the "
                f"stored {dtype}.")


def formattable_kinds() -> FrozenSet[str]:
    """The slot kinds a deployment may choose a storage format for — what a
    cache precision rule's `slots: all` stands for."""
    return frozenset(n for n, k in slot_kinds().items() if k.storage_format)


def refuse_storage_format(kind: str) -> None:
    """Raise when a storage format is asked of a kind none applies to: a
    precision rule that names it, or an entry of it that would take the
    deployment's KV dtype."""
    if kind not in slot_kinds():
        raise KeyError(f"cache slot kind {kind!r} is not registered")
    if not slot_kinds()[kind].storage_format:
        raise ValueError(
            f"cache slot kind {kind!r} takes no storage format: "
            f"{slot_kinds()[kind].meaning} Its entries keep the dtype they "
            f"declare; choose a format for one of "
            f"{sorted(formattable_kinds())}.")


def slot_kinds_stored(storage: str) -> FrozenSet[str]:
    """The slot kinds that live under one storage law."""
    out = frozenset(n for n, k in slot_kinds().items() if k.storage == storage)
    if not out:
        raise KeyError(f"no slot kind is stored {storage!r}")
    return out
