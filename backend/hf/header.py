"""A checkpoint's safetensors header: every tensor the file holds, by the name
the file gives it.

Names are kept verbatim. Stored as written they would run to millions of
strings across the corpus (a 384-expert, 61-layer bank is 70k names), so the
header is kept as GROUPS: the tensors whose names differ only in their numeric
segments and that share one dtype and shape. A group is its name with `{}` at
each numeric segment and the index tuples present there; the encoding is
exact, `names()` gives back every name the file holds.

What an index MEANS is not this module's business. A spec names its tensors
from its declarations, with `X` at each position a declaration indexes
(`model.layers.X.mlp.experts.X.w1.weight`) and the literal segment everywhere
else (`model.layers.X.mlps.0.gate_proj.weight`); `select` answers which header
tensors such a name stands for: `X` takes any index the file holds at that
position, a literal digit only its own, and an index set in the notation
`display` writes (`{0-2,5}`) each index it lists. A name that writes an index
the file does not have at that position, or omits one it does, selects
nothing.
"""
from __future__ import annotations

from collections import defaultdict
import re
from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple)

from huggingface_hub import errors as hf_errors

# The Hub could not be read: offline mode, or a failure of the hub's own HTTP
# client (DNS, connect, timeout, proxy, a dropped stream). That client is
# httpx2, whose errors are neither `httpx.HTTPError` nor `OSError`; a status
# answer (`HfHubHTTPError`, incl. not-found / gated) subclasses
# `hf_errors.HTTPError` too, so a caller that tells a status apart catches it
# first. The one definition every Hub reader in `backend/` catches.
HUB_UNREACHABLE = (
    hf_errors.OfflineModeIsEnabled,     # HF_HUB_OFFLINE
    hf_errors.HTTPError,                # the hub's HTTP client, any failure
)

#: the index placeholder a spec-composed name writes at a declared index
INDEX = "X"
_SLOT = "{}"

Box = Tuple[Tuple[int, ...], ...]


def _template(name: str) -> Tuple[str, Tuple[int, ...]]:
    """`name` with `{}` at each numeric segment, and those segments' values."""
    segments = name.split(".")
    values = tuple(int(s) for s in segments if s.isdigit())
    return ".".join(_SLOT if s.isdigit() else s for s in segments), values


def _boxes(tuples: Iterable[Tuple[int, ...]], width: int) -> List[Box]:
    """An exact partition of a set of index tuples into products of per-slot
    index sets: tuples grouped by their first index, first indices merged
    where the rest of their tuples coincide."""
    if width == 0:
        return [()]
    by_first: Dict[int, set] = defaultdict(set)
    for t in tuples:
        by_first[t[0]].add(t[1:])
    by_rest: Dict[frozenset, List[int]] = defaultdict(list)
    for first, rest in by_first.items():
        by_rest[frozenset(rest)].append(first)
    out: List[Box] = []
    for rest, firsts in by_rest.items():
        out.extend((tuple(sorted(firsts)), *sub)
                   for sub in _boxes(rest, width - 1))
    return sorted(out)


def _ranges(values: Sequence[int]) -> str:
    """`[0, 1, 2, 5]` → `"0-2,5"`."""
    parts: List[str] = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}-{prev}" if prev != start else str(start))
        start = prev = v
    parts.append(f"{start}-{prev}" if prev != start else str(start))
    return ",".join(parts)


def _parse_ranges(text: str) -> Tuple[int, ...]:
    out: List[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return tuple(out)


@dataclass(frozen=True)
class HeaderGroup:
    """Tensors of one dtype and shape whose names differ only in their numeric
    segments: `template` with each `{}` filled from one index tuple of
    `boxes` (each box a product of per-slot index sets)."""

    template: str
    dtype: str
    shape: Tuple[int, ...]
    boxes: Tuple[Box, ...] = ((),)

    @property
    def count(self) -> int:
        return sum(prod(len(axis) for axis in box) for box in self.boxes)

    @property
    def elements(self) -> int:
        """Elements of ONE tensor of the group."""
        return prod(self.shape)

    @cached_property
    def segments(self) -> Tuple[str, ...]:
        return tuple(self.template.split("."))

    @property
    def slot_positions(self) -> Tuple[int, ...]:
        """The segment positions holding an index."""
        return tuple(i for i, s in enumerate(self.segments) if s == _SLOT)

    @property
    def query_segments(self) -> Tuple[str, ...]:
        """The template as a spec-composed name writes it: `X` at each index."""
        return tuple(INDEX if s == _SLOT else s for s in self.segments)

    def indices(self) -> Iterator[Tuple[int, ...]]:
        for box in self.boxes:
            yield from _product(box)

    def names(self) -> Iterator[str]:
        for index in self.indices():
            yield self.name_at(index)

    def name_at(self, index: Tuple[int, ...]) -> str:
        it = iter(index)
        return ".".join(str(next(it)) if s == _SLOT else s
                        for s in self.segments)

    def count_where(self, fixed: Mapping[int, frozenset]) -> int:
        """How many of the group's tensors carry one of `fixed[slot]` at each
        fixed slot (slot = ordinal of the numeric segment)."""
        total = 0
        for box in self.boxes:
            n = 1
            for slot, axis in enumerate(box):
                if slot in fixed:
                    n *= sum(1 for i in axis if i in fixed[slot])
                else:
                    n *= len(axis)
            total += n
        return total

    def display(self, indices: Optional[Iterable[Tuple[int, ...]]] = None) -> str:
        """The template with each slot's index set written in, for a report
        (`model.layers.{0-60}.mlp.experts.{0-255}.w1.weight`). `indices`
        restricts it to a subset of the group's tuples."""
        boxes = (self.boxes if indices is None
                 else _boxes(indices, self.segments.count(_SLOT)))
        return " | ".join(_fill(self.segments, box) for box in boxes)

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.template, "dtype": self.dtype,
                               "shape": list(self.shape)}
        if self.boxes != ((),):
            out["index"] = [[_ranges(axis) for axis in box]
                            for box in self.boxes]
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "HeaderGroup":
        boxes = tuple(tuple(_parse_ranges(axis) for axis in box)
                      for box in raw.get("index", [[]]))
        return cls(template=raw["name"], dtype=str(raw["dtype"]),
                   shape=tuple(int(d) for d in raw["shape"]), boxes=boxes)


def _product(box: Box) -> Iterator[Tuple[int, ...]]:
    if not box:
        yield ()
        return
    for head in box[0]:
        for tail in _product(box[1:]):
            yield (head, *tail)


def _fill(segments: Sequence[str], box: Box) -> str:
    it = iter(box)
    return ".".join("{" + _ranges(next(it)) + "}" if s == _SLOT else s
                    for s in segments)


@dataclass(frozen=True)
class Selection:
    """The header tensors one spec-composed name stands for: per group, how
    many."""

    name: str
    parts: Tuple[Tuple[HeaderGroup, int], ...]
    #: how many tensors the name lists when it writes no `X` (the product of
    #: its index sets); None when an `X` leaves the count to the file
    listed: Optional[int] = None

    @property
    def count(self) -> int:
        return sum(n for _g, n in self.parts)

    @property
    def complete(self) -> bool:
        """Whether the file holds every tensor the name lists."""
        return bool(self.parts) and (self.listed is None or self.count == self.listed)

    @property
    def elements(self) -> int:
        return sum(g.elements * n for g, n in self.parts)

    def __bool__(self) -> bool:
        return bool(self.parts)

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(sorted({g.shape for g, _n in self.parts}))

    @property
    def dtypes(self) -> Tuple[str, ...]:
        return tuple(sorted({g.dtype for g, _n in self.parts}))


class ReadTensors:
    """The tensors read off a checkpoint's shard headers, by name.

    A name is one tensor of the model wherever it is stored, so reading it
    twice is reading the same tensor: an exporter may write a copy of one
    into a second shard (Step-5-Preview keeps `embed_tokens` in both its
    first and its last shard, and its whole MTP layer in both the last
    shard and the MTP shard). Keyed by name, the tensor set and the byte
    total therefore agree by construction — summing every occurrence
    instead inflates the bytes by each copy and makes a correct spec fail
    the byte gate.

    A copy that disagrees with the first reading of the name (another
    dtype, shape, or length) is not a copy: the checkpoint contradicts
    itself and `add` raises rather than let one of the two win silently.
    """

    def __init__(self) -> None:
        self._tensors: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
        self._nbytes: Dict[str, int] = {}

    def add(self, name: str, dtype: str, shape: Sequence[int], nbytes: int,
            *, shard: str = "") -> None:
        """One tensor as a shard header gives it. `shard` names the file, for
        the message a contradicting copy raises."""
        entry = (str(dtype), tuple(int(d) for d in shape))
        seen = self._tensors.get(name)
        if seen is not None and (seen, self._nbytes[name]) != (entry, int(nbytes)):
            where = f" in {shard}" if shard else ""
            raise ValueError(
                f"tensor {name!r} is stored twice{where} and the copies "
                f"disagree: {seen} at {self._nbytes[name]}B, then {entry} "
                f"at {int(nbytes)}B")
        self._tensors[name] = entry
        self._nbytes[name] = int(nbytes)

    @property
    def names(self) -> frozenset:
        return frozenset(self._tensors)

    @property
    def total_bytes(self) -> int:
        """The bytes the checkpoint's distinct tensors occupy."""
        return sum(self._nbytes.values())

    def header(self, **facts: Any) -> "SafetensorsHeader":
        """The header these tensors make, its `total_size` their bytes."""
        return SafetensorsHeader.from_tensors(self._tensors,
                                              total_size=self.total_bytes,
                                              **facts)


@dataclass(frozen=True)
class SafetensorsHeader:
    """Every tensor a checkpoint file holds, grouped (see the module doc).

    `revision` is the repo commit the header was read at; `total_size` the
    bytes the file's tensors occupy, summed from the shard headers'
    `data_offsets` — the one byte figure measured off the file rather than
    reconstructed from a dtype table (an index's `metadata.total_size` is
    the exporter's to write, and several write a parameter count there).
    Each tensor counts once, however many shards hold a copy of it
    (`ReadTensors`). `read_from` names the public repo a gated checkpoint's
    byte-identical shard files were read from, when they were."""

    groups: Tuple[HeaderGroup, ...]
    revision: Optional[str] = None
    total_size: Optional[int] = None
    read_from: Optional[str] = None

    # -- building ------------------------------------------------------------
    @classmethod
    def from_tensors(cls, tensors: Mapping[str, Tuple[str, Sequence[int]]],
                     **facts: Any) -> "SafetensorsHeader":
        """From `{name: (dtype, shape)}` — the header as read."""
        by_key: Dict[Tuple[str, str, Tuple[int, ...]], List[Tuple[int, ...]]] = (
            defaultdict(list))
        for name, (dtype, shape) in tensors.items():
            template, values = _template(name)
            by_key[(template, str(dtype), tuple(int(d) for d in shape))].append(values)
        groups = tuple(
            HeaderGroup(template, dtype, shape,
                        tuple(_boxes(values, template.split(".").count(_SLOT))))
            for (template, dtype, shape), values in sorted(by_key.items()))
        return cls(groups=groups, **facts)

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "SafetensorsHeader":
        return cls(groups=tuple(HeaderGroup.from_json(g) for g in raw["tensors"]),
                   revision=raw.get("revision"),
                   total_size=raw.get("total_size"),
                   read_from=raw.get("read_from"))

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"revision": self.revision,
                               "total_size": self.total_size}
        if self.read_from:
            out["read_from"] = self.read_from
        out["tensors"] = [g.to_json() for g in self.groups]
        return out

    # -- reading -------------------------------------------------------------
    def __iter__(self) -> Iterator[HeaderGroup]:
        return iter(self.groups)

    def __bool__(self) -> bool:
        return bool(self.groups)

    @property
    def tensor_count(self) -> int:
        return sum(g.count for g in self.groups)

    def names(self) -> Iterator[Tuple[str, HeaderGroup]]:
        for g in self.groups:
            for name in g.names():
                yield name, g

    @cached_property
    def _by_template(self) -> Mapping[str, Tuple[HeaderGroup, ...]]:
        out: Dict[str, List[HeaderGroup]] = defaultdict(list)
        for g in self.groups:
            out[g.template].append(g)
        return {k: tuple(v) for k, v in out.items()}

    def select(self, name: str) -> Selection:
        """The header tensors the spec-composed `name` stands for (module
        doc): `X` takes any index at its position, a literal digit its own."""
        template, fixed = _query(name)
        parts = []
        for g in self._by_template.get(template, ()):
            n = g.count_where(fixed)
            if n:
                parts.append((g, n))
        listed = (prod(len(v) for v in fixed.values())
                  if len(fixed) == template.count(_SLOT) else None)
        return Selection(name, tuple(parts), listed)

    def under(self, module: str) -> Tuple[HeaderGroup, ...]:
        """The groups holding a tensor stored inside `module` (a spec-composed
        module path: `X` any index, a digit its own)."""
        head = module.split(".")
        out = []
        for g in self.groups:
            tail = list(g.query_segments[len(head):])
            if not tail:
                continue
            if any(h is g for h, _n in self.select(".".join([*head, *tail])).parts):
                out.append(g)
        return tuple(out)

    def unclaimed(self, names: Iterable[str]
                  ) -> Tuple[Tuple[HeaderGroup, Tuple[Tuple[int, ...], ...]], ...]:
        """The header tensors none of `names` selects: per group, the index
        tuples left."""
        whole: set = set()
        partial: Dict[int, List[Mapping[int, int]]] = defaultdict(list)
        for name in names:
            _template_of, fixed = _query(name)
            for g, _n in self.select(name).parts:
                if fixed:
                    partial[id(g)].append(fixed)
                else:
                    whole.add(id(g))
        out = []
        for g in self.groups:
            if id(g) in whole:
                continue
            claims = partial.get(id(g), ())
            left = tuple(t for t in g.indices()
                         if not any(all(t[s] in v for s, v in c.items())
                                    for c in claims))
            if left:
                out.append((g, left))
        return tuple(out)


#: an index set as `display` writes it: `{0-2,5}`
_INDEX_SET = re.compile(r"\{\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*\}")


def _query(name: str) -> Tuple[str, Dict[int, frozenset]]:
    """A spec-composed name as a group template, and the index set each slot
    is fixed to: a literal digit its own index, `{0-2,5}` the indices it
    lists (`X` fixes none)."""
    fixed: Dict[int, frozenset] = {}
    template: List[str] = []
    for s in name.split("."):
        listed = _INDEX_SET.fullmatch(s) is not None
        if s == INDEX or s.isdigit() or listed:
            if s != INDEX:
                fixed[template.count(_SLOT)] = frozenset(
                    _parse_ranges(s[1:-1]) if listed else (int(s),))
            template.append(_SLOT)
        else:
            template.append(s)
    return ".".join(template), fixed
