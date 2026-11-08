"""The checkpoint tensors a spec declares unread (`checkpoint_unread:`), held
to the header and split off it.

A spec says what one path through a checkpoint computes; the modules it
declares unread are the rest of the file (`schema.UnreadCheckpointModule`).
A spec is compared against the part of the file its path reads: this module
splits the header in two by the declarations, and everything that compares
the spec with the file — the validator's per-tensor join, unmatched-tensor
gate and byte totals, the analyzer's storage figures — runs on the read
part (`UnreadSplit.read_part`). The unread part is reported per entry, so
its share of the file stays visible.

The declaration is held to the header three ways: an entry that selects no
tensor is stale, two entries that select one tensor say it twice, and a
tensor a spec node stores that an entry also selects is absent from the read
part, which the name join then reports as a stored tensor the checkpoint
does not hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.arch.dtypes import dtype_bytes
from backend.arch.quant_schemes import SCHEME_NONE, scheme_named
from backend.hf.header import SafetensorsHeader
from backend.hf.metadata import summarize


@dataclass(frozen=True)
class UnreadPart:
    """One declaration and the header tensors it selects."""

    entry: Any
    tensors: int
    elements: int
    bytes: int

    def as_dict(self) -> Dict[str, Any]:
        return {"module": self.entry.module, "layers": self.entry.layers,
                "kind": self.entry.kind, "reason": self.entry.reason,
                "source_anchor": self.entry.source_anchor,
                "tensors": self.tensors, "params": self.elements,
                "bytes": self.bytes}


@dataclass(frozen=True)
class UnreadSplit:
    """The header split by a spec's `checkpoint_unread:`."""

    read: Optional[SafetensorsHeader]
    parts: Tuple[UnreadPart, ...]
    #: One line per declaration the header contradicts.
    errors: Tuple[str, ...]

    @property
    def unread_bytes(self) -> int:
        return sum(part.bytes for part in self.parts)

    def read_part(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """`meta` (a `hf.metadata` fetch: the header and its totals) for the
        part of the file the spec's path reads, the totals summed again
        under the scheme the whole was. Unchanged when nothing is unread."""
        if not self.parts:
            return meta
        scheme = (scheme_named(meta["quant_scheme"])
                  if meta.get("quant_scheme") else SCHEME_NONE)
        totals = summarize(self.read, scheme=scheme,
                           num_bits=int(meta.get("num_bits") or 0))
        return {**meta, **{key: totals[key] for key in (
            "header", "total_params", "model_storage_bytes",
            "quantization_metadata_bytes", "checkpoint_storage_bytes",
            "index_total_size")}}

    @classmethod
    def of(cls, header: Optional[SafetensorsHeader],
           entries: Sequence[Any]) -> "UnreadSplit":
        """The split of `header` by `entries`. With nothing declared there is
        nothing to split, header or not."""
        if not entries:
            return cls(read=header, parts=(), errors=())
        if header is None:
            raise ValueError(
                "checkpoint_unread is declared and no checkpoint header was "
                "read to hold it to")
        kept: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
        selected: List[List[Tuple[int, int]]] = [[] for _ in entries]
        errors: List[str] = []
        for name, group in header.names():
            owners = [i for i, entry in enumerate(entries)
                      if entry.selects(name)]
            if not owners:
                kept[name] = (group.dtype, group.shape)
                continue
            if len(owners) > 1:
                errors.append(
                    f"checkpoint_unread: {name!r} is selected by "
                    f"{[entries[i].module for i in owners]}; declare each "
                    f"unread tensor once")
            selected[owners[0]].append(
                (group.elements,
                 int(group.elements * dtype_bytes(group.dtype))))
        parts = []
        for entry, picked in zip(entries, selected, strict=True):
            if not picked:
                where = entry.module + (f" layers {entry.layers}"
                                        if entry.layers else "")
                errors.append(
                    f"checkpoint_unread: {where} selects no tensor of the "
                    f"checkpoint; the module path (or its index set) does not "
                    f"match the header")
            parts.append(UnreadPart(
                entry=entry, tensors=len(picked),
                elements=sum(e for e, _ in picked),
                bytes=sum(b for _, b in picked)))
        total = header.total_size
        read = SafetensorsHeader.from_tensors(
            kept, revision=header.revision, read_from=header.read_from,
            total_size=(None if total is None
                        else int(total) - sum(p.bytes for p in parts)))
        return cls(read=read, parts=tuple(parts), errors=tuple(errors))
