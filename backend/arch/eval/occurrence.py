"""Occurrences: one execution of a node, and the graph of who feeds whom.

The walk evaluates a block template once for all the iterations it stands
for. That is right for a per-template shape and state, and it forgets which
iteration supplied a value — which is exactly what a pipeline cut needs: a cut
is a block partition of the layer loop's iteration space, crossed by the edges
whose two occurrences fall in different blocks.

The walk binds, beside every value an input reads, a `Reference`: for each
iteration of the reading stage, which occurrences produced it — none for a free
input (a request input), one for an ordinary read, m for a history read. The
walk folds those over a template's iterations into one value class
(`Evaluator._element`, `_history`, the carried fixed point); the graph keeps
them apart.

`OccurrenceGraph` holds one edge per (sink occurrence, input, k) → (source
occurrence, port), k indexing the fan-in. Edges that stay within one
iteration of one template are stored once per template and never unrolled;
every other edge — the carry chain, stream reads, named stage reads, edges into
or out of a model-level op — is stored as an explicit occurrence pair.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import (TYPE_CHECKING, Any, Dict, FrozenSet, Iterator, List,
                    Mapping, Optional, Tuple)
from backend.code_trace.memo import memo_property

if TYPE_CHECKING:
    from .program import Program


@dataclass(frozen=True)
class Occurrence:
    """One execution of an IR node: `iteration` is the iteration of the
    stage's own index space it belongs to, None for a model-level op (run
    once)."""

    uid: int
    iteration: Optional[int]


@dataclass(frozen=True)
class Source:
    """What one input reads at one iteration: the producing occurrence and
    port, and how the reference resolved to it (`OccurrenceGraph`'s
    projections select edges by it).

    `via` is one of the reference forms of the spec language:
    - `src`: a sibling's output, same iteration;
    - `carry`: formal 0 of a carrying stack, the previous iteration's exit
      (or the stack's input before its first iteration);
    - `named`: a stage named in `inputs:`, its last iteration's exit;
    - `element`: an appended stream's element paired with this iteration;
    - `history`: an appended stream's elements read at once, in order;
    - `join`: an element a lowered join reads;
    - `stream`: an inner stream, produced by the layer this iteration's
      `reads_from` names (an edge `in:` names, or a `reads:` declaration).
    """

    occurrence: Occurrence
    port: str
    via: str


@dataclass(frozen=True)
class Reference:
    """How the walk bound one value a node input reads: the occurrences that
    produced it, at each iteration of the reading stage (`at`).

    The walk evaluates a template once for all its iterations and binds one
    value per input; the reference it binds beside that value keeps which
    occurrence each iteration reads (`Carried.origin`). It is made where the
    walk binds the value — a sibling's output, a stage's `inputs:` name, the
    carried stream, an appended stream's element, an inner stream — and
    travels with the value through every component formal, so a read in a
    component body has the reference its caller bound.

    `table` holds the sources per reading iteration; a read the same at
    every iteration holds them once, under `None`. `local` marks a sibling
    read: the same iteration of the same template, an edge that never
    crosses a cut. `model_input` names the model input the value is, for
    the iterations whose read is that input."""

    table: Tuple[Tuple[Optional[int], Tuple[Source, ...]], ...] = ()
    local: bool = False
    model_input: Optional[str] = None

    @classmethod
    def fixed(cls, sources: Tuple[Source, ...] = (), *,
              model_input: Optional[str] = None) -> "Reference":
        """A read the same at every iteration: `()` for a free input."""
        return cls(table=((None, tuple(sources)),), model_input=model_input)

    @classmethod
    def sibling(cls, uid: int, port: str) -> "Reference":
        """A `src` edge inside one block template: the same iteration."""
        return cls(table=((None, (Source(Occurrence(uid, None), port,
                                         "src"),)),), local=True)

    def at(self, iteration: Optional[int]) -> Tuple[Source, ...]:
        """The sources iteration `iteration` reads."""
        if self.local:
            ((_, (found,)),) = self.table
            return (Source(Occurrence(found.occurrence.uid, iteration),
                           found.port, found.via),)
        for at, sources in self.table:
            if at is None or at == iteration:
                return sources
        raise ValueError(
            f"a read bound for iterations "
            f"{[at for at, _sources in self.table]} is asked at iteration "
            f"{iteration!r}")

    def via(self, via: str) -> "Reference":
        """The same sources, read by form `via` (an appended stream's
        element read paired, as a history, or by a join)."""
        return replace(self, table=tuple(
            (at, tuple(replace(found, via=via) for found in sources))
            for at, sources in self.table))


@dataclass(frozen=True)
class ReferenceTable:
    """What one walk bound to every input of every node it evaluated
    (`Analysis.input_references`), over the program it walked.

    `arity` is how many inputs the walk bound per evaluated node — none for
    a source op, whatever edge a caller wired to it; a component call's
    `in:` refs, then the streams its body reads with no edge naming them.
    `references` holds exactly one `Reference` per `(uid, input)` those
    counts name: a table with an input missing, or one more, is refused.
    The keys are `program`'s uids; a rewritten program's nodes are not in
    it."""

    program: "Program"
    arity: Mapping[int, int]
    references: Mapping[Tuple[int, int], Reference]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arity", MappingProxyType(dict(self.arity)))
        object.__setattr__(self, "references",
                           MappingProxyType(dict(self.references)))
        expected = {(uid, index) for uid, count in self.arity.items()
                    for index in range(count)}
        missing = sorted(expected - set(self.references))
        extra = sorted(set(self.references) - expected)
        if missing or extra:
            raise ValueError(
                f"{self.program.spec.name}: the walk's reference table "
                f"misses inputs {missing[:4]} and holds unbound inputs "
                f"{extra[:4]}; every input the walk bound has one reference")

    def covers(self, uid: int) -> bool:
        """Whether the walk evaluated node `uid` (and bound its inputs)."""
        return uid in self.arity

    def at(self, uid: int, index: int) -> Reference:
        """What input `index` of node `uid` reads, as the walk bound it."""
        if uid not in self.arity:
            raise ValueError(
                f"{self.program.spec.name}: uid {uid} is no node this walk "
                f"evaluated")
        if not 0 <= index < self.arity[uid]:
            raise ValueError(
                f"{self.program.spec.name}: {self.program.where(uid)} has "
                f"{self.arity[uid]} bound inputs, not an input {index}")
        return self.references[(uid, index)]


@dataclass(frozen=True)
class OccurrenceEdge:
    """Input `input_index` of `sink` reads `source`'s `port` as the `k`-th
    member of its fan-in (the history position for a history read)."""

    sink: Occurrence
    input_index: int
    k: int
    source: Occurrence
    port: str
    via: str


@dataclass(frozen=True)
class StageCrossing:
    """One hop: the value `producer` of occurrence `source_iteration`, sent
    across cut `cut` (from stage `cut` to stage `cut + 1`), once for all
    the sinks that read it past that cut (`sinks`: occurrence and input).
    `serves` names the stacks those sinks run in: a hop is timed when a
    stack it serves is."""

    producer: Tuple[int, str]
    source_iteration: Optional[int]
    cut: int
    sinks: Tuple[Tuple[Occurrence, int], ...]
    serves: FrozenSet[str]
    #: A value that is cache-resident (another layer's cache, read through
    #: `reads_from`): what its readers read of that cache this step
    #: (`layout.stages.CacheRead`), which is what crosses instead of the
    #: producer's output. None for every other hop.
    cache_read: Optional[Any] = None


@dataclass(frozen=True)
class OccurrenceGraph:
    """Every edge of a program's forward, at occurrence granularity, as one
    walk bound it (`Reference`).

    `local` holds the edges inside one iteration of one block template, once
    per template, as `(sink uid, input, k, source uid, port)`: they never
    cross a cut, and holding them per iteration would be O(nodes × layers).
    `pairs` holds every other edge explicitly."""

    program: "Program"
    local: Tuple[Tuple[int, int, int, int, str], ...]
    pairs: Tuple[OccurrenceEdge, ...]

    @classmethod
    def of(cls, table: ReferenceTable) -> "OccurrenceGraph":
        """The graph of the references one walk bound (`ReferenceTable`),
        input by input: every input of every node the walk evaluates. A node
        the program evaluates and the table does not cover is refused."""
        program = table.program
        local: List[Tuple[int, int, int, int, str]] = []
        pairs: List[OccurrenceEdge] = []
        for uid, iterations in program.occurrence_domains():
            if not table.covers(uid):
                raise ValueError(
                    f"{program.spec.name}: {program.where(uid)} is evaluated "
                    f"by the program and bound by no walk")
            for index in range(table.arity[uid]):
                reference = table.at(uid, index)
                if reference.local:
                    local.extend((uid, index, k, found.occurrence.uid,
                                  found.port)
                                 for k, found in enumerate(
                                     reference.at(iterations[0])))
                    continue
                for iteration in iterations:
                    sink = Occurrence(uid, iteration)
                    pairs.extend(
                        OccurrenceEdge(sink=sink, input_index=index, k=k,
                                       source=found.occurrence,
                                       port=found.port, via=found.via)
                        for k, found in enumerate(reference.at(iteration)))
        return cls(program=program, local=tuple(local), pairs=tuple(pairs))

    def edges(self, via: Optional[str] = None) -> Iterator[OccurrenceEdge]:
        """The explicit edges, those that resolved `via` one form when
        given."""
        return (edge for edge in self.pairs
                if via is None or edge.via == via)

    @memo_property
    def fan_in(self) -> Dict[Tuple[Occurrence, int],
                             Tuple[OccurrenceEdge, ...]]:
        """Each explicitly resolved input's sources, in fan-in order."""
        out: Dict[Tuple[Occurrence, int], List[OccurrenceEdge]] = {}
        for edge in self.pairs:
            out.setdefault((edge.sink, edge.input_index), []).append(edge)
        return {key: tuple(sorted(edges, key=lambda e: e.k))
                for key, edges in out.items()}
