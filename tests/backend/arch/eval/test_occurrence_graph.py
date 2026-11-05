"""The occurrence graph against the walk and the pairing tables.

The walk binds, beside each value an input reads, which occurrences produced
it at each iteration (`Analysis.input_references`). Two things must hold of
it on every model:

- folded over a template's iterations, the producers hold the one value the
  walk bound to that input (same shape and state, and the walk's
  representative is one of the producers) — the carried formal at every
  iteration included;
- its projections are the tables the engine already keeps:
  `stream_pairings` (element reads), `whole_stream_pairings` (history fan-ins,
  element by element, and joins) and the inner streams' declared reads
  (`SpecIRTree.stream_reads` / `stream_writes`).

Each holds of every walk a model has: the language model's step, and each
encoder's encode walk. The language model reads an encoder's output as a
handed-over tensor (`Program.handoffs`): free in its graph, like a model
input, so a pairing whose writer the walk does not evaluate has no edge.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import pytest

from backend.arch.eval import evaluator as evaluator_mod
from backend.arch.eval import transforms as transforms_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.occurrence import Occurrence
from backend.arch.eval.program import BlockStage, OpStage, Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, shared_loader, shared_walk


class _Recording(evaluator_mod.Evaluator):
    """The walk, keeping what it bound to each input and its last `Walk`."""

    def _inputs(self, node, scope, walk):
        out = super()._inputs(node, scope, walk)
        self.bound.setdefault(node.uid, out)
        self.walk = walk
        return out


@contextlib.contextmanager
def _recording():
    seen: List[_Recording] = []

    class Recording(_Recording):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.bound: Dict[int, List[Any]] = {}
            seen.append(self)

    # Patched where the walk names it: `Deployment._walk` in `transforms`.
    original = transforms_mod.Evaluator
    transforms_mod.Evaluator = Recording
    try:
        yield seen
    finally:
        transforms_mod.Evaluator = original


def walk_of(program: Program, phase: str = "prefill", *, media=None,
            stack=None, **parallel):
    """`(analysis, recording evaluator)` of one walk: the language model's
    step, or (`phase="encode"`) encoder `stack` on `media`."""
    deployment = Deployment(program).at(batch=1, input_tokens=16,
                                        media=media or {})
    if phase == "encode":
        deployment = deployment.during(phase, stack=stack)
    elif phase != "prefill":
        deployment = deployment.during(phase, cache_len=64)
    if parallel:
        deployment = replace(deployment, config=ParallelConfig(**parallel))
    with _recording() as seen:
        analysis = shared_walk(deployment)
    return analysis, seen[-1]


def _extents(shape, port: str):
    if shape is None:
        return None
    dims = shape.out if port == "out" else shape.ports.get(port)
    return None if dims is None else tuple(value_of(d) for d in dims)


class FoldOracle:
    """Folds the bound references over each template's iterations and
    compares with what the walk bound (`Evaluator._inputs`)."""

    def __init__(self, analysis, recorded):
        self.analysis = analysis
        self.program = analysis.program
        self.walk = recorded.walk
        self.bound = recorded.bound
        self.rule = analysis.rule

    def handed(self, uid: int, port: str) -> Tuple[Any, Any]:
        """`(extents, state)` of the value `(uid, port)` hands its readers."""
        closed = self.walk.handed_on.get((uid, port))
        if closed is not None:
            return _extents(closed.shape, "out"), closed.state
        return (_extents(self.walk.shapes.get(uid), port),
                self.walk.state_at(uid, port=port))

    def problems(self) -> List[str]:
        out: List[str] = []
        program = self.program
        for uid, iterations in program.occurrence_domains():
            bound = self.bound.get(uid)
            if bound is None:
                out.append(f"{program.where(uid)}: the walk bound nothing")
                continue
            for index, carried in enumerate(bound):
                out.extend(self._input(uid, index, iterations, carried))
        return out

    def _layer_stages(self, exit_uid):
        stage, _root = self.program.evaluating_stage(exit_uid)
        return [other for stack in self.program.stacks
                for other in stack.stages if other.layer_path == stage.layer_path]

    def _input(self, uid, index, iterations, carried) -> List[str]:
        program = self.program
        where = f"{program.where(uid)} input {index}"
        reference = self.analysis.input_reference(uid, index)
        members = [reference.at(i) for i in iterations]
        if carried.shape is None and carried.state is None:
            if any(members):
                return [f"{where}: the walk bound nothing, sources answer "
                        f"{members[0]}"]
            return []
        if not all(members):
            # Free in the graph and bound by the walk is a model input
            # (`model_inputs:`): a tensor nothing in the graph produced.
            if any(members) or carried.producer is not None:
                return [f"{where}: free in sources, bound by the walk"]
            return []
        widths = {len(found) for found in members}
        values = [[self.handed(s.occurrence.uid, s.port) for s in found]
                  for found in members]
        flat = [value for row in values for value in row]
        if any(value != flat[0] for value in flat[1:]):
            return [f"{where}: the producers' values disagree: "
                    f"{sorted(set(map(repr, flat)))[:3]}"]
        extents, state = flat[0]
        history = any(s.via == "history" for found in members for s in found)
        if history:
            (m,) = widths
            expected = (*extents[:-1], m, extents[-1])
            if not self.rule.is_trivial:
                state = self.rule.history_state(state)
        else:
            if widths != {1}:
                return [f"{where}: {sorted(widths)} sources, not a history"]
            expected = extents
            producers = {(s.occurrence.uid, s.port)
                         for found in members for s in found}
            # A stage read by name is its last iteration's exit, and a
            # carried formal the previous iteration's; the walk binds the
            # exit of the variant it walked last (`Evaluator.run`'s
            # `produced_by`, `_eval_link`'s entry) — another occurrence of
            # the same value class, which the fixed point holds equal.
            producers |= {(_exit(other).uid, s.port)
                          for found in members for s in found
                          if s.via in ("named", "carry")
                          for other in self._layer_stages(s.occurrence.uid)}
            if carried.producer is not None and \
                    carried.producer not in producers:
                return [f"{where}: the walk bound {carried.producer}, "
                        f"sources name {sorted(producers)}"]
        port = (carried.producer[1] if carried.producer is not None
                and not history else "out")
        got = _extents(carried.shape, "out")
        if got != expected and _extents(carried.shape, port) != expected:
            return [f"{where}: the walk bound extents {got}, the fold "
                    f"{expected}"]
        if carried.state != state:
            return [f"{where}: the walk bound {carried.state!r}, the fold "
                    f"{state!r}"]
        return []


def _exit(stage):
    """The node a stage hands on: a block's last declared output root, an
    op stage's op."""
    if isinstance(stage, BlockStage):
        return stage.template.output_roots()[-1]
    return stage.node


class ProjectionOracle:
    """The graph's projections against the engine's tables."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.program = analysis.program
        self.graph = analysis.occurrence_graph
        self.walked = self.program.walked_stack_names

    def _template_of(self, stage_name, stack, iteration):
        return next(stage for stack_ in self.program.stacks
                    for stage in stack_.stages
                    if isinstance(stage, BlockStage)
                    and stage.stack == stack and stage.name == stage_name
                    and iteration in stage.template.layer_indices)

    def _element(self, stream, j):
        """Element `j` of appended stream `stream`: the exit of the
        iteration it is appended after, or the appending stack's input."""
        after = self.program.ir.streams[stream].iterations[j]
        if after >= 0:
            (uid, _port), _state = self.analysis.appended[stream][j]
            return Occurrence(uid, after)
        (start,) = {stage.upstream[0] for stack in self.program.stacks
                    for stage in stack.stages
                    if isinstance(stage, BlockStage)
                    and stage.template.meta.layer_path
                    == self.program.ir.streams[stream].layer_path}
        entry = next(stage for stack in self.program.stacks
                     for stage in stack.stages if stage.name == start)
        return Occurrence(_exit(entry).uid,
                          None if isinstance(entry, OpStage)
                          else max(entry.template.layer_indices))

    def problems(self) -> List[str]:
        return [*self.element_reads(), *self.histories(), *self.joins(),
                *self.stream_reads()]

    def element_reads(self) -> List[str]:
        """`stream_pairings`, per (reading stage, iteration)."""
        table: Dict[Tuple, set] = {}
        for row in self.program.stream_pairings:
            # A row whose writer this walk does not evaluate is a handoff
            # (or its reader is not walked either): no edge.
            if not {row.writer_stack, row.reader_stack} <= self.walked:
                continue
            table.setdefault((row.reader_stack, row.reader,
                              row.reader_iteration), set()).add(
                self._element(row.stream, row.element))
        graph: Dict[Tuple, set] = {}
        for edge in self.graph.edges("element"):
            stage, _root = self.program.evaluating_stage(edge.sink.uid)
            graph.setdefault((stage.stack, stage.name, edge.sink.iteration),
                             set()).add(edge.source)
        return [f"element reads at {key}: table {table.get(key)}, graph "
                f"{graph.get(key)}"
                for key in sorted(set(table) | set(graph), key=repr)
                if table.get(key) != graph.get(key)]

    def histories(self) -> List[str]:
        """Each history fan-in, element by element in history order."""
        out: List[str] = []
        expected: Dict[Tuple, Tuple] = {}
        for row in self.program.whole_stream_pairings:
            if (row.representation != "history"
                    or row.reader_stack not in self.walked):
                continue
            elements = tuple(self._element(row.stream, j)
                             for j in range(row.visible))
            for iteration in (row.reader_iterations or (None,)):
                stage = (self._template_of(row.reader, row.reader_stack,
                                           iteration)
                         if row.reader_iterations else None)
                expected[(row.reader_stack, row.reader, iteration,
                          id(stage))] = elements
        seen: Dict[Tuple, set] = {}
        for (sink, _index), edges in self.graph.fan_in.items():
            if not any(edge.via == "history" for edge in edges):
                continue
            stage, _root = self.program.evaluating_stage(sink.uid)
            key = (stage.stack, stage.name, sink.iteration,
                   id(stage) if isinstance(stage, BlockStage) else id(None))
            fan = tuple(edge.source for edge in edges)
            seen.setdefault(key, set()).add(fan)
            if expected.get(key) != fan:
                out.append(f"history at {sink}: table {expected.get(key)}, "
                           f"graph {fan}")
        out.extend(f"history at {key} has no reader in the graph"
                   for key in expected if key not in seen)
        return out

    def joins(self) -> List[str]:
        """Each lowered join reads the k elements in the pairing's order."""
        out: List[str] = []
        streams = self.program.resolved.appended_streams()
        for row in self.program.whole_stream_pairings:
            if (row.representation != "concat" or row.through_uid is None
                    or row.reader_stack not in self.walked):
                continue
            node = self.program.by_uid(row.through_uid)
            iterations = (row.reader_iterations or (None,))
            for iteration in iterations:
                got = [self._joined(node.uid, index, iteration)
                       for index in range(len(node.inputs_obj))]
                want = [self._element(row.stream, j)
                        for j in range(len(streams[row.stream].iterations))]
                if node.op == "concat" and got != want:
                    out.append(f"join {row.stream} at {node.display_path} "
                               f"iteration {iteration}: {got} != {want}")
        return out

    def _joined(self, uid, index, iteration):
        """The element a join's input reads: directly, or through the
        crop the lowering put before it."""
        (found,) = self.analysis.input_reference(uid, index).at(iteration)
        if found.via == "join":
            return found.occurrence
        crop = self.program.by_uid(found.occurrence.uid)
        assert crop.provenance == "lowering", crop.display_path
        return self._joined(crop.uid, 0, found.occurrence.iteration)

    def stream_reads(self) -> List[str]:
        """An inner stream's reader at layer r of its stack reads what the
        layer its `reads_from` names wrote (`SpecIRTree.stream_reads`,
        `stream_writes`)."""
        ir = self.program.ir

        def produced(at, name):
            return Occurrence(ir.stream_writes[(*at, name)], at[1])

        want: Dict[Tuple, set] = {}
        for (stack, layer, name), at in ir.stream_reads.items():
            want.setdefault((stack, layer), set()).add(produced(at, name))
        got: Dict[Tuple, set] = {}
        for edge in self.graph.edges("stream"):
            stage, _root = self.program.evaluating_stage(edge.sink.uid)
            owner = stage.template.meta.iteration_owner
            got.setdefault((owner, edge.sink.iteration), set()).add(
                edge.source)
        return [f"stream read at {key}: graph {got.get(key)}, declared "
                f"{want.get(key)}"
                for key in sorted(set(want) | set(got), key=repr)
                if want.get(key) != got.get(key)]


def check(program: Program, phase: str = "prefill", *, media=None,
          **parallel) -> List[str]:
    """Every oracle-1 disagreement for one model under one mesh: its
    language model's walk, and each encoder's encode walk on `media` (none
    when no media is given)."""
    walks = [walk_of(program, phase, **parallel)]
    if media is not None:
        walks += [walk_of(program, "encode", media=media, stack=stack.name,
                          **parallel)
                  for stack in program.stacks if stack.is_encoder]
    return [problem for analysis, recorded in walks
            for problem in (*FoldOracle(analysis, recorded).problems(),
                            *ProjectionOracle(analysis).problems())]


#: One image, for the encoders of the vision-language models below.
IMAGE = {"pixel_values": {"n_img": 1, "img_h": 672, "img_w": 672}}


def _program(model_id: str) -> Program:
    return Program.of(resolve_snapshot(model_id, shared_loader()))




@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-1B",
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.2",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V3",
    "XiaomiMiMo/MiMo-V2.6-Pro-RL",
    "openai/gpt-oss-20b",
])
def test_the_graph_folds_to_what_the_walk_bound(model_id):
    program = _program(model_id)
    media = IMAGE if any(s.is_encoder for s in program.stacks) else None
    assert check(program, media=media, tp_attention=4, tp_dense_ffn=4,
                 tp_moe_expert=4) == []


def test_kimi_k3_output_attn_res_reads_eight_elements_in_history_order():
    """`output_attn_res` reads the whole `block_residual` bank: element 0 is
    the stack's input (`embed_merge`) and element j the exit of iteration
    12j - 1, in that order."""
    analysis, _recorded = walk_of(_program("moonshotai/Kimi-K3"))
    program = analysis.program
    (stage,) = [s for stack in program.stacks for s in stack.stages
                if isinstance(s, OpStage)
                and s.name.endswith("output_attn_res")]
    node = stage.node
    (index,) = [k for k, ref in enumerate(node.inputs_obj)
                if ref.external_sid == "block_residual"]
    found = analysis.input_reference(node.uid, index).at(None)
    assert [s.via for s in found] == ["history"] * 8
    merge = next(s for stack in program.stacks for s in stack.stages
                 if s.name.endswith("embed_merge"))
    assert found[0].occurrence == Occurrence(merge.node.uid, None)
    assert [s.occurrence.iteration for s in found[1:]] == [
        12 * j - 1 for j in range(1, 8)]


def test_qwen3_vl_deepstack_reader_reads_three_handed_elements():
    """The deepstack reader template is one evaluation for layers 0, 1 and 2,
    which read elements 0, 1 and 2 of the mergers' outputs. The mergers run
    in the vision encoder's walk, so each element reaches the language model
    handed over: the reader's sources are free at every iteration, and the
    pairing table still names one element per iteration."""
    analysis, _recorded = walk_of(_program("Qwen/Qwen3-VL-235B-A22B-Instruct"))
    program = analysis.program
    (stage,) = [s for stack in program.stacks for s in stack.stages
                if isinstance(s, BlockStage) and not stack.is_aux
                and 0 in s.template.layer_indices]
    (node,) = [n for n in stage.template.nodes()
               if any(r.external_sid == "deepstack_feat"
                      for r in n.inputs_obj or ())]
    (index,) = [k for k, r in enumerate(node.inputs_obj)
                if r.external_sid == "deepstack_feat"]
    (handoff,) = [h for h in program.handoffs if h.name == "deepstack_feat"]
    assert (handoff.stream, handoff.domain, handoff.defines) == (
        True, "pixel_values", False)
    reference = analysis.input_reference(node.uid, index)
    assert [reference.at(i)
            for i in stage.template.layer_indices] == [()] * len(
        stage.template.layer_indices)
    rows = sorted((r.element, r.reader_iteration, r.reader_uid)
                  for r in program.stream_pairings
                  if r.stream == "deepstack_feat")
    assert [(e, i) for e, i, _uid in rows] == [(0, 0), (1, 1), (2, 2)]
    assert len({uid for _e, _i, uid in rows}) == 1


def test_glm_shared_reader_reads_the_full_layer_it_names():
    """GLM-5.2's shared-indexer layers reuse `dsa_topk` through a `reads:`
    declaration no edge shows: the call reads it as a declared input, from
    the nearest full layer before it (`reads_from: {nearest: {from:
    config.indexer_types, is: full}}`; full at 0-2 and 6, 10, ..., 74)."""
    analysis, _recorded = walk_of(_program("zai-org/GLM-5.2"))
    graph = analysis.occurrence_graph
    readers = {edge.sink.iteration: edge.source.iteration
               for edge in graph.edges("stream")}
    assert readers[3] == 2 and readers[5] == 2 and readers[7] == 6
    assert readers[77] == 74


@pytest.mark.parametrize("build", ["llava", "eagle"])
def test_a_lowered_join_reads_each_element_in_order(build):
    """A join is an ordinary `concat` whose inputs are single element reads
    (`Program.lowered`): a projector in the vision encoder, checked in its
    encode walk (Llava), and a head stack's reading node (EAGLE-3)."""
    from tests.backend.arch.test_concat_streams import _eagle, _llava
    resolved = (_llava("occ_llava", after="[-2, -5]", k=2, order=True)
                if build == "llava" else _eagle("occ_eagle"))
    program = Program.of(resolved)
    media = ({"pixel_values": {"n_img": 1, "img_h": 2, "img_w": 2}}
             if build == "llava" else None)
    assert check(program, media=media, tp_attention=2, tp_dense_ffn=2) == []
    analysis, _recorded = (
        walk_of(program, "encode", media=media, stack="vis")
        if build == "llava" else walk_of(program))
    assert len(list(analysis.occurrence_graph.edges("join"))) >= 2


def test_a_reference_table_missing_an_input_is_refused():
    """Every input the walk bound has one reference: a table that lost one
    is refused rather than drawn as a graph with one edge fewer, and a
    table that lost a whole node refuses the graph."""
    from backend.arch.eval.occurrence import OccurrenceGraph, ReferenceTable
    analysis, _recorded = walk_of(_program("Qwen/Qwen3-8B"))
    table = analysis.reference_table
    (uid, index) = next(key for key in table.references if key[1] > 0)
    with pytest.raises(ValueError, match="misses inputs"):
        ReferenceTable(program=table.program, arity=table.arity,
                       references={key: found for key, found
                                   in table.references.items()
                                   if key != (uid, index)})
    dropped = ReferenceTable(
        program=table.program,
        arity={u: n for u, n in table.arity.items() if u != uid},
        references={key: found for key, found in table.references.items()
                    if key[0] != uid})
    with pytest.raises(ValueError, match="bound by no walk"):
        OccurrenceGraph.of(dropped)

