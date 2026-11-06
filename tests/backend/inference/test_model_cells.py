"""A request reads the cells an earlier request of the same model,
deployment and precision walked (`capacity._model_cells`): another workload
is another point of the same walk, and the estimate there is the one a
fresh process computes."""
from __future__ import annotations

from dataclasses import replace

from backend.arch.eval import transforms
from backend.inference import capacity, walk_registry
from backend.inference.parallel_config import ParallelConfig
from tests.sweep_parity import Deployment, diff, report

LLAMA = "meta-llama/Llama-3.2-3B"
TP2 = ParallelConfig(tp_attention=2, tp_dense_ffn=2)


def test_another_workload_reads_the_walked_cells(monkeypatch):
    dep = Deployment(LLAMA, parallel=TP2)
    walk_registry.WALK_CELLS.clear()
    dep.direct(avg_input_tokens=4096, prefill_batch=1, decode_batch=1)

    walks = []
    real = transforms.Deployment.analyze
    monkeypatch.setattr(transforms.Deployment, "analyze",
                        lambda self, *a, **k: walks.append(self) or real(self, *a, **k))
    later = replace(dep, knobs={"s_prefix_cached": 3072}).direct(
        avg_input_tokens=8192, prefill_batch=4, decode_batch=8)
    assert walks == [], "a request at another workload walked again"
    monkeypatch.undo()

    walk_registry.WALK_CELLS.clear()
    fresh = replace(dep, knobs={"s_prefix_cached": 3072}).direct(
        avg_input_tokens=8192, prefill_batch=4, decode_batch=8)
    diffs = diff(later, fresh)
    assert not diffs, report(diffs)


def test_another_card_reads_the_cells_budget_basis(monkeypatch):
    """A preparation on another card reads what the first preparation of
    the same cells built off them alone (`capacity.CellBasis`): the budget
    walk and the cache views are not built again, and the estimate there is
    the one a fresh process computes."""
    dep = Deployment(LLAMA, parallel=TP2)
    walk_registry.WALK_CELLS.clear()
    dep.direct(avg_input_tokens=4096)

    built = []
    real_walk = capacity._capacity_walk
    monkeypatch.setattr(capacity, "_capacity_walk",
                        lambda *a, **k: built.append("walk") or real_walk(*a, **k))
    later = dep.on("B200").direct(avg_input_tokens=4096)
    assert built == [], "another card walked the budget again"
    monkeypatch.undo()

    walk_registry.WALK_CELLS.clear()
    fresh = dep.on("B200").direct(avg_input_tokens=4096)
    diffs = diff(later, fresh)
    assert not diffs, report(diffs)


def test_another_deployment_walks_its_own_cells(monkeypatch):
    dep = Deployment(LLAMA, parallel=TP2)
    walk_registry.WALK_CELLS.clear()
    dep.direct(avg_input_tokens=1024)
    walks = []
    real = transforms.Deployment.analyze
    monkeypatch.setattr(transforms.Deployment, "analyze",
                        lambda self, *a, **k: walks.append(self) or real(self, *a, **k))
    replace(dep, parallel=ParallelConfig()).direct(avg_input_tokens=1024)
    assert walks, "another deployment read the first deployment's cells"


def test_a_refined_spec_is_other_cells():
    """A refine that changes the spec's content is another key: the cells
    of the old spec are not read for the new one."""
    from tests.snapshots import resolve_snapshot

    resolved = resolve_snapshot(LLAMA)
    precision = capacity.PrecisionProfile.from_request(None)
    walk_registry.WALK_CELLS.clear()
    cells = capacity._model_cells(resolved, TP2, precision, model_id=LLAMA).cells
    again = capacity._model_cells(resolved, TP2, precision, model_id=LLAMA).cells
    assert again is cells
    refined = resolved.with_spec(resolved.spec.model_copy(
        update={"unresolved_note": "refined"}))
    other = capacity._model_cells(refined, TP2, precision, model_id=LLAMA).cells
    assert other is not cells


def test_every_thread_reads_the_same_cells():
    """The key names what the cells are of, not who reads them: a request on
    another thread reads the cells this one walked."""
    import threading

    from tests.snapshots import resolve_snapshot

    resolved = resolve_snapshot(LLAMA)
    precision = capacity.PrecisionProfile.from_request(None)
    walk_registry.WALK_CELLS.clear()
    here = capacity._model_cells(resolved, TP2, precision, model_id=LLAMA).cells
    there = []
    thread = threading.Thread(target=lambda: there.append(capacity._model_cells(
        resolved, TP2, precision, model_id=LLAMA).cells))
    thread.start()
    thread.join()
    assert there == [here]


def test_what_a_cell_holds_is_bounded():
    """A cell keeps the `READINGS_KEPT` readings of each kind most recently
    asked, and a registry the `CELLS_KEPT` cells: asked for more, the least
    recently read goes, and is built again if asked again."""
    from collections import OrderedDict

    from backend.inference import step_eval

    table = OrderedDict()
    for i in range(step_eval.READINGS_KEPT + 5):
        step_eval._kept(table, i, lambda i=i: i)
    assert len(table) == step_eval.READINGS_KEPT
    assert step_eval._kept(table, step_eval.READINGS_KEPT + 4, lambda: -1) == (
        step_eval.READINGS_KEPT + 4)
    assert step_eval._kept(table, 0, lambda: "again") == "again"

    cells = step_eval.StepCells()
    spec = object()
    first = cells.cell(spec, TP2, phase="prefill")
    for i in range(step_eval.CELLS_KEPT):
        cells.cell(spec, ParallelConfig(tp_attention=2 ** (i + 2)), phase="prefill")
    assert len(cells._cells) == step_eval.CELLS_KEPT
    assert cells.cell(spec, TP2, phase="prefill") is not first


def test_liveness_and_rows_write_one_book():
    """A cell's activation liveness reads the serving layers its rows are
    read from and writes its laws into the same book: a law both state is
    one law of the cell."""
    dep = Deployment(LLAMA, parallel=TP2)
    walk_registry.WALK_CELLS.clear()
    dep.direct(avg_input_tokens=1024)
    (entry,) = _entries()
    cells = entry.cells
    (prefill,) = [c for c in cells._cells if c.phase == "prefill"]
    (liveness,) = prefill._liveness.values()
    assert liveness._book is prefill.laws
    assert all(serving.book is prefill.laws
               for serving in prefill._serving.values())


def _entries():
    # Each entry is stored beside its trace's id (`measure.cache.KeyedCache`).
    return [stored["value"]
            for stored in walk_registry.WALK_CELLS._backend._store.values()]


def test_a_step_that_speculates_is_another_walk():
    """Whether a decode step speculates is a fact of its walk: a step with
    no speculative verify walks its verify width as the number 1, a step
    that verifies K + 1 tokens walks it as the symbol, and the two never
    share a cell (`Workload.walked_as_number`, `StepCells.cell`)."""
    import sympy as sp

    from backend.arch.eval.dim import value_of
    from backend.inference.step_eval import StepCells, evaluate_step
    from tests.snapshots import shared_loader
    import json
    import pathlib

    name = "meta-llama__Llama-3.2-3B"
    snap = pathlib.Path(__file__).resolve().parents[3] / "backend" / "arch" \
        / "snapshots" / f"{name}.json"
    config = json.loads(snap.read_text())["config"]
    spec = shared_loader().match(config, model_id=name.replace("__", "/", 1))
    cells = StepCells()

    def step(s_verify):
        return evaluate_step(spec, None, phase="decode", batch=2,
                             input_tokens=1024, output_tokens=64,
                             s_verify=s_verify, cells=cells)

    plain, spec_step = step(1), step(4)
    assert plain.cell is not spec_step.cell
    assert step(1).cell is plain.cell
    assert value_of(plain.walk.binding["n_verify"]) == 1
    verify = value_of(spec_step.walk.binding["n_verify"])
    assert isinstance(verify, sp.Basic) and verify.free_symbols
    assert plain.point["s_verify"] == 1


def test_one_predicate_says_whether_a_step_speculates():
    """`Workload.walked_as_number` is the one statement: the number 1 is a
    step that does not speculate; a wider width, a bool, a SymPy number and
    a symbol are not. The cell key is its negation, and a cell refuses a
    first step whose width is not the width its key claims."""
    import sympy as sp
    import pytest
    from backend.arch.eval.transforms import Workload
    from backend.inference.step_eval import Cell, evaluate_step
    from tests.snapshots import shared_loader
    import json
    import pathlib

    assert Workload.walked_as_number("s_verify", 1)
    for value in (2, True, sp.Integer(1),
                  sp.Symbol("s_verify", integer=True, positive=True)):
        assert not Workload.walked_as_number("s_verify", value), value
    assert not Workload.walked_as_number("batch", 1)

    name = "meta-llama__Llama-3.2-3B"
    snap = pathlib.Path(__file__).resolve().parents[3] / "backend" / "arch" \
        / "snapshots" / f"{name}.json"
    spec = shared_loader().match(json.loads(snap.read_text())["config"],
                                 model_id=name.replace("__", "/", 1))
    step = evaluate_step(spec, None, phase="decode", batch=1,
                         input_tokens=256, output_tokens=8, s_verify=1)
    assert step.cell.speculative is False
    wrong = Cell(spec=step.cell.spec, parallel=step.cell.parallel,
                 phase="decode", speculative=True)
    with pytest.raises(ValueError, match="verify width 1"):
        wrong.walk(step)


def test_a_folded_verify_width_is_not_moved():
    """Rows of a walk that took the verify width as the number 1 are not
    read at a wider verify: that point is another walk, and reading it off
    these laws would price speculation as plain decode."""
    import pytest
    from backend.arch.eval.analysis import Point
    from backend.inference.batch_sweep import rows_at
    from backend.inference.comm_cost import Fabric
    from backend.inference.pipeline import analyze
    from tests.snapshots import shared_loader
    import json
    import pathlib

    name = "meta-llama__Llama-3.2-3B"
    snap = pathlib.Path(__file__).resolve().parents[3] / "backend" / "arch" \
        / "snapshots" / f"{name}.json"
    spec = shared_loader().match(json.loads(snap.read_text())["config"],
                                 model_id=name.replace("__", "/", 1))
    result = analyze(spec, parallel=None, batch_size=2, avg_input_tokens=256,
                     avg_output_tokens=8, phase="decode", breakdown=False)
    rows = result.per_op_timing
    fabric = Fabric(None, None, None)
    rows_at(rows, Point({"B": 4}), cell=result, peak_flops=None,
            peak_bw=None, fabric=fabric, breakdown=False)
    with pytest.raises(ValueError, match="another walk"):
        rows_at(rows, Point({"s_verify": 4}), cell=result, peak_flops=None,
                peak_bw=None, fabric=fabric, breakdown=False)
