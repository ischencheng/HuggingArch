"""A sweep of many cells is drawn as its cells land."""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from backend.api import offload, sweep_runtime


@dataclasses.dataclass(frozen=True)
class _Plan:
    cells: tuple


def _sweep():
    return dataclasses.replace(
        sweep_runtime._PARALLEL,
        cells=lambda plan: list(plan.cells),
        assemble=lambda plan, results: {
            "cells": list(plan.cells), "results": list(results)})


def _start(plan, reused=None):
    return sweep_runtime._SweepStart(
        None, None, plan, "digest", False, None, reused or {})


def test_a_partial_answer_is_the_plan_narrowed_to_the_cells_it_has():
    """The answer of some cells is the whole sweep's assembly over just
    those cells, in plan order — the absent ones are absent, not blank."""
    plan = _Plan(cells=("c0", "c1", "c2", "c3"))
    got = sweep_runtime._partial_answer(
        _sweep(), _start(plan), {3: "r3", 1: "r1"})
    assert got == {"cells": ["c1", "c3"], "results": ["r1", "r3"]}


def test_only_a_sweep_of_many_cells_assembles_partially():
    assert sweep_runtime._PARALLEL.assembles_partially
    assert not sweep_runtime._ESTIMATE.assembles_partially


def test_the_published_partial_reaches_a_waiting_client(monkeypatch):
    """While cells land, the publisher puts the assembly of those finished
    in the work's partial head, which the 503 carries."""
    sweep = _sweep()
    plan = _Plan(cells=("c0", "c1", "c2"))
    start = _start(plan)
    monkeypatch.setattr(sweep_runtime, "_PARTIAL_EVERY_S", 0.01)

    async def run():
        have: dict = {}
        moved = asyncio.Event()
        publisher = asyncio.ensure_future(
            sweep_runtime._publish_partials(sweep, start, have, moved))
        try:
            for i in (2, 0):
                have[i] = f"r{i}"
                moved.set()
                await asyncio.sleep(0.05)
            await asyncio.sleep(1.0)
        finally:
            publisher.cancel()
        return "done"

    async def scenario():
        with pytest.raises(offload.StillComputing) as info:
            await offload.within_deadline("sweep-partial", run)
        return info.value.partial

    monkeypatch.setattr(offload, "_deadline_s", 0.3)
    partial = asyncio.run(scenario())
    assert partial["head"] == {"cells": ["c0", "c2"], "results": ["r0", "r2"]}
