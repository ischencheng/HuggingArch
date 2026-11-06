"""Oracles for each cache layer a verifying trace passes through.

A layer is sound when a value served from it names all the code that
computing the value cold would run. Each case computes one thing two ways,
in fresh processes that share only the caches on disk:

  * the result cache — a Context × Batch sweep narrowed to one point, cold
    through the warm job's entry, and in a new process answered from the
    disk tier that run filled (`measure.cache.KeyedCache`: the hit merges
    the entry's trace). The same sweep entered through the HTTP route keeps
    the same trace: what a computation names does not depend on who asked;
  * the compute pool — an engine function here, and in spawned workers
    (`measure.pool.run_parallel`: each worker's trace travels back);
  * the SOL pool (regression) — one record walked as one of a class's
    layouts after the ones before it warmed every memo, against the same
    layout walked alone in a process that did nothing else.

The cold route's items must all be on the cached route, except what the
cache's own read and write path runs after a value is final — measured, not
listed: the trace of a throwaway value put and read back through the same
cache. Results must match too.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODEL = "meta-llama/Llama-3.2-3B"
GPU = "H100-SXM"


def _child(tmp: Path, *args: str, pool: bool = False, sweep_dir: str = "sweep",
           pool_dir: str = "sol-pool") -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update(HA_CODE_TRACE="functions", HA_CODE_TRACE_STRICT="1",
               HA_SWEEP_CACHE_DIR=str(tmp / sweep_dir),
               HA_SOL_POOL_DIR=str(tmp / pool_dir),
               HA_GALLERY_CACHE_DIR=str(tmp / "gallery"))
    if pool:
        env.update(HA_SWEEP_POOL_WORKERS="2")
    else:
        env.update(HUGGINGARCH_NO_SWEEP_POOL="1")
    out = subprocess.run(
        [sys.executable, "-m", "tests.backend.code_trace.cache_child", *args],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=3600,
        check=False)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _items(trace) -> set:
    return {tuple(item) for item in trace}


@pytest.fixture(scope="module")
def point(tmp_path_factory):
    """The one-point sweep: cold through the warm entry (filling the cache),
    then a hit off that cache, then cold through the route elsewhere."""
    tmp = tmp_path_factory.mktemp("point")
    cold = _child(tmp, "point", MODEL, "warm", sweep_dir="kept")
    hit = _child(tmp, "point", MODEL, "warm", sweep_dir="kept")
    route = _child(tmp, "point", MODEL, "route", sweep_dir="route")
    return cold, hit, route


def test_a_cached_answer_names_all_the_code_its_cold_computation_ran(point):
    cold, hit, _route = point
    assert hit["result"] == cold["result"]
    hidden = sorted(_items(cold["trace"]) - _items(hit["trace"])
                    - _items(hit["plumbing"]))
    assert hidden == [], f"code a cache hit skipped without naming: {hidden}"


def test_a_computation_names_the_same_code_whoever_asks(point):
    """The warm job and the HTTP route reach the sweep through different
    frames; the answer each keeps names one trace — the computation's, not
    the path to it."""
    cold, _hit, route = point
    assert route["result"] == cold["result"]
    assert route["stored"] == cold["stored"] and len(cold["stored"]) == 1


def test_a_pool_job_names_its_code_to_the_computation_that_asked(tmp_path):
    here = _child(tmp_path, "pool", "here")
    pooled = _child(tmp_path, "pool", "pooled", pool=True)
    assert pooled["result"] == here["result"]
    hidden = sorted(_items(here["trace"]) - _items(pooled["trace"]))
    assert hidden == [], f"code a worker ran that its requester lost: {hidden}"


@pytest.mark.regression
def test_a_sol_record_names_all_the_code_its_layout_ran_alone(tmp_path):
    kept = _child(tmp_path, "keep-walk", MODEL, GPU)
    records = kept["records"]
    assert records and all(t is not None for t in records.values())
    last = str(len(records) - 1)    # walked after every other layout
    alone = _child(tmp_path, "layout-law", json.dumps(kept["items"][last]),
                   pool_dir="alone")["trace"]
    hidden = sorted(_items(alone) - _items(records[last]))
    assert hidden == [], f"code a record does not name: {hidden}"
