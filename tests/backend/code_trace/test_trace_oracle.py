"""Oracle for verifying traces: a memo hit must not hide code.

A sweep cell computed cold — first in a fresh process, every memo it
reaches missing — names the code it ran. The same cell computed again in
that process takes the other routes: with only the walk
registry dropped (`0!`), every memo of ours and of SymPy's is hot; with the
registry kept, the walks themselves are hits (`walk_registry`'s growing
entry). Each route's trace must cover the cold one's; any item missing is
code a hit skipped without naming (a memo outside `backend.code_trace.memo`,
a growing entry not attached, a third-party memo calling back into ours
outside rule 3).

The default case is the cheapest that goes red when a memo hit or a growing
entry stops naming its code (checked by mutation when it was written):
Llama-3.2-3B, one layout, one card. Rule 3's own default guard is
`test_provider`'s SymPy fixture. The regression cases run the expensive
routes: DeepSeek-V3, whose MLA laws reach SymPy's cache — its cold route
runs in a process of its own with SymPy's cache off (`SYMPY_USE_CACHE=no`)
so every call SymPy makes into repo code is recorded, the hit route with it
unbounded — at a few hash
seeds (which queries a walk asks follows set order), and Qwen3-30B-A3B over
several layouts.
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


def _cells(model: str, cells: str, *, provider: str, seed: int, gpus: str,
           sympy_cache: bool = True) -> list:
    env = dict(os.environ, HA_CODE_TRACE=provider, HA_CODE_TRACE_STRICT="1",
               PYTHONHASHSEED=str(seed),
               **({"SYMPY_CACHE_SIZE": "none"} if sympy_cache
                  else {"SYMPY_USE_CACHE": "no"}))
    out = subprocess.run(
        [sys.executable, "-m", "tests.backend.code_trace.sweep_child", model,
         "dp_ep,tp_ep,tp_tp", cells, gpus],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
        check=False)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])["cells"]


def _items(cell: dict) -> set:
    return {tuple(i) for i in cell["trace"]}


@pytest.mark.parametrize("model, cell, warm, gpus, seed, cold_alone", [
    # Cell 0 cold, then again with every memo hot, then off the kept walks:
    # one process, its first cell the cold route.
    ("meta-llama/Llama-3.2-3B", "0", "0,0!,0", "H100-SXM", 1, False),
    ("deepseek-ai/DeepSeek-V3", "10", "10,10!", "H100-SXM,B200", 1, True),
    ("deepseek-ai/DeepSeek-V3", "10", "10,10!", "H100-SXM,B200", 0, True),
    ("deepseek-ai/DeepSeek-V3", "10", "10,10!", "H100-SXM,B200", 2, True),
    ("Qwen/Qwen3-30B-A3B", "8", "0,15,8,8", "H100-SXM,B200", 1, True),
])
def test_a_memo_hit_route_names_all_the_code_the_cold_route_ran(
        model, cell, warm, gpus, seed, cold_alone):
    routes = _cells(model, warm, provider="functions", seed=seed, gpus=gpus)
    first = warm.split(",").index(cell)
    cold = (_cells(model, cell, provider="functions", seed=seed, gpus=gpus,
                   sympy_cache=False)[-1]
            if cold_alone else routes[first])
    for route in routes[first + 1:]:
        hidden = sorted(_items(cold) - _items(route))
        assert hidden == [], (
            f"{route['cell']}: code a memo hit skipped without naming: {hidden}")
        assert route["result"] == cold["result"]


@pytest.mark.regression
def test_both_providers_answer_the_same():
    """`TracedFunctions` changes what is recorded, never what is computed."""
    args = ("Qwen/Qwen3-30B-A3B", "8")
    traced = _cells(*args, provider="functions", seed=1, gpus="H100-SXM,B200")
    plain = _cells(*args, provider="global", seed=1, gpus="H100-SXM,B200")
    assert traced[-1]["result"] == plain[-1]["result"]
