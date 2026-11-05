"""`SpecIRNode.bound` — the frame a node sits in, read before the walk —
gives what the walk binds there.

Two routes to one fact: the tree builder's frame stack (`bound`) and the
evaluator's scopes (`Analysis.node_dims`). Every head-count label an
attention kernel reads is checked on every node the walk ran. The first
case, GLM-5.3-Flash, has an indexer score whose own `dims:` rebinds
`n_heads` to `n_indexer_heads`: the case where the node's own binding must
win over its frame.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import dims as dims_registry
from backend.arch.axis_propagation import signature_for
from backend.arch.eval.dim import value_of
from backend.arch.eval.transforms import Deployment
from backend.arch.ops import OpSpec
from tests.snapshots import shared_program, shared_walk
from tools.corpus_pins.corpus import Corpus

FIRST = "zai-org__GLM-5.3-Flash"
STEMS = [FIRST, *(s for s in Corpus.stems() if s != FIRST)]


def _head_labels(node):
    spec = node.declaration
    if not isinstance(spec, OpSpec) or spec.attention_kernel is None:
        return ()
    signature = signature_for(spec, node, len(spec.shape_out or ()))
    if signature is None:
        return ()
    labels = set()
    for template in signature.ins:
        for label in template:
            symbol = dims_registry.get(label)
            if symbol is not None and symbol.kind == "head_count":
                labels.add(label)
    return tuple(sorted(labels))


@pytest.mark.parametrize("stem", STEMS)
def test_bound_is_what_the_walk_binds(stem):
    program = shared_program(stem)
    analysis = shared_walk(
        Deployment(program).at(batch=1, input_tokens=4))
    checked = 0
    for node in analysis.program.nodes():
        walked = analysis.node_dims.values.get(node.uid)
        if walked is None or not node.frame:
            continue
        for label in _head_labels(node):
            if label not in walked:
                continue
            assert int(value_of(node.bound(label))) == int(
                value_of(walked[label])), (node.display_path, label)
            checked += 1
    if stem == FIRST:
        assert checked > 0
