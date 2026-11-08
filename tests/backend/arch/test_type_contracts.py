"""Unit tests for type-contract checks (validate/_type_contracts.py).

Field-consistency contracts (kv_label / kv_cached / kv_growth storage axiom)
on small in-memory specs. Source ⟺ structure equivalence is the agent's
judgment, not a tool check — there is no source-keyword layer here.
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock as mock
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.loader import ResolvedSpec
from backend.arch.schema import (
    ArchSpec,
    BlockGraph,
    CacheSpec,
    GraphNode,
    ModelLayer,
)

pytestmark = pytest.mark.regression  # walks every snapshot


from backend.arch.validate._type_contracts import (
    check_field_contracts,
)


def _attn(role="attention", family=None, entries=None, **kw):
    cache = (CacheSpec(family=family, entries=entries or [])
             if (family is not None or entries is not None) else None)
    return BlockGraph(
        inputs=["x"],
        graph={"wq": GraphNode(op="linear", inputs=["x"], dims={"in": "d_model", "out": "d_model"})},
        outputs=["wq"],
        role=role,
        cache=cache,
        **kw,
    )


def _make_resolved(components):
    graph = {}
    prev = "x"
    for i, name in enumerate(components):
        nid = f"c{i}"
        graph[nid] = GraphNode(op=name, inputs=[prev])
        prev = nid
    block = BlockGraph(inputs=["x"], graph=graph, outputs=[prev])
    spec = ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"}},
        name="demo",
        dims={"d_model": "config.hidden_size", "n_layers": "1"},
        model=[
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "d_model", "d_model": "d_model"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["embed"]),
        ],
        blocks={"block": block},
        components=dict(components),
    )
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 1}, {}, {})


def _make_resolved_with_encoder(decoder_components, encoder_components):
    """`_make_resolved` plus a vision encoder group (`aux_kind: vision`, no
    `inputs:`) whose block invokes `encoder_components`, its output merged
    into the prompt's image positions."""
    def _chain(names, prefix):
        graph, prev = {}, "x"
        for i, name in enumerate(names):
            nid = f"{prefix}{i}"
            graph[nid] = GraphNode(op=name, inputs=[prev])
            prev = nid
        return BlockGraph(inputs=["x"], graph=graph, outputs=[prev])

    spec = ArchSpec(
        model_inputs={"input_ids": {"kind": "tokens"},
                      "pixel_values": {"kind": "image",
                                       "axes": ["n_img", "img_h", "img_w"]}},
        name="demo",
        dims={"d_model": "config.hidden_size", "n_layers": "1"},
        model=[
            ModelLayer(name="vision", group=True, aux_kind="vision", layers=[
                ModelLayer(name="vision.patchify", op="patchify",
                           inputs=["pixel_values"],
                           dims={"patch": "1", "temporal": "1",
                                 "channels": "d_model", "align": "1"},
                           segment="item",
                           source_anchor="a test processor"),
                ModelLayer(name="tower", template="vblock",
                           inputs=["vision.patchify"]),
            ]),
            ModelLayer(name="embed", op="embedding", inputs=["input_ids"],
                       boundary_role="input_embedding",
                       dims={"n_vocab": "d_model", "d_model": "d_model"}),
            ModelLayer(name="merge", op="scatter_rows_replace",
                       inputs=["embed", "tower"],
                       prompt_slots={"domain": "pixel_values",
                                     "defines": True,
                                     "source_anchor": "a test merge"}),
            ModelLayer(name="layers", template="block", repeat="n_layers",
                       inputs=["merge"]),
        ],
        blocks={"block": _chain(list(decoder_components), "c"),
                "vblock": _chain(list(encoder_components), "v")},
        components={**decoder_components, **encoder_components},
    )
    return ResolvedSpec(spec, {"d_model": 16, "n_layers": 1}, {}, {})


def _kinds(warnings):
    return {w.kind for w in warnings}


class TestFieldContracts(unittest.TestCase):
    def test_full_gqa_is_clean(self):
        comps = {"gqa": _attn(family="GQA", entries=[{"node": "wq", "kind": "kv"}])}
        self.assertEqual(check_field_contracts(_make_resolved(comps)), [])

    def test_missing_kv_label(self):
        comps = {"a": _attn(entries=[{"node": "wq", "kind": "kv"}])}  # no cache family
        self.assertIn("attention_missing_kv_label",
                      _kinds(check_field_contracts(_make_resolved(comps))))

    def test_missing_kv_cached(self):
        comps = {"a": _attn(family="GQA")}  # no cache entries
        self.assertIn("attention_missing_kv_cached",
                      _kinds(check_field_contracts(_make_resolved(comps))))

    def test_encoder_attention_without_cache_is_clean(self):
        # A ViT attention the vision encoder alone reaches caches nothing:
        # the library `vit_attention` form (`family: ViT, entries: []`).
        comps = {"gqa": _attn(family="GQA", entries=[{"node": "wq", "kind": "kv"}])}
        vit = {"vit_attn": _attn(family="ViT", entries=[])}
        self.assertEqual(
            check_field_contracts(_make_resolved_with_encoder(comps, vit)), [])

    def test_decoder_attention_without_cache_still_fails(self):
        # The same empty cache on a decode-path attention is still refused,
        # whatever family it declares — beside an encoder or not.
        comps = {"a": _attn(family="GQA", entries=[])}
        vit = {"vit_attn": _attn(family="ViT", entries=[])}
        findings = check_field_contracts(_make_resolved_with_encoder(comps, vit))
        self.assertEqual(
            [(f.kind, f.template) for f in findings
             if f.kind == "attention_missing_kv_cached"],
            [("attention_missing_kv_cached", "a")])
        vit_on_decode = {"vit_attn": _attn(family="ViT", entries=[])}
        self.assertIn("attention_missing_kv_cached", _kinds(
            check_field_contracts(_make_resolved(vit_on_decode))))

    def test_a_family_that_keeps_no_cache_owes_no_entries(self):
        comps = {"dit": _attn(family="DiT", entries=[])}
        self.assertNotIn("attention_missing_kv_cached", _kinds(
            check_field_contracts(_make_resolved(comps))))

    def test_non_attention_ignored(self):
        comps = {"moe": _attn(role="ffn_moe")}
        # ffn_moe isn't an attention component — no kv_* contracts apply.
        self.assertEqual(check_field_contracts(_make_resolved(comps)), [])


class TestSwaWindowGates(unittest.TestCase):
    """End-to-end: KV/attention type contracts gate `valid`.

    Linear/SSM components avoid a false positive by declaring their fixed
    recurrent state; transparent attention adapters inherit the nested
    component's cache contract.
    """

    def _find_swa_snapshot(self):
        import glob
        import json
        from tests.snapshots import shared_loader
        loader = shared_loader()
        for sp in sorted(glob.glob(str(PROJECT_ROOT / "backend/arch/snapshots/*.json"))):
            with open(sp) as snapshot_file:
                cfg = json.load(snapshot_file)
            r = loader.match(cfg["config"], model_id=cfg["model_id"])
            if r and any(getattr(bg, "kv_label", None) == "SWA"
                         for bg in (r.spec.components or {}).values()):
                return cfg["model_id"]
        return None

    def test_windowless_swa_flips_valid_false(self):
        mid = self._find_swa_snapshot()
        from backend.arch.validate import _axioms
        from backend.arch.validate._orchestrator import validate

        # Baseline: the real spec (window intact) is not failed by our contract.
        base = validate(mid)
        self.assertFalse(any("sliding-window" in e for e in (base.get("errors") or [])))

        # Every cached stream grows without a cap — an author who labels SWA
        # but never windows the cache in the graph (the 128TB blowup).
        with mock.patch.object(_axioms, "_storage_is_bounded", return_value=False):
            broken = validate(mid)
        self.assertFalse(broken["valid"])
        self.assertTrue(any("sliding-window" in e for e in broken["errors"]),
                        f"windowless SWA did not gate valid for {mid}")


if __name__ == "__main__":
    unittest.main()
