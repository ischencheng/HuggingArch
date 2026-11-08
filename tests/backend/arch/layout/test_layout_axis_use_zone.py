"""Gates for phase 1b: the tagged axis_use registry must agree with
`couples_on` exactly (custom ops included — the gate-equals-implementation
discipline), and every dispatch graph in the corpus must satisfy the three
routed-zone rules that replaced the route state machine. Synthetic
violations prove each rule can actually fire, and the full classification
is pinned verbatim: a drift is a design change and fails here first."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.ops import OP_REGISTRY
from backend.arch.layout import axis_use as au
from backend.arch.layout.zone import (
    graph_boundary, iter_component_graphs, validate_dispatch_region)
from tools.corpus_pins.checks.route_zone import graph_violations
from tools.corpus_pins.pins.route_components import ROUTE_COMPONENTS

ARCH_ROOT = PROJECT_ROOT / "backend" / "arch"


def _custom_declarations():
    """(source, name) -> (CustomOpSpec, OpSpec) for every custom op the YAML
    corpus declares. Parse failures raise: a gate that skips is no gate."""
    from backend.arch.schema.custom_ops import CustomOpSpec
    found = {}
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(ARCH_ROOT.rglob(pattern)):
            doc = yaml.safe_load(path.read_text())
            if not isinstance(doc, dict):
                continue
            for name, raw in (doc.get("custom_ops") or {}).items():
                cop = CustomOpSpec(**raw)
                found[(str(path.relative_to(ARCH_ROOT)), name)] = (cop, cop)
    return found

#: The route-component census, pinned as RELATIVE PATHS — a bare count
#: lets an addition and a removal cancel; same-named files in different
#: directories must not collapse. Pin P6, `tests/golden/pins/route_components.json`.
ROUTE_COMPONENT_FILES = set(ROUTE_COMPONENTS.load())

#: (op, axis) -> (category, param, plan), pinned verbatim.
SNAPSHOT = {'attention_apply': {'s_kv': ('contract',
                              'online_attention',
                              'attention_kernel'),
                     's_query': ('separable', '', 'attention_kernel')},
 'attention_score': {'d_head': ('contract', 'sum', ''),
                     's_kv': ('separable', '', 'attention_kernel'),
                     's_query': ('separable', '', 'attention_kernel')},
 'attn_res_combine': {'Sar': ('normalize', 'softmax', ''),
                      'd_model': ('contract', 'sum', '')},
 'combine': {'n_expert_copies': ('route', '', ''), 'M': ('route', '', '')},
 'conv1d': {'C_in': ('contract', 'sum', ''),
            's_ctx': ('window', 'kernel_size-1 halo', '')},
 'conv2d': {'C_in': ('contract', 'sum', ''),
            's_ctx': ('window', 'kernel window halo', '')},
 'conv1d_weight_norm': {'C_in': ('contract', 'sum', ''),
                        's_ctx': ('window', 'kernel_size-1 halo', '')},
 'conv_transpose1d': {'C_in': ('contract', 'sum', ''),
                      's_ctx': ('window', 'kernel_size taps scattered per input row', '')},
 'conv_transpose1d_weight_norm': {'C_in': ('contract', 'sum', ''),
                                  's_ctx': ('window', 'kernel_size taps scattered per input row', '')},
 'dsa_attention_apply': {'s_kv': ('contract',
                                  'online_attention',
                                  'attention_kernel'),
                         's_query': ('separable', '', 'attention_kernel')},
 'dsa_attention_score': {'d_head': ('contract', 'sum', ''),
                         's_kv': ('separable', '', 'attention_kernel'),
                         's_query': ('separable', '', 'attention_kernel')},
 'dsa_softmax': {'s_kv': ('normalize', 'softmax', 'attention_kernel')},
 'dsa_topk_mask': {'s_kv': ('normalize', 'topk', 'attention_kernel')},
 'embedding': {'n_vocab': ('contract', 'sum', '')},
 'gated_delta_rule': {'d_lin_k_head': ('recurrent', 'state', ''),
                      's_ctx': ('recurrent', 'd_lin_k_head', '')},
 'gather': {'in': ('route', '', '')},
 'group_max': {'in': ('contract', 'group_max', '')},
 'group_mask_apply': {'in': ('route', '', ''),
                      'n_groups_active': ('route', '', '')},
 'group_top2_sum': {'in': ('contract', 'top2_sum', '')},
 'hash_embedding': {'n_vocab': ('contract', 'sum', '')},
 'hash_route': {'n_vocab': ('route', '', '')},
 'hc_mix': {'n_hidden_copies': ('contract', 'sum', '')},
 'hc_split_sinkhorn': {'d_hc_mix': ('normalize', 'sinkhorn', '')},
 'indexer_logits': {'d_indexer_head': ('contract', 'sum', ''),
                    'n_indexer_heads': ('contract', 'sum', ''),
                    's_kv': ('separable', '', 'attention_kernel'),
                    's_query': ('separable', '', 'attention_kernel')},
 'l2norm': {'dim': ('normalize', 'l2', '')},
 'hadamard_transform': {'dim': ('normalize', 'hadamard', '')},
 'layernorm': {'dim': ('normalize', 'mean_var', '')},
 'linear': {'in': ('contract', 'sum', '')},
 'linear_external_weight': {'in': ('contract', 'sum', '')},
 'linear_param': {'in': ('contract', 'sum', '')},
 'mamba2_scan': {'d_ssm_state': ('recurrent', 'state', ''),
                 's_ctx': ('recurrent', 'd_ssm_state', '')},
 'mrope': {'dim': ('window', 'pairing dim/2', '')},
 'normalize_sum_one': {'in': ('normalize', 'sum', '')},
 'per_head_linear': {'in': ('contract', 'sum', '')},
 'permute': {'B': ('route', '', ''), 's_ctx': ('route', '', '')},
 'quantise_activation': {'dim': ('normalize', 'amax', '')},
 'rmsnorm': {'dim': ('normalize', 'rms', '')},
 'rmsnorm_unscaled': {'dim': ('normalize', 'rms', '')},
 'rope': {'dim': ('window', 'pairing dim/2', '')},
 'rope_2d': {'dim': ('window', 'pairing dim/2', '')},
 'rope_3d': {'dim': ('window', 'pairing dim/2', '')},
 'rope_emb': {'dim': ('window', 'pairing dim/2', '')},
 'rope_emb_inverse': {'dim': ('window', 'pairing dim/2', '')},
 'xdrope': {'dim': ('window', 'pairing dim/2', '')},
 'kv_cache_view': {'s_in': ('window', 's_kv', '')},
 'sliding_window_kv': {'s_in': ('window', 's_window', '')},
 'softmax': {'s_kv': ('normalize', 'softmax', 'attention_kernel')},
 'softmax_act': {'dim': ('normalize', 'softmax', '')},
 'softmax_with_sink': {'s_kv': ('normalize',
                                 'softmax',
                                 'attention_kernel')},
 'sum_reduce': {'in': ('contract', 'sum', '')},
 'swa_attention_apply': {'s_attend': ('contract',
                                      'online_attention',
                                      'attention_kernel'),
                         's_kv': ('contract',
                                  'online_attention',
                                  'attention_kernel'),
                         's_query': ('separable', '', 'attention_kernel')},
 'swa_attention_score': {'d_head': ('contract', 'sum', ''),
                         's_kv': ('window', 's_window', ''),
                         's_query': ('separable', '', 'attention_kernel')},
 'swa_softmax': {'s_attend': ('normalize', 'softmax', 'attention_kernel')},
 'swa_softmax_key0_bias': {'s_attend': ('normalize',
                                    'softmax',
                                    'attention_kernel')},
 'swa_softmax_with_sink': {'s_attend': ('normalize',
                                    'softmax',
                                    'attention_kernel')},
 'topk': {'in': ('normalize', 'topk', '')}}

#: Same pin for the spec-scoped custom ops.
CUSTOM_SNAPSHOT = {'gla_simple_recurrence': {'d_head': ('recurrent', 'state', ''),
                           's_ctx': ('recurrent', 'd_head', '')},
 'mxfp4_fused_linear': {'In': ('contract', 'sum', '')},
 'packed_expert_linear': {'In': ('contract', 'sum', '')}}


class TestAxisUseGate(unittest.TestCase):
    def test_registry_matches_couples_on_exactly(self):
        self.assertEqual(au.registry_violations(OP_REGISTRY), [])

    def test_every_custom_op_is_declared_and_tagged(self):
        # Scan the YAML source of truth — BOTH extensions, parse failures
        # RAISE (a gate that skips its inputs is no gate), duplicates
        # across sources are detected by (source, op). The tags come from
        # the same declaration (`axis_use`, derived for the linear family),
        # so the only thing left to check is that they cover `couples_on`
        # exactly and that every declaration parses.
        found = _custom_declarations()
        names = [name for _, name in found]
        self.assertEqual(len(names), len(set(names)),
                         "same custom op declared in two sources — the "
                         "gate must see every declaration separately")
        self.assertGreaterEqual(len(found), 3)
        for (src, name), (cop, spec) in sorted(found.items()):
            tagged = au.custom_axis_use(spec)
            self.assertTrue(tagged or not cop.couples_on,
                            f"custom op {name} in {src} couples but tags nothing")
            self.assertEqual(set(tagged), set(cop.couples_on),
                             f"custom op {name}: tags vs couples_on differ")

    def test_categories_are_closed_and_parameterized(self):
        custom = {name: au.custom_axis_use(spec)
                  for (_, name), (_, spec) in _custom_declarations().items()}
        for table in (au.AXIS_USE, custom):
            for op, axes in table.items():
                for axis, use in axes.items():
                    self.assertIn(use.category, au.CATEGORIES)
                    if use.category in (au.CONTRACT, au.NORMALIZE,
                                        au.WINDOW):
                        self.assertTrue(use.param, f"{op}.{axis}")

    def test_explicit_separable_carries_its_justification(self):
        for op, axes in au.AXIS_USE.items():
            for axis, use in axes.items():
                if use.category == au.SEPARABLE:
                    self.assertTrue(use.note, f"{op}.{axis}: bare separable")
                    self.assertTrue(use.plan, f"{op}.{axis}: a separable "
                                    "coupling must name its owning plan")

    def test_produces_partial_is_structural(self):
        # The generic sharded-contract -> Partial rule is answered by the
        # type, not by prose: exotic statistics and plan-owned axes say no.
        self.assertTrue(au.AXIS_USE["linear"]["in"].admits_deferred_reduction)
        self.assertTrue(au.AXIS_USE["embedding"]["n_vocab"].admits_deferred_reduction)
        self.assertFalse(
            au.AXIS_USE["group_top2_sum"]["in"].admits_deferred_reduction)
        self.assertFalse(
            au.AXIS_USE["attention_apply"]["s_kv"].admits_deferred_reduction)
        self.assertFalse(au.AXIS_USE["topk"]["in"].admits_deferred_reduction)
        self.assertFalse(
            au.AXIS_USE["swa_attention_apply"]["s_kv"].admits_deferred_reduction)
        # normalize axes whose online form exists carry the owning plan —
        # machine-readable eligibility, not prose.
        for op in ("softmax", "softmax_with_sink", "dsa_softmax", "swa_softmax",
                   "swa_softmax_with_sink", "swa_softmax_key0_bias"):
            axis = next(iter(au.AXIS_USE[op]))
            self.assertEqual(au.AXIS_USE[op][axis].plan, "attention_kernel",
                             op)

    def test_geometry_exemption_is_exact(self):
        for op in au.GEOMETRY_OPS:
            self.assertIn(op, OP_REGISTRY)
            self.assertNotIn(op, au.AXIS_USE)

    def test_gate_rejects_empty_tag_and_geometry_drift(self):
        class FakeSpec:
            layout_family = None

            def __init__(self, couples):
                self.couples_on = couples
        fake = {op: FakeSpec(tuple(au.AXIS_USE[op])) for op in au.AXIS_USE}
        fake["silu"] = FakeSpec(())
        au.AXIS_USE["silu"] = {}
        try:
            self.assertTrue(any("silu" in v
                                for v in au.registry_violations(fake)))
        finally:
            del au.AXIS_USE["silu"]
        fake2 = dict(fake)
        fake2["concat"] = FakeSpec(("in_a", "in_b", "smuggled"))
        self.assertTrue(any("geometry couples_on drifted" in v
                            for v in au.registry_violations(fake2)))

    def test_classification_snapshot(self):
        custom = {name: au.custom_axis_use(spec)
                  for (_, name), (_, spec) in _custom_declarations().items()}
        for table, snap in ((au.AXIS_USE, SNAPSHOT),
                            (custom, CUSTOM_SNAPSHOT)):
            actual = {op: {ax: (u.category, u.param, u.plan)
                           for ax, u in axes.items()}
                      for op, axes in table.items()}
            self.assertEqual(actual, snap)


class TestRoutedZoneCorpus(unittest.TestCase):
    def test_every_dispatch_graph_in_corpus_is_zone_clean(self):
        violations = graph_violations(iter_component_graphs(ARCH_ROOT))
        self.assertEqual(violations, [])
        self.assertEqual(set(ROUTE_COMPONENTS.measure_all()), ROUTE_COMPONENT_FILES)


class TestRoutedZoneRulesFire(unittest.TestCase):
    def _clean(self):
        return {
            "gate": {"op": "router_softmax_topk", "in": "x"},
            "permute": {"op": "permute", "in": ["x", "gate"]},
            "experts": {"op": "swiglu_ffn", "in": "permute"},
            "combine": {"op": "combine", "in": ["experts", "gate"]},
            "shared": {"op": "swiglu_ffn", "in": "x"},
            "moe_out": {"op": "add", "in": ["combine", "shared"]},
        }

    def test_clean_graph_passes(self):
        self.assertEqual(validate_dispatch_region(self._clean(), ("moe_out",)),
                         [])

    def test_r1_mismatched_routing_operand_fires(self):
        g = self._clean()
        g["gate2"] = {"op": "router_softmax_topk", "in": "x"}
        g["combine"] = {"op": "combine", "in": ["experts", "gate2"]}
        self.assertTrue(any("same routing SSA operand" in v
                            for v in validate_dispatch_region(g, ())))

    def test_r1_extra_or_smuggled_operand_fires(self):
        g = self._clean()
        g["combine"] = {"op": "combine", "in": ["experts", "gate",
                                                "experts"]}
        self.assertTrue(any("exactly" in v
                            for v in validate_dispatch_region(g, ())))
        g2 = self._clean()
        g2["combine"] = {"op": "combine", "in": ["experts", "permute"]}
        self.assertTrue(any("same routing SSA operand" in v
                            for v in validate_dispatch_region(g2, ())))

    def test_unrelated_combine_is_not_a_pair(self):
        g = self._clean()
        g["combine"] = {"op": "combine", "in": ["shared", "gate"]}
        self.assertTrue(any("does not bracket" in v
                            for v in validate_dispatch_region(g, ())))

    def test_r2_outside_token_tensor_into_zone_fires(self):
        g = self._clean()
        g["experts"] = {"op": "swiglu_ffn", "in": ["permute", "x"]}
        self.assertTrue(any("outside the routed zone" in v
                            for v in validate_dispatch_region(g, ())))

    def test_r2_name_disguise_does_not_pass(self):
        g = self._clean()
        g["x.weight"] = {"op": "add", "in": "x"}
        g["experts"] = {"op": "swiglu_ffn", "in": ["permute", "x.weight"]}
        self.assertTrue(any("outside the routed zone" in v
                            for v in validate_dispatch_region(g, ())))

    def test_r2_parameters_are_welcome_inside(self):
        g = self._clean()
        g["experts.bias"] = {"op": "parameter_vector", "dim": "d"}
        g["experts"] = {"op": "swiglu_ffn",
                        "in": ["permute", "experts.bias"]}
        self.assertEqual(validate_dispatch_region(g, ("moe_out",)), [])

    def test_escape_by_mixing_is_r2(self):
        g = self._clean()
        g["leak"] = {"op": "add", "in": ["experts", "x"]}
        self.assertTrue(any("outside the routed zone" in v
                            for v in validate_dispatch_region(g, ())))

    def test_escape_by_flow_through_taints_the_boundary(self):
        g = self._clean()
        g["moe_out"] = {"op": "add", "in": ["experts", "shared"]}
        self.assertTrue(any("must exit through combine" in v
                            for v in validate_dispatch_region(g, ("moe_out",))))

    def test_dispatch_output_is_itself_routed(self):
        # The dispatch's own output IS routed rows: exposing it (or one of
        # its ports) as any boundary tensor escapes combine.
        g = self._clean()
        for boundary in (("permute",), ("permute.some_port",)):
            r = validate_dispatch_region(g, boundary)
            self.assertTrue(any("must exit through combine" in v for v in r),
                            boundary)

    def test_boundary_maps_and_cache_entries_are_producer_refs(self):
        # stream_out / secondary_outputs are alias -> ref maps and cache
        # entries are persistent endpoints: a zone tensor in any of them
        # escapes.
        g = self._clean()
        doc = {"graph": g, "outputs": ["moe_out"],
               "stream_out": {"alias": "experts"},
               "cache": {"entries": ["moe_out", {"node": "experts"}]}}
        boundary = graph_boundary(doc)
        self.assertIn("experts", boundary)
        r = validate_dispatch_region(g, boundary)
        self.assertTrue(any("must exit through combine" in v for v in r))

    def test_dead_routed_branch_fires(self):
        g = self._clean()
        g["dead"] = {"op": "silu", "in": "experts"}
        self.assertTrue(any("never reaches combine" in v
                            for v in validate_dispatch_region(g, ("moe_out",))))

    def test_port_references_resolve_to_their_producer(self):
        g = self._clean()
        g["tap"] = {"op": "add", "in": "experts.partial_port"}
        r = validate_dispatch_region(g, ("tap",))
        self.assertTrue(any("must exit through combine" in v for v in r), r)

    def test_nested_pairs_are_loud_not_guessed(self):
        g = self._clean()
        g["permute2"] = {"op": "permute", "in": ["combine", "gate"]}
        g["combine2"] = {"op": "combine", "in": ["permute2", "gate"]}
        r = validate_dispatch_region(g, ())
        self.assertEqual(len(r), 1)
        self.assertIn("nested routing", r[0])


if __name__ == "__main__":
    unittest.main()
