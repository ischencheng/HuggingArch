"""A role's attributes live on the role, in `roles.yaml`.

Three private tables restated the vocabulary — the loader's slot filter,
sharding's weight classes, the activation check's FFN-gate roles — and
each had drifted from it. Two are attribute columns now; the slot filter
is gone, because it was the reason `per_role.router` never reached a
router leaf.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import backend.arch.roles as roles
from tests.snapshots import shared_loader
from backend.arch.spec_parsing import _parse_block_graph

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"


class TestAttributesAreColumns(unittest.TestCase):
    def test_weight_classes_and_gate_roles(self):
        self.assertEqual(roles.weight_classes(),
                         {"attention", "attention_swa", "attention_indexer",
                          "attention_linear", "attention_output",
                          "attention_kv_up", "attention_compressor",
                          "attention_position_embedding",
                          "attention_short_conv", "attention_refiner",
                          "ffn_dense", "ffn_moe", "ffn_moe_shared",
                          "ffn_moe_latent", "ffn_refiner"})
        self.assertEqual(roles.ffn_gate_roles(),
                         {"ffn_dense", "ffn_moe", "ffn_moe_shared",
                          "ffn_refiner"})
        for name in roles.weight_classes() | roles.ffn_gate_roles():
            self.assertTrue(roles.is_registered(name))


class TestRefinedRoles(unittest.TestCase):
    def test_a_per_role_rule_reaches_the_roles_refining_it(self):
        """A finer role takes the broader role's rule unless it is named."""
        self.assertEqual(roles.selection_order("attention_output", {}),
                         ("attention_output", "attention"))
        self.assertEqual(
            roles.selection_order("attention_swa", {"attention_swa": "attention"}),
            ("attention_swa", "attention"))
        for name, role in roles.all_roles().items():
            if role.refines:
                self.assertEqual(roles.kind_of(role.refines), role.kind, name)


class TestTheSchemaAsksTheVocabulary(unittest.TestCase):
    def test_an_unregistered_role_is_refused_at_load(self):
        with self.assertRaisesRegex(ValueError, "roles.yaml"):
            _parse_block_graph({"role": "mlp", "graph": {"n": {"op": "rmsnorm", "in": "x", "dim": "d_model"}},
                                "outputs": ["n"]}, identity="bad_role")


class TestRouterIsAContributedRole(unittest.TestCase):
    def test_deepseek_v3_router_leaves_carry_the_router_role(self):
        """DSv3 writes `per_role.router: {method: none}`; the slot filter
        used to drop `router` from the roles a component contributes, so no
        leaf ever matched that row."""
        config = json.loads((SNAPSHOTS / "deepseek-ai__DeepSeek-V3.json").read_text())["config"]
        resolved = shared_loader().match(config, model_id="deepseek-ai/DeepSeek-V3")
        self.assertIn("router", resolved.spec.quant_context.per_role)
        from backend.arch.spec_tree import iter_leaves
        router_leaves = [lt for lt in iter_leaves(resolved) if lt.role == "router"]
        self.assertTrue(router_leaves, "no leaf carries role=router")


if __name__ == "__main__":
    unittest.main()
