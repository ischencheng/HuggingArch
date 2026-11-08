"""The role-classification gate — a new role is loud, never silent.

`roles.yaml` replaced four name-keyed tables that had to agree by hand:
`_ATTN_ROLES` / `_FFN_ROLES` / `_WRAPPER_ROLES` in
`eval/projection/result.py` and the `startswith("attention_")` test in
`inference/sharding.py`. They had drifted — `attention_swa` passed the prefix
test and failed the frozenset — and the failure was silent in the worst
direction: a frozenset that does not contain a name reports zero, and zero is
a number the rest of the pipeline happily carries.

So this gate asks staleness to fail the suite from both ends: a role the
corpus uses that the registry does not classify, and a hand-kept table naming
a role the registry never heard of.
"""
from __future__ import annotations

import sys
import unittest
import pytest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import roles
from backend.arch.loader import SpecLoader
from backend.inference.sharding import is_attention_role

pytestmark = pytest.mark.regression  # walks every snapshot

MODELS = PROJECT_ROOT / "backend" / "arch" / "models"


class TestRoleRegistryCoverage(unittest.TestCase):
    maxDiff = None

    def test_every_corpus_role_is_classified(self):
        """Declared component roles AND model slot names, both.

        `spec.component_roles` carries two vocabularies — a library
        component's own `role:` and the slot name a model's `components:`
        map binds it to — and every consumer downstream reads them through
        the same key, so both must be classified.
        """
        loader = SpecLoader()
        used = {r for spec in loader.all_specs().values()
                for r in (spec.component_roles or {}).values() if r}
        self.assertEqual(
            sorted(used - set(roles.all_roles())), [],
            "roles the corpus uses that roles.yaml does not classify")

    def test_every_declared_component_role_is_classified(self):
        """Including components no model binds today.

        The corpus check above only reaches components some spec resolves.
        A draft or adapters-tier component declaring an unclassified role is
        a landmine for whoever binds it next.
        """
        declared = set()
        for path in (PROJECT_ROOT / "backend" / "arch"
                     / "components").rglob("*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            stack = [doc]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    if isinstance(item.get("role"), str):
                        declared.add(item["role"])
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
        self.assertEqual(
            sorted(declared - set(roles.all_roles())), [],
            "component `role:` declarations roles.yaml does not classify")

    def test_every_model_slot_name_is_classified(self):
        """A `components:` slot key is a role name too."""
        slots = set()
        for path in sorted(MODELS.glob("*.yaml")):
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            slots |= set((doc.get("components") or {}).keys())
        # `norm` and the attention/ffn slots are roles; anything else a model
        # invents must be classified before it can reach component_roles.
        self.assertEqual(
            sorted(slots - set(roles.all_roles())), [],
            "model `components:` slot names roles.yaml does not classify")

    def test_weight_classes_are_an_attribute_of_the_registry(self):
        """`weight_class` answers a genuinely different question from
        `roles.of_kind("ffn")` — `router` is feed-forward work whose gate
        weight is replicated, so it is not a weight class. It used to be a
        literal in `sharding.py`; now it is a column of the same yaml, so
        it cannot name a role the registry does not have."""
        self.assertEqual(roles.weight_classes(),
                         {"attention", "attention_swa", "attention_indexer",
                          "attention_linear", "attention_output",
                          "attention_kv_up", "attention_compressor",
                          "attention_position_embedding",
                          "attention_short_conv", "attention_refiner",
                          "ffn_refiner",
                          "ffn_dense", "ffn_moe", "ffn_moe_shared",
                          "ffn_moe_latent"})
        self.assertNotIn("router", roles.weight_classes())
        self.assertIn("router", roles.of_kind("ffn"))

    def test_attention_family_is_the_registry_not_a_prefix(self):
        attn = roles.of_kind("attention")
        self.assertIn("attention", attn)
        self.assertIn("attention_swa", attn)
        for name in attn:
            self.assertTrue(is_attention_role(name))
        # A plausible-looking name nobody declared is not attention by
        # spelling — it is a load error, which is where it belongs.
        self.assertFalse(is_attention_role("attention_linear_v2"))
        self.assertFalse(is_attention_role(None))
        self.assertFalse(is_attention_role(""))

    def test_kinds_partition_the_registry(self):
        by_kind = {k: roles.of_kind(k) for k in roles.KINDS}
        union = set().union(*by_kind.values())
        self.assertEqual(union, set(roles.all_roles()))
        seen = set()
        for kind, members in by_kind.items():
            overlap = seen & members
            self.assertEqual(overlap, set(),
                             f"{kind} shares roles with an earlier kind")
            seen |= members


if __name__ == "__main__":
    unittest.main()
