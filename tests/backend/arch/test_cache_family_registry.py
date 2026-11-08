"""`cache.family` is a registered vocabulary, and its kind is where the
sliding-window contract reads.

`CacheSpec.family` was free text: a typo was a new family, and the
validator kept its own `{"SWA"}` to know which families are windows.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import backend.arch.cache_families as families
from backend.arch.spec_parsing import _parse_block_graph


class TestTheRegistry(unittest.TestCase):
    def test_the_corpus_families_are_registered_with_kinds(self):
        self.assertEqual(set(families.all_families()),
                         {"MHA", "GQA", "MLA", "SWA", "ViT", "DiT", "GLA-Linear", "GatedDeltaNet", "Mamba2"})
        self.assertEqual(families.of_kind("windowed"), {"SWA"})
        self.assertEqual(families.of_kind("recurrent"), {"GLA-Linear", "GatedDeltaNet", "Mamba2"})

    def test_an_unregistered_family_is_refused_at_load(self):
        with self.assertRaisesRegex(ValueError, "cache_families.yaml"):
            _parse_block_graph({
                "role": "attention",
                "cache": {"family": "SWQ", "entries": []},
                "graph": {"q": {"op": "linear", "in": "x", "dims": {"in": "d_model", "out": "d_model"}}},
                "outputs": ["q"],
            }, identity="typo_family")


if __name__ == "__main__":
    unittest.main()
