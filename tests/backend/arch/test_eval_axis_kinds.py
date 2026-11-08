"""The axis-kind coverage gate — a new dim symbol is loud, never silent.

`dims.yaml` replaced the policy vocabulary that used to scatter across 42
per-component `shard_symbols` lists. The scatter's failure mode was silence:
a block that carried no list (MiniMax) or half of one (Ling) simply did not
shard, and nothing said so. A registry can fail the same way — by going stale
— so this gate makes staleness fail the suite in both directions:

  * a WEIGHT-AXIS symbol the corpus uses that the registry does not name
    fails the sweep — the author of a new spec is forced to classify the axis
    at birth, which is the entire design;
  * a component `shard_symbols` declaration the registry does not call
    TP-divisible fails the consistency check — two vocabularies drifting
    apart silently is precisely the hole this closes.

The `whole` half of the classification used to live HERE, as a frozenset the
gate subtracted, which made the test a second declaration point: a symbol was
divisible in `transforms.py` or whole in a test file, and the glossary that
explained it was a third. All three are now `dims.yaml`, so this file only
asks whether the registry covers the corpus.
"""
from __future__ import annotations

import glob
import json
import sys
import unittest
import pytest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import dims
from backend.arch.eval.transforms import tp_divides
from tests.snapshots import shared_loader
from backend.arch.spec_tree import iter_leaves

pytestmark = pytest.mark.regression  # walks every snapshot

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"
MODELS = PROJECT_ROOT / "backend" / "arch" / "models"


def _corpus_weight_axes() -> set:
    out: set = set()
    for snap in sorted(SNAPSHOTS.glob("*.json")):
        resolved = shared_loader().match(json.loads(snap.read_text())["config"],
                                         model_id=snap.stem.replace("__", "/", 1))
        for leaf in iter_leaves(resolved):
            out |= set(getattr(leaf, "dim_symbols", ()) or ())
    return out


class TestAxisKindCoverage(unittest.TestCase):
    maxDiff = None

    def test_every_weight_axis_is_classified(self):
        unclassified = _corpus_weight_axes() - set(dims.symbols())
        self.assertEqual(
            sorted(unclassified), [],
            "weight-axis symbols the registry does not name. Classify each "
            "at birth in backend/arch/dims.yaml — a divisible kind, or "
            "`whole` — or its tensors silently never shard.")

    def test_every_bound_dim_is_registered(self):
        """A model may not bind a symbol nothing has classified.

        The weight-axis gate above only sees symbols that reach a tensor
        shape. A `dims:` entry that feeds a FLOP expression or a KV formula
        reaches neither, and it is exactly as unclassified.
        """
        bound: set = set()
        for path in sorted(MODELS.glob("*.yaml")):
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            bound |= set((doc.get("dims") or {}).keys())
        self.assertEqual(
            sorted(bound - set(dims.symbols())), [],
            "model `dims:` blocks binding symbols absent from dims.yaml")

    def test_component_declarations_agree_with_the_registry(self):
        """No component may declare a shard the registry does not grant.

        The corpus declares none today — `shard_symbols` survives in the
        schema and in the spec-author prompt, so this guards what that path
        may still produce rather than anything currently on disk.
        """
        declared: set = set()
        for pattern in ("backend/arch/components/**/*.yaml",
                        "backend/arch/models/*.yaml"):
            for path in glob.glob(str(PROJECT_ROOT / pattern),
                                  recursive=True):
                try:
                    with open(path) as fh:
                        doc = yaml.safe_load(fh)
                except Exception:
                    continue
                stack = [doc]
                while stack:
                    item = stack.pop()
                    if isinstance(item, dict):
                        par = item.get("parallelism")
                        if isinstance(par, dict):
                            declared |= {str(s) for s in
                                         (par.get("shard_symbols") or [])}
                        stack.extend(item.values())
                    elif isinstance(item, list):
                        stack.extend(item)
        contradictions = declared - tp_divides()
        self.assertEqual(
            sorted(contradictions), [],
            "component shard_symbols declarations the registry does not "
            "divide — one of the two vocabularies is wrong, loudly")


class TestEngineSuppliedIsDeclared(unittest.TestCase):
    """Who supplies a symbol's value is a fact about the symbol.

    `reject_divisible_parameter_names` refuses a custom op that names a
    parameter after a symbol some parallelism divides, and exempts the ones
    the engine supplies — a signature writing `[B, s_ctx, In]` names the real
    sequence axis rather than borrowing a name for a per-call width. The
    exemption used to be spelled as membership in two glossary sections, so
    the day an engine-supplied symbol was filed under a new heading the rule
    would have started rejecting it, quietly and for the wrong reason. It
    reads `engine_supplied` now; these hold that line.
    """
    maxDiff = None

    #: What the engine supplies: `B` and `s_ctx` ride in with the analyze
    #: request, the phase ones are bound by `phases.py:phase_binding` once the
    #: phase is known, and `Sar` — how many entries of a `read: all` history
    #: a layer sees — by the derived group of layers that see that many.
    #: Everything else in the registry is bound by a model's `dims:` block or
    #: by a component call site.
    ENGINE_SUPPLIED = frozenset({
        "B", "s_ctx",
        "s_sampled", "s_query", "s_kv", "s_attend", "s_cache", "n_verify",
        # the keys before a step's first query: the phase binds it with
        # s_query and s_kv (`Phase.resolve`), for a selection's pairs law
        "s_past",
        # the step's live (query, key) pairs: the phase binds it
        # (`STEP_LAWS`), and a node that scores a matrix of its own rebinds it
        "s_pairs",
        "Sar",
        # a media input's axes, given by the workload (`Workload.media`) and
        # bound by the encode phase
        "n_img", "img_h", "img_w", "n_crops", "crop_h", "crop_w", "n_aud",
        "s_audio",
        # a diffusion pipeline's: its prompt, the latents the denoise loop
        # starts from, and the latents the decoders read at its end
        "n_prompt", "s_prompt",
        "n_vlat", "vlat_t", "vlat_h", "vlat_w", "n_alat", "alat_t",
        "n_vden", "vden_t", "vden_h", "vden_w", "n_aden", "aden_t",
    })

    @staticmethod
    def _op(shape):
        from backend.arch.schema.custom_ops import CustomOpSpec
        return CustomOpSpec(
            signature={"x": shape}, shape_kind="preserve", couples_on=[],
            description="fixture: exercises the parameter-name rule")

    def test_the_registry_declares_exactly_these(self):
        declared = {n for n, s in dims.symbols().items() if s.engine_supplied}
        self.assertEqual(
            sorted(declared), sorted(self.ENGINE_SUPPLIED),
            "the set of symbols no `dims:` block and no call site binds has "
            "changed. That is a real change to who owns a number, not a "
            "bookkeeping detail — say so here as well as in dims.yaml")

    def test_the_exemption_follows_the_declaration_not_the_section(self):
        """Filed under an ordinary heading, still exempt."""
        from unittest import mock
        from dataclasses import replace
        from backend.arch.ops_base import reject_divisible_parameter_names

        registry = dict(dims.base_symbols())
        registry["s_ctx"] = replace(registry["s_ctx"],
                                    group="per_arch_one_offs")
        with mock.patch.object(dims, "base_symbols", lambda: registry):
            reject_divisible_parameter_names("op", self._op(["B", "s_ctx"]))

    def test_a_section_member_that_does_not_declare_it_is_rejected(self):
        """Filed under the engine's heading, but a parameter — still refused."""
        from unittest import mock
        from dataclasses import replace
        from backend.arch.ops_base import reject_divisible_parameter_names

        registry = dict(dims.base_symbols())
        registry["n_heads"] = replace(registry["n_heads"],
                                      group="engine_dims")
        with mock.patch.object(dims, "base_symbols", lambda: registry), \
                self.assertRaises(ValueError) as caught:
            reject_divisible_parameter_names("op", self._op(["B", "n_heads"]))
        self.assertIn("n_heads", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
