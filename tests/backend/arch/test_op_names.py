"""Every op name a module matches against must be an op that exists.

A `frozenset` of op names is a branch condition written as data, and it fails
in the one way data cannot report: a name nobody registered matches nothing,
forever, while reading exactly like a live rule. Four sets were in that state
when this file was written, and each missing name marked a real gap rather
than a
harmless typo:

  * `validate._activation._ACT_OPS` named `swiglu` / `geglu` / `gelu_tanh`,
    which are FFN compounds and not activations, and did not name
    `relu_squared`, which is one — so NemotronH's squared-ReLU FFN was
    invisible to the activation check.
  * `structural.spec_graph._TENSOR_MATH_OPS` named `activation`. Activations
    are registered individually, so the list covered none of them and every
    activation node read to the structural comparator as a named submodule
    call in the forward.
  * the former fusion implementation named `view` and `reshape`, which
    do not exist, alongside `transpose`, which the `OpSpec.is_view_op` branch
    immediately below already handled with identical code.
  * `axis_propagation._RANK_STRICT` named `matmul`. There is no generic
    contraction op.

The check needs no list of its own: `op_names()` / `op_keyed()` register what
they build, so importing the modules is what populates it.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.op_names import _DECLARED, universe, verify_all


def _import_backend() -> None:
    """Import every backend module, so every declaration has run.

    Modules that fail to import on their own (optional deps, side effects)
    are skipped rather than failing this test — a set they declare is then
    simply not checked, which is the same coverage the old hand-written check
    had and no worse.
    """
    import backend

    for info in pkgutil.walk_packages(backend.__path__, prefix="backend."):
        try:
            importlib.import_module(info.name)
        except Exception:
            continue


class TestDeclaredOpNames(unittest.TestCase):
    def test_no_set_names_an_op_that_does_not_exist(self):
        _import_backend()
        self.assertGreater(len(_DECLARED), 0, "nothing registered — did "
                                              "`op_names` stop being used?")
        self.assertEqual(verify_all(), {})

    def test_the_universe_is_the_two_things_an_op_ref_can_be(self):
        """A spec node's `op` resolves to a registered primitive or a declared
        component, and nothing else. If this ever shrinks to one of the two,
        every compound name in every set starts reading as a name nobody
        registered."""
        from backend.arch.ops import OP_REGISTRY

        known = universe()
        self.assertTrue(set(OP_REGISTRY) <= known)
        self.assertIn("swiglu_ffn", known, "component names must be in scope")
        self.assertIn("linear", known, "primitive names must be in scope")

    def test_the_view_only_components_are_named(self):
        """`view_only_components()` is derived off the same library scan, and a
        derived set fails silently in one more way than a hand-kept one: the
        scan stops matching and the answer is empty rather than wrong. It read
        a `name:` key for a while after that key was retired, so it was empty,
        and the two byte-walkers that ask it stopped crossing a head reshape
        with nothing said. Name what it must contain: the three head reshapes,
        CSA2's three slot fillers that hand on what an earlier layer
        published (a shared latent, its compressed rows, a shared top-k), and
        the rotation slot that rotates nothing."""
        from backend.arch.op_names import view_only_components

        self.assertEqual(view_only_components(),
                         frozenset({"head_split", "head_merge",
                                    "head_broadcast", "csa2_shared_latent",
                                    "csa2_compress_borrowed",
                                    "csa2_shared_selection",
                                    "identity_rope"}))

    def test_is_view_name_reads_the_registry_not_a_second_table(self):
        from backend.arch.op_names import is_view_name

        self.assertTrue(is_view_name("transpose"))
        self.assertTrue(is_view_name("view_split"))
        self.assertTrue(is_view_name("head_merge"))
        self.assertFalse(is_view_name("concat"))
        self.assertFalse(is_view_name("linear"))


if __name__ == "__main__":
    unittest.main()
