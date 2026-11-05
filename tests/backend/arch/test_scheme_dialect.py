"""The quant-scheme YAML dialect: `when:` predicates and cardinality formulas."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.scheme_dialect import (
    eval_formula,
    eval_predicate,
)


class TestPredicateEval(unittest.TestCase):

    def test_simple_equality(self):
        self.assertTrue(eval_predicate("symmetric == false", {"symmetric": False}))
        self.assertFalse(eval_predicate("symmetric == false", {"symmetric": True}))

    def test_string_comparison(self):
        self.assertTrue(eval_predicate("type == 'int'", {"type": "int"}))
        self.assertFalse(eval_predicate("type == 'int'", {"type": "float"}))

    def test_compound_and(self):
        bindings = {"type": "int", "num_bits": 4}
        self.assertTrue(eval_predicate("type == 'int' and num_bits == 4", bindings))
        self.assertFalse(eval_predicate("type == 'int' and num_bits == 8", bindings))

    def test_compound_or(self):
        bindings = {"num_bits": 4}
        self.assertTrue(eval_predicate(
            "num_bits == 4 or num_bits == 8", bindings,
        ))
        self.assertFalse(eval_predicate(
            "num_bits == 16 or num_bits == 32", bindings,
        ))

    def test_negation(self):
        self.assertTrue(eval_predicate("not (num_bits == 8)", {"num_bits": 4}))

    def test_true_false_keywords(self):
        self.assertTrue(eval_predicate("true", {}))
        self.assertFalse(eval_predicate("false", {}))

    def test_unbound_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "unbound name"):
            eval_predicate("foo == 1", {})

    def test_unsupported_node_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            eval_predicate("len(x) == 1", {"x": [1, 2]})


class TestFormulaEval(unittest.TestCase):

    def test_simple_product(self):
        self.assertEqual(
            eval_formula("in * out", {"in": 7168, "out": 3072}),
            22020096,
        )

    def test_division_by_group_size(self):
        self.assertEqual(
            eval_formula(
                "(in * out) / group_size",
                {"in": 7168, "out": 3072, "group_size": 32},
            ),
            22020096 // 32,
        )

    def test_per_tile_2d(self):
        self.assertEqual(
            eval_formula(
                "(in / block_h) * (out / block_w)",
                {"in": 7168, "out": 7168, "block_h": 128, "block_w": 128},
            ),
            56 * 56,
        )

    def test_constant_one(self):
        self.assertEqual(eval_formula("1", {}), 1)

    def test_unbound_symbol_rejected(self):
        with self.assertRaisesRegex(ValueError, "unbound symbol"):
            eval_formula("in * mystery", {"in": 4})

    def test_floor_div_works(self):
        self.assertEqual(eval_formula("in // 2", {"in": 7}), 3)

    def test_true_division_refuses_a_fractional_axis(self):
        with self.assertRaisesRegex(ValueError, "did not evaluate to an integer"):
            eval_formula("in / group_size", {"in": 33, "group_size": 32})

    def test_true_division_is_exact_above_float_precision(self):
        self.assertEqual(
            eval_formula(
                "(in * out) / group_size",
                {"in": 9007199254740993, "out": 32, "group_size": 32},
            ),
            9007199254740993,
        )


class TestEveryFormatHasALabelOfItsOwn(unittest.TestCase):
    """A quantized tensor's dtype is its FORMAT, never its container.

    MXFP4 ships E2M1 pairs in a `U8` and its UE8M0 exponent in another
    `U8`, because safetensors has a header spelling for neither. Labelling
    such a tensor by the slot says `U8`, which is the packaging — and the
    measurement corpus has no token for a container at all, so every
    gpt-oss expert GEMM was declined for a dtype environment nothing could
    spell.
    """

    def test_a_numeric_format_with_no_label_is_not_its_container(self):
        from backend.arch.eval.scheme_dialect import logical_dtype_label
        self.assertEqual(logical_dtype_label("mx_e2m1", 4, "U8"), "MXFP4")
        self.assertEqual(logical_dtype_label("fp_e2m1", 4, "U8"), "FP4")
        # Unquantized: no format was selected, so the storage dtype IS the
        # dtype and that is the one case it may be returned.
        self.assertEqual(logical_dtype_label(None, 16, "BF16"), "BF16")
        with self.assertRaisesRegex(ValueError, "no dtype label"):
            logical_dtype_label("fp_e3m2", 6, "U8")

    def test_mx_and_nv_fp4_are_two_labels(self):
        # Same element type, different format: NVFP4's scale is an FP8
        # value per 16 and MXFP4's a power of two per 32, so a weight
        # placed on one grid is not on the other and a kernel written for
        # one declines the other. One token each, so they cannot share a
        # measurement — and each scheme says which it is rather than the
        # distinction being inferred from the scale it happens to carry.
        from backend.arch.dtypes import corpus_token
        from backend.arch.eval.scheme_dialect import (
            logical_dtype_label, select_numeric_format,
        )
        from backend.arch.quant_schemes import scheme_named
        got = {}
        for name in ("mxfp4", "mxfp4-pack-quantized", "mxfp4-dot-suffix",
                     "nvfp4", "fp4"):
            scheme = scheme_named(name)
            fmt = select_numeric_format(scheme, dict(scheme.deployment_fields))
            got[name] = corpus_token(logical_dtype_label(fmt, 4, "U8"))
        self.assertEqual(got["mxfp4"], "mxfp4")
        self.assertEqual(got["mxfp4-pack-quantized"], "mxfp4")
        # The third MX layout: DeepSeek's `.weight` / `.scale` convention,
        # which is `fp4`'s layout exactly and MX's numeric format. The
        # generic `fp4` beside it keeps `fp4_e2m1`, because whether a
        # dot-suffix FP4 deployment implements MX is the model's claim to
        # make and not this scheme's.
        self.assertEqual(got["mxfp4-dot-suffix"], "mxfp4")
        self.assertEqual(got["nvfp4"], "fp4_e2m1")
        self.assertEqual(got["fp4"], "fp4_e2m1")

    def test_the_mx_variant_changes_no_byte_of_its_layout(self):
        # The DeepSeek MX scheme is `fp4`'s layout with MX's numeric
        # format, and that is the whole difference: same storage dtype and
        # packing (so the same weight bytes), same scale dtype and shape
        # (so the same metadata bytes), same activation reference. If those
        # ever diverge, the claim "only the identity moved" stops holding.
        from backend.arch.quant_schemes import scheme_named
        mx, fp4 = scheme_named("mxfp4-dot-suffix"), scheme_named("fp4")
        for name in ("format_variants", "scale_strategy",
                     "metadata_siblings", "activation", "required_fields",
                     "optional_fields", "default_fields", "valid_status"):
            self.assertEqual(getattr(mx, name), getattr(fp4, name), name)
        self.assertNotEqual(mx.numeric_format, fp4.numeric_format)

    def test_every_scheme_scale_resolves_to_a_corpus_dtype(self):
        # The scale's dtype travels into the measurement identity
        # (`access._scale_dtype` → the row's `metadata_dtype` → the
        # corpus key), so a scheme whose scale resolves to something the
        # corpus cannot spell silently declines every cell that uses it.
        # Asked of every scheme, so a new one cannot land without a
        # spelling.
        from backend.arch.dtypes import UnknownDtype, corpus_token
        from backend.arch.quant_schemes import QUANT_SCHEMES
        unspellable = {}
        for scheme in QUANT_SCHEMES.values():
            for sib in scheme.metadata_siblings:
                if sib.kind != "scale":
                    continue
                dtype = sib.logical_dtype or sib.dtype
                try:
                    corpus_token(str(dtype))
                except UnknownDtype:
                    unspellable[scheme.name] = (sib.suffix, sib.dtype,
                                                sib.logical_dtype)
        self.assertEqual(unspellable, {})


if __name__ == "__main__":
    unittest.main()
