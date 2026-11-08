"""What an op is, asked of its registration.

`OpSpec` carries the author's facts — stored tensors, arity, whether the
math is a GEMM, which half of an attention kernel it is — and consumers
used to keep their own lists of op names that restated those facts and
drifted (`_GEMM_OPS` in two files with different members, a bias table
that forgot `conv1d`, a zero-input check that knew one op). These pin the
projections the lists were replaced with.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import backend.arch.ops_registry  # noqa: F401  (populates the registry)
from backend.arch.ops import (OP_REGISTRY, OpSpec, TensorDecl, is_gemm_op,
                              is_identity_source, is_tensor_source)
from backend.inference.op_signatures import op_bindings


class TestGemmIsAnAuthorFact(unittest.TestCase):
    def test_the_linear_family_is_declared_gemm(self):
        for op in ("linear", "linear_param", "linear_external_weight", "per_head_linear"):
            self.assertTrue(is_gemm_op(op), op)

    def test_every_registered_op_measured_as_gemm_is_declared_gemm(self):
        """The kernel side's own map names which ops run the GEMM
        signature; each of those that is a registered primitive must say
        so on its registration. A subset, not an equality: `per_head_linear`
        is a GEMM in math and `per_head_gemm` in measurement."""
        measured = {op for op, sig in op_bindings().items()
                    if sig == "gemm" and op in OP_REGISTRY}
        self.assertTrue(measured)
        self.assertEqual(sorted(measured - {o for o in OP_REGISTRY if is_gemm_op(o)}), [])


class TestTensorSources(unittest.TestCase):
    def test_a_tensor_source_is_an_identity_source_or_a_buffer(self):
        identity = {o for o in OP_REGISTRY if is_identity_source(o)}
        sources = {o for o in OP_REGISTRY if is_tensor_source(o)}
        self.assertTrue(identity <= sources)
        self.assertEqual(sorted(sources - identity),
                         ["filter_buffer", "rotary_inv_freq_buffer"])


class TestStoredTensorsAreOneDeclaration(unittest.TestCase):
    def test_per_head_linear_declares_the_bias_its_params_count(self):
        decl = OP_REGISTRY["per_head_linear"].bias_decl()
        self.assertIsNotNone(decl)
        self.assertEqual((decl.when, tuple(decl.shape)), ("bias", ("n_heads", "out")))

    def test_layernorm_bias_is_on_unless_switched_off_and_rmsnorm_has_none(self):
        """nn.LayerNorm stores beta unless built with bias=False (MiMo-V2.6's
        merger ln_q checkpoint holds only its weight)."""
        bias = OP_REGISTRY["layernorm"].bias_decl()
        self.assertEqual((bias.flag({}), bias.flag({"bias": 0})), (1, 0))
        self.assertIsNone(OP_REGISTRY["rmsnorm"].bias_decl())

    def test_a_weight_shape_that_disagrees_with_its_tensor_is_refused(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            OpSpec("bad_op", flops_fn=lambda d: 0, params_fn=lambda d: 0,
                   shape_in=("B", "s_ctx", "in"), shape_out=("B", "s_ctx", "out"),
                   weight_shape=("out", "in"),
                   tensors=(TensorDecl(".weight", ("in", "out")),))

    def test_every_registered_primary_tensor_matches_weight_shape(self):
        for name, spec in OP_REGISTRY.items():
            if spec.tensors and spec.weight_shape:
                self.assertEqual(tuple(spec.tensors[0].shape), tuple(spec.weight_shape), name)


class TestAttentionKernelHalves(unittest.TestCase):
    def test_three_score_apply_pairs_are_declared(self):
        score = {o for o, s in OP_REGISTRY.items() if s.attention_kernel == "score"}
        apply = {o for o, s in OP_REGISTRY.items() if s.attention_kernel == "apply"}
        self.assertEqual(score, {"attention_score", "swa_attention_score", "dsa_attention_score"})
        self.assertEqual(apply, {"attention_apply", "swa_attention_apply", "dsa_attention_apply"})


if __name__ == "__main__":
    unittest.main()
