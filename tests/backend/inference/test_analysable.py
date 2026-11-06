"""The analysis replay: which deployments, what a refusal says, the memo,
and the worker grant."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.corpus_pins.corpus import CorpusModel

from backend.arch.eval.refusal import node_context
from backend.arch.validate._analysable import ENGINE_KIND, OPEN_KIND, SPEC_KIND
from backend.inference import analysable
from backend.inference.parallel_config import ParallelConfig


def _resolved(stem):
    model = CorpusModel(stem)
    return model.config, model.resolved()


def _labels(stem):
    return [d.label for d in analysable.product_deployments(_resolved(stem)[1])]


class TestProductDeployments(unittest.TestCase):
    def test_dense(self):
        self.assertEqual(_labels("Qwen__Qwen3-8B"), [
            "single", "tp8", "audit ep8+dp8", "cp8 ulysses", "cp8 pcp",
            "cp8 layersplit"])

    def test_moe_takes_the_top_legal_expert_rung(self):
        self.assertEqual(_labels("deepseek-ai__DeepSeek-V3"), [
            "single", "tp8", "ep8+dp8", "cp8 ulysses", "cp8 pcp",
            "cp8 layersplit"])

    def test_pp_is_not_replayed(self):
        """A spec declares nothing about pipeline parallelism: every input
        the walk binds is resolved per occurrence at pp = 1, and a stage is
        placement arithmetic over that, so a pp failure is the engine's, not
        the spec's. No replayed deployment pipelines."""
        for stem in ("Qwen__Qwen3-8B", "meituan-longcat__LongCat-Flash-Omni",
                     "Qwen__Qwen3.8-Flash-Next"):
            with self.subTest(stem):
                self.assertTrue(all(
                    d.parallel.pp == 1 for d in analysable.product_deployments(
                        _resolved(stem)[1])))

    def test_the_replay_asks_no_framework_binding(self):
        """The replay's estimate resolves no binding: no framework
        legality, fusion rewrite or pipeline wire enters validate."""
        from backend.inference.calibration_overlay import (
            resolve_framework_binding)
        self.assertEqual(resolve_framework_binding(
            framework=analysable.REPLAY_WORKLOAD.framework), ("theory", None))

    def test_a_hybrid_cache_gets_no_layersplit(self):
        labels = _labels("openai__gpt-oss-20b")
        self.assertNotIn("cp8 layersplit", labels)
        self.assertIn("cp8 pcp", labels)


class TestReplayOne(unittest.TestCase):
    DEPLOYMENT = analysable.ReplayDeployment("tp8", ParallelConfig(tp_attention=8))

    def _replay_raising(self, exc):
        def estimate(**kwargs):
            with node_context("lm_head"):
                raise exc
        with mock.patch.object(analysable, "estimate_capacity", estimate):
            return analysable.replay_one({}, None, self.DEPLOYMENT)

    def test_an_untyped_refusal_claims_no_owner_and_names_its_node(self):
        refusal = self._replay_raising(ValueError("gather extent"))
        self.assertEqual((refusal.kind, refusal.node, refusal.deployment),
                         (OPEN_KIND, "lm_head", "tp8"))

    def test_a_not_implemented_refusal_is_the_engines(self):
        def estimate(**kwargs):
            raise NotImplementedError("DCP sparse top-k")
        with mock.patch.object(analysable, "estimate_capacity", estimate):
            refusal = analysable.replay_one({}, None, self.DEPLOYMENT)
        self.assertEqual(refusal.kind, ENGINE_KIND)

    def _replay_returning(self, result):
        with mock.patch.object(analysable, "estimate_capacity",
                               lambda **kwargs: result):
            return analysable.replay_one({}, None, self.DEPLOYMENT)

    @staticmethod
    def _result(row, **phase):
        closed = {"total_bytes": 10, "flops_per_step": 4, "mem_write_bytes": 1}
        return {"phases": {"prefill": {**closed, **phase}, "decode": closed},
                "per_op_timing": [row], "per_op_timing_decode": [
                    {"aggregation_id": "o_proj", "access_bytes_per_rank": 10,
                     "weight_read_bytes_per_rank": 6, "act_in_bytes_per_rank": 3,
                     "act_out_bytes_per_rank": 1, "flops_per_rank": 4}]}

    def test_a_closed_result_is_accepted(self):
        row = {"aggregation_id": "q_proj", "access_bytes_per_rank": 10,
               "weight_read_bytes_per_rank": 5, "metadata_bytes_per_rank": 1,
               "act_in_bytes_per_rank": 3, "act_out_bytes_per_rank": 1,
               "flops_per_rank": 4}
        self.assertIsNone(self._replay_returning(self._result(row)))

    def test_a_returned_calibration_error_is_a_refusal(self):
        # estimate_capacity RETURNS the error rather than raising it; the
        # SOL closure refuses the result, as `measure_case` does.
        refusal = self._replay_returning({"calibration_error": "no fit"})
        self.assertEqual((refusal.kind, refusal.stage, refusal.deployment,
                          refusal.exception),
                         (ENGINE_KIND, "closure", "tp8", "SolClosureError"))
        self.assertTrue(refusal.site.startswith("sol_closure.py:"))
        self.assertIn("no fit", refusal.message)

    def test_a_row_that_does_not_partition_names_its_node(self):
        row = {"aggregation_id": "q_proj", "access_bytes_per_rank": 10,
               "weight_read_bytes_per_rank": 5, "act_in_bytes_per_rank": 3,
               "act_out_bytes_per_rank": 1, "flops_per_rank": 4}
        refusal = self._replay_returning(self._result(row))
        self.assertEqual((refusal.node, refusal.stage), ("q_proj", "closure"))

    def test_rows_that_do_not_sum_to_the_phase_are_a_refusal(self):
        row = {"aggregation_id": "q_proj", "access_bytes_per_rank": 10,
               "weight_read_bytes_per_rank": 6, "act_in_bytes_per_rank": 3,
               "act_out_bytes_per_rank": 1, "flops_per_rank": 4}
        refusal = self._replay_returning(self._result(row, flops_per_step=5))
        self.assertEqual(refusal.stage, "closure")
        self.assertIn("FLOPs closure", refusal.message)

    def test_infrastructure_raises(self):
        def estimate(**kwargs):
            raise OSError("disk")
        with mock.patch.object(analysable, "estimate_capacity", estimate), \
                self.assertRaises(OSError):
            analysable.replay_one({}, None, self.DEPLOYMENT)


class TestReplayFindsSpecRefusals(unittest.TestCase):
    #: `up` -> `square` -> `down` after the final norm, `WIDTH` wide in the
    #: middle; `square` is a `WIDTH -> WIDTH` projection split on its output
    #: only (vLLM ColumnParallelLinear(d, d)).
    CHAIN = (
        "\n  - { name: up, op: linear, dims: { in: d_model, out: WIDTH }, "
        "weight_axes: [{ replicated: true }, { replicated: true }], inputs: [norm] }"
        "\n  - { name: square, op: linear, dims: { in: WIDTH, out: WIDTH }, "
        "weight_axes: [{ world: dense_ffn, parts: WIDTH, fit: exact }, "
        "{ replicated: true }], inputs: [up] }"
        "\n  - { name: down, op: linear, dims: { in: WIDTH, out: d_model }, "
        "weight_axes: [{ replicated: true }, { world: dense_ffn, parts: WIDTH, "
        "fit: exact }], inputs: [square] }")

    def _replay(self, width, dims=""):
        config, resolved = _resolved("Qwen__Qwen3-8B")
        text = (PROJECT_ROOT / "backend" / "arch" / "models" / "qwen3.yaml").read_text()
        norm = ("{ name: norm, op: rmsnorm, dim: d_model, weight_axes: "
                "[{ replicated: true }] , inputs: [layers]}")
        self.assertIn(norm, text)
        text = text.replace(norm, norm + self.CHAIN.replace("WIDTH", width)) \
            .replace("inputs: [norm]}", "inputs: [down]}") \
            .replace("\ndims:\n", "\ndims:\n" + dims, 1)
        text = text.replace("model_type: [qwen3]", "model_id: [Qwen/Qwen3-8B]")
        from backend.arch.loader import SpecLoader, overlay_kwargs
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "models").mkdir()
            (Path(tmp) / "models" / "qwen3_glmsol_case.yaml").write_text(text)
            loader = SpecLoader(**overlay_kwargs(tmp))
            case = loader.resolve(loader.load_model_spec(
                str(Path(tmp) / "models" / "qwen3_glmsol_case.yaml")),
                config, model_id="Qwen/Qwen3-8B")
        self.assertIsNot(case.spec, resolved.spec)
        return analysable.replay(
            config, case, [analysable.ReplayDeployment(
                "tp8", ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                                      tp_moe_expert=8))])

    def test_a_symbol_on_two_differently_sharded_axes_replays(self):
        """`d_ffn` sizes both axes of `square`, only the output split: each
        axis keeps its own extent."""
        self.assertEqual(self._replay("d_ffn"), [])

    def test_exact_parts_that_do_not_divide_the_degree(self):
        """A width of 1001 does not split exactly over eight ranks."""
        (refusal,) = self._replay("d_ffn_shared",
                                  dims='  d_ffn_shared: "1001"\n')
        self.assertEqual(refusal.kind, SPEC_KIND)
        self.assertEqual(refusal.node, "square")
        self.assertEqual(refusal.declaration, "weight_axes")


def _case(stem, model_file, edit, model_id):
    """A library spec with one edit, resolved against its snapshot."""
    from backend.arch.loader import SpecLoader, overlay_kwargs
    config, _ = _resolved(stem)
    text = (PROJECT_ROOT / "backend" / "arch" / "models" / model_file).read_text()
    edited = edit(text)
    assert edited != text
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models" / "glmsol_case.yaml"
        path.parent.mkdir()
        path.write_text(edited)
        loader = SpecLoader(**overlay_kwargs(tmp))
        return config, loader.resolve(loader.load_model_spec(str(path)),
                                      config, model_id=model_id)


class TestPositionTableTriage(unittest.TestCase):
    """Qwen3.8-27B at 8e148681 refused at EP8+DP8 with an invalid collective
    at `model.visual.pos_add`, raised deep in the comm lowering. The cause was
    the spec: its learned position table was an `embedding` lookup, whose
    rows follow the requests a rank owns, where the add reads the table
    whole. The upstream fix declared the table a `parameter_tensor`. A lookup
    reads the ids it names in `inputs:`, so a table written as one with no
    ids is refused where it is loaded, naming the node and the parameter
    form."""

    def test_an_embedding_position_table_is_a_spec_refusal(self):
        # The parameter node is named for its tensor (`.weight`); an
        # `embedding` node is named for its module and composes `.weight`.
        table = ("- {name: visual.pos_embed.weight, op: parameter_tensor,\n"
                 "         shape: [n_vis_pos_slots, d_vis_model],")
        lookup = ("- {name: visual.pos_embed, op: embedding,\n"
                  "         dims: {n_vocab: n_vis_pos_slots, d_model: d_vis_model},")
        reader = "inputs: [visual.patch_embed.proj, visual.pos_embed.weight]"
        with self.assertRaisesRegex(
                Exception, r"'visual\.pos_embed' reads a tensor(.|\n)*"
                           r"Only a zero-input op \(a parameter"):
            _case(
                "Qwen__Qwen3-VL-235B-A22B-Instruct",
                "Qwen__Qwen3-VL-235B-A22B-Instruct.yaml",
                lambda text: text.replace(table, lookup).replace(
                    reader,
                    "inputs: [visual.patch_embed.proj, visual.pos_embed]"),
                "Qwen/Qwen3-VL-235B-A22B-Instruct")


class TestMemoAndGrant(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {
            "HA_VALIDATE_REPLAY_DIR": self.tmp.name,
            "HA_VALIDATE_REPLAY_SLOTS": "4",
            "HA_VALIDATE_REPLAY_WAIT_SECONDS": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_an_unchanged_spec_reads_the_memo(self):
        config, resolved = _resolved("Qwen__Qwen3-8B")
        calls = []
        with mock.patch.object(analysable, "replay",
                               lambda *args: calls.append(1) or []):
            first = analysable.replay_result(config, resolved)
            again = analysable.replay_result(config, resolved)
        self.assertEqual(first, again)
        self.assertEqual(first.report()["deployments"][0],
                         {"label": "single", "parallel": "one GPU", "refusals": []})
        self.assertEqual(len(calls), 1)

    def test_an_unreadable_memo_entry_is_a_miss(self):
        config, resolved = _resolved("Qwen__Qwen3-8B")
        key = analysable.memo_key(config, resolved)
        analysable._memo_path(key).write_text("{\"deployments\": [", encoding="utf-8")
        calls = []
        with mock.patch.object(analysable, "replay",
                               lambda *args: calls.append(1) or []):
            result = analysable.replay_result(config, resolved)
        self.assertEqual(len(calls), 1)
        self.assertEqual(analysable._memo_read(key), result)

    def test_an_entry_whose_code_trace_is_stale_is_a_miss(self):
        # The code is not in the key: an entry names the code it was
        # computed by (its trace), and code that no longer satisfies it
        # replays — under either provider.
        from backend.code_trace.provider import Trace
        config, resolved = _resolved("Qwen__Qwen3-8B")
        key = analysable.memo_key(config, resolved)
        calls = []
        with mock.patch.object(analysable, "replay",
                               lambda *args: calls.append(1) or []):
            result = analysable.replay_result(config, resolved)
            analysable._memo_write(key, result, Trace((("<engine>", "stale"),
                                                       ("<import-time>", "stale"))))
            analysable.replay_result(config, resolved)
        self.assertEqual(len(calls), 2)

    def test_any_input_changes_the_key(self):
        config, resolved = _resolved("Qwen__Qwen3-8B")
        base = analysable.memo_key(config, resolved)
        self.assertNotEqual(base, analysable.memo_key(
            {**config, "num_hidden_layers": 1}, resolved))
        with mock.patch.object(analysable, "data_version", lambda r: "other"):
            self.assertNotEqual(base, analysable.memo_key(config, resolved))
        patch = Path(self.tmp.name) / "proposal" / "patch.py"
        patch.parent.mkdir()
        patch.write_text("OP_REGISTRATIONS = ''\n")
        with mock.patch.dict(os.environ, {"HA_PENDING_OP_PATCH": str(patch)}):
            self.assertNotEqual(base, analysable.memo_key(config, resolved))

    def test_a_full_host_waits_then_times_out_as_infrastructure(self):
        with analysable.worker_grant(4) as held:
            self.assertEqual(held, 4)
            with self.assertRaises(analysable.ReplayCapacityTimeout), \
                    analysable.worker_grant(4):
                pass
        with analysable.worker_grant(8) as held:
            self.assertEqual(held, 4)


if __name__ == "__main__":
    unittest.main()
