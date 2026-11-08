import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx2  # the hub's HTTP client: what model_info raises


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from huggingface_hub import errors as hf_errors
from huggingface_hub.hf_api import ModelInfo

from backend.hf import task_gate


# Model-info responses recorded from the hub (date in the file), trimmed to
# the fields the gate requests: pipeline_tag, library_name,
# transformersInfo, config (architectures / model_type / auto_map), siblings.
# `corpus` holds every curated snapshot id; `outside` holds repos the gate
# must refuse (with their file listings); `own_forward` holds repos admitted
# only for the forward source they ship; `pipelines` holds diffusers
# pipeline repos, admitted by their `model_index.json`.
_RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "model_info_task_gate.json").read_text()
)
_RECORDS = {**_RECORDED["corpus"], **_RECORDED["outside"],
            **_RECORDED["own_forward"], **_RECORDED["pipelines"]}


def _hub_task(model_id):
    return task_gate.HubTask.from_model_info(ModelInfo(**_RECORDS[model_id]))


class TestPrettifyTask(unittest.TestCase):
    """The refusal quotes the task in HF's own model-page display form —
    the label the user just saw on huggingface.co."""

    def test_hf_display_form(self):
        self.assertEqual(
            task_gate.prettify_task("image-text-to-video"),
            "Image-Text-to-Video",
        )
        self.assertEqual(
            task_gate.prettify_task("image-text-to-text"),
            "Image-Text-to-Text",
        )
        self.assertEqual(
            task_gate.prettify_task("text-generation"),
            "Text-Generation",
        )
        self.assertEqual(task_gate.prettify_task("any-to-any"), "Any-to-Any")


class TestRecordedHubAnswers(unittest.TestCase):
    """The gate decides from how Hugging Face resolves the checkpoint, not
    from the author-set page task."""

    def setUp(self):
        task_gate._hub_model_info.cache_clear()
        self.addCleanup(task_gate._hub_model_info.cache_clear)
        no_snapshot = mock.patch(
            "backend.arch.snapshots.load_config", return_value=None
        )
        no_snapshot.start()
        self.addCleanup(no_snapshot.stop)

    def _refusal(self, model_id):
        info = ModelInfo(**_RECORDS[model_id])
        with mock.patch("huggingface_hub.HfApi.model_info", return_value=info):
            return task_gate.unsupported_task_refusal(model_id)

    def test_every_curated_model_passes_on_its_hub_answer_alone(self):
        # The curated-snapshot skip is a round-trip saving, not a bypass:
        # 52 resolve to a supported task, 5 register AutoModelForCausalLM
        # for their remote code (ChatGLM x2, Kimi x2, LongCat-Thinking),
        # 1 has no Transformers resolution but a causal-LM config
        # (Mistral-Nemo).
        self.assertEqual(len(_RECORDED["corpus"]), 58)
        refused = {m: msg for m in _RECORDED["corpus"]
                   if (msg := self._refusal(m)) is not None}
        self.assertEqual(refused, {})

    def test_omni_model_resolving_to_text_generation_is_admitted(self):
        # Page task any-to-any; HF resolves the checkpoint to
        # AutoModelForCausalLM / text-generation.
        self.assertIsNone(self._refusal("meituan-longcat/LongCat-Flash-Omni"))

    def test_causal_lm_is_admitted(self):
        self.assertIsNone(self._refusal("Qwen/Qwen3-8B"))

    def test_remote_code_registration_outranks_hf_resolution(self):
        # HF resolves Kimi-K2.5 to AutoModel / feature-extraction (the
        # first auto_map entry); the repo registers AutoModelForCausalLM.
        task = _hub_task("moonshotai/Kimi-K2.5")
        self.assertEqual(task.resolved_pipeline, "feature-extraction")
        self.assertIsNone(self._refusal("moonshotai/Kimi-K2.5"))

    def test_causal_lm_config_under_other_library_label_is_admitted(self):
        # library_name `vllm`, no transformers_info, MistralForCausalLM.
        self.assertIsNone(self._refusal("mistralai/Mistral-Nemo-Base-2407"))

    def test_repo_shipping_its_own_forward_is_admitted(self):
        # No Transformers resolution, no auto_map, a config head Transformers
        # does not know (Step5ForConditionalGeneration); the forward is the
        # repo's reference implementation in `inference/model.py`.
        task = _hub_task("stepfun-ai/Step-5-Preview")
        self.assertFalse(task.resolved)
        self.assertIn("inference/model.py", task.files)
        self.assertIsNone(self._refusal("stepfun-ai/Step-5-Preview"))

    def test_audio_out_omni_model_is_refused(self):
        msg = self._refusal("Qwen/Qwen3-Omni-30B-A3B-Instruct")
        self.assertIn("Any-to-Any model that Transformers resolves to "
                      "Text-to-Audio", msg)
        self.assertIn("text-generation and image-text-to-text models only", msg)

    def test_speech_model_with_non_causal_config_is_refused(self):
        msg = self._refusal("mistralai/Voxtral-Mini-4B-Realtime-2602")
        self.assertIn("Automatic-Speech-Recognition model", msg)
        self.assertIn("vllm", msg)

    def test_embedding_models_are_refused(self):
        # bge-m3: no Transformers resolution, XLMRobertaModel backbone.
        msg = self._refusal("BAAI/bge-m3")
        self.assertIn("Sentence-Similarity model", msg)
        self.assertIn("sentence-transformers", msg)
        # all-MiniLM: resolved to AutoModel / feature-extraction.
        msg = self._refusal("sentence-transformers/all-MiniLM-L6-v2")
        self.assertIn("Sentence-Similarity model", msg)
        self.assertIn("Feature-Extraction", msg)

    def test_diffusion_pipeline_is_admitted(self):
        # library `diffusers`, no Transformers resolution, no config: a
        # `model_index.json` naming its components is what admits it.
        task = _hub_task("stabilityai/stable-diffusion-xl-base-1.0")
        self.assertFalse(task.resolved)
        self.assertIn("model_index.json", task.files)
        self.assertIsNone(self._refusal("stabilityai/stable-diffusion-xl-base-1.0"))

    def test_a_refusal_names_the_pipelines_in_scope(self):
        msg = self._refusal("BAAI/bge-m3")
        self.assertIn("diffusers pipelines", msg)


class TestHubCannotAnswer(unittest.TestCase):
    """Gated / private / missing / offline repos are unknown, never refused
    here — the config-fetch classifier owns those words."""

    def setUp(self):
        task_gate._hub_model_info.cache_clear()
        self.addCleanup(task_gate._hub_model_info.cache_clear)
        no_snapshot = mock.patch(
            "backend.arch.snapshots.load_config", return_value=None
        )
        no_snapshot.start()
        self.addCleanup(no_snapshot.stop)

    @staticmethod
    def _http_error(kind, status):
        resp = httpx2.Response(status, request=httpx2.Request("GET", "https://hf.co"))
        return kind("probe", response=resp)

    def test_unanswered_lookups_pass_through(self):
        for exc in (
            self._http_error(hf_errors.GatedRepoError, 403),
            self._http_error(hf_errors.RepositoryNotFoundError, 404),
            self._http_error(hf_errors.HfHubHTTPError, 429),
            self._http_error(hf_errors.HfHubHTTPError, 500),
            self._http_error(hf_errors.HfHubHTTPError, 400),
            hf_errors.OfflineModeIsEnabled("offline"),
            httpx2.ConnectError("connection refused"),
        ):
            with self.subTest(exc=type(exc).__name__), \
                 mock.patch("huggingface_hub.HfApi.model_info", side_effect=exc):
                self.assertIsNone(task_gate.unsupported_task_refusal("org/model"))

    def test_failures_are_not_cached(self):
        info = ModelInfo(**_RECORDS["Qwen/Qwen3-8B"])
        outage = self._http_error(hf_errors.HfHubHTTPError, 503)
        with mock.patch("huggingface_hub.HfApi.model_info",
                        side_effect=[outage, info]) as hub:
            self.assertIsNone(task_gate.fetch_hub_task("org/model"))
            self.assertIsNotNone(task_gate.fetch_hub_task("org/model"))
        self.assertEqual(hub.call_count, 2)

    def test_gate_bugs_are_not_swallowed(self):
        with mock.patch("huggingface_hub.HfApi.model_info",
                        side_effect=AttributeError("bug")), \
             self.assertRaises(AttributeError):
            task_gate.unsupported_task_refusal("org/model")


class TestGateSkipsTheHub(unittest.TestCase):
    def test_custom_ids_never_leave_the_gate(self):
        with mock.patch("backend.hf.task_gate._hub_model_info") as hub:
            self.assertIsNone(
                task_gate.unsupported_task_refusal("custom:my-model")
            )
        hub.assert_not_called()

    def test_curated_snapshots_skip_the_hub(self):
        # Built-ins are known-good; the gate must not round-trip for them.
        with mock.patch(
            "backend.arch.snapshots.load_config", return_value={"model_type": "qwen3"}
        ), mock.patch("backend.hf.task_gate._hub_model_info") as hub:
            self.assertIsNone(
                task_gate.unsupported_task_refusal("Qwen/Qwen3-8B")
            )
        hub.assert_not_called()


if __name__ == "__main__":
    unittest.main()
