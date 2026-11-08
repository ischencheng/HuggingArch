"""A diffusers pipeline repo reads as one model: its config is the index with
each component's own config in its place, and its checkpoint names every
tensor under its component's folder."""
from __future__ import annotations

from backend.hf.pipeline import (compose_config, components,
                                 is_pipeline_repo, weight_files)

INDEX = {
    "_class_name": "MiniMaxH3ModularPipeline",
    "_diffusers_version": "0.36.0.dev0",
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration", {}],
    "tokenizer": ["transformers", "Qwen2TokenizerFast", {}],
    "transformer": ["diffusers", "MiniMaxH3Transformer3DModel", {}],
    "scheduler": ["diffusers", "MiniMaxH3Scheduler", {}],
}

FILES = {
    "model_index.json", "README.md",
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
}


def test_a_root_index_without_a_root_checkpoint_is_a_pipeline():
    assert is_pipeline_repo(FILES)
    assert not is_pipeline_repo({"config.json", "model.safetensors"})
    # A transformers checkpoint that also ships an index is read as itself.
    assert not is_pipeline_repo(FILES | {"model.safetensors.index.json"})


def test_components_are_the_index_entries_naming_a_library_and_class():
    assert components(INDEX) == {
        "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration"),
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
        "transformer": ("diffusers", "MiniMaxH3Transformer3DModel"),
        "scheduler": ("diffusers", "MiniMaxH3Scheduler"),
    }


def test_a_component_stores_weights_sharded_single_or_not_at_all():
    assert weight_files(FILES, "text_encoder") == (
        "text_encoder/model.safetensors.index.json", None)
    assert weight_files(FILES, "transformer") == (
        None, "transformer/diffusion_pytorch_model.safetensors")
    assert weight_files(FILES, "tokenizer") is None
    assert weight_files(FILES, "scheduler") is None


def test_the_config_holds_each_component_config_under_its_name():
    config = compose_config(INDEX, {
        "transformer": {"hidden_size": 5376},
        "scheduler": {"shift": 12.0},
    })
    assert config["transformer"] == {"hidden_size": 5376}
    assert config["scheduler"] == {"shift": 12.0}
    assert config["_class_name"] == "MiniMaxH3ModularPipeline"
    assert config["_components"]["text_encoder"] == [
        "transformers", "Qwen3VLForConditionalGeneration"]
    # A component with no config file keeps only its library and class.
    assert "tokenizer" not in config
