"""A diffusers pipeline repo, read as one model.

A pipeline repo is not one checkpoint but several. Its root `model_index.json`
names the components — a transformer, a text encoder, autoencoders, the
tokenizer and the schedulers — and each component lives in its own subfolder
with its own config and, when it has weights, its own safetensors. MiniMax-H3
is one: `transformer/`, `text_encoder/` (a Qwen3-VL checkpoint), `vae/`,
`audio_vae/`.

HuggingArch reads such a repo as one model, the way it reads a VLM's vision
tower and language model as one:

  * its config is the index with each component's own config file in place of
    the component's entry (`config.transformer.hidden_size`,
    `config.text_encoder.text_config.num_hidden_layers`), and the index's
    `[library, class]` pair of every component under `_components`;
  * its checkpoint is every weighted component's tensors, each name prefixed
    by the component's folder (`transformer.proj_in.weight`,
    `text_encoder.model.language_model.embed_tokens.weight`). A spec
    addresses a component the way it addresses `model.` in a transformers
    checkpoint, through `checkpoint: {root: <component>}`.

The prefix is the folder name, so two components can never write the same
tensor name, and the file's byte total is the sum of the components' own.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .header import HUB_UNREACHABLE, ReadTensors

#: The file that makes a repo a diffusers pipeline.
PIPELINE_INDEX = "model_index.json"

#: A component's weights, as diffusers and transformers save them: a sharded
#: checkpoint's index, else the single file.
_WEIGHT_FILES = (
    ("diffusion_pytorch_model.safetensors.index.json",
     "diffusion_pytorch_model.safetensors"),
    ("model.safetensors.index.json", "model.safetensors"),
)


def is_pipeline_repo(files: Iterable[str]) -> bool:
    """Whether a repo listing is a diffusers pipeline's: a root
    `model_index.json` and no root transformers checkpoint beside it."""
    files = set(files)
    return PIPELINE_INDEX in files and not any(
        index in files or single in files for index, single in _WEIGHT_FILES)


def components(index: Mapping[str, Any]) -> Dict[str, Tuple[str, str]]:
    """The components `model_index.json` names: `{folder: (library, class)}`.
    An entry is a component when its value is `[library, class, ...]`; the
    `_`-prefixed keys are the pipeline's own facts."""
    out: Dict[str, Tuple[str, str]] = {}
    for name, value in index.items():
        if name.startswith("_") or not isinstance(value, list) or len(value) < 2:
            continue
        library, cls = value[0], value[1]
        if isinstance(library, str) and isinstance(cls, str):
            out[name] = (library, cls)
    return out


def weight_files(files: Iterable[str], component: str
                 ) -> Optional[Tuple[str, Optional[str]]]:
    """`(index file or None, single file or None)` of a component's weights
    in a repo listing, or None when the component stores none (a tokenizer,
    a scheduler)."""
    files = set(files)
    for index, single in _WEIGHT_FILES:
        if f"{component}/{index}" in files:
            return f"{component}/{index}", None
        if f"{component}/{single}" in files:
            return None, f"{component}/{single}"
    return None


def compose_config(index: Mapping[str, Any],
                   component_configs: Mapping[str, Mapping[str, Any]]
                   ) -> Dict[str, Any]:
    """The pipeline's config: the index's own `_` facts, each component's
    config under its name, and every component's `[library, class]` under
    `_components`. A component with no config file (a tokenizer) keeps only
    its `_components` entry."""
    named = components(index)
    out: Dict[str, Any] = {k: v for k, v in index.items() if k.startswith("_")}
    out["_components"] = {name: list(pair) for name, pair in named.items()}
    for name in named:
        if name in component_configs:
            out[name] = dict(component_configs[name])
    return out


def pipeline_config(model_id: str, revision: Optional[str] = None
                    ) -> Dict[str, Any]:
    """`compose_config` of the pipeline at `model_id` / `revision`, read off
    the Hub: the index, then each component's config file."""
    from huggingface_hub import HfApi, hf_hub_download

    files = set(HfApi().list_repo_files(model_id, revision=revision))
    with open(hf_hub_download(model_id, PIPELINE_INDEX, revision=revision),
              encoding="utf-8") as f:
        index = json.load(f)
    configs: Dict[str, Dict[str, Any]] = {}
    for name in components(index):
        # A model's `config.json`, a scheduler's `scheduler_config.json`.
        for fname in ("config.json", "scheduler_config.json"):
            path = f"{name}/{fname}"
            if path in files:
                with open(hf_hub_download(model_id, path, revision=revision),
                          encoding="utf-8") as f:
                    configs[name] = json.load(f)
                break
    return compose_config(index, configs)


def read_pipeline_header(model_id: str, revision: Optional[str] = None):
    """Every weighted component's shard headers at `revision`, each tensor
    named `<component>.<name>` (`header.SafetensorsHeader`)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.model_info(model_id, revision=revision, files_metadata=True)
    files = {s.rfilename for s in info.siblings}
    with open(hf_hub_download(model_id, PIPELINE_INDEX, revision=info.sha),
              encoding="utf-8") as f:
        index = json.load(f)
    read = ReadTensors()
    for component in sorted(components(index)):
        found = weight_files(files, component)
        if found is None:
            continue
        index_file, single = found
        if index_file is not None:
            with open(hf_hub_download(model_id, index_file, revision=info.sha),
                      encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map") or {}
            shards = sorted({f"{component}/{s}" for s in weight_map.values()})
        else:
            weight_map, shards = None, [single]
        names = set()
        for shard in shards:
            last_err: Optional[Exception] = None
            for attempt in range(3):
                try:
                    meta = api.parse_safetensors_file_metadata(
                        model_id, shard, revision=info.sha)
                    break
                except (OSError, ValueError, *HUB_UNREACHABLE) as e:
                    last_err = e
                    time.sleep(0.4 * (attempt + 1))
            else:
                raise OSError(
                    f"{model_id}: shard {shard} header unreadable: {last_err}")
            for name, t in meta.tensors.items():
                names.add(name)
                read.add(f"{component}.{name}", str(t.dtype), t.shape,
                         int(t.data_offsets[1] - t.data_offsets[0]),
                         shard=shard)
        if weight_map is not None and set(weight_map) != names:
            raise ValueError(
                f"{model_id}: {component}'s index and its shard headers name "
                f"different tensors ({len(set(weight_map) ^ names)} differ)")
    return read.header(revision=info.sha)
