"""Task-type sanity check for model ids — Layer 1 (I/O boundary).

HuggingArch derives architecture from a Transformers ``config.json``, so it
answers only for models that load as a text-generation or
image-text-to-text Transformers model — and for diffusers pipelines, whose
``model_index.json`` names the components it reads together
(``hf.pipeline``). A repo outside that set (an audio model, an embedding
encoder) used to fail downstream with a vague "no config.json" or "No spec
matches" — ``unsupported_task_refusal`` asks the
model-info API and turns every out-of-scope answer into a precise refusal
at the boundary instead.

The decision reads what the hub declares about how the checkpoint loads:
Hugging Face's own resolution of the repo to a Transformers task
(``transformers_info.pipeline_tag``), the auto classes the repo registers
for its remote code (``config.auto_map``), and — when Hugging Face did not
resolve the repo to Transformers — whether its ``config.architectures``
names a head that skeleton extraction (``backend.hf.model_info``) builds,
or the repo ships its own forward source (a reference implementation such
as ``inference/model.py``, which spec generation reads instead).
The author-set page ``pipeline_tag`` only words the refusal: an omni model
tagged ``any-to-any`` whose checkpoint resolves to text-generation is in
scope; one that resolves to text-to-audio is not.

Custom ids and curated snapshots pass through untouched. A lookup the hub
cannot answer (gated or private repo, missing repo, offline, rate limit)
yields None so the config-fetch classifier keeps owning those words; every
answer the hub does give settles admit or refuse.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from huggingface_hub import errors as hf_errors

from backend.custom_id import is_custom_id
from backend.hf.metadata import HUB_UNREACHABLE
from backend.hf.forward_source import own_forward_files
from backend.hf.model_info import SKELETON_AUTO_CLASS, causal_lm_heads
from backend.hf.pipeline import is_pipeline_repo


logger = logging.getLogger(__name__)


# The single vocabulary of what a spec can describe: the tasks Hugging Face
# resolves a Transformers checkpoint to (``transformers_info.pipeline_tag``)
# whose language stack HuggingArch analyzes. This is the only place the set
# is defined — routes and workers ask the gate.
SUPPORTED_PIPELINES = frozenset({
    "text-generation",     # causal LMs, incl. omni models whose checkpoint
                           # loads as a causal LM (LongCat-Flash-Omni)
    "image-text-to-text",  # VLMs; audio-out omni models resolve to
                           # text-to-audio and stay out (Qwen3-Omni)
})

# Hub failures where the hub cannot answer for this id. The gate stays
# fail-open on all of them (unknown, never a refusal); transient and
# unexpected ones are logged so an outage is visible.
_HUB_NO_SUCH_REPO = (
    hf_errors.HFValidationError,        # not an owner/name id
    hf_errors.RepositoryNotFoundError,  # missing, private, or gated
)

# Connector words HF keeps lowercase in its model-page task labels.
_TASK_SMALL_WORDS = frozenset({"to", "and", "of", "with", "for"})

_SCOPE = ("HuggingArch supports diffusers pipelines and text-generation and "
          "image-text-to-text models only.")


def prettify_task(tag: str) -> str:
    """HF's model-page display form: ``image-text-to-video`` →
    ``Image-Text-to-Video`` (connector words stay lowercase)."""
    return "-".join(
        word if word in _TASK_SMALL_WORDS else word.capitalize()
        for word in tag.split("-")
    )


def _a(label: str) -> str:
    return "an" if label[:1].lower() in "aeiou" else "a"


@dataclass(frozen=True)
class HubTask:
    """What the hub's model-info call says a repo loads as."""

    page_tag: Optional[str]           # author-set page task; wording only
    library: Optional[str]            # author-set library_name
    resolved: bool                    # HF resolved the repo to Transformers
    resolved_pipeline: Optional[str]  # transformers_info.pipeline_tag
    registered: frozenset             # config.auto_map keys (remote code)
    architectures: tuple              # config.architectures
    files: frozenset                  # repo-relative paths (siblings)

    @classmethod
    def from_model_info(cls, info) -> "HubTask":
        ti = info.transformers_info
        config = info.config or {}
        return cls(
            page_tag=info.pipeline_tag,
            library=info.library_name,
            resolved=ti is not None,
            resolved_pipeline=ti.pipeline_tag if ti is not None else None,
            registered=frozenset(config.get("auto_map") or ()),
            architectures=tuple(config.get("architectures") or ()),
            files=frozenset(s.rfilename for s in info.siblings or ()),
        )

    def refusal(self, model_id: str) -> Optional[str]:
        """Precise "not supported" message, or None when the repo is in
        scope. Every hub answer settles the question one way or the other."""
        if self.resolved_pipeline in SUPPORTED_PIPELINES:
            return None
        if is_pipeline_repo(self.files):
            # A diffusers pipeline (`model_index.json`, one folder per
            # component): read as one model, its components' configs and
            # checkpoints together (`hf.pipeline`). Its page task
            # (text-to-video, image-text-to-video) says what the pipeline
            # makes, not which Transformers head it loads as — it loads as
            # none.
            return None
        if SKELETON_AUTO_CLASS in self.registered:
            # The repo registers its remote code under the class skeleton
            # extraction instantiates. For remote-code repos HF's resolution
            # can stop at the generic AutoModel (Kimi-K2.5 → feature-
            # extraction) while the repo registers a causal-LM class; that
            # registration is executable and is what extraction loads, so
            # it decides.
            return None
        if self.resolved:
            label = self._page_label()
            resolved = (prettify_task(self.resolved_pipeline)
                        if self.resolved_pipeline
                        else "a bare backbone with no task head")
            if label is None:
                return (f"'{model_id}' resolves in Transformers to {resolved} "
                        f"— {_SCOPE}")
            if label == resolved:
                return f"'{model_id}' is {_a(label)} {label} model — {_SCOPE}"
            return (f"'{model_id}' is {_a(label)} {label} model that "
                    f"Transformers resolves to {resolved} — {_SCOPE}")
        # HF did not resolve the repo to Transformers (library label is not
        # `transformers`). A config naming a causal-LM head still loads
        # through skeleton extraction.
        if self.architectures and set(self.architectures) & causal_lm_heads():
            return None
        # Nor did it register remote code; a repo that ships its own forward
        # (Step-5-Preview's `inference/model.py`) is still read by spec
        # generation from that source, the same files forward_source fetches.
        if own_forward_files(self.files):
            return None
        library = self.library or "unknown"
        label = self._page_label()
        if label is not None:
            return (f"'{model_id}' is {_a(label)} {label} model from the "
                    f"{library} library, not a Transformers text-generation "
                    f"model — {_SCOPE}")
        return (f"'{model_id}' is a {library} repo, not a Transformers "
                f"text-generation model — {_SCOPE}")

    def _page_label(self) -> Optional[str]:
        return prettify_task(self.page_tag) if self.page_tag else None


@lru_cache(maxsize=512)
def _hub_model_info(model_id: str):
    """One model-info round-trip per id. Failures raise and are not cached,
    so a transient outage does not stick."""
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id, expand=[
        "pipeline_tag", "library_name", "transformersInfo", "config",
        "siblings",
    ])


def fetch_hub_task(model_id: str) -> Optional[HubTask]:
    """The repo's HubTask, or None when it cannot be determined (custom
    id, curated snapshot, or a lookup the hub cannot answer)."""
    if is_custom_id(model_id):
        return None
    # Curated snapshots are known-good built-ins; skip the round-trip.
    from backend.arch.snapshots import load_config as load_snapshot_config
    if load_snapshot_config(model_id) is not None:
        return None
    try:
        info = _hub_model_info(model_id)
    except _HUB_NO_SUCH_REPO:
        return None
    except hf_errors.HfHubHTTPError as exc:
        # A status answer, caught before HUB_UNREACHABLE (whose HTTPError it
        # subclasses). 429 / 5xx are transient; any other status is
        # unexpected here.
        logger.warning("task gate: hub answered HTTP %s for %s: %s",
                       exc.response.status_code, model_id, exc)
        return None
    except HUB_UNREACHABLE as exc:
        logger.warning("task gate: hub unreachable for %s: %s", model_id, exc)
        return None
    return HubTask.from_model_info(info)


def unsupported_task_refusal(model_id: str) -> Optional[str]:
    """Precise "not supported" message when the hub's answer puts the model
    outside SUPPORTED_PIPELINES; None when it is supported or could not be
    determined (never refuse on an unknown)."""
    task = fetch_hub_task(model_id)
    return task.refusal(model_id) if task is not None else None
