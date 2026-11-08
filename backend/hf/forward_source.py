"""Best-effort fetch of forward() source text for structural validation.

Multi-strategy chain (first non-empty wins):

  1. Custom model (`custom:<name>`)
     → read user-provided `forward.py` from the local cache dir.

  2. HF Hub direct download — the repo's own code, which is the model's
     forward wherever the repo ships one
     → pull the `auto_map` files, root `modeling_*.py`, or reference code
       (`inference/model.py`) from the model repo (DeepSeek V3 / V4, Kimi,
       GLM, Nemotron, MiniMax ship source alongside the checkpoint).

  3. Local transformers native model, for a repo that ships no code
     → import `transformers.models.<model_type>.modeling_<model_type>`
       and return `inspect.getsource(module)`. Python's `inspect` resolves
       cross-class and cross-file inheritance automatically, so
       nested / subclassed forward methods are captured.

  4. model_info cache (written by `backend/tools/model_info.py` after
     a successful run — typically populated by spec_worker during
     spec generation)
     → read `~/.cache/huggingarch/model_info/<model_id>.json` and
       concatenate `forward_methods` values.

  Returns '' on total failure — callers treat this as "structural
  check unavailable" and skip silently.

When the repo cannot be read (offline, outage, gated), strategies 3-4 are
not the repo's code: `describe_forward_source` still returns them for the
structural hints, marked `degraded` with the fetch error and led by a
`# DEGRADED SOURCE` banner. The snapshot refresh (`hf.snapshot`) and the
spec-gen module source (`spec_worker._ground_truth.full_module_sources`)
never stand in: a failed repo fetch is their error. Offline, both work from
what is already captured — a snapshot's frozen `forward_source`, or repo
files in the local HF cache named by `auto_map` — and the transformers
module is used only once the Hub has answered that the repo ships no code.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from backend.custom_id import is_custom_id, model_path, strip_prefix


_CACHE_DIR = Path.home() / ".cache" / "huggingarch" / "model_info"

# A relative import of one of these class-name shapes names a model backbone /
# decoder — the load-bearing forward. If the sibling module it comes from isn't
# among the fetched files, the actual decoder forward is MISSING from the
# source we hand the agent (it only got the config + whatever wrapper/tower
# lives in the fetched files). Kimi-K3 is the canonical case: its
# `KimiK3ForConditionalGeneration` does `from .modeling_kimi_linear import
# KimiLinearForCausalLM`, and `modeling_kimi_linear.py` — where the AttnRes /
# LatentMoE / attention forward actually lives — is a sibling file the
# auto_map fetch never pulled.
_DECODER_IMPORT_RE = re.compile(
    r'^\s*from\s+\.+([\w.]+)\s+import\s+(.+)$', re.MULTILINE,
)
_DECODER_NAME_RE = re.compile(
    r'\b(\w*(?:ForCausalLM|ForConditionalGeneration|DecoderLayer|Model))\b',
)


def _is_framework_name(name: str) -> bool:
    """A transformers factory / base / output class that is NOT a model's own
    decoder: `AutoModel*` (registry factories), `*PreTrainedModel` (bases),
    `Base*` / `*Output*` (output dataclasses like `BaseModelOutputWithPast`).
    These ride along in `inspect.getsource(module)` of a transformers-native
    modeling file and must not be mistaken for a missing sibling decoder."""
    return (
        name.startswith(("Auto", "Base"))
        or "Output" in name
        or name.endswith("PreTrainedModel")
    )


def _unresolved_decoder_imports(
    files: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Relative imports of a *Model / *ForCausalLM / *DecoderLayer whose
    sibling module is NOT among the fetched files — i.e. the decoder forward
    the agent needs is elsewhere. Returns `[{module, names}, ...]`.

    Only meaningful for `hf_modeling_files` source (individually-fetched repo
    files that can miss a sibling). `local_transformers` / `model_info_cache`
    are self-contained, so callers gate on `source` — see
    `describe_forward_source`."""
    have = {Path(f.get("path", "")).stem for f in files if f.get("path")}
    joined = "\n".join(f.get("text", "") for f in files)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for mod, imported in _DECODER_IMPORT_RE.findall(joined):
        leaf = mod.rsplit(".", 1)[-1]
        if leaf in have or leaf in seen:
            continue
        names = [n for n in _DECODER_NAME_RE.findall(imported)
                 if not _is_framework_name(n)]
        if names:
            seen.add(leaf)
            out.append({"module": leaf, "names": sorted(set(names))})
    return out


#: Upper bound on the files one HF fetch hands the agent, siblings included.
_MAX_SOURCE_FILES = 8

_ABSOLUTE_IMPORT_RE = re.compile(
    r'^\s*(?:from\s+(\w+)\s+import\s|import\s+(\w+)\s*(?:as\s+\w+\s*)?$)',
    re.MULTILINE,
)


def _sibling_imports(results, fetched, listing):
    """`(repo, repo-relative path)` of each sibling module the fetched files
    import and nothing has asked for yet, in import order.

    `results` carry `repo` / `repo_path`; `listing(repo)` is the repo's file
    set. A relative import of a backbone/decoder name resolves against the
    importing file's directory; an absolute import names a sibling exactly
    when `<importer dir>/<module>.py` is in the repo — the repo's file list
    says so, whatever this host happens to have installed."""
    out: list = []
    for r in results:
        repo, rel = r.get("repo"), r.get("repo_path")
        if not repo or rel is None:
            continue
        here = PurePosixPath(rel).parent
        text = r.get("text", "")
        for mod, imported in _DECODER_IMPORT_RE.findall(text):
            names = [n for n in _DECODER_NAME_RE.findall(imported)
                     if not _is_framework_name(n)]
            if not names:
                continue
            fname = (here / f"{mod.rsplit('.', 1)[-1]}.py").as_posix()
            if (repo, fname) not in fetched and fname in listing(repo):
                out.append((repo, fname))
        for frm, imp in _ABSOLUTE_IMPORT_RE.findall(text):
            mod = frm or imp
            fname = (here / f"{mod}.py").as_posix()
            if (repo, fname) in fetched:
                continue
            if fname in listing(repo):
                out.append((repo, fname))
    return list(dict.fromkeys(out))


def _incomplete_source_banner(files: List[Dict[str, str]]) -> str:
    """A comment-safe banner (each line `#`, so `ast.parse` ignores it) warning
    that the decoder forward is missing. Empty string when the source looks
    complete. Spec-gen prep does not stop at the banner: it refuses the
    source (`_ground_truth.full_module_sources`)."""
    missing = _unresolved_decoder_imports(files)
    if not missing:
        return ""
    refs = "; ".join(
        f"`{m['module']}.py` (provides {', '.join(m['names'])})" for m in missing
    )
    return (
        "# ⚠️ INCOMPLETE FORWARD SOURCE — the decoder/backbone forward is NOT below.\n"
        f"# The fetched files import the load-bearing model from a sibling module\n"
        f"# that was NOT pulled: {refs}.\n"
        "# The code below is only the config + wrapper/vision tower. Do NOT infer\n"
        "# the text-decoder block wiring (residual/norm placement, gating,\n"
        "# cross-block accumulation, MoE routing) from what's here — you'll guess\n"
        "# wrong. Spec generation refuses to start on source like this.\n"
        "# ── end warning ──────────────────────────────────────────────────────"
    )


def model_info_cache_path(model_id: str) -> Path:
    """Canonical on-disk path for a model_info result cache entry."""
    return _CACHE_DIR / f"{model_id.replace('/', '__')}.json"


def describe_forward_source(model_id: str, resolved_spec: Any) -> Dict[str, Any]:
    """Return `{source, files, paths, text}` describing where forward() came from.

    - `source`: which strategy supplied the text (one of
      `custom_model`, `local_transformers`, `model_info_cache`,
      `hf_modeling_files`, or `` when nothing matched).
    - `files`: list of `{path, text}` pairs — one per physical file
      contributing source. Frontends render each in its own frame.
    - `paths`: flat list of paths (same order as `files`). Derived.
    - `text`: concatenated text with per-file headers. Kept so AST
      callers can consume a single module-shaped string.
    """
    repo_fetch_error = ""
    for label, fn in (
        ("snapshot", lambda: _try_snapshot(model_id)),
        ("custom_model", lambda: _try_custom_model(model_id)),
        ("hf_modeling_files", lambda: _try_hf_modeling_files(model_id, resolved_spec)),
        ("local_transformers", lambda: _try_local_transformers(model_id, resolved_spec)),
        ("model_info_cache", lambda: _try_model_info_cache(model_id)),
    ):
        # The documented strategy contract is "returns [] on any failure" —
        # forward source feeds only structural WARNINGS, which never gate, so
        # a fetch crash (e.g. httpx.ConnectError on an offline/sandboxed host)
        # must degrade to "try the next strategy", not kill the validation.
        # A failed repo fetch is remembered: what a later strategy supplies is
        # then a stand-in for the repo's own code, and the payload says so.
        try:
            files = fn()
        except Exception as exc:
            if label == "hf_modeling_files":
                repo_fetch_error = f"{type(exc).__name__}: {exc}"
            continue
        if files and any(f.get("text") for f in files):
            paths = [f["path"] for f in files if f.get("path")]
            # Joined text for AST consumers (structural validator).
            text = _joined(files)
            # Missing-sibling detection is ONLY valid for `hf_modeling_files`
            # (individually-fetched repo files that can miss a sibling). A
            # `local_transformers` source is `inspect.getsource(module)` — a
            # self-contained module whose relative imports (`from ...auto import
            # AutoModel`, `from ..llama.modeling_llama import LlamaModel`) point
            # at OTHER installed modules, not a missing decoder; `model_info_cache`
            # is a method dump. Snapshots preserve the ORIGINAL source, so read
            # it back rather than trusting the "snapshot" label.
            orig_source = label
            if label == "snapshot":
                try:
                    from backend.arch.snapshots import load_forward_source
                    orig_source = (load_forward_source(model_id) or {}).get(
                        "source") or "snapshot"
                except Exception:
                    orig_source = "snapshot"
            # If the decoder forward is imported from a sibling module the fetch
            # didn't pull, lead with a warning so the agent goes and gets it
            # instead of hallucinating block wiring from the config + tower.
            # Comment-safe: `ast.parse` (structural validator) ignores it.
            banner = (_incomplete_source_banner(files)
                      if orig_source == "hf_modeling_files" else "")
            if banner:
                text = f"{banner}\n\n{text}"
            degraded = ""
            if repo_fetch_error and label in ("local_transformers", "model_info_cache"):
                degraded = (f"the HF repo could not be read ({repo_fetch_error}); "
                            f"this source is the {label} stand-in, not the "
                            f"repo's own modeling code")
                text = f"# DEGRADED SOURCE: {degraded}\n\n{text}"
            return {
                "source": label,
                "files": files,
                "paths": paths,
                "text": text,
                "incomplete": bool(banner),
                "degraded": degraded,
            }
    return {"source": "", "files": [], "paths": [], "text": "", "incomplete": False,
            "degraded": ""}


def auto_model_class(model_type: str) -> Optional[str]:
    """The class transformers' auto registry builds for `model_type`
    (`AutoModel.from_config(cfg)` with `cfg.model_type`), or None when the
    registry is not installed or has no entry for it."""
    if importlib.util.find_spec("transformers") is None:
        return None
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
    return MODEL_MAPPING_NAMES.get(model_type)


def fetch_forward_source(model_id: str, resolved_spec: Any) -> str:
    """Thin wrapper returning only the joined source text.
    Kept for callers that just need the AST input."""
    return describe_forward_source(model_id, resolved_spec)["text"]


def recorded_forward_files(model_id: str) -> List[Dict[str, str]]:
    """The forward source files recorded for this model — its snapshot's,
    or a custom model's own — as `{path, text}`; empty when none is
    recorded. Unlike `describe_forward_source` it never reads the host's
    installed transformers or the Hub, so a check against it gives the same
    answer on every host and version."""
    return _try_snapshot(model_id) or _try_custom_model(model_id)


def recorded_forward_source(model_id: str) -> Optional[str]:
    """`recorded_forward_files`, joined like `describe_forward_source`;
    None when none is recorded."""
    files = recorded_forward_files(model_id)
    return _joined(files) if files else None


def _joined(files: List[Dict[str, str]]) -> str:
    return "\n\n".join(f"# ==== {f.get('path') or ''} ====\n{f.get('text', '')}"
                       for f in files)


# ---------------------------------------------------------------------------
# Strategies (each returns '' on any failure)
# ---------------------------------------------------------------------------

def _try_snapshot(model_id: str) -> List[Dict[str, str]]:
    """Read forward_source directly from the git-managed snapshot. Authoritative
    for built-in models: freezes the exact forward code that was analyzed,
    making structural validation reproducible and fully offline-capable."""
    try:
        from backend.arch.snapshots import load_forward_source
    except ImportError:
        return []
    fs = load_forward_source(model_id)
    if not fs:
        return []
    files = fs.get("files") or []
    return [
        {"path": f.get("path", ""), "text": f.get("text", "")}
        for f in files if f.get("text")
    ]


def _try_custom_model(model_id: str) -> List[Dict[str, str]]:
    if not is_custom_id(model_id):
        return []
    try:
        from backend.custom_model import load
    except ImportError:
        return []
    try:
        name = strip_prefix(model_id)
        text = load(name).get("forward_source", "") or ""
        if not text:
            return []
        path = str(model_path(name).parent / "forward.py")
        return [{"path": path, "text": text}]
    except (FileNotFoundError, OSError):
        return []


def _try_local_transformers(model_id: str, resolved_spec: Any) -> List[Dict[str, str]]:
    """Import the modeling module from the installed transformers
    package. Covers all transformers-native models without network or
    subprocess. The package comes from transformers' own model_type →
    module mapping, so text sub-configs (`gemma3_text` → `gemma3`,
    `llama4_text` → `llama4`) and special names (`openai-gpt`) resolve.
    """
    config = _resolved_config(resolved_spec)
    model_type = (config.get("model_type") if isinstance(config, dict) else None) or ""
    if not model_type:
        return []
    try:
        from transformers.models.auto.configuration_auto import model_type_to_module_name
    except ImportError:
        return []
    package = model_type_to_module_name(model_type)
    leaf = package.rsplit(".", 1)[-1]
    import importlib
    candidates = [f"transformers.models.{package}.modeling_{leaf}"]
    for modname in candidates:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):
            continue
        try:
            path = inspect.getfile(mod)
        except (TypeError, OSError):
            path = ""
        return [{"path": path, "text": src}]
    return []


def _try_pipeline_sources(model_id: str, resolved_spec: Any) -> List[Dict[str, str]]:
    """A diffusers pipeline's forward: each component's module from the
    library its index names (`pipeline.compose_config` keeps the pair under
    `_components`) — a transformers component's modeling module by its
    config's `model_type`, a diffusers component's by the file of the
    installed `diffusers` package that defines its class. The diffusers files
    are read as text and never imported. Empty for a non-pipeline config."""
    config = _resolved_config(resolved_spec)
    named = config.get("_components") if isinstance(config, dict) else None
    if not isinstance(named, dict):
        return []
    out: List[Dict[str, str]] = []
    diffusers_root: Optional[Path] = None
    spec = importlib.util.find_spec("diffusers")
    if spec is not None and spec.submodule_search_locations:
        diffusers_root = Path(next(iter(spec.submodule_search_locations)))
    for name, pair in sorted(named.items()):
        library, cls = [*pair, None, None][:2]
        component = config.get(name)
        if library == "transformers" and isinstance(component, dict):
            out.extend(_try_local_transformers(
                model_id, type("_Shim", (), {"config": component})()))
        elif library == "diffusers" and diffusers_root is not None and cls:
            pattern = re.compile(rf"^class {re.escape(cls)}\b", re.MULTILINE)
            for path in sorted((diffusers_root / "models").rglob("*.py")):
                text = path.read_text(encoding="utf-8")
                if pattern.search(text):
                    rel = PurePosixPath("site-packages", "diffusers",
                                        path.relative_to(diffusers_root).as_posix())
                    out.append({"path": str(rel), "text": text})
                    break
    # One file per module, however many components it serves.
    seen: set = set()
    return [f for f in out if not (f["path"] in seen or seen.add(f["path"]))]


def _try_model_info_cache(model_id: str) -> List[Dict[str, str]]:
    path = model_info_cache_path(model_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    forwards = (data.get("source") or {}).get("forward_methods") or data.get("forward_methods") or {}
    if not isinstance(forwards, dict) or not forwards:
        return []
    # Cache mixes multiple class sources into a single JSON entry; treat
    # the whole cache file as one logical "file" but include per-class
    # section headers inside the text.
    parts = []
    for label, src in forwards.items():
        parts.append(f"# ==== {label} ====\n{src}")
    return [{"path": str(path), "text": "\n\n".join(parts)}]


def _auto_map_files(auto_map: Any, model_id: str) -> List[tuple]:
    """`(repo_id, filename)` for each `config.auto_map` class reference.
    `module.Class` lives in the model's own repo; `org/repo--module.Class`
    names the repo that holds the code (transformers' dynamic-module
    convention — quantized / fine-tuned repos point at upstream code)."""
    out: list = []
    if not isinstance(auto_map, dict):
        return out
    for ref in auto_map.values():
        if isinstance(ref, str) and "." in ref:
            stem = ref.rsplit(".", 1)[0]
            repo, _, module = stem.rpartition("--")
            out.append((repo or model_id, f"{module}.py"))
    # AutoConfig / AutoModelForCausalLM often name the same file — dedupe in
    # first-seen order.
    return list(dict.fromkeys(out))


def own_forward_files(files) -> List[str]:
    """The repo's own forward-pass source among `files` (repo-relative
    paths of one repo listing), in fetch order: the root `modeling_*.py`
    files of the standard transformers layout, else the reference-
    implementation paths. Empty when the repo ships no forward source."""
    files = set(files)
    candidates = sorted(f for f in files
                        if f.startswith("modeling_") and f.endswith(".py"))
    if candidates:
        return candidates
    # Where publishers that ship no root `modeling_*.py` put their forward
    # (DeepSeek, Kimi, Moonlight, StepFun, …): a small priority list of
    # likely names + dirs rather than globbing the repo, model.py first so
    # it dominates when several match.
    reference_paths = (
        "inference/model.py",
        "inference/kernel.py",
        "inference/modeling.py",
        "src/model.py",
        "code/model.py",
        "model.py",
        "modeling.py",
    )
    return [f for f in reference_paths if f in files]


def _try_hf_modeling_files(model_id: str, resolved_spec: Any) -> List[Dict[str, str]]:
    """Pull forward source from the model's HF repo. Three discovery
    layers, tried in order:

      a. `config.auto_map` references — for trust_remote_code models the
         class is declared like `modeling_x.MyForCausalLM` (or
         `org/repo--modeling_x.MyForCausalLM`, code in another repo); we
         download each `.py` file from the repo that holds it.
      b. Root-level `modeling_*.py` — the standard transformers layout
         (also the fallback when every auto_map file is missing).
      c. Reference-implementation directories — DSv3.2 / DSv3.2-Exp ship
         their forward in `inference/model.py` (with `inference/kernel.py`
         containing the DSA / TileLang custom kernels), Kimi-K2.5 the
         same. We pick a small ordered priority list rather than
         globbing the whole repo to avoid downloading tests / tooling.

    Unlike the other strategies this one does not swallow fetch failures:
    returns [] only when the Hub answers that no modeling file exists (a
    named file is a 404), and raises when the Hub cannot be read (network,
    outage, offline cache miss, missing / gated repo). Spec-gen prep needs
    that difference to tell "retry" from "this model has no source";
    `describe_forward_source` catches it like any other strategy failure.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from huggingface_hub.errors import RemoteEntryNotFoundError
    except ImportError:
        return []

    results: List[Dict[str, str]] = []
    # (repo, repo-relative filename) of each result, same order, plus every
    # name already asked for so a 404 is not asked twice.
    fetched: set = set()

    def _fetch(repo: str, fname: str) -> bool:
        fetched.add((repo, fname))
        try:
            path = hf_hub_download(repo_id=repo, filename=fname)
        except RemoteEntryNotFoundError:  # a named file the repo lacks
            return False
        with open(path) as _f:
            results.append({"path": path, "text": _f.read(),
                            "repo": repo, "repo_path": fname})
        return True

    config = _resolved_config(resolved_spec)
    for repo, fname in _auto_map_files(config.get("auto_map"), model_id)[:3]:
        _fetch(repo, fname)  # cap to avoid huge downloads
    listings: Dict[str, set] = {}

    def _listing(repo: str) -> set:
        if repo not in listings:
            listings[repo] = set(list_repo_files(repo_id=repo))
        return listings[repo]

    if not results:
        candidates = own_forward_files(_listing(model_id))
        for fname in candidates[:3]:  # cap to avoid huge downloads
            _fetch(model_id, fname)

    # Follow imports to sibling `.py` files nothing named. Two shapes:
    #  - a relative import of the backbone/decoder (`from .modeling_dec import
    #    DecForCausalLM`): the wrapper an auto_map points at often just glues a
    #    config + vision tower and pulls the load-bearing decoder from a
    #    sibling — Kimi-K3's `modeling_kimi_k3.py` does `from
    #    .modeling_kimi_linear import KimiLinearForCausalLM`;
    #  - an absolute import of a module that sits in the importing file's own
    #    directory (`from engram import NgramHashState`): reference
    #    implementations run as scripts from their directory, so their local
    #    modules are imported absolutely. DeepSeek-V4.1's `inference/model.py`
    #    takes Engram and the vision tower this way from `inference/engram.py`
    #    and `inference/vision.py`.
    # Both resolve against the importing file's directory, not the repo root.
    # Without this the source we hand the agent is missing whole blocks and
    # it writes them from tensor names. Bounded: at most 2 hops and
    # `_MAX_SOURCE_FILES` files; a repo is listed when a fetched file imports
    # a module nothing has pulled yet. Whatever stays unresolved is visible to
    # `_unresolved_decoder_imports` downstream.
    for _hop in range(2):
        added = False
        for repo, fname in _sibling_imports(results, fetched, _listing):
            if len(results) >= _MAX_SOURCE_FILES:
                break
            if (repo, fname) in fetched:
                continue
            fetched.add((repo, fname))
            if _fetch(repo, fname):
                added = True
        if not added:
            break
    return [{"path": r["path"], "text": r["text"]} for r in results]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolved_config(resolved_spec: Any) -> dict:
    """Pull the config dict off a ResolvedSpec-like object; never raise."""
    if resolved_spec is None:
        return {}
    for attr in ("config", "_config", "raw_config"):
        val = getattr(resolved_spec, attr, None)
        if isinstance(val, dict):
            return val
    return {}
