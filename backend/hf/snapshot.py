"""Freeze a model's config.json + HF tensor metadata to the git-managed
snapshot dir so the analyzer can run on the model offline.

Usage:
    python -m backend.hf.snapshot <model_id>
    python -m backend.hf.snapshot --all         # refresh every ID in supported_models.txt
    python -m backend.hf.snapshot --missing     # only IDs without a snapshot

Example:
    # Requires the runner to have HF token + license acceptance for gated repos.
    python -m backend.hf.snapshot meta-llama/Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from backend.arch.quant_schemes import resolve_scheme
from backend.hf.metadata import HUB_UNREACHABLE, metadata_record
from backend.hf.pipeline import is_pipeline_repo, pipeline_config

# Ensure project root is on sys.path (mirrors other tools).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def fetch_config_raw(model_id: str) -> dict:
    """Raw HF fetch — bypasses the snapshot lookup in `hf.metadata.fetch_config`
    (otherwise refreshing a snapshot would read from itself). The spec-gen
    side's one config read, beside `fetch_metadata_raw`."""
    from huggingface_hub import HfApi, hf_hub_download
    if is_pipeline_repo(HfApi().list_repo_files(model_id)):
        return pipeline_config(model_id)
    p = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_metadata_raw(model_id: str, config: dict) -> dict:
    """Raw HF metadata fetch — calls the HF path directly so a stale
    snapshot doesn't short-circuit the refresh.

    This is the spec-generation side, the one place `config.quantization_config`
    is read: the scheme it names classifies the checkpoint's tensors for the
    snapshot, and `quant_method_unknown` travels to the spec-gen prompt."""
    from backend.hf.metadata import _fetch_metadata_hf
    scheme, num_bits, unknown_method = resolve_scheme(config)
    return _fetch_metadata_hf(
        model_id, scheme=scheme, num_bits=num_bits, unknown_method=unknown_method)


def _fetch_forward_source_raw(model_id: str, config: dict) -> dict:
    """Describe forward source bypassing the snapshot (so refresh captures
    the CURRENT transformers package / HF source, not the old snapshot)."""
    from backend.hf.forward_source import (
        _try_custom_model, _try_local_transformers,
        _try_model_info_cache, _try_hf_modeling_files,
        _try_pipeline_sources,
    )
    # Lightweight shim mimicking ResolvedSpec.config access
    class _ResolvedShim:
        def __init__(self, cfg):
            self.config = cfg
    shim = _ResolvedShim(config or {})
    # The repo's own code is the model's forward; the transformers module
    # for its model_type stands in only when the repo ships none.
    for label, fn in (
        ("custom_model", lambda: _try_custom_model(model_id)),
        # A pipeline's components each name their library's module; the
        # repo's own `.py` files (H3 ships its VAEs' reference code under
        # workflow folders) are not the pipeline the index loads.
        ("pipeline_components", lambda: _try_pipeline_sources(model_id, shim)),
        ("hf_modeling_files", lambda: _try_hf_modeling_files(model_id, shim)),
        ("local_transformers", lambda: _try_local_transformers(model_id, shim)),
        ("model_info_cache", lambda: _try_model_info_cache(model_id)),
    ):
        files = fn()
        if files and any(f.get("text") for f in files):
            return {
                "source": label,
                "paths": [f["path"] for f in files if f.get("path")],
                "files": files,
            }
    return {}


def _fetch_processor_raw(model_id: str) -> dict:
    """The processor configs the repo ships beside config.json, verbatim, by
    file name (`preprocessor_config.json`, `video_preprocessor_config.json`,
    `processor_config.json`); a file the repo does not ship is absent."""
    from huggingface_hub import HfApi, hf_hub_download
    shipped = set(HfApi().list_repo_files(model_id))
    out = {}
    for name in ("preprocessor_config.json", "video_preprocessor_config.json",
                 "processor_config.json"):
        if name not in shipped:
            continue
        with open(hf_hub_download(repo_id=model_id, filename=name), "r",
                  encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def _load_curated_ids() -> list:
    """Read supported_models.txt (one ID per line, # comments)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "arch", "supported_models.txt",
    )
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
    return ids


def snapshot_one(model_id: str, *, skip_if_exists: bool = False) -> str:
    from backend.arch.snapshots import snapshot_path, write_snapshot

    p = snapshot_path(model_id)
    if skip_if_exists and p.exists():
        return f"skip (exists): {model_id}"

    try:
        cfg = fetch_config_raw(model_id)
    except (OSError, ValueError, *HUB_UNREACHABLE) as e:
        return f"FAIL config: {model_id}  ({e.__class__.__name__}: {str(e)[:120]})"
    try:
        meta = fetch_metadata_raw(model_id, cfg)
    except (OSError, ValueError, *HUB_UNREACHABLE) as e:
        return f"FAIL metadata: {model_id}  ({e.__class__.__name__}: {str(e)[:120]})"
    if not meta:
        return f"FAIL metadata: {model_id}  (empty — checkpoint may lack safetensors)"

    # No snapshot without its forward source (`write_snapshot` refuses one):
    # a fetch that failed, or found no code anywhere, is a FAIL line like the
    # ones above rather than a stop of an `--all` run. The HF strategy
    # raises on every Hub failure.
    try:
        fs = _fetch_forward_source_raw(model_id, cfg)
    except Exception as e:
        return f"FAIL forward_source: {model_id}  ({e.__class__.__name__}: {str(e)[:120]})"
    from backend.arch.snapshots import has_forward_source
    if not has_forward_source(fs):
        return (f"FAIL forward_source: {model_id}  (no modeling code found: "
                f"not in the repo, and the installed transformers has no "
                f"module for model_type {cfg.get('model_type')!r} — upgrade "
                f"transformers or add the code, then refresh)")

    try:
        processor = _fetch_processor_raw(model_id)
    except (OSError, ValueError, *HUB_UNREACHABLE) as e:
        return f"FAIL processor: {model_id}  ({e.__class__.__name__}: {str(e)[:120]})"

    model_str = ""
    # A pipeline has no one module to print: its components are separate
    # models, and the skeleton loader builds a transformers causal LM.
    try:
        if "_components" in cfg:
            raise LookupError("pipeline: no single skeleton")
        from backend.hf.model_info import extract_model_info
        info = extract_model_info(model_id)
        model_str = info.get("model_str", "") or ""
    except Exception:
        # extract_model_info runs arbitrary modeling code written by
        # model authors; failure modes are unbounded (ValueError on
        # unknown model_type, AttributeError on missing config attrs,
        # KeyError, etc.). Skeleton is best-effort metadata for the
        # snapshot — config + safetensors above are the load-bearing
        # parts.
        pass

    from backend.arch.snapshots import release_dates_for
    out = write_snapshot(model_id, cfg, metadata_record(meta),
                         forward_source=fs,
                         model_str=model_str or None,
                         release_date=release_dates_for([model_id]).get(model_id),
                         processor=processor or None)
    n_groups = len(meta["header"].groups)
    n_fs_files = len(fs.get("files", []))
    size_kb = out.stat().st_size / 1024
    fs_str = f"  fs={n_fs_files}f({fs.get('source','?')})"
    ms_str = f"  ms={len(model_str)//1024}KB" if model_str else "  ms=-"
    return f"OK  {model_id:50s}  {n_groups:>4d} tensor groups{fs_str}{ms_str}  {size_kb:>6.1f} KB  → {out.name}"


def main() -> int:
    p = argparse.ArgumentParser(description="Snapshot HF model config + tensor metadata to git.")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("model_id", nargs="?", help="HF model id, e.g. meta-llama/Llama-3.1-8B")
    grp.add_argument("--all", action="store_true", help="Refresh every ID in supported_models.txt")
    grp.add_argument("--missing", action="store_true", help="Only snapshot IDs that don't have a file yet")
    args = p.parse_args()

    if args.all or args.missing:
        ids = _load_curated_ids()
        print(f"Snapshotting {len(ids)} curated IDs...")
        for i, mid in enumerate(ids, 1):
            print(f"[{i:>2}/{len(ids)}] ", end="", flush=True)
            print(snapshot_one(mid, skip_if_exists=args.missing))
            time.sleep(0.2)  # be polite to HF
    else:
        print(snapshot_one(args.model_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
