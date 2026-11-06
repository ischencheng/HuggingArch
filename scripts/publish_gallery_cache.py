"""Publish deterministic Gallery response entries as an indexed artifact.

Measured views are deliberately excluded: they belong to a Kernel Bench
snapshot and are folded live. Model facts and frontier entries are the
expensive deterministic projections a new machine can safely reuse.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import urllib.request

from backend.code_trace.provider import TRACES
from backend.fingerprint import cache_engine_version
from backend.measure.cache import TRACE_FIELD
from backend.release_artifacts import INDEX_SCHEMA
from backend.supabase_env import configured, service_role


def _tar_info(name: str, size: int, *, directory: bool = False):
    info = tarfile.TarInfo(name)
    info.size = 0 if directory else size
    info.mode = 0o755 if directory else 0o644
    info.mtime = 0
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    return info


def build(root: Path, destination: Path) -> int:
    files = sorted(
        [*root.glob("*.json"), *root.glob("frontier/*.json")],
        key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"no shareable Gallery entries under {root}")
    with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(
            fileobj=zipped, mode="w") as tar:
        tar.addfile(_tar_info("gallery", 0, directory=True))
        tar.addfile(_tar_info("gallery/frontier", 0, directory=True))
        traces = set()
        for path in files:
            relative = path.relative_to(root)
            data = path.read_bytes()
            tar.addfile(_tar_info(
                f"gallery/{relative.as_posix()}", len(data)), io.BytesIO(data))
            try:
                tid = json.loads(data).get(TRACE_FIELD)
            except (ValueError, AttributeError):
                tid = None
            if tid:
                traces.add(tid)
        # Each trace an entry names, once: what a pulling machine checks the
        # entries against (`gallery_cache_artifact._install`).
        tar.addfile(_tar_info("gallery/code-traces", 0, directory=True))
        for tid in sorted(traces):
            trace = TRACES.get(tid)
            if trace is None:
                continue
            data = json.dumps(trace.to_json()).encode()
            tar.addfile(_tar_info(f"gallery/code-traces/{tid}.json", len(data)),
                        io.BytesIO(data))
    return len(files)


def _upload(url: str, key: str, data: bytes, content_type: str) -> None:
    request = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": content_type,
        "x-upsert": "true",
    })
    with urllib.request.urlopen(request, timeout=900) as response:
        response.read()


def publish(root: Path) -> str:
    if not configured():
        raise RuntimeError("Supabase is not configured")
    base, key = service_role()
    base = base.rstrip("/")
    bucket = os.environ.get("HA_ARTIFACT_BUCKET", "sol-pool")
    engine = cache_engine_version()
    with tempfile.TemporaryDirectory(prefix="gallery-cache-artifact.") as temporary:
        archive = Path(temporary) / "cache.tar.gz"
        files = build(root, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        path = f"gallery-cache-v1/{engine}/{digest}.tar.gz"
        base_url = f"{base}/storage/v1/object/{bucket}"
        _upload(f"{base_url}/{path}.sha256", key,
                f"{digest}\n".encode(), "text/plain")
        _upload(f"{base_url}/{path}", key, archive.read_bytes(),
                "application/octet-stream")
        index = {
            "schema": INDEX_SCHEMA,
            "engine_version": engine,
            "artifacts": [{
                "path": path,
                "sha256": digest,
                "files": files,
            }],
        }
        _upload(f"{base_url}/gallery-cache-v1/{engine}/index.json", key,
                json.dumps(index, sort_keys=True).encode() + b"\n",
                "application/json")
        return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(publish(arguments.cache_dir.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
