"""Bootstrap a shareable Gallery response cache from a versioned artifact.

The canonical SOL pool and this cache are different layers: the pool owns
immutable laws, while these are pre-materialized response entries. They are
deterministic for one engine/spec snapshot, and can be expensive to draw, so
a release may publish a partial cache (for example completed Gallery GPUs)
and let runtime warm extend it on the persistent volume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import urllib.request

from backend.code_trace.provider import TRACES, Trace
from backend.fingerprint import cache_engine_version
from backend.release_artifacts import artifact_to_pull


DEFAULT_BUCKET = "sol-pool"


def _cache_root() -> Path:
    return Path(os.environ.get(
        "HA_GALLERY_CACHE_DIR",
        str(Path.home() / ".cache/huggingarch/gallery")))


def _index_path() -> str:
    return f"gallery-cache-v1/{cache_engine_version()}/index.json"


def _request(url: str, key: str, timeout: float):
    request = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Accept": "application/octet-stream",
    })
    return urllib.request.urlopen(request, timeout=timeout)


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _safe_archive(archive: Path) -> int:
    """Accept only regular JSON files below ``gallery/``."""
    files = 0
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name.rstrip("/")
            if name != "gallery" and not name.startswith("gallery/"):
                raise ValueError(f"gallery cache archive member {name!r}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"gallery cache archive member is not a file: {name}")
            if not name.endswith(".json"):
                raise ValueError(
                    f"gallery cache archive member is not JSON: {name}")
            files += 1
        if not any(member.name.rstrip("/") == "gallery" for member in members):
            raise ValueError("gallery cache archive has no gallery/ root")
        if not files:
            raise ValueError("gallery cache archive has no JSON entries")
    return files


def _marker_matches(root: Path, artifact: str, digest: str) -> bool:
    try:
        marker = json.loads(
            (root / ".bootstrap.json").read_text(encoding="utf-8"))
        expected = int(marker["files"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    if marker.get("artifact") != artifact or marker.get("sha256") != digest:
        return False
    # Runtime warm may add more entries. A smaller tree means this artifact
    # still has files the volume lacks, so it must be downloaded and merged.
    entries = sum(1 for path in [
        *root.glob("*.json"), *root.glob("frontier/*.json")])
    return expected > 0 and entries >= expected


def _write_marker(root: Path, artifact: str, digest: str, files: int) -> None:
    marker = root / ".bootstrap.json"
    staging = root / f".bootstrap.json.{os.getpid()}.tmp"
    try:
        with staging.open("w", encoding="utf-8") as out:
            json.dump({"artifact": artifact, "sha256": digest, "files": files},
                      out, sort_keys=True)
            out.write("\n")
        os.replace(staging, marker)
    finally:
        staging.unlink(missing_ok=True)


def _install(archive: Path, root: Path, artifact: str, digest: str) -> int:
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gallery-cache.", dir=root.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        unpacked = staging / "gallery"
        files = [path for path in unpacked.rglob("*.json") if path.is_file()
                 and path.parent != unpacked / "code-traces"]
        if not files:
            raise ValueError("gallery cache archive has no JSON entries")
        # Measured views are machine-local: a new deterministic artifact must
        # not leave drawings from an older Kernel Bench snapshot in front of
        # the freshly merged entries.
        views = root / "frontier-views"
        if views.exists():
            for view in views.glob("*.json"):
                view.unlink(missing_ok=True)
        for source in files:
            relative = source.relative_to(unpacked)
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.tmp")
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        # The traces the entries name travel with them (`gallery/code-traces`,
        # `scripts.publish_gallery_cache.build`): an entry is served only
        # while the running code satisfies its trace, and this machine never
        # recorded those.
        for body in (unpacked / "code-traces").glob("*.json"):
            try:
                trace = Trace.from_json(json.loads(body.read_text(
                    encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                continue
            if trace.id == body.stem:
                TRACES.put(trace)
        _write_marker(root, artifact, digest, len(files))
        return len(files)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def pull(timeout: float = 900.0) -> int:
    from backend.supabase_env import configured, service_role
    if not configured():
        print("gallery cache artifact: Supabase is not configured",
              file=sys.stderr, flush=True)
        return 0
    base, key = service_role()
    bucket = os.environ.get("HA_ARTIFACT_BUCKET", DEFAULT_BUCKET)
    base = base.rstrip("/")
    selected = artifact_to_pull(
        os.environ.get("HA_GALLERY_CACHE_ARTIFACT", ""), _index_path(),
        cache_engine_version(), base, key, bucket, min(30.0, timeout), _request)
    if selected is None:
        print("gallery cache artifact: none for this engine; "
              "runtime warm will build it", flush=True)
        return 0
    artifact, expected = selected
    object_url = f"{base}/storage/v1/object/{bucket}/{artifact}"

    root = _cache_root()
    if _marker_matches(root, artifact, expected):
        print(f"gallery cache artifact: {root} already has {artifact}",
              flush=True)
        return 0

    root.parent.mkdir(parents=True, exist_ok=True)
    archive = root.parent / f".{root.name}.{os.getpid()}.tar.gz"
    try:
        print(f"gallery cache artifact: downloading {artifact}", flush=True)
        with _request(object_url, key, timeout) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        actual = _digest(archive)
        if actual != expected:
            raise ValueError(
                f"gallery cache artifact digest mismatch: {actual} != {expected}")
        files = _safe_archive(archive)
        installed = _install(archive, root, artifact, expected)
        print(f"gallery cache artifact: merged {installed}/{files} entries "
              f"into {root}", flush=True)
    finally:
        archive.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pull"])
    parser.add_argument("--timeout", type=float, default=900.0)
    arguments = parser.parse_args(argv)
    return pull(arguments.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
