"""Bootstrap the local SOL pool from a versioned storage artifact.

The canonical pool is a set of immutable, content-addressed bundles.  A
machine therefore needs only one download per engine version: fetch the
compressed archive, verify its digest, and unpack it on the persistent volume.
Per-record remote reads belong in ``SolPool`` itself (the next step for this
layer); this module is the smaller startup boundary that lets a new machine
reuse a complete warm corpus without baking it into the image.
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

from backend.fingerprint import cache_engine_version
from backend.release_artifacts import artifact_to_pull


DEFAULT_BUCKET = "sol-pool"


def _pool_root() -> Path:
    return Path(os.environ.get(
        "HA_SOL_POOL_DIR", str(Path.home() / ".cache/huggingarch/sol-pool")))


def _index_path() -> str:
    from backend.inference.sol_data import walkRecordSchema
    return f"{walkRecordSchema}/{cache_engine_version()}/index.json"


def _request(url: str, key: str, timeout: float):
    request = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Accept": "application/octet-stream",
    })
    return urllib.request.urlopen(request, timeout=timeout)


def _download(url: str, key: str, destination: Path, timeout: float) -> None:
    with _request(url, key, timeout) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive(archive: Path) -> None:
    """Refuse an archive that is not exactly the canonical ``sol/`` tree."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            name = member.name.rstrip("/")
            if name == "sol" or name.startswith("sol/"):
                continue
            raise ValueError(f"SOL pool archive has non-pool member {name!r}")
        if not any(member.name.rstrip("/") == "sol" for member in members):
            raise ValueError("SOL pool archive has no sol/ root")


def _write_bootstrap(root: Path, artifact: str, digest: str,
                     manifest_count: int) -> None:
    marker = root / ".bootstrap.json"
    staging = root / f".bootstrap.json.{os.getpid()}.tmp"
    try:
        with staging.open("w", encoding="utf-8") as out:
            json.dump({
                "artifact": artifact,
                "sha256": digest,
                "manifests": manifest_count,
            }, out, sort_keys=True)
            out.write("\n")
        os.replace(staging, marker)
    finally:
        staging.unlink(missing_ok=True)


def _has_bootstrap(root: Path, artifact: str, digest: str) -> bool:
    """Whether ``root`` is exactly a previously installed artifact.

    Manifest presence alone cannot answer that question. A runtime warm that
    started before an artifact existed leaves valid immutable bundles behind,
    and that partial state must not make later starts skip the complete,
    digest-verified corpus.
    """
    try:
        marker = json.loads(
            (root / ".bootstrap.json").read_text(encoding="utf-8"))
        expected = int(marker["manifests"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    if marker.get("artifact") != artifact or marker.get("sha256") != digest:
        return False
    return expected > 0 and len(list(root.glob("*/*/manifest.json"))) == expected


def _unpack(archive: Path, root: Path, artifact: str, digest: str) -> None:
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(staging, filter="data")
        unpacked = staging / "sol"
        manifests = list(unpacked.glob("*/*/manifest.json"))
        if not manifests:
            raise ValueError("SOL pool archive has no manifests")
        if root.exists() or root.is_symlink():
            if root.is_symlink():
                raise RuntimeError(f"SOL pool path {root} is a symlink")
            backup = Path(tempfile.mkdtemp(
                prefix=f".{root.name}.previous.", dir=root.parent))
            backup.rmdir()
            root.rename(backup)
        else:
            backup = None
        try:
            unpacked.rename(root)
        except BaseException:
            if backup is not None and not root.exists():
                backup.rename(root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        _write_bootstrap(root, artifact, digest, len(manifests))
        print(f"SOL pool artifact: unpacked {len(manifests)} bundles to {root}",
              flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def pull(timeout: float = 900.0) -> int:
    """Fetch, verify and unpack the configured artifact; 0 on success/no-op."""
    from backend.supabase_env import configured, service_role
    if not configured():
        print("SOL pool artifact: Supabase is not configured",
              file=sys.stderr, flush=True)
        return 0
    base, key = service_role()
    base = base.rstrip("/")
    bucket = os.environ.get("HA_ARTIFACT_BUCKET",
                            os.environ.get("HA_SOL_POOL_BUCKET", DEFAULT_BUCKET))
    selected = artifact_to_pull(
        os.environ.get("HA_SOL_POOL_ARTIFACT", ""), _index_path(),
        cache_engine_version(), base, key, bucket, min(30.0, timeout), _request)
    if selected is None:
        print("SOL pool artifact: none for this engine; "
              "runtime warm will build it", flush=True)
        return 0
    artifact, expected = selected
    object_url = f"{base}/storage/v1/object/{bucket}/{artifact}"

    root = _pool_root()
    if _has_bootstrap(root, artifact, expected):
        print(f"SOL pool artifact: {root} already has this complete corpus",
              flush=True)
        return 0

    archive = root.parent / f".{root.name}.{os.getpid()}.tar.gz"
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"SOL pool artifact: downloading {artifact}", flush=True)
        _download(object_url, key, archive, timeout)
        actual = _digest(archive)
        if actual != expected:
            raise ValueError(
                f"SOL pool artifact digest mismatch: {actual} != {expected}")
        _safe_archive(archive)
        _unpack(archive, root, artifact, expected)
    finally:
        archive.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pull"])
    parser.add_argument("--timeout", type=float, default=900.0)
    arguments = parser.parse_args(argv)
    if arguments.command == "pull":
        return pull(arguments.timeout)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
