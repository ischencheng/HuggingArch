"""Publish the complete canonical SOL pool through its engine-version index."""
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

from backend.fingerprint import cache_engine_version
from backend.inference.sol_data import walkRecordSchema
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
    manifests = sorted(root.glob("*/*/manifest.json"))
    if not manifests:
        raise ValueError(f"no SOL bundles under {root}")
    # A bundle being written lives beside its key until it is swapped in
    # (`SolPool._swapIn`: `<key>.<pid>.<thread>.new` / `.old`, the writer's
    # lock `<key>.writer`); a key itself has no dot. Only bundles ship.
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {
            ".bootstrap.json", ".writer"}
        and "." not in path.relative_to(root).parts[1]
    )
    with destination.open("wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0) as zipped, tarfile.open(
            fileobj=zipped, mode="w") as tar:
        tar.addfile(_tar_info("sol", 0, directory=True))
        directories = {"sol"}
        for path in files:
            relative = path.relative_to(root)
            parts = relative.parts
            for depth in range(1, len(parts)):
                directory_name = "sol/" + "/".join(parts[:depth])
                if directory_name not in directories:
                    directories.add(directory_name)
                    tar.addfile(_tar_info(directory_name, 0, directory=True))
            data = path.read_bytes()
            tar.addfile(_tar_info(f"sol/{relative.as_posix()}", len(data)),
                        io.BytesIO(data))
    return len(manifests)


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
    with tempfile.TemporaryDirectory(prefix="sol-pool-artifact.") as temporary:
        archive = Path(temporary) / "pool.tar.gz"
        manifests = build(root, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        path = f"{walkRecordSchema}/{engine}/{digest}.tar.gz"
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
                "manifests": manifests,
            }],
        }
        _upload(f"{base_url}/{walkRecordSchema}/{engine}/index.json", key,
                json.dumps(index, sort_keys=True).encode() + b"\n",
                "application/json")
        return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(publish(arguments.pool_dir.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
