"""Which cached results an edit would make stale — before deploying it.

Every cached result names the code it was computed by (its trace,
`backend.code_trace.provider`): each function it executed, by the hash of
its normalized source, and one `<import-time>` item for what ran at import.
This tool hashes every `backend/` def at two revisions (a commit, or the
working tree), takes the items whose hash moved, and reads the stored traces
against them:

  * the SOL pool's records (`HA_SOL_POOL_DIR`): one line per model × card
    class × layout that would be walked again;
  * the result caches on disk (the Scaling sweep's, the Gallery's): one line
    per entry, labelled with what the entry says about itself.

A trace goes stale when an item it names moved, or when its import-time
trace does — any item of that one (a module's top level, a def added or
removed, rule 3's methods, `uv.lock`, the recording machinery) moves every
trace. What it cannot see is code that would *newly* run at import; a def
added anywhere moves a skeleton, which it does see.

    .venv/bin/python -m tools.code_trace_dryrun --from HEAD~1 --to HEAD
    .venv/bin/python -m tools.code_trace_dryrun            # HEAD → working tree
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess
import sys
from typing import Dict, Iterator, List, Optional, Set, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.code_trace.provider import (
    IMPORT_ITEM, LOCK_ITEM, TRACES, Trace)
from backend.code_trace.source import FileIdentity, digest
from backend.measure.cache import TRACE_FIELD


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def _sources(rev: Optional[str]) -> Iterator[Tuple[str, str]]:
    """(path under backend/, source) of every backend `.py` at `rev` (None:
    the working tree)."""
    if rev is None:
        for path in sorted((ROOT / "backend").rglob("*.py")):
            yield (path.relative_to(ROOT / "backend").as_posix(),
                   path.read_text(encoding="utf-8"))
        return
    for name in _git("ls-tree", "-r", "--name-only", rev, "--", "backend").split():
        if name.endswith(".py"):
            yield name[len("backend/"):], _git("show", f"{rev}:{name}")


def hash_table(rev: Optional[str]) -> Dict[str, str]:
    """Every item a trace can name, at `rev`: each def and each skeleton
    (`FileIdentity`), and `uv.lock`."""
    table: Dict[str, str] = {}
    for rel, src in _sources(rev):
        table.update(FileIdentity(rel, src).items())
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8") if rev is None \
        else _git("show", f"{rev}:uv.lock")
    table[LOCK_ITEM] = digest(lock)
    return table


def moved(before: Dict[str, str], after: Dict[str, str]) -> Set[str]:
    return {k for k in before.keys() | after.keys()
            if before.get(k) != after.get(k)}


class Verdicts:
    """Whether each stored trace survives the edit, by id."""

    def __init__(self, changed: Set[str]) -> None:
        self.changed = changed
        self._known: Dict[str, Optional[Tuple[str, ...]]] = {}

    def stale_by(self, tid: str) -> Optional[Tuple[str, ...]]:
        """The moved items that make trace `tid` stale; () when it holds;
        None when its body is not kept here."""
        if tid not in self._known:
            trace = TRACES.get(tid)
            self._known[tid] = None if trace is None else self._of(trace)
        return self._known[tid]

    def _of(self, trace: Trace) -> Tuple[str, ...]:
        hits = sorted(k for k, _h in trace.items
                      if k != IMPORT_ITEM and k in self.changed)
        for k, h in trace.items:
            if k == IMPORT_ITEM:
                shared = self.stale_by(h)
                if shared is None:
                    hits.append("<import-time: body not kept>")
                elif shared:
                    hits.append(f"<import-time: {len(shared)} moved>")
        return tuple(hits)


def _pool_records(pool: pathlib.Path) -> Iterator[Tuple[str, str]]:
    """(label, trace id) of every SOL pool record."""
    for manifest_path in sorted(pool.glob("*/*/manifest.json")):
        bundle = manifest_path.parent
        if "." in bundle.name:
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index = json.loads((bundle / "index.json").read_text(encoding="utf-8"))
            bodies = json.loads((bundle / "traces.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for body in bodies.values():
            TRACES.put(Trace.from_json(body))
        content = (manifest.get("identity") or {}).get("content") or {}
        where = " ".join(f"{k}={content[k]}" for k in ("model", "card_class")
                         if isinstance(content, dict) and k in content)
        for record_id, tid in (index.get("traces") or {}).items():
            yield f"sol-pool {where or bundle.name} record {record_id[:12]}", tid


def _cache_entries(directory: pathlib.Path) -> Iterator[Tuple[str, str]]:
    """(label, trace id) of every entry of a result cache on disk."""
    for path in sorted(directory.rglob("*.json")):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(stored, dict) or TRACE_FIELD not in stored:
            continue
        value = stored.get("value")
        said = ""
        if isinstance(value, dict):
            said = " ".join(f"{k}={value[k]}" for k in ("model_id", "mode")
                            if isinstance(value.get(k), str))
        yield (f"{directory.name}/{path.relative_to(directory).as_posix()[:20]} "
               f"{said}".rstrip(), stored[TRACE_FIELD])


def main(argv: Optional[List[str]] = None) -> int:
    home = pathlib.Path.home() / ".cache" / "huggingarch"
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from", dest="before", default="HEAD")
    ap.add_argument("--to", dest="after", default=None,
                    help="a revision; default the working tree")
    ap.add_argument("--pool", default=os.environ.get(
        "HA_SOL_POOL_DIR", str(home / "sol-pool")))
    ap.add_argument("--cache", action="append", default=None,
                    help="a result cache directory (repeatable); default the "
                         "sweep and gallery caches")
    ap.add_argument("--list", action="store_true",
                    help="every stale entry, not only the counts")
    args = ap.parse_args(argv)
    changed = moved(hash_table(args.before), hash_table(args.after))
    print(f"{len(changed)} items moved between {args.before} and "
          f"{args.after or 'the working tree'}")
    for item in sorted(changed)[:30]:
        print(f"  {item}")
    verdicts = Verdicts(changed)
    caches = args.cache or [os.environ.get("HA_SWEEP_CACHE_DIR",
                                           str(home / "sweep")),
                            os.environ.get("HA_GALLERY_CACHE_DIR",
                                           str(home / "gallery"))]
    sources = [("SOL pool", _pool_records(pathlib.Path(args.pool)))]
    sources += [(c, _cache_entries(pathlib.Path(c))) for c in caches]
    for name, entries in sources:
        counts: collections.Counter = collections.Counter()
        why: collections.Counter = collections.Counter()
        stale: List[str] = []
        for label, tid in entries:
            verdict = verdicts.stale_by(tid)
            state = ("unknown" if verdict is None
                     else "stale" if verdict else "valid")
            counts[state] += 1
            if verdict:
                why.update(verdict)
                stale.append(label)
        print(f"{name}: {dict(counts)}")
        for item, n in why.most_common(10):
            print(f"  {n:6d} × {item}")
        if args.list:
            for label in stale:
                print(f"  stale: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
