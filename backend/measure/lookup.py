"""Per op, exact: the key form of a physical op, the client of the Kernel
Bench service's lookup, and the coverage question the kernel_bench harness
asks (docs/measured_calibration.md §7.2; the package docstring maps the
rest).

The Kernel Bench service (`python -m backend.service.bench.server`, bound to
127.0.0.1) is the one process that opens the lake. An estimate does not copy
the lake: it asks the service for the candidate rows of the physical ops it
prices, in one batch, at one pinned lake snapshot.

The service is configured by the environment the API and its compute workers
share: `HUGGINGARCH_BENCH_URL` (e.g. http://127.0.0.1:8100) and the shared
secret (`HUGGINGARCH_BENCH_SECRET`, or a 0600 file named by
`HUGGINGARCH_BENCH_SECRET_FILE`) every request carries in `X-HA-Internal`.
A harness that contributes its rows under an account names it in
`HUGGINGARCH_BENCH_OWNER` (the account's uuid), so its coverage question
counts that account's own rows too (`covered`).

A key is anything with the five fields of `op_calibration.PhysicalKey`
(`signature`, `shape_json` — the projected shape —, `dtype_env_json`, `gpu`,
`phase`); `WireKey` is its wire form — the one declaration of the key's
fields the client sends (`key_json`), the service's request model validates
and its query unnests. This module imports no estimator
code: the estimator imports it.
"""
from __future__ import annotations

import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Sequence, Set, Tuple, TypedDict

import httpx

from .scope import NONE, MeasurementScope

#: The header carrying the shared secret on every request to the service.
INTERNAL_HEADER = "X-HA-Internal"
#: The headers carrying the caller's identity, set by the API only.
USER_HEADER = "X-HA-User"
ADMIN_HEADER = "X-HA-Admin"

class WireKey(TypedDict):
    """One lookup key on the wire, in `PhysicalKey` order (`op_calibration`):
    the measurement signature, the shape projected to its identifying axes,
    the dtype environment, the card and the phase. The service's request
    model (`routes.LookupBody`) validates against it and its query
    (`service.bench.lookup`) unnests its fields."""
    signature: str
    physical_shape: str
    dtype_env_json: str
    gpu: str
    phase: str


class Key(NamedTuple):
    """One lookup key as plain data: `op_calibration.PhysicalKey`'s five
    fields in its order, so a key crosses a process boundary, a cache entry
    or a JSON payload as a tuple of strings and is read back exactly
    (`PhysicalKey(*key)`, `Key(**dataclasses.asdict(physical))`)."""
    signature: str
    shape_json: str
    dtype_env_json: str
    gpu: str
    phase: str


def secret() -> str:
    """The shared secret: `HUGGINGARCH_BENCH_SECRET`, else the content of the
    file `HUGGINGARCH_BENCH_SECRET_FILE` names; "" when neither is set. A
    file named but unreadable is a deployment error, raised as such
    (`RuntimeError`), never an empty secret."""
    value = os.environ.get("HUGGINGARCH_BENCH_SECRET", "").strip()
    if value:
        return value
    path = os.environ.get("HUGGINGARCH_BENCH_SECRET_FILE", "").strip()
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError as e:
        raise RuntimeError(f"HUGGINGARCH_BENCH_SECRET_FILE ({path}) cannot be read: "
                           f"{type(e).__name__}: {e}") from e


def service_url() -> str:
    """The service's base URL, "" when no service is configured."""
    return os.environ.get("HUGGINGARCH_BENCH_URL", "").strip().rstrip("/")


def key_json(key) -> WireKey:
    """A lookup key (`op_calibration.PhysicalKey`) as the service reads it."""
    return WireKey(signature=key.signature, physical_shape=key.shape_json,
                   dtype_env_json=key.dtype_env_json, gpu=key.gpu, phase=key.phase)


# ---------------------------------------------------------------------------
# The client: three outcomes, never confused.
#
#   * no service configured (`HUGGINGARCH_BENCH_URL` unset) — no measured
#     rows: the scope is `NONE`, the estimate is SOL and says so;
#   * a configured service answering — measured where it has rows (an empty
#     answer is a normal answer, cached like any other);
#   * a configured service failing, timing out (5 s, plus the time its size
#     allows: `lookup_timeout`) or answering something malformed — `MeasurementsUnavailable`: the estimate fails with 503
#     "measurements unavailable" and nothing is cached.
# ---------------------------------------------------------------------------

#: Seconds a lookup or a version request may take.
TIMEOUT_S = 5.0

#: Seconds a lookup may take per key it sends, beside its floor. The service's
#: work grows with the keys asked: a Kimi-K3 parallel sweep's one lookup sends
#: 5,505 keys and took 3.2 s on a cold service, 1.0 s warm (2026-10-31) —
#: ~0.6 ms a key at worst, so 2 ms leaves the busy service headroom. An
#: estimate's few hundred keys keep about the floor; a sweep's thousands are
#: not failed as "measurements unavailable" for being a sweep.
TIMEOUT_S_PER_KEY = 0.002

#: Seconds the harness's coverage question may take per batch. Not the
#: estimate's budget: a sweep asks thousands of keys at once, from a bench box
#: through a tunnel, before any GPU work — and a coverage answer that times
#: out turns the skip off for the whole run (27 of 65 env starts on the H100,
#: 2026-10-24), which re-measures every cell the lake already has.
COVERAGE_TIMEOUT_S = 120.0

#: Tests route the client to the service in process.
transport: Optional[Any] = None
_clients: dict = {}
_clients_lock = threading.Lock()


class MeasurementsUnavailable(Exception):
    """The configured Kernel Bench service did not answer (down, slow, an
    error, a malformed answer): the estimate cannot say what was measured.
    A 503 the client may retry; nothing is cached."""

    reason = "measurements_unavailable"


def _client() -> "httpx.Client":
    """One keep-alive client per process (compute workers each have theirs)."""
    pid = os.getpid()
    with _clients_lock:
        client = _clients.get(pid)
        if client is None:
            # Never through an environment proxy (http_proxy / HTTPS_PROXY):
            # the service is on loopback and every request carries the
            # shared secret.
            client = _clients[pid] = httpx.Client(
                transport=transport, timeout=TIMEOUT_S, follow_redirects=False,
                trust_env=False)
        return client


def _call(method: str, path: str, *, timed: bool = False, **kw: Any):
    url = service_url()
    started = time.perf_counter()
    try:
        r = _client().request(method, url + path,
                              headers={INTERNAL_HEADER: secret()}, **kw)
    except httpx.HTTPError as e:
        raise MeasurementsUnavailable(
            f"measurements unavailable: the Kernel Bench service did not answer "
            f"({type(e).__name__})") from None
    if r.status_code != 200:
        try:
            detail = str(r.json().get("detail") or "")[:200]
        except ValueError:
            detail = "(no JSON body)"
        raise MeasurementsUnavailable(
            f"measurements unavailable: the Kernel Bench service answered "
            f"{r.status_code}{': ' + detail if detail else ''}")
    try:
        value = r.json()
    except ValueError:
        raise MeasurementsUnavailable(
            "measurements unavailable: the Kernel Bench service answered no JSON") from None
    return (value, time.perf_counter() - started) if timed else value


def resolve_scope(*, reads: bool, owner: Optional[str]) -> MeasurementScope:
    """The scope an estimate is computed and cached under. `reads`: whether
    its mode reads measurements (`theoretical_only` does not: `NONE`, no
    request). No service configured: `NONE`. Otherwise one
    `GET /internal/bench/version` names the lake, the snapshot the lookup
    will be pinned at, the shared rows' snapshot and the rules' revision."""
    if not reads or not service_url():
        return NONE
    version = _call("GET", "/internal/bench/version")
    try:
        return MeasurementScope.from_version(version, owner)
    except (KeyError, TypeError, ValueError):
        raise MeasurementsUnavailable(
            "measurements unavailable: the Kernel Bench service's version is "
            "malformed") from None


@dataclass(frozen=True)
class Lookup:
    """One lookup's answer: `rows[i]` the candidate rows of `keys[i]` (the
    registry's cells as dicts, each with its `origin`), and how many of the
    owner's rows the current validator refused among those fetched now."""
    keys: tuple
    rows: tuple
    excluded_mine: int
    request_s: float = 0.0
    service_s: float = 0.0


#: The rows of keys already looked up in this process, by the scope's cache
#: identity and the key. Exact, not a heuristic: an owner's scope names its
#: pinned snapshot, and a shared scope names the snapshot its shared rows
#: last changed at (every pin with that `shared_snapshot` answers the same
#: rows, `MeasurementScope.key`), so an entry can never be stale for its key.
#: Bounded by rows held (each a dict of the registry's 22 cells, the
#: strings shared with the response's distinct texts): an evicted key is
#: asked again.
ROWS_KEPT = 50_000
_rows: "OrderedDict[tuple, tuple]" = OrderedDict()
_rows_held = 0
_rows_lock = threading.Lock()


def forget_rows() -> None:
    """Drop every held row (tests; a process that wants its memory back)."""
    global _rows_held
    with _rows_lock:
        _rows.clear()
        _rows_held = 0


def _held(scope: MeasurementScope, keys: Sequence[Any]) -> dict:
    with _rows_lock:
        out = {}
        for k in keys:
            got = _rows.get((scope.key, k))
            if got is not None:
                _rows.move_to_end((scope.key, k))
                out[k] = got
        return out


def _hold(scope: MeasurementScope, found: dict) -> None:
    global _rows_held
    with _rows_lock:
        for k, rows in found.items():
            old = _rows.pop((scope.key, k), None)
            _rows_held -= len(old or ())
            _rows[(scope.key, k)] = rows
            _rows_held += len(rows)
        while _rows_held > ROWS_KEPT and _rows:
            _key, rows = _rows.popitem(last=False)
            _rows_held -= len(rows)


def lookup_timeout(n_keys: int, floor: float = TIMEOUT_S) -> float:
    """The seconds a lookup sending `n_keys` keys may take: `floor` (a hung
    service still fails fast) plus `TIMEOUT_S_PER_KEY` for each key."""
    return floor + n_keys * TIMEOUT_S_PER_KEY


def lookup(scope: MeasurementScope, keys: Sequence[Any], *,
           timeout: float = TIMEOUT_S) -> Lookup:
    """The candidate rows of each key (`op_calibration.PhysicalKey`) under
    `scope`: the keys this process has not looked up under the scope's
    identity yet go in one request pinned at `scope.snapshot`; the rest are
    the rows held (`_rows`). `NONE` asks nothing and answers nothing.
    `timeout` is the floor; the request may take what its size allows
    beside it (`lookup_timeout`)."""
    keys = tuple(dict.fromkeys(keys))
    if not scope.reads or not keys:
        return Lookup(keys, tuple(() for _ in keys), 0)
    held = _held(scope, keys)
    missing = [k for k in keys if k not in held]
    excluded = 0
    if missing:
        started = time.perf_counter()
        out, request_s = _call("POST", "/internal/bench/lookup", timed=True, json={
            "keys": [key_json(k) for k in missing], "owner": scope.owner,
            "snapshot": scope.snapshot}, timeout=lookup_timeout(len(missing), timeout))
        service_s = float(out.get("service_s") or 0.0)
        try:
            if (str(out["lake_id"]) != scope.lake_id or int(out["snapshot"]) != scope.snapshot
                    or len(out["rows"]) != len(missing)):
                raise ValueError("not the scope asked for")
            columns, values = list(out["columns"]), list(out["values"])
            fetched = {k: tuple(dict(zip(columns, (values[i] for i in cells), strict=True))
                                for cells in per_key)
                       for k, per_key in zip(missing, out["rows"], strict=True)}
            excluded = int(out["excluded_mine"])
        except (KeyError, TypeError, ValueError, IndexError) as e:
            raise MeasurementsUnavailable(
                f"measurements unavailable: the Kernel Bench service's lookup is "
                f"malformed ({e})") from None
        _hold(scope, fetched)
        held.update(fetched)
    return Lookup(keys, tuple(held[k] for k in keys), excluded,
                  request_s=(time.perf_counter() - started) if missing else 0.0,
                  service_s=service_s if missing else 0.0)


def covered(keys: Sequence[Any]) -> Dict[Any, Set[Tuple[str, str, str]]]:
    """The coverage question the kernel_bench harness asks: per key, the
    implementations whose rows already measure it —
    `(framework, framework_version, kernel_impl)` — through the same lookup
    (empty with no service configured).

    Whose rows: the shared ones, and the contributing account's own when the
    harness names it (`HUGGINGARCH_BENCH_OWNER`) — the scope a signed-in
    estimate reads. Shared rows alone left out everything a box had measured
    but not yet published, so a re-planned sweep measured it again: 82% of
    the H100's new rows on 2026-10-24 (63,743 of 77,286) repeated a cell
    and implementation the same account already had."""
    owner = os.environ.get("HUGGINGARCH_BENCH_OWNER", "").strip() or None
    scope = resolve_scope(reads=True, owner=owner)
    found = lookup(scope, keys, timeout=COVERAGE_TIMEOUT_S)
    return {k: {(r["framework"], r["framework_version"], r["kernel_impl"]) for r in rows}
            for k, rows in zip(found.keys, found.rows, strict=True)}
