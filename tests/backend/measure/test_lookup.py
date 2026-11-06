"""The Kernel Bench lookup client (`backend.measure.lookup`,
docs/measured_calibration.md §7.3): three outcomes never confused — no
service, an answer (empty is an answer), a failure — and the registry built
from a lookup: each row under the key that selected it, the owner's over the
shared, today's lookups unchanged."""
from __future__ import annotations

import json
import os

import httpx
import pytest

import tests.corpus_row as _corpus
from backend.measure import lookup as measurements
from backend.measure.lookup import MeasurementsUnavailable
from backend.measure.scope import NONE, MeasurementScope
from backend.inference.op_calibration import (
    BenchmarkKey,
    PhysicalKey,
    latest_time_for,
    merge_validated_rows,
    physical_time_for,
    registry_from_lookup,
    CalibrationRegistry,
    ROW_COLUMNS,
    registry_for,
)
from backend.inference import benchmarks as _benchmarks

SCOPE = MeasurementScope(owner=None, lake_id="lake-1", snapshot=9, shared_snapshot=4,
                         rules_rev="r")
_PEAKS = dict(current_peak_tflops_at=lambda dtype: 989.0, current_peak_bw_gbps=3350.0)


def _row(**over) -> dict:
    return _benchmarks.validate_text(_corpus.to_csv([_corpus.row(**over)]), source="t")[1][0]


def _key(row: dict) -> PhysicalKey:
    return BenchmarkKey.from_csv_row(row).to_physical()


@pytest.fixture(autouse=True)
def _no_rows_held(monkeypatch):
    measurements.forget_rows()


@pytest.fixture
def service(monkeypatch):
    """A stand-in service: `answer(request) -> httpx.Response`; records
    the requests."""
    state = {"answer": None, "seen": []}

    def handler(request):
        state["seen"].append(request)
        return state["answer"](request)
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    monkeypatch.setenv("HUGGINGARCH_BENCH_SECRET", "s")
    monkeypatch.setattr(measurements, "_clients", {
        os.getpid(): httpx.Client(transport=httpx.MockTransport(handler))})
    return state


def _answer(keys, rows_per_key, *, lake="lake-1", snapshot=9, excluded=0):
    """The service's compact answer: the registry's columns, each cell an
    index into the distinct texts."""
    cols = [*ROW_COLUMNS, "origin"]
    values: dict = {}
    rows = [[[values.setdefault(r[c], len(values)) for c in cols] for r in per]
            for per in rows_per_key]
    return httpx.Response(200, json={
        "lake_id": lake, "snapshot": snapshot, "columns": cols, "values": list(values),
        "excluded_mine": excluded, "rows": rows})


class TestOutcomes:
    def test_no_service_reads_nothing_and_asks_nothing(self, monkeypatch):
        monkeypatch.delenv("HUGGINGARCH_BENCH_URL", raising=False)
        assert measurements.resolve_scope(reads=True, owner="o") is NONE
        assert len(registry_for(NONE, [_key(_row())])) == 0


def test_a_lookup_splits_service_time_from_the_round_trip(service):
    """The fold's timing vocabulary splits what the service measured from
    HTTP transport; a cache hit asks for neither."""
    key = _key(_row())
    service["answer"] = lambda _request: httpx.Response(200, json={
        **_answer([key], [[]]).json(), "service_s": 0.0001})
    got = measurements.lookup(SCOPE, [key])
    assert got.service_s == 0.0001
    assert got.request_s >= got.service_s
    assert measurements.lookup(SCOPE, [key]).request_s == 0.0

    def test_theory_asks_nothing(self, service):
        assert measurements.resolve_scope(reads=False, owner=None) is NONE
        assert service["seen"] == []

    def test_the_version_names_the_scope(self, service):
        service["answer"] = lambda r: httpx.Response(200, json={
            "lake_id": "lake-1", "snapshot": 9, "shared_snapshot": 4, "rules_rev": "r",
            "snapshot_content": "t9", "shared_snapshot_content": "t4"})
        got = measurements.resolve_scope(reads=True, owner="alice")
        assert got == MeasurementScope("alice", "lake-1", 9, 4, "r", "t9", "t4")
        (req,) = service["seen"]
        assert req.url.path == "/internal/bench/version"
        assert req.headers["x-ha-internal"] == "s" and "x-ha-user" not in req.headers

    def test_an_empty_answer_is_an_answer(self, service):
        k = _key(_row())
        service["answer"] = lambda r: _answer([k], [[]])
        assert len(registry_for(SCOPE, [k])) == 0
        body = json.loads(service["seen"][0].content)
        assert body == {"keys": [measurements.key_json(k)], "owner": None, "snapshot": 9}

    @pytest.mark.parametrize("answer", [
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
        lambda r: httpx.Response(503, json={"detail": "Kernel Bench store unavailable: x"}),
        lambda r: httpx.Response(500, json={"detail": "shared row fails the validator"}),
        lambda r: httpx.Response(200, text="not json"),
        lambda r: httpx.Response(200, json={"rows": []}),
        lambda r: _answer([None], [[]], lake="another-lake"),
        lambda r: _answer([None], [[]], snapshot=10),
        lambda r: _answer([None], [[], []]),
    ])
    def test_a_failure_is_unavailable_never_empty(self, service, answer):
        service["answer"] = answer
        with pytest.raises(MeasurementsUnavailable, match="measurements unavailable"):
            measurements.lookup(SCOPE, [_key(_row())])

    def test_one_request_for_the_union_of_keys(self, service):
        a, b = _row(), _row(gpu="H200")
        service["answer"] = lambda r: _answer(None, [[{**a, "origin": "shared"}], []])
        reg = registry_for(SCOPE, [_key(a), _key(b), _key(a)])
        assert len(service["seen"]) == 1 and len(reg) == 1
        assert len(json.loads(service["seen"][0].content)["keys"]) == 2

    def test_held_rows_are_bounded_by_rows(self, service, monkeypatch):
        a = _row()
        service["answer"] = lambda r: _answer(None, [[{**a, "origin": "shared"}] * 3])
        monkeypatch.setattr(measurements, "ROWS_KEPT", 4)
        k1 = _key(a)
        k2 = PhysicalKey(k1.signature, k1.shape_json, k1.dtype_env_json, "H200", k1.phase)
        measurements.lookup(SCOPE, [k1])
        measurements.lookup(SCOPE, [k2])          # 6 rows > 4: the first key goes
        assert measurements._rows_held == 3 and len(measurements._rows) == 1
        measurements.lookup(SCOPE, [k1])
        assert len(service["seen"]) == 3

    def test_keys_looked_up_under_a_scope_are_not_asked_again(self, service):
        """The rows of a key are held by the scope's cache identity: a later
        estimate under the same shared rows (another pin, the same
        `shared_snapshot`) asks only for the keys it has not seen; another
        shared revision or another owner asks again."""
        a, b = _row(), _row(gpu="H200")

        def answer(request):
            body = json.loads(request.content)
            return _answer(None, [[{**a, "origin": "shared"}] if k["gpu"] == a["gpu"] else []
                                  for k in body["keys"]], snapshot=body["snapshot"])
        service["answer"] = answer
        measurements.lookup(SCOPE, [_key(a)])
        later_pin = MeasurementScope(None, "lake-1", 12, 4, "r")
        got = measurements.lookup(later_pin, [_key(a), _key(b)])
        assert [len(r) for r in got.rows] == [1, 0]
        assert [len(json.loads(r.content)["keys"]) for r in service["seen"]] == [1, 1]
        measurements.lookup(later_pin, [_key(a), _key(b)])
        assert len(service["seen"]) == 2
        for other in (MeasurementScope(None, "lake-1", 13, 13, "r"),
                      MeasurementScope("alice", "lake-1", 12, 4, "r")):
            measurements.lookup(other, [_key(a)])
        assert len(service["seen"]) == 4


def test_coverage_is_the_shared_rows_implementations(service):
    """The kernel_bench harness's question: which implementations already
    measure each key — the shared rows, through the same lookup."""
    a, b = _row(framework="vllm", kernel_impl="fa3"), _row(gpu="H200")

    def answer(request):
        if request.url.path.endswith("/version"):
            return httpx.Response(200, json={"lake_id": "lake-1", "snapshot": 9,
                                             "shared_snapshot": 4, "rules_rev": "r",
                                             "snapshot_content": "t9", "shared_snapshot_content": "t4"})
        assert json.loads(request.content)["owner"] is None
        return _answer(None, [[{**a, "origin": "shared"}], []])
    service["answer"] = answer
    got = measurements.covered([_key(a), _key(b)])
    assert got == {_key(a): {("vllm", a["framework_version"], "fa3")}, _key(b): set()}


def test_coverage_counts_the_contributing_accounts_own_rows(service, monkeypatch):
    """A harness that names the account it contributes under
    (`HUGGINGARCH_BENCH_OWNER`) asks under that owner's scope — its own
    private and nominated rows count as measured, as they do for a
    signed-in estimate — and with the harness's longer budget, not the
    estimate's: shared rows alone sent a re-planned sweep back over 82% of
    the cells it had already measured."""
    a = _row(framework="vllm", kernel_impl="fa3")
    owner = "07a6e2a7-54b4-4e52-ba0d-21ce44f06b09"
    monkeypatch.setenv("HUGGINGARCH_BENCH_OWNER", owner)
    asked = []

    def answer(request):
        if request.url.path.endswith("/version"):
            return httpx.Response(200, json={"lake_id": "lake-1", "snapshot": 9,
                                             "shared_snapshot": 4, "rules_rev": "r",
                                             "snapshot_content": "t9", "shared_snapshot_content": "t4"})
        asked.append((json.loads(request.content)["owner"],
                      request.extensions["timeout"]["read"]))
        return _answer(owner, [[{**a, "origin": "mine"}]])
    service["answer"] = answer
    got = measurements.covered([_key(a)])
    assert got == {_key(a): {("vllm", a["framework_version"], "fa3")}}
    assert asked == [(owner, measurements.lookup_timeout(1, measurements.COVERAGE_TIMEOUT_S))]
    assert measurements.COVERAGE_TIMEOUT_S > measurements.TIMEOUT_S


def test_a_lookup_may_take_the_time_its_size_needs(service):
    """The read budget grows with the keys a lookup sends: an estimate's few
    keys keep about the 5 s floor, a sweep's thousands get the seconds the
    service needs for them (a Kimi-K3 parallel sweep sends 5,505 and took
    3.2 s cold) instead of failing as "measurements unavailable"."""
    asked = []

    def answer(request):
        if request.url.path.endswith("/version"):
            return httpx.Response(200, json={"lake_id": "lake-1", "snapshot": 9,
                                             "shared_snapshot": 4, "rules_rev": "r",
                                             "snapshot_content": "t9", "shared_snapshot_content": "t4"})
        sent = json.loads(request.content)["keys"]
        asked.append((len(sent), request.extensions["timeout"]["read"]))
        return _answer(None, [[] for _ in sent])
    service["answer"] = answer
    scope = measurements.resolve_scope(reads=True, owner=None)
    base = _row()
    keys = [PhysicalKey(signature=_key(base).signature,
                        shape_json=json.dumps({"B": i + 1}), dtype_env_json=base["dtype_env_json"],
                        gpu=base["gpu"], phase=base["phase"]) for i in range(5505)]
    measurements.lookup(scope, keys[:3])
    measurements.lookup(scope, keys[3:])
    (few, t_few), (many, t_many) = asked
    assert (few, many) == (3, 5502)
    assert t_few == pytest.approx(measurements.TIMEOUT_S, abs=0.1)
    assert t_many > 3.2 * 3


class TestTheRegistry:
    def test_a_row_is_indexed_under_the_key_that_selected_it(self):
        """A row whose stored shape projects elsewhere (a retired vocabulary
        the service's annotations migrate) is a candidate under the key the
        estimate asked for — not under its own projection."""
        stored = _row()
        current = PhysicalKey(signature=_key(stored).signature,
                              shape_json='{"B":1,"d_head":128,"n_heads":24,"s_query":2048}',
                              dtype_env_json=stored["dtype_env_json"], gpu=stored["gpu"],
                              phase=stored["phase"])
        assert current != _key(stored)
        reg = registry_from_lookup([current], [[{**stored, "origin": "shared"}]])
        assert reg.get_physical(_key(stored)) == []
        hit = physical_time_for(reg, current, **_PEAKS)
        assert hit is not None and hit.t_op_s == pytest.approx(387.5e-6)

    def test_the_owners_row_wins_its_identity_and_says_mine(self):
        shared = _row()
        mine = _row(latency_us="100.0", contributor="alice")
        k = _key(shared)
        reg = registry_from_lookup([k], [[{**shared, "origin": "shared"},
                                          {**mine, "origin": "mine"}]])
        (row,) = reg.rows_by_key.values()
        assert row.origin == "mine" and row.latency_us == 100.0

    def test_identical_to_the_corpus_files(self):
        """The rows the shared corpus file and the owner's file carried,
        indexed the old way (`merge_validated_rows`, to_physical), and the
        same rows from a lookup: the same candidates, the same answers."""
        rows = [_row(framework="vllm", framework_version=v, kernel_impl="fa3", latency_us=lat,
                     date=d)
                for v, lat, d in (("0.20.0", "400.0", "2026-05-01"),
                                  ("0.21.0", "380.0", "2026-06-01"))]
        rows.append(_row(framework="flashinfer", kernel_impl="fi", latency_us="350.0"))
        mine = _row(framework="vllm", framework_version="0.22.0", kernel_impl="fa3",
                    latency_us="360.0", contributor="alice")
        old = merge_validated_rows(CalibrationRegistry(), rows, source="shared", origin="shared")
        old = merge_validated_rows(old, [mine], source="mine", origin="mine")
        k = _key(rows[0])
        new = registry_from_lookup([k], [[{**r, "origin": "shared"} for r in rows]
                                         + [{**mine, "origin": "mine"}]])
        declared = {"state": "declared", "library": "vllm", "impl_id": "fa3"}
        for fn, kw in ((physical_time_for, {}),
                       (latest_time_for, {"framework": "vllm",
                                          "kernel_dispatch": declared})):
            a, b = fn(old, k, **kw, **_PEAKS), fn(new, k, **kw, **_PEAKS)
            assert (a.t_op_s, a.source, a.source_badge) == (b.t_op_s, b.source, b.source_badge)

    def test_the_registry_reads_only_the_sent_columns(self):
        """A lookup sends the registry's columns only (`ROW_COLUMNS`); a row
        cut to them indexes and times as the whole row does."""
        a = _row()
        k = _key(a)
        cut = {c: a[c] for c in ROW_COLUMNS}
        whole = registry_from_lookup([k], [[{**a, "origin": "shared"}]])
        sent = registry_from_lookup([k], [[{**cut, "origin": "shared"}]])
        assert (physical_time_for(whole, k, **_PEAKS).source_badge
                == physical_time_for(sent, k, **_PEAKS).source_badge)

    def test_a_repeated_shared_identity_is_an_integrity_error(self):
        a = _row()
        with pytest.raises(_benchmarks.BenchmarkValidationError):
            registry_from_lookup([_key(a)], [[{**a, "origin": "shared"},
                                              {**a, "origin": "shared", "latency_us": "1.0"}]])


def test_the_internal_clients_never_use_an_environment_proxy(monkeypatch):
    """The service is on loopback and every request carries the shared
    secret: with http_proxy / HTTPS_PROXY / ALL_PROXY pointing at a dead
    proxy, the estimator's client and the API's proxy (its health probe)
    still reach the service directly."""
    import asyncio
    import http.server
    import threading

    class Service(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"lake_id": "l", "snapshot": 1, "shared_snapshot": 1,
                               "snapshot_content": "t", "shared_snapshot_content": "t",
                               "rules_rev": "r", "status": "ok", "lake": {}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Service)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                     "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(name, "http://127.0.0.1:9")
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HUGGINGARCH_BENCH_URL", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setenv("HUGGINGARCH_BENCH_SECRET", "s")
        monkeypatch.setattr(measurements, "_clients", {})
        assert measurements.resolve_scope(reads=True, owner=None).lake_id == "l"

        from backend.service import bench_proxy
        monkeypatch.setattr(bench_proxy, "transport", None)
        monkeypatch.setattr(bench_proxy, "_clients", type(bench_proxy._clients)())
        assert asyncio.run(bench_proxy.bench_health())["state"] == "up"
    finally:
        server.shutdown()


def test_an_unreadable_secret_file_is_an_error_not_an_empty_secret(monkeypatch, tmp_path):
    from backend.measure import lookup as L
    monkeypatch.delenv("HUGGINGARCH_BENCH_SECRET", raising=False)
    monkeypatch.setenv("HUGGINGARCH_BENCH_SECRET_FILE", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="cannot be read"):
        L.secret()
