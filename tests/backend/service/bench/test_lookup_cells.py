"""The lookup's cells file (`lookup.Cells`, docs/measured_calibration.md
§7.2): a lookup reads the cells of its rows from the file, not the lake's
parquet, when the file holds the lookup's snapshot and content; the file
follows the lake by its change feed and then holds exactly what the lake
holds; at any other snapshot or content it answers nothing and the lookup
reads the lake, with the same answer."""
from __future__ import annotations

import threading
import time
import uuid

import pytest

from backend.inference.op_calibration import ROW_COLUMNS
from backend.service.bench import lookup as LK
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig
from tests.backend.service.bench.conftest import make_annotator

_LAKE_CELLS = "FROM lake.measurements m AT (VERSION => ?) WHERE m.input_id IN"


def _upload(lake, annotate, owner, rows, *, status="shared"):
    store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                 rows=rows, annotate=annotate, status=status,
                 decided_by=owner if status == "shared" else None)
    return lake.snapshot_id()


def _key(annotate, row) -> dict:
    ann = annotate(row)
    return {"signature": ann["signature"], "physical_shape": ann["physical_shape"],
            "dtype_env_json": row["dtype_env_json"], "gpu": row["gpu"], "phase": row["phase"]}


class _Reads:
    """Counts the lake reads of a lookup's wide cells from the lake's files."""

    def __init__(self, lake, monkeypatch):
        self.lake_cells = 0
        real = lake.read

        def read(sql, params=()):
            self.lake_cells += _LAKE_CELLS in sql
            return real(sql, params)
        monkeypatch.setattr(lake, "read", read)


_SELECTION = LK.Selection()


def _lookup(lake, keys, snap, cells=None, owner=None):
    return LK.lookup(lake, keys, owner=owner, snapshot=snap, verdicts=LK.Verdicts(),
                     selection=_SELECTION, cells=cells)


def _file(lake) -> dict:
    a = LK.Cells.ALIAS
    return {r["id"]: tuple(LK._cell(r[c]) for c in ROW_COLUMNS) for r in lake.read(
        f"SELECT id::VARCHAR AS id, * EXCLUDE (id) FROM {a}.rows")}


def _lake_at(lake, snap) -> dict:
    """The second route: every row's cells read from the lake itself."""
    return {r["id"]: tuple(LK._cell(r[c]) for c in ROW_COLUMNS) for r in lake.read(
        f"SELECT id::VARCHAR AS id, {LK._SENT} FROM lake.measurements m "
        "AT (VERSION => ?)", [snap])}


def test_a_lookup_reads_its_cells_from_the_file_not_the_lakes_files(
        lake, annotate, make_rows, new_owner, monkeypatch):
    rows = make_rows(4)
    snap = _upload(lake, annotate, new_owner(), rows)
    keys = [_key(annotate, r) for r in rows]
    cells = LK.Cells()
    assert cells.refresh(lake)["how"] == "rebuilt"
    reads = _Reads(lake, monkeypatch)
    got = _lookup(lake, keys, snap, cells)
    assert reads.lake_cells == 0
    assert got == _lookup(lake, keys, snap) and all(got["rows"])
    assert reads.lake_cells == 1               # without the file: the lake's files


def test_the_file_follows_the_change_feed_and_holds_what_the_lake_holds(
        lake, annotate, make_rows, new_owner):
    alice = new_owner()
    rows = make_rows(6)
    _upload(lake, annotate, alice, rows[:3])
    cells = LK.Cells()
    assert cells.refresh(lake)["how"] == "rebuilt"
    _upload(lake, annotate, alice, rows[3:], status="private")      # inserts
    doomed = sorted(_file(lake))[0]
    store.delete_rows(lake, owner=alice, ids=[doomed])                # a delete
    edited = sorted(_file(lake))[1]
    store.restamp(lake, owner=alice, rows=[{"id": edited, "achieved_tflops": 1.25}])
    snap = lake.snapshot_id()
    assert cells.refresh(lake) == {"snapshot": snap, "how": "changes", "rows": 5}
    held = _file(lake)
    assert held == _lake_at(lake, snap) and len(held) == 5
    assert doomed not in held
    assert held[edited][ROW_COLUMNS.index("achieved_tflops")] == "1.25"
    assert cells.refresh(lake)["how"] == "current"


def test_the_file_answers_only_its_own_snapshot_and_content(
        lake, annotate, make_rows, new_owner, monkeypatch):
    owner = new_owner()
    rows = make_rows(3)
    s1 = _upload(lake, annotate, owner, rows[:2])
    cells = LK.Cells()
    cells.refresh(lake)
    ids = sorted(_file(lake))
    assert set(cells.read(lake, ids, s1)) == set(ids)
    s2 = _upload(lake, annotate, owner, rows[2:])
    assert cells.read(lake, ids, s2) == {}       # the lake moved: nothing
    keys = [_key(annotate, r) for r in rows]
    assert _lookup(lake, keys, s2, cells) == _lookup(lake, keys, s2)
    cells.refresh(lake)
    assert cells.read(lake, ids, s1) == {}       # an older pin: nothing
    # Another content at the same snapshot (a restored or diverged copy).
    monkeypatch.setattr(lake, "content_id", lambda n: "another lake")
    assert cells.read(lake, ids, s2) == {}


def test_a_reopened_lake_resumes_from_the_file_it_left(tmp_path, make_rows, new_owner):
    annotate = make_annotator()
    config = LakeConfig(catalog=str(tmp_path / "catalog.ducklake"),
                        data=str(tmp_path / "data"), cache_dir=str(tmp_path / "cache"))
    owner = new_owner()
    rows = make_rows(3)
    lk = Lake.open(config)
    store.ensure_schema(lk)
    store.annotate_all(lk, annotate)
    _upload(lk, annotate, owner, rows[:2])
    assert LK.Cells().refresh(lk)["how"] == "rebuilt"
    lk.close()
    lk = Lake.open(config)
    try:
        store.annotate_all(lk, annotate)
        snap = _upload(lk, annotate, owner, rows[2:])
        # A new process: its history is checked by content, then followed.
        assert LK.Cells().refresh(lk) == {"snapshot": snap, "how": "changes", "rows": 1}
        assert _file(lk) == _lake_at(lk, snap)
    finally:
        lk.close()


def test_a_lookup_behind_the_file_wakes_the_follower(lake, annotate, make_rows, new_owner):
    owner = new_owner()
    rows = make_rows(2)
    _upload(lake, annotate, owner, rows[:1])
    cells = LK.Cells()
    cells.follow(lake)
    s1 = lake.snapshot_id()
    deadline = time.monotonic() + 20
    while not cells.read(lake, sorted(_lake_at(lake, s1)), s1):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    s2 = _upload(lake, annotate, owner, rows[1:])
    ids = sorted(_lake_at(lake, s2))
    while len(cells.read(lake, ids, s2)) < len(ids):   # each miss pokes it
        assert time.monotonic() < deadline
        time.sleep(0.05)


@pytest.mark.parametrize("n", [4])
def test_lookups_read_while_the_file_refreshes(lake, annotate, make_rows, new_owner, n):
    owner = new_owner()
    rows = make_rows(n)
    snaps = [_upload(lake, annotate, owner, [r]) for r in rows]
    keys = [_key(annotate, r) for r in rows]
    cells = LK.Cells()
    errors: list = []

    def run():
        try:
            for _ in range(5):
                out = _lookup(lake, keys, snaps[-1], cells)
                assert sum(map(len, out["rows"])) == n
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    cells.refresh(lake)
    for t in threads:
        t.join(30)
    assert errors == []
