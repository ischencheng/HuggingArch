"""The one memo mechanism of the traced trees.

A memo hit returns a value without running the code that computed it, so a
recording (`provider.TracedFunctions`) would not see that code. Each memo
here keeps, beside its value, the trace recorded when the value was
computed, and hands it to every active recording on each hit (`merge`).
Under `GlobalHash` they are exactly `functools.lru_cache` /
`functools.cached_property` and a plain bounded table: there is nothing to
record.

  * `memoize`       — `functools.lru_cache`, same arguments;
  * `memo_property` — `functools.cached_property`;
  * `Memo`          — a keyed table (`get(key, build)`), bounded,
                      single-flight per key.

A value filled after it is built (a registry of walks walked lazily) holds a
`provider.Growing` and `attach`es it on every read instead
(`backend.inference.walk_registry`).

`tests/meta/test_code_trace_gate.py` keeps `functools` memos out of the
traced trees, so a memo cannot skip code without saying what it skipped.
"""
from __future__ import annotations

import functools
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, Optional

from .provider import provider

_P = provider()
TRACED = _P.traced


def _traced_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """`fn(*args, **kwargs)` and the trace of what it ran."""
    rec = _P.begin(nested=True)
    try:
        value = fn(*args, **kwargs)
    finally:
        trace = _P.end(rec)
    return value, trace


def memoize(maxsize: Any = 128, typed: bool = False) -> Any:
    """`functools.lru_cache`, whose hits merge the trace of their miss."""
    if callable(maxsize) and not isinstance(maxsize, bool):
        return memoize(128, typed)(maxsize)
    if not TRACED:
        return functools.lru_cache(maxsize=maxsize, typed=typed)

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        cached = functools.lru_cache(maxsize=maxsize, typed=typed)(
            functools.partial(_traced_call, fn))

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            value, trace = cached(*args, **kwargs)
            _P.merge(trace)
            return value

        wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
        return wrapper

    return decorate


class _Slot:
    """A `memo_property` value with the trace of its computation."""

    __slots__ = ("trace", "value")

    def __init__(self, value: Any, trace: Any) -> None:
        self.value = value
        self.trace = trace


class _TracedProperty(functools.cached_property):
    """`cached_property` that is a data descriptor, so a hit passes through
    `__get__` and merges the trace of the miss. The value lives where
    `cached_property` keeps it (the instance `__dict__`, under the
    attribute's name); a value put there directly reads back as itself."""

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self.attrname] = value

    def __delete__(self, instance: Any) -> None:
        instance.__dict__.pop(self.attrname, None)

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            return self
        slot = instance.__dict__.get(self.attrname, _MISSING)
        if slot is _MISSING:
            value, trace = _traced_call(self.func, instance)
            instance.__dict__[self.attrname] = _Slot(value, trace)
            return value
        if type(slot) is _Slot:
            _P.merge(slot.trace)
            return slot.value
        return slot


_MISSING = object()

memo_property: Any = _TracedProperty if TRACED else functools.cached_property


class _Entry:
    __slots__ = ("trace", "value")

    def __init__(self, value: Any, trace: Any) -> None:
        self.value = value
        self.trace = trace


class Memo:
    """A keyed memo table: `get(key, build)` is the value `build()` made for
    `key`, built once (single-flight per key) and kept among the `maxsize`
    most recently read (None: all)."""

    def __init__(self, maxsize: Optional[int] = None) -> None:
        self.maxsize = maxsize
        self._table: "OrderedDict[Hashable, _Entry]" = OrderedDict()
        self._lock = threading.Lock()
        self._building: Dict[Hashable, threading.Lock] = {}

    def _hit(self, key: Hashable) -> Optional[_Entry]:
        with self._lock:
            entry = self._table.get(key)
            if entry is not None:
                self._table.move_to_end(key)
            return entry

    def get(self, key: Hashable, build: Callable[[], Any]) -> Any:
        entry = self._hit(key)
        if entry is None:
            with self._lock:
                gate = self._building.setdefault(key, threading.Lock())
            with gate:
                entry = self._hit(key)
                if entry is None:
                    entry = self._build(key, build)
            with self._lock:
                if self._building.get(key) is gate:
                    del self._building[key]
        _P.merge(entry.trace)
        return entry.value

    def _build(self, key: Hashable, build: Callable[[], Any]) -> _Entry:
        entry = (_Entry(*_traced_call(build)) if TRACED
                 else _Entry(build(), None))
        with self._lock:
            self._table[key] = entry
            while self.maxsize is not None and len(self._table) > self.maxsize:
                self._table.popitem(last=False)
        return entry

    def clear(self) -> None:
        with self._lock:
            self._table.clear()

    def __len__(self) -> int:
        return len(self._table)
