"""A Supabase cache RPC answered by a non-JSON body (a proxy's error page) is
a miss, as a network error is — never an exception the route turns into the
caller's 400."""
from __future__ import annotations

import pytest


class _Resp:
    status_code = 200
    content = b"<html>bad gateway</html>"
    text = "<html>bad gateway</html>"

    def json(self):
        raise ValueError("not json")


class _Client:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, *a, **k):
        return _Resp()


@pytest.mark.parametrize("module", ["backend.service.sweep_cache", "backend.service.shared_spec_cache"])
def test_a_non_json_reply_is_a_miss(monkeypatch, module):
    import importlib

    import httpx

    from backend import supabase_env, supabase_rest
    mod = importlib.import_module(module)
    # `_client` keeps one client per process: without this the stub outlives
    # the test and answers every later RPC, or a real client already cached
    # answers this one.
    monkeypatch.setattr(supabase_rest, "_clients", {})
    monkeypatch.setattr(supabase_env, "configured", lambda: True)
    monkeypatch.setattr(supabase_env, "service_role", lambda: ("https://x", "k"))
    monkeypatch.setattr(httpx, "Client", _Client)
    assert mod._rpc("any_fn", {}) is None
