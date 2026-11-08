"""`checkpoint_unread:` splits the modules a spec's path never runs off the
checkpoint header: the spec is compared with the rest, and each declaration
is held to the header."""
from __future__ import annotations

import pytest

from backend.arch.checkpoint_unread import UnreadSplit
from backend.arch.schema import UnreadCheckpointModule
from backend.hf.header import SafetensorsHeader

TENSORS = {
    "text_encoder.model.embed_tokens.weight": ("BF16", [16, 8]),
    **{f"text_encoder.model.layers.{i}.mlp.up_proj.weight": ("BF16", [32, 8])
       for i in range(6)},
    "text_encoder.lm_head.weight": ("BF16", [16, 8]),
    "vae.encoder.conv_in.weight": ("F32", [4, 3, 3]),
    "vae.decoder.proj_in.weight": ("F32", [8, 4]),
}


def _header():
    return SafetensorsHeader.from_tensors(TENSORS, revision="r",
                                          total_size=2 * 16 * 8 + 2 * 6 * 32 * 8
                                          + 2 * 16 * 8 + 4 * 36 + 4 * 32)


def _entry(module, layers=None):
    return UnreadCheckpointModule(module=module, layers=layers,
                                  kind="load_skipped", reason="r",
                                  source_anchor="f.py:1")


def test_the_read_part_keeps_what_no_entry_selects():
    split = UnreadSplit.of(_header(), [
        _entry("text_encoder.model.layers", "4-5"),
        _entry("text_encoder.lm_head"),
        _entry("vae.encoder"),
    ])
    assert split.errors == ()
    read = {name for name, _group in split.read.names()}
    assert read == {
        "text_encoder.model.embed_tokens.weight",
        *(f"text_encoder.model.layers.{i}.mlp.up_proj.weight"
          for i in range(4)),
        "vae.decoder.proj_in.weight",
    }
    parts = {p.entry.module: (p.tensors, p.elements, p.bytes)
             for p in split.parts}
    assert parts["text_encoder.model.layers"] == (2, 2 * 256, 2 * 512)
    assert parts["vae.encoder"] == (1, 36, 144)
    # The read part's byte total is the file's less what is split off.
    assert split.read.total_size == _header().total_size - split.unread_bytes


def test_a_module_path_is_matched_by_whole_segments():
    # `vae.enc` is no prefix of `vae.encoder`'s path segments.
    split = UnreadSplit.of(_header(), [_entry("vae.enc")])
    assert any("selects no tensor" in e for e in split.errors)


def test_an_entry_can_name_a_bare_parameter():
    entry = _entry("vae.decoder.mask_token")
    assert entry.selects("vae.decoder.mask_token")
    assert not entry.selects("vae.decoder.mask_token_2")


def test_sidecars_the_config_names_no_scheme_for_are_filed_as_sidecars():
    from types import SimpleNamespace

    from backend.arch.checkpoint_unread import undeclared_sidecars
    from backend.hf.header import SafetensorsHeader
    from backend.hf.metadata import summarize
    from backend.arch.quant_schemes import SCHEME_NONE
    row = SimpleNamespace(is_metadata=True, name="a.linear::_scale",
                          tensor=SimpleNamespace(suffix="::_scale"))
    weight = SimpleNamespace(is_metadata=False, name="a.linear::_w",
                             tensor=SimpleNamespace(suffix="::_w"))
    analysis = SimpleNamespace(iter_checkpoint=lambda: (row, weight))
    assert undeclared_sidecars(analysis, "none") == ("::_scale",)
    header = SafetensorsHeader.from_tensors(
        {"a.linear::_w": ("U8", (8, 2)), "a.linear::_scale": ("U8", (8, 1))})
    totals = summarize(header, scheme=SCHEME_NONE, num_bits=0,
                       sidecars=("::_scale",))
    assert totals["model_storage_bytes"] == 16
    assert totals["quantization_metadata_bytes"] == 8


def test_an_entry_selecting_nothing_is_stale():
    split = UnreadSplit.of(_header(), [_entry("text_encoder.model.layers", "7-9")])
    assert len(split.errors) == 1 and "selects no tensor" in split.errors[0]


def test_a_tensor_two_entries_select_is_declared_twice():
    split = UnreadSplit.of(_header(), [_entry("text_encoder"),
                                       _entry("text_encoder.lm_head")])
    assert any("declare each unread tensor once" in e for e in split.errors)


def test_nothing_declared_leaves_the_fetch_alone():
    meta = {"header": _header(), "model_storage_bytes": 1}
    assert UnreadSplit.of(_header(), []).read_part(meta) is meta


def test_the_declaration_states_why_and_cites_it():
    with pytest.raises(ValueError, match="say why"):
        UnreadCheckpointModule(module="a.b", kind="load_skipped", reason=" ",
                               source_anchor="f.py:1")
    with pytest.raises(ValueError, match="index set"):
        UnreadCheckpointModule(module="a.b", layers="50..63",
                               kind="load_skipped", reason="r",
                               source_anchor="f.py:1")


def test_the_reconciliation_compares_the_read_part():
    from types import SimpleNamespace

    from backend.arch.eval.checkpoint import Reconciliation
    from backend.arch.quant_schemes import SCHEME_NONE
    from backend.hf.metadata import summarize
    header = _header()
    entries = [_entry("text_encoder"), _entry("vae.encoder")]
    analysis = SimpleNamespace(
        checkpoint_tensors=lambda: (),
        iter_checkpoint=lambda: (),
        program=SimpleNamespace(spec=SimpleNamespace(checkpoint_unread=entries)))
    whole = summarize(header, scheme=SCHEME_NONE, num_bits=0)
    result = Reconciliation.of(analysis, whole)
    assert result.header_weight_bytes == 4 * 32  # vae.decoder alone
