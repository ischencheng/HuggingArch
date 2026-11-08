"""Spec-node ↔ HF-tensor matching + per-tensor diff.

Kept in one file because they all join the same two name sets — the stored
tensors `Analysis.iter_checkpoint()` names, and the checkpoint's safetensors
header — on the name:

  • `_classify_ratio`           — turn spec/hf ratio into a hint string
                                  ("0.5×" → "one dim halved").
  • `_get_safetensors_data`     — the checkpoint header and its totals
                                  (`hf.metadata.fetch_metadata`).
  • `_flatten_block_nodes`      — engine block result → flat list of
                                  leaf nodes for matching.
  • `CheckpointJoin`            — the two name sets joined once: header
                                  tensors no stored tensor is named (the ship
                                  gate) and stored tensors the header lacks.
  • `match_checkpoint_to_hf`    — matched tensors in physical-element
                                  units, for the per-tensor diagnostic.
  • `bind_leaves_to_hf`         — spec leaves bound to the header tensors
                                  their stored tensors name; the estimate
                                  path's `spec_vs_tensors_check` reads it.

The header side is the file's own names, verbatim (`hf.header`). The spec
side names each tensor from its declarations, with `X` at each position a
declaration indexes; `SafetensorsHeader.select` reads such a name as every
header tensor that has the name's literal segments and some index wherever
it writes `X`.

`bind_leaves_to_hf` / `HFBinding` / `BindResult` carry object references,
so a caller can correlate a binding back to spec IR nodes instead of
round-tripping through `tensor_name` strings.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from backend.arch.spec_tree import LeafTensor


class CheckpointUnavailable(RuntimeError):
    """Safetensors / checkpoint metadata could not be read, so there is
    no weight anchor to validate against. Not a spec problem."""


def _classify_ratio(r: float) -> str | None:
    """Turn a spec/hf ratio into a short human hint when it's far from 1.0.
    Returns None for ratios within ~5% of 1 (no meaningful hint)."""
    if r is None:
        return None
    # Common multiplier mistakes from mis-bound dims:
    # 0.5 / 2.0 → a dim got halved/doubled
    # (d_ffn_expert vs d_ffn_expert*2, d_ffn_shared vs d_ffn_shared/2, …)
    # 0.33 / 3.0 → a repeat factor is off by one (3 → 1 or 1 → 3)
    # 0.125 / 8.0 → INT4-packed elem count leaked into logical count
    if abs(r - 1.0) < 0.05:
        return None
    for target in (0.125, 0.25, 0.333, 0.5, 2.0, 3.0, 4.0, 8.0):
        if abs(r - target) / target < 0.05:
            return f"{target}×"
    return f"{r:.3g}×"


def _get_safetensors_data(model_id: str, overlay_dir=None) -> dict:
    """The checkpoint header (`header`) and the totals it implies, from
    `fetch_metadata` — the snapshot when there is one (shared, then the
    overlay's), so validation runs offline for snapshotted models."""
    from backend.hf.metadata import fetch_metadata
    meta = fetch_metadata(model_id, overlay_dir)
    # An empty fetch (network / Hub failure swallowed by fetch_metadata)
    # is not a 0-byte model: the orchestrator treats a missing
    # `model_storage_bytes` key as checkpoint infrastructure.
    if not meta:
        return {}
    return {
        "header": meta["header"],
        "total_params": meta.get("total_params", 0),
        # Physical disk byte total — sums `numel × dtype_bytes` across
        # every tensor in the header that is not a quant sidecar. Unlike
        # `total_params` (which depends on `packed_unpack_factor` correctly
        # identifying every tensor's quant scheme — fragile for mixed-
        # precision vendors who split methods across multiple config
        # fields, e.g. DSv4's `expert_dtype` outside `quantization_config`),
        # this value reads safetensors' own `dtype + shape` headers and
        # is the unambiguous physical truth on disk. Bytes-vs-bytes is
        # the canonical comparison the orchestrator uses for the headline
        # diff; total_params stays for the per-tensor diagnostic.
        "model_storage_bytes": meta.get("model_storage_bytes", 0),
        "quantization_metadata_bytes": meta.get("quantization_metadata_bytes", 0),
        "checkpoint_storage_bytes": meta.get("checkpoint_storage_bytes", 0),
        "index_total_size": meta.get("index_total_size"),
        # The checkpoint's `quant_method` when no registered scheme is named
        # for it (`hf.metadata`); the spec's quant_context must describe it.
        "quant_method_unknown": meta.get("quant_method_unknown"),
        # The scheme the totals were summed under, so a part of the header
        # (`_checkpoint_unread`) is summed the same way.
        "quant_scheme": meta.get("quant_scheme"),
        "num_bits": meta.get("num_bits", 0),
    }


def _flatten_block_nodes(block_result) -> list:
    """Flatten a BlockResult into leaf nodes with names."""
    leaves = []
    for node in block_result.nodes:
        _flatten_node(node, "", leaves, 1)
    return leaves


def _repeat_mult(node) -> int:
    """The resolved `repeat:` count of a NodeReport, or 1. A sub-stack repeated
    INSIDE a block (e.g. a vision tower's `repeat: n_vis_layers` block nested in a
    singleton `aux_block` wrapper) holds the aggregate param count on the
    component node but PER-INSTANCE params on its leaf children — so the flatten
    must re-apply the count to each descendant leaf. `repeat_display` carries the
    resolved count as `(symbol, "n_heads")`; the top-level block-template repeat
    (num_layers) is applied separately by the orchestrator via `block_results`
    and never appears here, so there is no double-count."""
    rep = node.repeat_display
    if isinstance(rep, (tuple, list)) and len(rep) == 2:
        if not isinstance(rep[1], int):
            raise ValueError(
                f"{node.name}: repeat_display {rep!r} carries a non-integer "
                f"multiplier; the coverage count cannot silently take 1")
        return rep[1]
    return 1


def _flatten_node(node, prefix, leaves, mult):
    """Recursively flatten a NodeReport into primitive rows, carrying a
    repeat multiplier down so nested-repeated sub-stacks contribute their true
    per-instance × count param total to the matcher."""
    full_name = f"{prefix}.{node.name}" if prefix else node.name
    sub_mult = mult * _repeat_mult(node)
    if node.children:
        for child in node.children:
            _flatten_node(child, full_name, leaves, sub_mult)
    else:
        leaves.append({"name": full_name, "op": node.op,
                       "params": node.params * mult})


def match_checkpoint_to_hf(analysis, header) -> list[dict]:
    """Matched checkpoint tensors in physical-element units.

    Includes quant-metadata siblings (scales, weight_shape, …) from
    `iter_checkpoint()`. The validator's headline parameter total is
    logical, but its per-tensor diagnostic faces safetensors shapes.
    Those are physical elements: an INT4 weight packed into I32 has
    fewer header elements than logical matrix entries.

    Rows sharing a name are summed against the header tensors that name
    selects: an MTP layer at the tail of the backbone's index space carries
    the backbone's names, and the two stacks' copies together are what the
    file holds.
    """
    from backend.arch.dtypes import UnknownDtype, dtype_bytes

    spec_by_name: Dict[str, dict] = {}
    for row in analysis.iter_checkpoint():
        if row.is_buffer:
            continue
        entry = spec_by_name.setdefault(row.name, {"params": 0})
        try:
            width = dtype_bytes(row.storage_dtype or "")
        except UnknownDtype:
            # The row stays in the table, marked: its physical element
            # count cannot be formed without a width. The byte gate reads
            # bytes and is not this table; dropping the row hid the dtype.
            entry["unknown_dtype"] = row.storage_dtype
            continue
        entry["params"] += round(
            row.storage_bytes / width
        ) * int(row.count)

    matched: list[dict] = []
    for name, spec in spec_by_name.items():
        selection = header.select(name)
        if not selection:
            continue
        spec_params = int(spec["params"])
        hf_params = int(selection.elements)
        matched.append({
            "spec_node": name,
            "hf_module": name,
            "spec_params": spec_params,
            "hf_params": hf_params,
            "diff": hf_params - spec_params,
        })
    return matched


@dataclass(frozen=True)
class HFBinding:
    """One stored-tensor name, the header tensors it selects, and the spec
    leaves that store under it.

    Several leaves bind one name where several stored tensors carry it: a
    heterogeneous stack's `input_layernorm` in the dense block (count=Ld)
    and the moe block (count=L−Ld), or an MTP layer appended to the
    backbone's index space. The rows are summed, which is what keeps such a
    name honest — pairing one leaf per name would report a spurious Ld/L
    ratio.

    `leaves` carries direct `LeafTensor` refs and `rows` the stored tensors
    (`Analysis.tensors()` rows), not strings, so a caller correlates a
    binding back to spec IR nodes without round-tripping through names.
    """
    leaves: Tuple["LeafTensor", ...]
    rows: Tuple[Any, ...]
    #: the header tensors `hf_name` selects
    selection: Any
    spec_node_label: str
    hf_name: str
    spec_params: int
    hf_params: int
    diff: int
    shape_ok: bool


@dataclass(frozen=True)
class BindResult:
    bindings: List[HFBinding]
    unmatched_spec: List["LeafTensor"]
    unmatched_hf: Dict[str, dict]


def bind_leaves_to_hf(header, analysis) -> BindResult:
    """Spec leaves → the header tensors their stored tensors name.

    The leaves are the walked graph's (`analysis.program`): the forms the
    phase runs, which store what the file stores (`module:`).
    A leaf's stored tensors are its node's rows of `Analysis.tensors()` —
    its weight and its bias, by the node rather than by a name composed from
    the weight's — and each binds the header tensors its name selects
    (`SafetensorsHeader.select`). Parameter counts are the rows' elements
    times their copies.
    """
    from backend.arch.spec_tree import iter_leaves

    rows_by_uid: Dict[int, List[Any]] = defaultdict(list)
    for row in analysis.tensors():
        if not (row.is_metadata or row.is_buffer):
            rows_by_uid[row.uid].append(row)

    leaves_by_name: Dict[str, List["LeafTensor"]] = defaultdict(list)
    rows_by_name: Dict[str, List[Any]] = defaultdict(list)
    for lt in iter_leaves(analysis.program.resolved):
        rows = rows_by_uid.get(lt.ir_node.uid)
        if not rows:
            raise ValueError(
                f"weight leaf {lt.tensor_name!r} (×{lt.multiplicity}) has no "
                f"tensor row; its parameter count cannot be read")
        for row in rows:
            leaves_by_name[row.name].append(lt)
            rows_by_name[row.name].append(row)

    bindings: List[HFBinding] = []
    unmatched_spec: List["LeafTensor"] = []
    for name, lts in leaves_by_name.items():
        selection = header.select(name)
        if not selection:
            unmatched_spec.extend(lts)
            continue
        rows = tuple(rows_by_name[name])
        spec_params = sum(int(row.elements) * int(row.count) for row in rows)
        hf_params = int(selection.elements)
        bindings.append(HFBinding(
            leaves=tuple(lts),
            rows=rows,
            selection=selection,
            spec_node_label=" + ".join(sorted({lt.tensor_name for lt in lts})),
            hf_name=name,
            spec_params=spec_params,
            hf_params=hf_params,
            diff=hf_params - spec_params,
            shape_ok=(hf_params == spec_params),
        ))

    return BindResult(
        bindings=bindings,
        unmatched_spec=unmatched_spec,
        unmatched_hf=_unclaimed_report(header, leaves_by_name),
    )


def _unclaimed_report(header, names) -> Dict[str, dict]:
    """The header tensors none of `names` selects, one report row per group
    (its name with the unclaimed index sets written in)."""
    out: Dict[str, dict] = {}
    for group, indices in header.unclaimed(names):
        out[group.display(indices)] = {
            "shape": list(group.shape),
            "dtype": group.dtype,
            "count": len(indices),
            "params": group.elements * len(indices),
        }
    return out


@dataclass(frozen=True)
class CheckpointJoin:
    """The stored tensors the spec declares ⋈ the header's tensor names.

    One join on the name, read in both directions:

      * `unmatched_header` — header tensors no stored tensor's name selects
        (the ship gate `unmatched_hf_tensors`);
      * `absent` — stored tensors whose name selects no header tensor
        (`checkpoint_name_absent`). A buffer is computed at load and may
        legitimately be missing.

    The spec side is `Analysis.iter_checkpoint()`: the weights, their quant
    siblings, the tied copies and split parts the file holds — every name
    composed once, from the declared addresses (`spec_tree`). The header
    side is the file's names verbatim (see the module docstring). Where
    several stored tensors share a name — an MTP layer at the tail of the
    backbone's index space (`checkpoint.index: {after: backbone}`) — the
    name cannot tell them apart, and their copy counts carry the check:
    `match_checkpoint_to_hf` holds the selection's elements to theirs summed.
    """

    unmatched_header: Dict[str, dict]
    absent: Tuple[Any, ...]

    @classmethod
    def of(cls, analysis, header) -> "CheckpointJoin":
        """Refuses a header with no tensors: an empty set is a missing
        fetch, and joining against it would pass every stored tensor."""
        if not header:
            raise CheckpointUnavailable(
                "the checkpoint header lists no tensors; there is nothing to "
                "join the stored tensors against")
        rows = list(analysis.iter_checkpoint())
        named = {row.name for row in rows}
        absent: List[Any] = []
        seen = set()
        for row in rows:
            if row.is_buffer or row.name in seen:
                continue
            seen.add(row.name)
            if not header.select(row.name):
                absent.append(row)
        return cls(unmatched_header=_unclaimed_report(header, named),
                   absent=tuple(absent))


def _unmatched_report_leaves(block_results, tensor_index) -> list:
    """Report leaves with stored params that no storage row covers.

    Empty under normal operation (spec_tree + the scheme registry cover the
    supported models 100%); populated when a new spec / scheme combination
    breaks coverage — a leaf with params and no row is priced at nothing.
    Derived from the projection reports, not from a rendered block graph:
    the report tree is the same leaves without the presentation detour.
    """
    out = []

    def _walk(nodes, template_name):
        for n in nodes:
            if n.children:
                _walk(n.children, template_name)
                continue
            params = n.params or 0
            if params <= 0:
                continue
            agg = (n.ir_node.aggregation_id
                   if n.ir_node is not None else n.name)
            if agg not in tensor_index:
                out.append({
                    "template": template_name,
                    "node_id": agg,
                    "params": params,
                })

    for tmpl_name, _count, br in block_results:
        _walk(br.nodes, tmpl_name)
    return out


def spec_vs_tensors_check(resolved, tensor_patterns_meta: dict) -> list[dict]:
    """Compare spec-computed leaf params against actual tensor elements.

    Policy: if caller provided checkpoint metadata with its `header`, we
    MUST validate — mismatches become visible warnings. If the metadata is
    absent, we return []. Never silent passes.

    Routes through `bind_leaves_to_hf` so `/inference/estimate` and
    `backend.arch.validate` consume the SAME name join and
    aggregation logic for the same inputs (CLAUDE.md §1: two consumers
    must not read different answers from the same spec).
    """
    header = tensor_patterns_meta.get("header")
    if not header:
        return []

    # materialize is the single quant-fusion entry (CLAUDE.md §2.3) — we
    # do NOT re-derive bytes here. The binding's rows carry their on-disk
    # bytes, which lets a binding whose logical param counts differ but
    # whose PHYSICAL bytes agree be recognised as a packing artifact rather
    # than a mismatch (CLAUDE.md §4: Layer-3 matching is bytes-vs-bytes;
    # logical numel diff is diagnostic only).
    from backend.arch.dtypes import dtype_bytes
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=8,
    ).checkpoint()
    bind = bind_leaves_to_hf(header, analysis)

    warnings: list[dict] = []
    for b in bind.bindings:
        if b.diff == 0:
            continue
        # Reconcile on physical bytes before flagging. A logical-numel
        # diff that vanishes in bytes is pure dtype packing (e.g. DSv4
        # fp4-in-I8 experts: spec counts 22M logical params/expert, the
        # checkpoint stores 11M packed I8 elements — same bytes). A real
        # shape error keeps the same dtype on both sides, so it can never
        # reconcile in bytes; this only ever suppresses packing artifacts.
        spec_bytes = sum(
            float(row.storage_bytes) * int(row.count) for row in b.rows)
        hf_bytes = float(sum(group.elements * n * dtype_bytes(group.dtype)
                             for group, n in b.selection.parts))
        if spec_bytes > 0 and hf_bytes > 0 and \
                abs(spec_bytes - hf_bytes) <= 1e-6 * max(spec_bytes, hf_bytes):
            continue
        warnings.append({
            "hf_module": b.hf_name,
            "spec_node": b.spec_node_label,
            "spec_params": b.spec_params,
            "hf_params": b.hf_params,
            "diff": b.diff,
            "ratio": (b.spec_params / b.hf_params) if b.hf_params else None,
            "spec_bytes": spec_bytes,
            "hf_bytes": hf_bytes,
        })
    return warnings
