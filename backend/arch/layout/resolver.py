"""Deployment resolution: ParallelConfig × phase → explicit families and
per-symbol axis layouts.

The worlds registry, phase-gated degrees and mesh stride/size tables are
resolved here into explicit `ProcessGroupFamily`
members and `Block` layouts under the named non-divisible policy
(`replicate_parts`: a 6-way request over a 4-way tiling resolves to four
shards with replicated cells).
"""
from __future__ import annotations

from typing import Dict, Optional

from backend.arch import dims
from backend.arch.eval.dim import common_divisor, same_law
from backend.code_trace.memo import Memo

from . import mesh

from .groups import ProcessGroupFamily
from .layout import AxisLayout, Replicated, block


def _groups_from_span(stride: int, size: int, world_size: int):
    """Enumerate the members of every group of one mesh axis: ranks whose
    other-axis coordinates agree, spaced by `stride`, `size` long. The
    enumeration formula is valid only when the axis tiles the stage —
    asserted loudly rather than assumed (a validated ParallelConfig
    guarantees it; an unvalidated one must not silently mis-enumerate)."""
    if world_size % (stride * size):
        raise ValueError(
            f"axis (stride={stride}, size={size}) does not tile the stage "
            f"of {world_size} ranks — resolve only validated deployments")
    groups = []
    seen = set()
    for base in range(world_size):
        if base in seen:
            continue
        member0 = base - ((base // stride) % size) * stride
        group = tuple(member0 + i * stride for i in range(size))
        if group[0] == base:
            groups.append(group)
        seen.update(group)
    return tuple(groups)


#: `(topology fields, phase)` -> resolved families, LRU-bounded. Families
#: are a pure function of the mesh degrees and subgroup sizes — nothing
#: else on `ParallelConfig` (cp_scheme, world_degrees, …)
#: reaches `_resolve_families`, so nothing else may enter the key: a wider
#: key stores equivalent entries once per irrelevant-field value, and a
#: dict-carrying field would make a legal config unhashable. A parallel
#: sweep asks per CELL (`capacity.py`), so entries are shared, read-only
#: objects.
_FAMILIES = Memo(maxsize=256)


def _deployment_key(pc, phase: str) -> tuple:
    degrees = mesh.axis_degrees(pc)
    return (
        phase,
        tuple(sorted((axis, int(value or 1))
                     for axis, value in degrees.items())),
        tuple((world, int(getattr(pc, world, 1) or 1))
              for world in mesh.SUBGROUP_AXES),
    )


def families_for(pc, phase: str) -> Dict[str, ProcessGroupFamily]:
    """world label → explicit-member family, for every world above degree
    one in this phase. Subgroup worlds (dcp) use their declared fill
    order via `mesh.subgroup_stride`. Cached per (topology, phase);
    callers treat the returned mapping as immutable."""
    return _FAMILIES.get(_deployment_key(pc, phase),
                         lambda: _resolve_families(pc, phase))


def _resolve_families(pc, phase: str) -> Dict[str, ProcessGroupFamily]:
    degrees = mesh.axis_degrees(pc)
    world_size = 1
    for region_axes in mesh.REGION_AXES.values():
        prod = 1
        for axis in region_axes:
            prod *= int(degrees.get(axis, 1) or 1)
        world_size = max(world_size, prod)
    out: Dict[str, ProcessGroupFamily] = {}
    spans = mesh.spans(degrees)
    for axis, (stride, size) in spans.items():
        out[axis] = ProcessGroupFamily(
            id=axis, groups=_groups_from_span(stride, size, world_size))
    for world in mesh.SUBGROUP_AXES:
        size = int(getattr(pc, world, 1) or 1)
        if size <= 1:
            continue
        stride = mesh.subgroup_stride(pc, world)
        out[world] = ProcessGroupFamily(
            id=world, groups=_groups_from_span(stride, size, world_size))
    return out


def pipeline_family_for(pc) -> Optional[ProcessGroupFamily]:
    """The `pp` world: one group per in-stage rank, holding that rank's
    counterpart on every stage, in stage order. A cut's point-to-point
    transfer runs between consecutive members — `per_stage_size` ranks apart,
    the outermost stride `mesh.GLOBAL_OUTER_AXES` reserves for it. None at
    `pp = 1`.

    Outside `families_for`: those families tile one stage, which is the
    universe every layout is relative to, and no tensor is cut across
    stages."""
    pp = int(pc.pp or 1)
    if pp <= 1:
        return None
    per_stage = int(pc.per_stage_size)
    return ProcessGroupFamily(
        id="pp", groups=tuple(
            tuple(rank + stage * per_stage for stage in range(pp))
            for rank in range(per_stage)))


def shards_under_policy(extent: int, degree: int) -> int:
    """Decision 5, as arithmetic: the largest shard count dividing BOTH the
    extent and the group size — replica cells must tile the group. The
    extent and group size. This guarantees that replica cells tile the group
    (12 heads over 8 ranks resolves to four shards). An extent in the
    workload's symbols is read under the walk's assumptions
    (`dim.common_divisor`)."""
    if degree <= 1 or same_law(extent, 0):
        return 1
    return common_divisor(extent, degree)


def layout_for_symbol(symbol: str, extent: int, pc, phase: str,
                      families: Optional[Dict] = None) -> Optional[AxisLayout]:
    """The declaration-direct axis layout of one symbol, or None when the
    symbol belongs to no world / its world is per-rank-bound (the B
    migration item, reported by the caller rather than silently skipped).

    This resolves static symbol placement. Propagation-derived facts such as
    pending reductions, segmented locality and routed rows remain attached to
    the evaluator's carried state."""
    world = dims.world_of(symbol)
    if world is None or world.bound_per_rank:
        return None
    degree = pc.degree_of_symbol(symbol, phase=phase)
    if degree <= 1:
        return None
    families = families if families is not None else families_for(pc, phase)
    family = families.get(world.label)
    if family is None:
        # A degree above one whose world the mesh declares no topology for
        # (no span, no subgroup stride) — `world_degrees` can name such a
        # world (the deployment API deliberately allows it), but layout
        # resolution needs explicit group members and has nowhere to get
        # them. A degree without group membership is unresolvable and must be
        # rejected instead of treated as a usable placement.
        raise ValueError(
            f"symbol {symbol!r} runs at degree {degree} in world "
            f"{world.label!r}, but the mesh declares no topology for that "
            "world — a degree without explicit group members cannot be "
            "resolved; declare the world's mesh placement first")
    num_shards = shards_under_policy(extent, degree)
    if num_shards <= 1:
        return Replicated(scope=family.ranks())
    replication = family.group_size // num_shards
    cells = tuple(tuple(range(j * replication, (j + 1) * replication))
                  for j in range(num_shards))
    return block(family, cells)
