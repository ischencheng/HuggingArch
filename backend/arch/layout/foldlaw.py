"""ReductionHomomorphism — the single predicate governing Partial passthrough.

The mathematical criterion is homomorphism of the op w.r.t. the reducer;
that is a property of the function body, invisible to shapes, so it is
REGISTERED — same epistemic position as axis_use. Two iron rules keep
this from being a policy table: the default (`requires_complete`) is
always correct, and every law is a falsifiable mathematical claim with a
counterexample test.

The evaluator has exactly one generic rule: axis_use `contract` on a sharded
axis PRODUCES a Partial (embedding is an instance). `normalize` never
produces; only an execution plan declaring an online form does, with
reducer/schema/group defined by that plan entry. This module only judges
passthrough.

Laws are keyed by the REAL primitive vocabulary of `ops.OP_REGISTRY` —
`linear` (bias is a dimension of the same op: the law carries a
`required_node_facts` flag the caller answers from the node, and a biased linear has
no passthrough: a backend that fuses one must finalize INSIDE the kernel
and then apply the affine), `combine`, `add`, `mul`, `scatter_rows_add`,
`concat`, and the payload-moving view ops.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Optional, Tuple

from backend.arch.op_names import op_keyed

from .value import Partial, PENDING_MERGE_KINDS, UnknownValue

# Closed coefficient predicate classes — nothing outside these four.
GROUP_INVARIANT = "g_invariant"            # equal value on every rank of G
PARTITION_OF_UNITY = "partition_of_unity"   # owner masks: sum_r chi_r == 1
INDEX_EQUIVARIANT = "index_equivariant"     # reindex / masks / segment injections
CONSTANT = "constant"
COEFFICIENT_CLASSES = (GROUP_INVARIANT, PARTITION_OF_UNITY,
                       INDEX_EQUIVARIANT, CONSTANT)

#: Closed set of payload transforms a law may name ("" = identity).
SCHEMA_MAPS = ("", "matmul_shape", "weighted_inverse_route", "concat",
               "reindex")

#: Sentinels, split by meaning: a law that holds for every pending-merge
#: kind, and a carrier declaration meaning "every data input".
ANY_REDUCER = "*any-reducer*"
ALL_INPUT_SLOTS = "*all-inputs*"


@dataclass(frozen=True)
class ReductionHomomorphism:
    """One registered claim: `op` commutes with `reducer` in its carrier
    slots. Operand slots are the op's REAL data inputs (arity checked
    against the corpus); conditions on implicit parameters (a linear
    layer's weight tensor is not a data input) live in
    `parameter_conditions`, and boolean node facts in
    `required_node_facts`."""

    op: str
    reducer: str                       # a pending-merge kind, or ANY_REDUCER
    carrier_slots: Tuple[str, ...]     # data-input slots folded together
    coefficient_slots: Tuple[Tuple[str, str], ...] = ()   # (data slot, class)
    parameter_conditions: Tuple[Tuple[str, str], ...] = ()  # (param, class)
    schema_map: str = ""
    required_node_facts: Tuple[str, ...] = ()
    # The op's value does not depend on the order of its data inputs, so
    # the law's slot names may be laid over them in any order: a pending
    # sum may arrive at either operand of `mul`.
    commutative: bool = False

    def __post_init__(self) -> None:
        if self.reducer != ANY_REDUCER and self.reducer not in PENDING_MERGE_KINDS:
            raise ValueError(f"unknown reducer {self.reducer!r}")
        if not self.carrier_slots:
            raise ValueError("a law without carrier slots folds nothing")
        if self.schema_map not in SCHEMA_MAPS:
            raise ValueError(f"unknown schema_map {self.schema_map!r}")
        for _, cls in (*self.coefficient_slots, *self.parameter_conditions):
            if cls not in COEFFICIENT_CLASSES:
                raise ValueError(f"coefficient class {cls!r} is not in the "
                                 "closed vocabulary")

    def slots(self, arity: int) -> Optional[Tuple[str, ...]]:
        """The op's data-input slot names in positional order — carriers,
        then coefficient operands — or None when the law does not describe
        a node with `arity` inputs. A wildcard law (`concat`) names nothing:
        its inputs are positional and all of them carry."""
        if self.carrier_slots == (ALL_INPUT_SLOTS,):
            return tuple(f"in_{i}" for i in range(arity))
        named = tuple(self.carrier_slots) + tuple(
            slot for slot, _ in self.coefficient_slots)
        return named if len(named) == arity else None

    def namings(self, arity: int) -> Tuple[Tuple[str, ...], ...]:
        """Every way the law's slot names may be laid over a node's inputs
        in positional order: the declared order, plus its permutations
        when the op is commutative. Empty when `slots` describes no node
        of this arity."""
        declared = self.slots(arity)
        if declared is None:
            return ()
        if not self.commutative:
            return (declared,)
        return tuple(dict.fromkeys(permutations(declared)))


def _linear_law(op: str) -> ReductionHomomorphism:
    # The weight is an implicit parameter, not a data input: its
    # group-invariance is a parameter condition, and the registry key
    # always equals law.op.
    return ReductionHomomorphism(
        op=op, reducer="sum", carrier_slots=("in",),
        parameter_conditions=(("weight", GROUP_INVARIANT),),
        schema_map="matmul_shape", required_node_facts=("bias_absent",))

def _view_law(op: str) -> ReductionHomomorphism:
    return ReductionHomomorphism(
        op=op, reducer=ANY_REDUCER, carrier_slots=("in",),
        parameter_conditions=(("index_map", INDEX_EQUIVARIANT),),
        schema_map="reindex")

#: The corpus passthrough table, exhaustively enumerated by REAL op names
#: (`op_keyed`: a key no registered op answers to is refused at load).
#: Ops absent here (biased linear via `required_node_facts`, activations,
#: narrowing casts) have NO law: arriving Partials finalize first.
HOMOMORPHISM_REGISTRY = op_keyed({
    "linear": _linear_law("linear"),
    "linear_param": _linear_law("linear_param"),
    # linear_external_weight takes its weight as a SECOND DATA INPUT
    # (ops.py: signature (in, weight); hc_pre wires a computed tensor in),
    # so the invariance condition sits on a coefficient slot, not on an
    # implicit parameter.
    "linear_external_weight": ReductionHomomorphism(
        op="linear_external_weight", reducer="sum", carrier_slots=("in",),
        coefficient_slots=(("weight", GROUP_INVARIANT),),
        schema_map="matmul_shape", required_node_facts=("bias_absent",)),
    "per_head_linear": _linear_law("per_head_linear"),
    # combine's REAL arity is (expert_out, routing) — the routing operand
    # IS the gate stream, one slot, matching the two-input seam contract.
    "combine": ReductionHomomorphism(
        op="combine", reducer="sum", carrier_slots=("expert_out",),
        coefficient_slots=(("routing", GROUP_INVARIANT),),
        schema_map="weighted_inverse_route"),
    "add": ReductionHomomorphism(
        op="add", reducer="sum", carrier_slots=("a", "b"),
        parameter_conditions=(("unit_scale", CONSTANT),), schema_map=""),
    # sum_r (x_r * g) = (sum_r x_r) * g when g is the same on every rank:
    # mul is linear in each operand separately. One carrier only — with
    # both operands pending, (sum_r x_r)(sum_r y_r) != sum_r x_r y_r, and
    # a cut factor scales each rank's part by a different value, so both
    # cases finalize first. A sigmoid gate on an unreduced shared expert
    # output (Qwen3-Next) is the corpus instance.
    "mul": ReductionHomomorphism(
        op="mul", reducer="sum", carrier_slots=("x",),
        coefficient_slots=(("factor", GROUP_INVARIANT),), schema_map="",
        commutative=True),
    # sum_r scatter_add(x_r, v_r) = scatter_add(sum_r x_r, sum_r v_r): the
    # row injection is the same on every rank. Its replace sibling has no
    # law, so the default (finalize first) holds for it.
    "scatter_rows_add": ReductionHomomorphism(
        op="scatter_rows_add", reducer="sum", carrier_slots=("x", "v"),
        parameter_conditions=(("row_injection", INDEX_EQUIVARIANT),),
        schema_map=""),
    "concat": ReductionHomomorphism(
        op="concat", reducer=ANY_REDUCER, carrier_slots=(ALL_INPUT_SLOTS,),
        parameter_conditions=(("segment_injections", INDEX_EQUIVARIANT),),
        schema_map="concat"),
    # Views that move a pending payload without touching its algebra.
    **{op: _view_law(op) for op in (
        "slice", "view_split", "view_merge", "spatial_merge", "transpose",
        "head_repeat")},
})


def may_defer_reduction(law: ReductionHomomorphism, inputs, coefficient_ok,
             node_facts: frozenset = frozenset()) -> bool:
    """The single passthrough predicate.

    inputs: (slot, state) for EVERY data input of the node — state is a
    `Partial` or anything else (treated as Complete). Passing only the
    Partial-carrying slots is a caller bug: the wildcard laws and the
    carrier_slots-set equality both need the full slot picture.
    coefficient_ok: callable (slot, predicate_class) -> bool, answered by
    the propagation layer relative to the pending group.
    node_facts: facts the law `required_node_facts` (e.g. "bias_absent"), read off
    the concrete node by the caller.

    Holds iff every required node fact is present, the set of
    Partial-carrying slots equals the law's carrier_slots (wildcard: every
    input carries), all pending states are field-for-field identical, and
    every coefficient satisfies its declared class."""
    # Caller-bug checks come FIRST and raise regardless of `required_node_facts`:
    # a mathematical rejection (False) and a broken call are different
    # answers, and the second must never masquerade as the first.
    inputs = tuple(inputs)
    for slot, v in inputs:
        if isinstance(v, UnknownValue):
            raise ValueError(f"UnknownValue reached may_defer_reduction at slot "
                             f"{slot!r} — bottom is loud at every consumer")
    provided = {s for s, _ in inputs}
    if law.carrier_slots != (ALL_INPUT_SLOTS,):
        if not set(law.carrier_slots) <= provided:
            raise ValueError(
                f"caller passed a partial slot picture: law {law.op!r} "
                f"declares carrier_slots {law.carrier_slots} but only "
                f"{provided} arrived — may_defer_reduction needs EVERY "
                "data input")
        missing = {slot for slot, _ in law.coefficient_slots} - provided
        if missing:
            raise ValueError(
                f"caller omitted declared coefficient operand(s) {missing} "
                f"of law {law.op!r} — coefficient slots are real data "
                "inputs (parameter conditions are checked separately)")
    if any(req not in node_facts for req in law.required_node_facts):
        return False
    partial_slots = {s for s, v in inputs if isinstance(v, Partial)}
    if not partial_slots:
        return True
    states = [v for _, v in inputs if isinstance(v, Partial)]
    first = states[0]
    if any(p != first for p in states[1:]):
        return False                    # mixed pending states never merge
    if law.reducer not in (ANY_REDUCER, first.reducer):
        return False
    if law.carrier_slots == (ALL_INPUT_SLOTS,):
        if partial_slots != {s for s, _ in inputs}:
            return False                # wildcard: ALL inputs must carry
    else:
        if partial_slots != set(law.carrier_slots):
            return False                # e.g. Partial + Complete into add
        declared = set(law.carrier_slots) | {s for s, _ in law.coefficient_slots}
        undeclared = provided - declared
        if undeclared:
            return False                # closure: no unmodeled operand
    return all(coefficient_ok(slot, cls)
               for slot, cls in (*law.coefficient_slots,
                                 *law.parameter_conditions))
