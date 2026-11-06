"""The role registry — one classification and one meaning per role.

`roles.yaml` beside this module is the declaration; this is the reader. A
component declares `role:`, which is a fact about that component; what the
role IS — which family of work it does — is a fact about the role, and it is
written once, here.

Four tables used to hold that between them: `_ATTN_ROLES`, `_FFN_ROLES` and
`_WRAPPER_ROLES` in `eval/projection/result.py`, and the
`role.startswith("attention_")` prefix test in `inference/sharding.py`. They
had already drifted, and `attention_swa` is the proof: the prefix test caught
it, the frozenset did not, so every sliding-window layer in gpt-oss,
MiMo-V2-Flash and DeepSeek-V4 reported `attention_flops = 0` while its FLOPs
sat undisturbed in the block total. Nothing said so, because a frozenset that
does not contain a name cannot be asked whether it should.

The distinction the four tables blurred is worth keeping visible: `attention`
is a role a component library entry declares, `attention_swa` is a SLOT name a
model's `components:` map binds. Both reach `spec.component_roles`, so both
must be classified; neither may be classified by its spelling.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Mapping, Optional, Tuple
from backend.code_trace.memo import memoize


_REGISTRY_PATH = Path(__file__).parent / "roles.yaml"

#: What a `kind` may say. `wrapper` is the one with teeth: a cost classifier
#: recurses through a wrapper rather than attributing to it, so misfiling a
#: container as work — or work as a container — drops a subtree.
KINDS: FrozenSet[str] = frozenset(
    {"attention", "ffn", "memory", "norm", "residual", "wrapper"})


@dataclass(frozen=True)
class Role:
    """One registered role or slot name."""
    name: str
    kind: str
    meaning: str
    #: A TP weight class (attention / dense FFN / MoE FFN).
    weight_class: bool = False
    #: The FFN whose gate nonlinearity `config.hidden_act` governs.
    ffn_gate: bool = False
    #: The broader role this one is a part of (`attention_output` is a part
    #: of `attention`). A quant `per_role` entry or a precision choice made
    #: for the broader role reaches this one unless it is named itself.
    refines: Optional[str] = None
    #: What a block whose own role this is shows as its name when it states
    #: no attention display (`block_label.slot_label`), e.g. `MoE FFN`.
    label: Optional[str] = None


@memoize(maxsize=1)
def _load() -> Dict[str, Role]:
    raw = load_yaml(_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    out: Dict[str, Role] = {}
    for name, entry in (raw.get("roles") or {}).items():
        role = Role(name=name, **entry)
        if role.kind not in KINDS:
            raise ValueError(
                f"roles.yaml: role {name!r} has kind {role.kind!r}, not one "
                f"of {sorted(KINDS)}."
            )
        if not role.meaning:
            raise ValueError(
                f"roles.yaml: role {name!r} has no meaning. What the role "
                f"does is the half of the registry a human reads."
            )
        out[name] = role
    for role in out.values():
        if role.refines is None:
            continue
        parent = out.get(role.refines)
        if parent is None or parent.kind != role.kind:
            raise ValueError(
                f"roles.yaml: role {role.name!r} refines {role.refines!r}, "
                f"which is not a registered role of kind {role.kind!r}.")
        seen = {role.name}
        while parent is not None:
            if parent.name in seen:
                raise ValueError(
                    f"roles.yaml: role {role.name!r} refines itself through "
                    f"{sorted(seen)}.")
            seen.add(parent.name)
            parent = out.get(parent.refines) if parent.refines else None
    return out


def all_roles() -> Mapping[str, Role]:
    """Every registered role, in declaration order."""
    return _load()


def get(name: str) -> Optional[Role]:
    return _load().get(name)


def is_registered(name: str) -> bool:
    return name in _load()


def require_registered(name: Optional[str]) -> Optional[str]:
    """`name` when it is a registered role (or None); else the ValueError a
    schema validator raises. The one gate for every place a spec writes
    `role:` — a component, a graph node, a model-level node."""
    if name is not None and not is_registered(name):
        raise ValueError(
            f"role {name!r} is not registered in backend/arch/roles.yaml "
            f"(known: {sorted(all_roles())})")
    return name


def kind_of(name: str) -> str:
    """The family a role belongs to, `""` if the role is unregistered.

    Unregistered answers `""` rather than raising: callers classify nodes
    that may carry no role at all, and the load-time gate in `SpecLoader` is
    where an unknown role is supposed to be caught.
    """
    role = _load().get(name)
    return role.kind if role else ""


def lineage(name: Optional[str]) -> Tuple[str, ...]:
    """`name`, then the role it refines, then that role's, to the broadest.

    Empty for no role; an unregistered name is its own lineage (the loader
    is where an unknown role is refused)."""
    out = []
    while name and name not in out:
        out.append(name)
        role = _load().get(name)
        name = role.refines if role is not None else None
    return tuple(out)


def selection_order(name: Optional[str],
                    aliases: Mapping[str, str]) -> Tuple[str, ...]:
    """The roles a per-role rule for a leaf carrying `name` is looked up
    under, most specific first: the leaf's own role, the role its slot's
    component declares (`Program.role_aliases`), then the roles each
    refines. A rule for `attention` reaches `attention_output` and an
    `attention_swa` slot unless they are named themselves."""
    alias = aliases.get(name) if name else None
    out = []
    for candidate in (*lineage(name)[:1], *lineage(alias)[:1],
                      *lineage(name)[1:], *lineage(alias)[1:]):
        if candidate not in out:
            out.append(candidate)
    return tuple(out)


def weight_classes() -> FrozenSet[str]:
    """Roles that are a TP weight class."""
    return frozenset(n for n, r in _load().items() if r.weight_class)


def ffn_gate_roles() -> FrozenSet[str]:
    """Roles whose gate nonlinearity `config.hidden_act` governs."""
    return frozenset(n for n, r in _load().items() if r.ffn_gate)


def of_kind(kind: str) -> FrozenSet[str]:
    """Every role in the named family."""
    if kind not in KINDS:
        raise ValueError(f"no such role kind {kind!r}; have {sorted(KINDS)}")
    return frozenset(name for name, role in _load().items()
                     if role.kind == kind)


def meanings() -> Dict[str, str]:
    """`{role: meaning}` — for prompts and docs."""
    return {name: role.meaning for name, role in _load().items()}
