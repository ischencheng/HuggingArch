"""Schema — Pydantic models for architecture spec YAML files.

Package facade: every public name stays importable from
`backend.arch.schema`; of the private helpers only the two that callers
reach through the package (`_all_graphs`, `_is_quant_metadata_suffix`)
are re-exported. The pieces:
guards / custom_ops / fusion / claims / cache / markers / weights /
graph / model / quant / spec.
"""
from __future__ import annotations

from .guards import (
    _is_quant_metadata_suffix,
)
from .custom_ops import (
    CustomOpSpec,
)
from .fusion import (
    PatternNode,
    ParallelPatternForm,
    NaivePattern,
    FusionOpSpec,
)
from .claims import (
    CLAIMS_V1,
    Axiom,
    AxiomSet,
    ComponentClaims,
)
from .cache import (
    CacheEntry,
    CacheSpec,
)
from .markers import (
    BINDING_ONLY,
    LOADER_DERIVED,
    RETIRED_GRAPH_NODE_NAMING,
    reject_retired_graph_node_naming,
    OWN_NAME,
)
from .weights import (
    WeightAxisPlacementSpec,
    WeightAxisSegmentSpec,
    WeightAxisSpec,
    ExpertWeight,
)
from .graph import (
    ParamSpec,
    GraphNode,
    parse_input_ref,
    resolve_block_input_ports,
    BlockGraph,
)
from .model import (
    BlockAssignment,
    CheckpointAddress,
    LayerMap,
    NearestLayer,
    StreamLayerRef,
    StreamLayerSet,
    CheckpointIndexAfter,
    TiedWeightSpec,
    UnreadCheckpointModule,
    DenoiseLatent,
    DenoiseLoop,
    resolve_config_bool,
    HISTORY_AXIS,
    ModelInputKind,
    ModelInputSpec,
    StreamSpec,
    ModelLayer,
)
from .quant import (
    PerRoleEntry,
    OverrideSelector,
    OverrideEntry,
    QuantContext,
)
from .spec import (
    _all_graphs,
    ValidationExemptions,
    ArchSpec,
)

__all__ = [
    "BINDING_ONLY",
    "CLAIMS_V1",
    "HISTORY_AXIS",
    "LOADER_DERIVED",
    "OWN_NAME",
    "RETIRED_GRAPH_NODE_NAMING",
    "ArchSpec",
    "Axiom",
    "AxiomSet",
    "BlockAssignment",
    "BlockGraph",
    "CacheEntry",
    "CacheSpec",
    "CheckpointAddress",
    "CheckpointIndexAfter",
    "ComponentClaims",
    "CustomOpSpec",
    "DenoiseLatent",
    "DenoiseLoop",
    "ExpertWeight",
    "FusionOpSpec",
    "GraphNode",
    "LayerMap",
    "ModelInputKind",
    "ModelInputSpec",
    "ModelLayer",
    "NaivePattern",
    "NearestLayer",
    "OverrideEntry",
    "OverrideSelector",
    "ParallelPatternForm",
    "ParamSpec",
    "PatternNode",
    "PerRoleEntry",
    "QuantContext",
    "StreamLayerRef",
    "StreamLayerSet",
    "StreamSpec",
    "TiedWeightSpec",
    "UnreadCheckpointModule",
    "ValidationExemptions",
    "WeightAxisPlacementSpec",
    "WeightAxisSegmentSpec",
    "WeightAxisSpec",
    "_all_graphs",
    "_is_quant_metadata_suffix",
    "parse_input_ref",
    "reject_retired_graph_node_naming",
    "resolve_block_input_ports",
    "resolve_config_bool",
]
