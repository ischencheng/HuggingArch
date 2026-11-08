"""The checkpoint invariant: what the spec SAYS is saved is what is saved.

Why this is a check and not a reconciliation
--------------------------------------------
A weight's dtype had three claimants, and they were treated as three numbers to
be reconciled:

  * `spec.quant_context` — per-role schemes, resolved by `FormatPolicy`;
  * a caller-supplied `weight_dtype_bytes` float, threaded from the API through
    `pipeline.analyze` into every byte calculation;
  * the safetensors header, counted per shard by `hf.metadata`.

They coincide under BF16 and diverge under quantisation, and the reconciling was
being done by whichever consumer noticed. `_matching.py` gave up on the spec side
entirely and reads the physical byte total straight off the header, saying so:
`total_params` would have to depend on `packed_unpack_factor` being right AND on
config fields that lie.

Splitting `saved` from `intent` collapses the three to two questions, and the
first has exactly one answer. `quant_context` and the header both describe the
FILE, so they are not two estimates to be averaged — they are an equality, and
an equality can be checked. The third claimant was never about the file at all:
it is the deployment's intent, which now lives where intents live.

What this module compares
-------------------------
The whole file:

    Sigma saved bytes over every tensor == metadata.checkpoint_storage_bytes

The TOTAL is the check, because it is drawn along no line at all. The header
splits its bytes by its quant scheme's sidecar suffixes and nothing else: a
serialised buffer (`rotary_emb.inv_freq`) is no sidecar, so it sits with the
weights. The two delta columns follow the header's line on purpose:
`weight_delta` compares declared tensors and buffers against
`model_storage_bytes`, and `metadata_delta` compares implied scales against
`quantization_metadata_bytes`, so a checkpoint that ships a position buffer is
exact in both columns when the file agrees.

The columns are reported because a total that disagrees says nothing
about where. So is the per-dtype histogram, and that is the real localiser:
both sides partition by the very thing under test, so "4.7 GB too much in BF16
and 2.35 GB too little in F8_E4M3" says a role rule is missing a family of
exactly 2.35 G elements — without ever naming a tensor.

No tensor-name matching anywhere, deliberately: a name join that is wrong looks
like a dtype disagreement, and this comparison holds without one.

What the other side actually is
-------------------------------
Not the file. `metadata.checkpoint_storage_bytes` is `model_storage_bytes +
quantization_metadata_bytes`, and both of those are `elements x
_DTYPE_BYTES[dtype]` summed over the header's tensor patterns — the same shape
of calculation this package performs, from a table maintained in a different
file. So the comparison is between two reconstructions, and two tables missing
the same row agree with each other while both are wrong. DeepSeek-V4-Pro is the
worked example: neither table had `F8_E8M0`, this side priced 49 G block scales
at zero and the other at four bytes each.

The file does carry a measured figure: each shard header's `data_offsets` say
how many bytes every tensor occupies, and their sum is what the shards hold
whatever any dtype table says. `hf.header` keeps it as `total_size` (carried
here as `index_total_size`), and `is_anchored` says whether a given comparison
has it. An index's own `metadata.total_size` is not that figure: the exporter
writes it, and several write a parameter count there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from ..checkpoint_unread import UnreadSplit
from ..dtypes import UnknownDtype, dtype_bytes


@dataclass(frozen=True)
class Bucket:
    """One dtype's share, on both sides."""

    dtype: str
    mine: float = 0.0
    header: float = 0.0

    @property
    def delta(self) -> float:
        return self.mine - self.header


@dataclass(frozen=True)
class Reconciliation:
    """What the spec claims the checkpoint holds, against what it holds."""

    model: str
    #: learnable tensors, at their saved format
    weight_bytes: float = 0.0
    #: the scales those formats imply
    metadata_bytes: float = 0.0
    #: precomputed tables — real bytes, not parameters. Kept apart from the
    #: weights, but compared in `weight_delta`: the header files them with
    #: the weights, being no quant sidecar.
    buffer_bytes: float = 0.0
    header_weight_bytes: float = 0.0
    header_metadata_bytes: float = 0.0
    # The bytes the file's tensors occupy, summed from the shard headers'
    # `data_offsets` — the ONLY byte figure on the other side measured off the
    # file rather than reconstructed from a dtype table. `None` when the header
    # carries none, and that is why `is_anchored` exists: a comparison without
    # it is this package's dtype table against `backend/hf/metadata`'s, and
    # two tables with the same gap agree.
    measured_total: Optional[float] = None
    #: dtype → both sides, for localising a disagreement without a name join
    by_dtype: Mapping[str, Bucket] = field(default_factory=dict)
    #: dtypes the header reports that no scheme has declared a width for
    unknown_dtypes: Tuple[str, ...] = ()

    @property
    def total(self) -> float:
        return self.weight_bytes + self.metadata_bytes + self.buffer_bytes

    @property
    def header_total(self) -> float:
        return self.header_weight_bytes + self.header_metadata_bytes

    @property
    def ratio(self) -> float:
        return self.total / self.header_total if self.header_total else 0.0

    @property
    def delta(self) -> float:
        return self.total - self.header_total

    @property
    def weight_delta(self) -> float:
        """The TENSORS the spec declares, against the file's own count of them.

        Buffers are here: the header files a serialised `inv_freq` table with
        the weights, since no quant scheme names it a sidecar.
        """
        return self.weight_bytes + self.buffer_bytes - self.header_weight_bytes

    @property
    def metadata_delta(self) -> float:
        """The scales a format IMPLIES, against the file's count of them.

        The two columns fail for unrelated reasons and are worth separating for
        that alone: a weight disagreement means the spec has the wrong tensor or
        the wrong dtype, while a metadata disagreement means a quant scheme's
        cardinality or its scale width is wrong and every weight is fine.
        Measured across the corpus the split is almost total — 12 of the 13
        disagreements are metadata-only, with the weight column exactly zero.

        It is also a column a gate that compares only `model_storage_bytes`
        cannot have — one total on each side never asks the question, and
        reports 0 for all 13, including a model whose implied scale bytes are
        out by 147 GB.
        """
        return self.metadata_bytes - self.header_metadata_bytes

    @property
    def is_anchored(self) -> bool:
        """Whether the other side of this comparison was MEASURED.

        False means the checkpoint figures were reconstructed the same way this
        package reconstructs them — same kind of table, same kind of gap — so
        agreement is weaker evidence than it looks and disagreement does not say
        which side is wrong. DeepSeek-V4-Pro is the demonstration: both tables
        were missing `F8_E8M0`, this one priced its scales at zero and that one
        at four bytes, and neither noticed.
        """
        return self.measured_total is not None

    @property
    def measured_delta(self) -> Optional[float]:
        """Against the file's own byte total, or `None` when there isn't one."""
        if self.measured_total is None:
            return None
        return self.total - self.measured_total

    @property
    def agrees(self) -> bool:
        """Exact, and there is no knob.

        A byte total is an integer count of a physical file, so anything but
        equality is a disagreement about the model. The `tolerance` parameter
        that used to be here was documented as "a way of not noticing one" and
        shipped anyway — an argument for its own removal, left in the signature
        where the next caller would have found it and used it.
        """
        return not self.header_total or self.delta == 0

    @classmethod
    def of(cls, analysis: Any, metadata: Mapping[str, Any],
           model: str = "") -> "Reconciliation":
        """Compare the analysis's SAVED formats against the checkpoint header.

        `metadata` is a snapshot's `metadata` block — `fetch_metadata`'s output,
        where `model_storage_bytes` and `quantization_metadata_bytes` are counted
        from the real shard headers rather than derived from any config.

        Reads `saved`, never `resident`. A deployment that dequantises changes what
        is in HBM and changes nothing about the file, so consulting the resident
        format here would make the invariant fail for a correct spec the moment
        somebody chose to serve it differently.

        The modules the spec declares unread (`checkpoint_unread:`) are split
        off first, as the validator and the analyzer split them: the spec
        says what one path through the file holds.
        """
        metadata = UnreadSplit.of(
            metadata.get("header"),
            analysis.program.spec.checkpoint_unread).read_part(metadata)
        weight = 0.0
        meta = 0.0
        buffers = 0.0
        buckets: Dict[str, Bucket] = {}

        for row in analysis.checkpoint_tensors():
            tensor, rep, count = row.tensor, row.rep, row.count
            if rep is None:
                continue
            fmt = rep.saved
            own = float(fmt.weight_bytes_for(tensor.elements)) * count
            if tensor.is_parameter:
                weight += own
            else:
                buffers += own
            meta += float(fmt.metadata_bytes) * count
            key = fmt.storage_dtype
            prior = buckets.get(key) or Bucket(key)
            buckets[key] = Bucket(key, prior.mine + own, prior.header)

        header_weight = float(metadata.get("model_storage_bytes") or 0)
        header_meta = float(metadata.get("quantization_metadata_bytes") or 0)
        unknown = _fill_header_side(buckets, metadata)
        measured = metadata.get("index_total_size")

        return cls(
            model=model,
            weight_bytes=weight, metadata_bytes=meta, buffer_bytes=buffers,
            header_weight_bytes=header_weight,
            header_metadata_bytes=header_meta,
            measured_total=(None if measured is None else float(measured)),
            by_dtype=buckets, unknown_dtypes=unknown)


def _fill_header_side(buckets: Dict[str, Bucket],
                      metadata: Mapping[str, Any]) -> Tuple[str, ...]:
    """The header's own per-dtype byte split: every tensor it holds, at its
    stored dtype — used to say WHICH dtype a disagreement sits in."""
    unknown: Dict[str, None] = {}
    for group in metadata.get("header") or ():
        dtype = str(group.dtype).upper()
        try:
            width = dtype_bytes(dtype)
        except UnknownDtype:
            unknown[dtype] = None
            continue
        prior = buckets.get(dtype) or Bucket(dtype)
        buckets[dtype] = Bucket(dtype, prior.mine, prior.header
                                + float(group.elements * group.count) * width)
    return tuple(sorted(unknown))
