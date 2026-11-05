"""A serving request's eligibility has one owner."""
from __future__ import annotations

import pytest

from backend.inference.card import Card
from backend.inference.defaults import (
    COMM_LATENCY_US_DEFAULT,
    GPU_MEMORY_FRACTION_DEFAULT,
    LAUNCH_FLOOR_US_DEFAULT,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.serving_projection import PublicServingLawRequest
from backend.inference.sol_data import (
    ServingWalkFacts,
    ServingWalkKey,
)


FACTS = ServingWalkFacts.producer()
WALK = ServingWalkKey.of_card(
    "version", model="org/model", card=Card.of("H100-SXM"), facts=FACTS)
LAYOUT = ParallelConfig().to_dict()


def request(**changes):
    axes = {
        "environment": FACTS.to_environment(),
        "contexts": (1024, 4096),
        "decode_batches": (1, None),
        "prefill_batches": (1, 1),
        "prefix_tokens": (0, 0),
        "verify_widths": (1, 1),
    }
    axes.update(changes)
    values = {"walk": WALK, "card": "H100-SXM", "layout": LAYOUT, **axes}
    return PublicServingLawRequest(**values)


def covers(environment=None, contexts=(1024, 4096),
           measurement_scope=None):
    return PublicServingLawRequest.covers(
        environment=environment or FACTS.to_environment(),
        contexts=contexts, measurement_scope=measurement_scope)


def test_a_public_request_names_its_law_record():
    law = request().law_key
    assert law.walk == WALK
    assert law.layout == LAYOUT


@pytest.mark.parametrize("environment", [
    {**FACTS.to_environment(),
     "comm_latency_us": COMM_LATENCY_US_DEFAULT + 1},
    {**FACTS.to_environment(),
     "launch_floor_us": LAUNCH_FLOOR_US_DEFAULT + 1},
])
def test_nondefault_view_points_are_law_substitutions(environment):
    assert covers(environment=environment)


def test_a_nondefault_hbm_budget_needs_its_own_walk():
    environment = {
        **FACTS.to_environment(), "gpu_memory_fraction":
            GPU_MEMORY_FRACTION_DEFAULT - .01}
    assert not covers(environment=environment)


@pytest.mark.parametrize("changes", [
    {"prefill_batches": (1, 2)},
    {"prefix_tokens": (0, 1)},
    {"verify_widths": (1, 2)},
    {"decode_batches": (1, 3)},
])
def test_any_positive_law_point_may_be_substituted(changes):
    request(**changes)


def test_only_a_producer_walk_and_precomputed_contexts_are_eligible():
    assert covers()
    assert not covers(contexts=(1024, 6144))
    assert not covers(measurement_scope=object())
    with pytest.raises(ValueError, match="pipeline"):
        request(layout=ParallelConfig(pp=2).to_dict())
