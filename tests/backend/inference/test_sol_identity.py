"""SOL projection identities derive from one owner, not each consumer."""
from __future__ import annotations

from dataclasses import replace

from backend.inference.card import Card
from backend.inference.parallel_config import ParallelConfig
from backend.inference.sol_data import (
    ServingWalkFacts,
    ServingWalkKey,
    StepLawKey,
    walkRecordSchema,
    recordId,
)
from backend.measure.scope import NONE


def test_producer_facts_round_trip_through_the_artifact_spelling():
    facts = ServingWalkFacts.producer()
    record = facts.to_json()
    assert ServingWalkFacts.from_record(record) == facts
    assert facts.to_anchor() == {"kind": "serving-law",
                                  "schema": walkRecordSchema, **record}


def test_environment_facts_use_the_same_owner_as_the_producer():
    environment = {
        "avg_output_tokens": 256,
        "gpu_memory_fraction": 0.9,
        "precision": ServingWalkFacts.producer().precision,
        "s_prefix_cached": 0,
        "s_verify": 1,
        "acceptance_rate": 1.0,
        "speculative_draft": None,
        "include_aux_weights": False,
        "host_offload": True,
        "framework": "",
        "framework_options": {"flag": 2},
        "kv_block_size": None,
        "prefill_chunk_size": 8192,
        "ep_load_factor": 1.0,
    }
    facts = ServingWalkFacts.from_environment(environment)
    expected = replace(ServingWalkFacts.producer(),
                       framework_options=(("flag", 2),))
    assert facts == expected
    assert facts.to_json()["frameworkOptions"] == {"flag": 2}
    assert ServingWalkFacts.from_environment(
        ServingWalkFacts.producer().to_environment()
    ) == ServingWalkFacts.producer()


def test_a_walk_key_names_the_card_class_once():
    facts = ServingWalkFacts.producer()
    key = ServingWalkKey.of_card("version", model="org/model",
                                 card=Card.of("H100-SXM"), facts=facts)
    assert key.card_class == Card.of("H100-SXM").klass.key()
    assert key.identity.content == {
        "model": "org/model", "layout": None,
        "card_class": key.card_class, "precision": facts.precision,
        "pp_stage": None, "anchor": facts.to_anchor(),
    }
    assert key.identity.scope == NONE

    assert key.precision == facts.precision


def test_a_layout_record_key_is_canonical_and_stable():
    layout = ParallelConfig(tp_attention=2, tp_dense_ffn=2,
                            tp_moe_expert=2, world_degrees={"dsa": 4})
    facts = ServingWalkFacts.producer()
    walk = ServingWalkKey.of_card("version", model="org/model",
                                  card=Card.of("H100-SXM"), facts=facts)
    key = StepLawKey.of_layout(walk, layout)
    assert key.layout == layout.to_dict()
    assert key.record_name == recordId("layout", layout.to_dict())
    assert StepLawKey.of_layout(walk, layout.to_dict()) == key

    assert key.identity.content["layout"] == layout.to_dict()
    assert key.identity.content["anchor"]["kind"] == "serving-law-step"
    assert key.bundle_key == walk.identity.key
