import json

import pytest

from memorygate.adapters import AdapterError, load_adapter
from memorygate.adapters.ollama import OllamaExtractor, parse_facts
from memorygate.adapters.rules import LooseExtractor, RuleExtractor
from memorygate.adapters.scoped import ScopedMemory
from memorygate.models import Conversation, Turn


def conv(*lines, cid="c1", customer="A", ts=0):
    return Conversation(id=cid, customer_id=customer, timestamp=ts,
                        turns=[Turn(speaker=s, text=t) for s, t in lines])


def test_rule_extractor_ignores_third_parties_and_assistant():
    c = conv(
        ("user", "My city is Paris."),
        ("user", "My sister's city is Rome."),
        ("assistant", "Our records say your plan is Basic. Is that still right?"),
        ("user", "Update: my plan is now Pro."),
        ("user", "Sorry, I got that wrong: my email is a.b@example.com."),
    )
    got = {(f.attribute, f.value) for f in RuleExtractor().extract(c)}
    assert got == {("city", "Paris"), ("plan", "Pro"), ("email", "a.b@example.com")}


def test_loose_extractor_makes_the_expected_mistakes():
    c = conv(("user", "My sister's city is Rome."), ("assistant", "Our records say your plan is Basic."))
    got = {(f.attribute, f.value) for f in LooseExtractor().extract(c)}
    assert got == {("city", "Rome"), ("plan", "Basic")}


def test_scoped_memory_last_write_wins_and_isolated():
    mem = ScopedMemory(RuleExtractor(), "t")
    mem.ingest("A", conv(("user", "My city is Paris."), cid="a1"))
    mem.ingest("B", conv(("user", "My city is Oslo."), cid="b1", customer="B"))
    mem.ingest("A", conv(("user", "I just moved to Rome."), cid="a2", ts=5))
    assert [(m.value, m.conversation_id) for m in mem.memories("A")] == [("Rome", "a2")]
    top = mem.retrieve("A", "What is my city?", 3)
    assert [m.value for m in top] == ["Rome"]
    assert mem.retrieve("A", "What is my phone number?", 3) == []
    assert mem.retrieve("nobody", "What is my city?", 3) == []


def test_better_beats_worse_on_leakage_and_staleness(dataset):
    from memorygate.runner import run_adapter

    good = run_adapter(load_adapter("better"), dataset)
    bad = run_adapter(load_adapter("worse"), dataset)
    assert good.metrics["leakage_rate"] == 0
    assert bad.metrics["leakage_rate"] > 0.1
    assert bad.metrics["stale_fact_rate"] > good.metrics["stale_fact_rate"]
    assert good.metrics["fact_f1"] > bad.metrics["fact_f1"]
    assert good.metrics["mrr"] > bad.metrics["mrr"]


def test_ollama_extractor_with_fake_transport():
    seen = {}

    def transport(payload):
        seen.update(payload)
        return json.dumps({"facts": [
            {"attribute": "city", "value": " Paris "},
            {"attribute": "shoe_size", "value": "42"},  # outside the schema
            {"attribute": "plan", "value": ""},
            "junk",
        ]})

    ex = OllamaExtractor(model="m", transport=transport)
    facts = ex.extract(conv(("user", "I live in Paris.")))
    assert [(f.attribute, f.value) for f in facts] == [("city", "Paris")]
    assert seen["model"] == "m" and seen["format"] == "json" and "I live in Paris." in seen["messages"][0]["content"]


@pytest.mark.parametrize("raw", ["not json", "[]", '{"facts": "x"}', "{}"])
def test_parse_facts_is_defensive(raw):
    assert parse_facts(raw) == []


def test_unreachable_ollama_raises_adapter_error():
    ex = OllamaExtractor(host="http://127.0.0.1:9", transport=None)
    with pytest.raises(AdapterError):
        ex.extract(conv(("user", "hi")))


def test_load_adapter_errors_and_custom_path():
    with pytest.raises(AdapterError):
        load_adapter("nope")
    with pytest.raises(AdapterError):
        load_adapter("memorygate.nothing:Thing")
    assert load_adapter("memorygate.adapters.naive:NaiveMemory").name == "worse"
