from memorygate.generator import generate
from memorygate.models import GeneratorConfig


def test_deterministic_for_same_seed():
    cfg = GeneratorConfig(seed=3, n_customers=10)
    assert generate(cfg).fingerprint() == generate(cfg).fingerprint()
    assert generate(cfg).fingerprint() != generate(GeneratorConfig(seed=4, n_customers=10)).fingerprint()


def test_shape_and_distractors(dataset):
    assert len(dataset.evaluated()) == 30
    assert len(dataset.customers) == 30 + round(30 * 0.25)
    assert {q.customer_id for q in dataset.queries} == {c.id for c in dataset.evaluated()}
    assert sum(len(c.facts) for c in dataset.evaluated()) == len(dataset.queries)


def test_ground_truth_is_planted_in_text(dataset):
    text_by_customer: dict[str, str] = {}
    for conv in dataset.conversations:
        text_by_customer[conv.customer_id] = text_by_customer.get(conv.customer_id, "") + " ".join(
            t.text for t in conv.turns if t.speaker == "user"
        )
    for c in dataset.customers:
        for value in c.facts.values():
            assert value in text_by_customer[c.id]
        for s in c.stale:
            assert s.value != c.facts[s.attribute]


def test_has_updates_and_contradictions(dataset):
    reasons = {s.reason for c in dataset.customers for s in c.stale}
    assert reasons == {"update", "contradiction"}


def test_conversations_sorted_and_owned(dataset):
    stamps = [(c.timestamp, c.id) for c in dataset.conversations]
    assert stamps == sorted(stamps)
    assert set(dataset.owners().values()) == {c.id for c in dataset.customers}
