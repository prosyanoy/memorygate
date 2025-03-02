import numpy as np
import pytest

from memorygate.metrics import FIELDS, metric_values, point_metrics, score_customer
from memorygate.models import Customer, MemoryItem, Query, StaleFact


def mem(attr, value, conv):
    return MemoryItem(attribute=attr, value=value, conversation_id=conv)


OWNERS = {"a1": "A", "b1": "B"}
CUSTOMER = Customer(
    id="A",
    facts={"city": "Paris", "plan": "Pro"},
    stale=[StaleFact(attribute="city", value="Rome", reason="update")],
)
QUERIES = [
    Query(customer_id="A", attribute="city", text="q", expected_value="Paris"),
    Query(customer_id="A", attribute="plan", text="q", expected_value="Pro"),
]


def test_fact_counts_with_stale_and_foreign():
    memories = [mem("city", "Rome", "a1"), mem("plan", "pro", "a1"), mem("city", "Oslo", "b1")]
    st = score_customer(CUSTOMER, memories, [], [], OWNERS, 3)
    assert (st.tp, st.fp, st.fn) == (1, 2, 1)  # plan ok; Rome stale + foreign Oslo; city missing
    assert (st.stale_hits, st.stale_total) == (1, 1)


def test_retrieval_hit_mrr_and_leakage():
    retrievals = [
        [mem("city", "Oslo", "b1"), mem("city", "Paris", "a1")],  # hit at rank 2, one leak
        [mem("plan", "Team", "a1")],  # miss
    ]
    st = score_customer(CUSTOMER, [], QUERIES, retrievals, OWNERS, 3)
    assert (st.queries, st.hits, st.rr_sum) == (2, 1, 0.5)
    assert (st.retrieved, st.leaked) == (3, 1)


def test_k_truncates_retrieval():
    retrievals = [[mem("city", "Oslo", "b1"), mem("city", "Paris", "a1")], []]
    st = score_customer(CUSTOMER, [], QUERIES, retrievals, OWNERS, 1)
    assert st.hits == 0 and st.retrieved == 1


def test_metric_values_formulas():
    row = dict.fromkeys(FIELDS, 0.0)
    row.update(tp=6, fp=2, fn=4, stale_hits=1, stale_total=4, leaked=3, retrieved=12, hits=8, rr_sum=6, queries=10)
    m = metric_values(np.array([row[f] for f in FIELDS]))
    assert m["fact_precision"] == pytest.approx(0.75)
    assert m["fact_recall"] == pytest.approx(0.6)
    assert m["fact_f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert m["stale_fact_rate"] == pytest.approx(0.25)
    assert m["leakage_rate"] == pytest.approx(0.25)
    assert m["hit_at_k"] == pytest.approx(0.8)
    assert m["mrr"] == pytest.approx(0.6)


def test_zero_denominators_are_zero_not_nan():
    m = metric_values(np.zeros(len(FIELDS)))
    assert all(v == 0 for v in m.values())


def test_point_metrics_sums_customers():
    from memorygate.models import CustomerStats

    pm = point_metrics({"a": CustomerStats(tp=1, fp=1), "b": CustomerStats(tp=3)})
    assert pm["fact_precision"] == pytest.approx(0.8)
