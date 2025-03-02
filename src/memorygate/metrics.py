"""Metrics: per-customer raw counts, and vectorised ratios of their sums.

Keeping raw counts per customer lets the gate bootstrap over customers (the
independent unit) without re-running the memory system.
"""

from __future__ import annotations

import numpy as np

from .models import Customer, CustomerStats, Dataset, MemoryItem, Query
from .schema import normalize

FIELDS = tuple(CustomerStats.model_fields)

# metric name -> higher is better?
HIGHER_IS_BETTER: dict[str, bool] = {
    "fact_precision": True,
    "fact_recall": True,
    "fact_f1": True,
    "stale_fact_rate": False,
    "leakage_rate": False,
    "hit_at_k": True,
    "mrr": True,
}


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros(np.shape(num), dtype=float), where=den > 0)


def metric_values(sums: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all metrics from summed counts of shape (..., len(FIELDS))."""
    c = {f: sums[..., i] for i, f in enumerate(FIELDS)}
    precision = _ratio(c["tp"], c["tp"] + c["fp"])
    recall = _ratio(c["tp"], c["tp"] + c["fn"])
    return {
        "fact_precision": precision,
        "fact_recall": recall,
        "fact_f1": _ratio(2 * precision * recall, precision + recall),
        "stale_fact_rate": _ratio(c["stale_hits"], c["stale_total"]),
        "leakage_rate": _ratio(c["leaked"], c["retrieved"]),
        "hit_at_k": _ratio(c["hits"], c["queries"]),
        "mrr": _ratio(c["rr_sum"], c["queries"]),
    }


def stats_matrix(per_customer: dict[str, CustomerStats], ids: list[str]) -> np.ndarray:
    return np.array([[getattr(per_customer[i], f) for f in FIELDS] for i in ids], dtype=float)


def point_metrics(per_customer: dict[str, CustomerStats]) -> dict[str, float]:
    ids = sorted(per_customer)
    sums = stats_matrix(per_customer, ids).sum(axis=0)
    return {k: float(v) for k, v in metric_values(sums).items()}


def _fact_key(item: MemoryItem, customer_id: str, owners: dict[str, str]) -> tuple[str, ...]:
    if owners.get(item.conversation_id) != customer_id:
        return ("!foreign", item.attribute or "", normalize(item.value))
    if item.attribute is None:
        return ("!unstructured", normalize(item.value))
    return (item.attribute, normalize(item.value))


def score_customer(
    customer: Customer,
    memories: list[MemoryItem],
    queries: list[Query],
    retrievals: list[list[MemoryItem]],
    owners: dict[str, str],
    k: int,
) -> CustomerStats:
    """Score one customer's extracted memories and retrieval results."""
    truth = {(a, normalize(v)) for a, v in customer.facts.items()}
    stale = {(s.attribute, normalize(s.value)) for s in customer.stale}
    got = {_fact_key(m, customer.id, owners) for m in memories}
    st = CustomerStats(
        tp=len(got & truth),
        fp=len(got - truth),
        fn=len(truth - got),
        stale_hits=len(got & stale),
        stale_total=len(stale),
    )
    for q, items in zip(queries, retrievals, strict=True):
        items = items[:k]
        st.queries += 1
        st.retrieved += len(items)
        st.leaked += sum(owners.get(m.conversation_id) != customer.id for m in items)
        for rank, m in enumerate(items, start=1):
            if (
                owners.get(m.conversation_id) == customer.id
                and m.attribute == q.attribute
                and normalize(m.value) == normalize(q.expected_value)
            ):
                st.hits += 1
                st.rr_sum += 1.0 / rank
                break
    return st


def score_dataset(
    dataset: Dataset,
    memories: dict[str, list[MemoryItem]],
    retrievals: dict[str, list[list[MemoryItem]]],
    k: int,
) -> dict[str, CustomerStats]:
    owners = dataset.owners()
    by_customer: dict[str, list[Query]] = {}
    for q in dataset.queries:
        by_customer.setdefault(q.customer_id, []).append(q)
    return {
        c.id: score_customer(c, memories[c.id], by_customer.get(c.id, []), retrievals[c.id], owners, k)
        for c in dataset.evaluated()
    }
