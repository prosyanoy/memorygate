"""Run a memory-system adapter against a dataset."""

from __future__ import annotations

from .adapters.base import MemorySystem
from .metrics import point_metrics, score_dataset
from .models import Dataset, RunResult


def run_adapter(adapter: MemorySystem, dataset: Dataset, k: int = 3) -> RunResult:
    # Ingest chronologically, all tenants interleaved in one shared system.
    for conv in sorted(dataset.conversations, key=lambda c: (c.timestamp, c.id)):
        adapter.ingest(conv.customer_id, conv)
    queries: dict[str, list] = {}
    for q in dataset.queries:
        queries.setdefault(q.customer_id, []).append(q)
    memories, retrievals = {}, {}
    for c in dataset.evaluated():
        memories[c.id] = adapter.memories(c.id)
        retrievals[c.id] = [adapter.retrieve(c.id, q.text, k) for q in queries.get(c.id, [])]
    per_customer = score_dataset(dataset, memories, retrievals, k)
    return RunResult(
        adapter=adapter.name,
        k=k,
        dataset_config=dataset.config,
        dataset_fingerprint=dataset.fingerprint(),
        metrics=point_metrics(per_customer),
        per_customer=per_customer,
    )
