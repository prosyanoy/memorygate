"""A tenant-isolated, last-write-wins memory store around any `Extractor`."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from ..models import Conversation, MemoryItem
from ..schema import ATTRIBUTE_LABELS
from .base import Extractor, MemorySystem


class ScopedMemory(MemorySystem):
    """Keeps one current value per (customer, attribute); retrieval is TF-IDF
    over that customer's memories only, so nothing can leak across tenants."""

    def __init__(self, extractor: Extractor, name: str):
        self.extractor = extractor
        self.name = name
        self._store: dict[str, dict[str, MemoryItem]] = {}

    def ingest(self, customer_id: str, conversation: Conversation) -> None:
        mine = self._store.setdefault(customer_id, {})
        for fact in self.extractor.extract(conversation):
            label = ATTRIBUTE_LABELS.get(fact.attribute, fact.attribute)
            mine[fact.attribute] = MemoryItem(
                attribute=fact.attribute,
                value=fact.value,
                conversation_id=conversation.id,
                text=f"{label}: {fact.value}",
            )

    def memories(self, customer_id: str) -> list[MemoryItem]:
        return list(self._store.get(customer_id, {}).values())

    def retrieve(self, customer_id: str, query: str, k: int) -> list[MemoryItem]:
        items = self.memories(customer_id)
        if not items:
            return []
        vec = TfidfVectorizer(stop_words="english")
        try:
            doc_m = vec.fit_transform([i.text for i in items])
        except ValueError:  # empty vocabulary
            return []
        scores = (doc_m @ vec.transform([query]).T).toarray().ravel()
        ranked = sorted(zip(scores, range(len(items))), key=lambda p: (-p[0], p[1]))
        return [items[i].model_copy(update={"score": float(s)}) for s, i in ranked[:k] if s > 0]
