"""A deliberately poor memory system for demonstrating a failing gate.

Flaws, all realistic: append-only (never supersedes stale facts), one global
store shared by every tenant, and retrieval that only *prefers* the asking
customer's memories instead of filtering to them.
"""

from __future__ import annotations

from ..models import Conversation, MemoryItem
from ..schema import ATTRIBUTE_LABELS
from .base import MemorySystem
from .rules import LooseExtractor


class NaiveMemory(MemorySystem):
    name = "worse"

    def __init__(self) -> None:
        self.extractor = LooseExtractor()
        self._items: list[tuple[str, MemoryItem]] = []  # (customer_id, item), insertion order

    def ingest(self, customer_id: str, conversation: Conversation) -> None:
        for fact in self.extractor.extract(conversation):
            label = ATTRIBUTE_LABELS[fact.attribute]
            self._items.append((customer_id, MemoryItem(
                attribute=fact.attribute, value=fact.value,
                conversation_id=conversation.id, text=f"{label}: {fact.value}")))

    def memories(self, customer_id: str) -> list[MemoryItem]:
        return [item for owner, item in self._items if owner == customer_id]

    def retrieve(self, customer_id: str, query: str, k: int) -> list[MemoryItem]:
        q = query.lower()
        scored = []
        for pos, (owner, item) in enumerate(self._items):
            if ATTRIBUTE_LABELS[item.attribute or ""] not in q:
                continue
            score = 1.0 + (0.5 if owner == customer_id else 0.0)
            scored.append((-score, pos, item.model_copy(update={"score": score})))
        scored.sort(key=lambda t: (t[0], t[1]))  # oldest first: stale values win ties
        return [item for _, _, item in scored[:k]]
