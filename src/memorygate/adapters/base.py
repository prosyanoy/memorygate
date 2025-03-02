"""The pluggable memory-system interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from ..models import Conversation, MemoryItem


class AdapterError(RuntimeError):
    """Raised when an adapter cannot do its job (e.g. Ollama is unreachable)."""


class MemorySystem(ABC):
    """What MemoryGate needs from a memory system under test.

    Conversations arrive chronologically for all customers in one shared
    instance, exactly like a multi-tenant production deployment.
    """

    name: str = "memory-system"

    @abstractmethod
    def ingest(self, customer_id: str, conversation: Conversation) -> None:
        """Observe one conversation belonging to `customer_id`."""

    @abstractmethod
    def memories(self, customer_id: str) -> list[MemoryItem]:
        """All memories currently held (believed active) for the customer."""

    @abstractmethod
    def retrieve(self, customer_id: str, query: str, k: int) -> list[MemoryItem]:
        """Top-k memories for a natural-language question, best first."""


@dataclass(frozen=True)
class ExtractedFact:
    attribute: str
    value: str


class Extractor(Protocol):
    name: str

    def extract(self, conversation: Conversation) -> list[ExtractedFact]:
        """Facts the customer states about themself, in conversation order."""
        ...
