"""Adapter lookup by name or by `package.module:factory` path."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from .base import AdapterError, MemorySystem
from .naive import NaiveMemory
from .ollama import OllamaExtractor
from .rules import RuleExtractor
from .scoped import ScopedMemory

BUILTIN: dict[str, Callable[[], MemorySystem]] = {
    "worse": NaiveMemory,
    "better": lambda: ScopedMemory(RuleExtractor(), "better"),
    "ollama": lambda: ScopedMemory(OllamaExtractor(), "ollama"),
}


def load_adapter(spec: str) -> MemorySystem:
    """`spec` is a built-in name or `module:callable` returning a MemorySystem."""
    if spec in BUILTIN:
        return BUILTIN[spec]()
    if ":" not in spec:
        raise AdapterError(f"unknown adapter {spec!r}; built-ins: {', '.join(BUILTIN)}")
    module, _, attr = spec.partition(":")
    try:
        factory = getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as e:
        raise AdapterError(f"cannot load adapter {spec!r}: {e}") from e
    adapter = factory()
    if not isinstance(adapter, MemorySystem):
        raise AdapterError(f"{spec!r} did not return a MemorySystem")
    return adapter
