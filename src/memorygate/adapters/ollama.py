"""Extractor backed by a local Ollama server (no API key, no cloud)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

from ..models import Conversation
from ..schema import ATTRIBUTE_LABELS
from .base import AdapterError, ExtractedFact

Transport = Callable[[dict], str]  # request payload -> raw model text


def _http_transport(host: str, timeout: float) -> Transport:
    def send(payload: dict) -> str:
        req = urllib.request.Request(
            host.rstrip("/") + "/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured local host
                return json.load(resp)["message"]["content"]
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            raise AdapterError(f"Ollama request to {host} failed: {e}") from e

    return send


class OllamaExtractor:
    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, transport: Transport | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        timeout = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
        self.transport = transport or _http_transport(host, timeout)

    def _prompt(self, conversation: Conversation) -> str:
        attrs = ", ".join(f'"{k}" ({v})' for k, v in ATTRIBUTE_LABELS.items())
        transcript = "\n".join(f"{t.speaker}: {t.text}" for t in conversation.turns)
        return (
            "Extract facts the USER states about THEMSELF in this support conversation. "
            "Ignore facts about other people and ignore the assistant's statements. "
            "If the user corrects themself, keep only the corrected value. "
            f"Allowed attributes: {attrs}.\n"
            'Reply with JSON only: {"facts": [{"attribute": "<key>", "value": "<exact text>"}]}\n\n'
            f"Conversation:\n{transcript}"
        )

    def extract(self, conversation: Conversation) -> list[ExtractedFact]:
        raw = self.transport({
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": self._prompt(conversation)}],
        })
        return parse_facts(raw)


def parse_facts(raw: str) -> list[ExtractedFact]:
    """Parse the model's JSON, dropping anything outside the schema."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = data.get("facts", []) if isinstance(data, dict) else []
    facts = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        attr, value = item.get("attribute"), item.get("value")
        if attr in ATTRIBUTE_LABELS and isinstance(value, str) and value.strip():
            facts.append(ExtractedFact(attr, value.strip()))
    return facts
