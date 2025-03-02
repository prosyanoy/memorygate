"""Rule-based extractor (the 'better' demo): user turns only, anchored phrases."""

from __future__ import annotations

import re

from ..models import Conversation
from .base import ExtractedFact

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# attribute -> phrases that introduce a first-person statement of the value
_PHRASES: dict[str, list[str]] = {
    "city": ["my city is (?:now )?", "i live in ", "i'm based in ", "i just moved to "],
    "employer": ["my employer is (?:now )?", "i work at ", "i switched jobs and now work at "],
    "plan": ["my plan is (?:now )?", "i signed up for ", "i switched my plan to "],
    "email": ["my (?:new )?email is (?:now )?", "you can email me at "],
    "favorite_color": ["my (?:new )?favorite color is (?:now )?", "i really like the color "],
    "language": ["my preferred language is (?:now )?", "please reply to me in ", "please switch to "],
    "timezone": ["my timezone is (?:now )?", "i'm in timezone ", "i relocated to timezone "],
    "phone": ["my phone number is (?:now )?", "you can call me on ", "my new number is "],
}
_TAIL = {"language": r"(?P<v>.+?)(?: from now on)?[.!?]?\s*$"}
_DEFAULT_TAIL = r"(?P<v>.+?)[.!?]?\s*$"
# optional lead-in such as "Update: " or "Sorry, I got that wrong: "
_PREFIX = r"^(?:[^:]{0,40}:\s*)?"

_PATTERNS = [
    (attr, re.compile(_PREFIX + "(?:" + "|".join(ph) + ")" + _TAIL.get(attr, _DEFAULT_TAIL), re.I))
    for attr, ph in _PHRASES.items()
]


class RuleExtractor:
    name = "rules"

    def extract(self, conversation: Conversation) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for turn in conversation.turns:
            if turn.speaker != "user":
                continue
            for sentence in _SENTENCE_SPLIT.split(turn.text.strip()):
                for attr, pat in _PATTERNS:
                    m = pat.match(sentence)
                    if m:
                        facts.append(ExtractedFact(attr, m.group("v").strip()))
                        break
        return facts


_LOOSE = re.compile(
    r"(?:\bmy|\byour|'s)\s+(?P<label>city|employer|plan|email|favorite color|preferred language|"
    r"timezone|phone number)\s+is\s+(?:now\s+)?(?P<v>.+?)[.!?]?\s*$",
    re.I,
)
_LOOSE_LABELS = {
    "city": "city", "employer": "employer", "plan": "plan", "email": "email",
    "favorite color": "favorite_color", "preferred language": "language",
    "timezone": "timezone", "phone number": "phone",
}
_LOOSE_FIXED = [
    (re.compile(r"\bI live in (?P<v>.+?)[.!?]?\s*$"), "city"),
    (re.compile(r"\bI work at (?P<v>.+?)[.!?]?\s*$"), "employer"),
    (re.compile(r"\bemail me at (?P<v>.+?)[.!?]?\s*$"), "email"),
]


class LooseExtractor:
    """Sloppy keyword matching (the 'worse' demo): reads every speaker, matches
    possessives of third parties ('my sister's city is ...') and repeats of
    the assistant's outdated records."""

    name = "loose"

    def extract(self, conversation: Conversation) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for turn in conversation.turns:
            for sentence in _SENTENCE_SPLIT.split(turn.text.strip()):
                m = _LOOSE.search(sentence)
                if m:
                    facts.append(ExtractedFact(_LOOSE_LABELS[m.group("label").lower()], m.group("v").strip()))
                    continue
                for pat, attr in _LOOSE_FIXED:
                    m = pat.search(sentence)
                    if m:
                        facts.append(ExtractedFact(attr, m.group("v").strip()))
                        break
        return facts
