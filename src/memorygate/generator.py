"""Seeded synthetic conversation generator with planted ground truth.

Every customer gets a handful of facts. Some facts are later *updated* (the old
value becomes stale), some are first stated *wrongly* and then corrected inside
the same conversation (a planted contradiction), and conversations also contain
statements about third parties ("my sister's city is ...") and assistant turns
that repeat outdated records. Distractor customers share the same attributes
and overlapping values, so a memory store that mixes tenants gets caught.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .models import Conversation, Customer, Dataset, GeneratorConfig, Query, StaleFact, Turn
from .schema import ATTRIBUTE_LABELS

CITIES = ["Paris", "Berlin", "New York", "Lisbon", "Toronto", "Osaka", "Nairobi", "Oslo",
          "Austin", "Sydney", "Dublin", "Mumbai", "Lima", "Prague"]
EMPLOYERS = ["Acme Corp", "Globex", "Initech", "Umbrella Ltd", "Hooli", "Stark Industries",
             "Wayne Enterprises", "Soylent", "Vandelay", "Cyberdyne", "Tyrell", "Wonka"]
PLANS = ["Basic", "Pro", "Team", "Enterprise", "Starter"]
COLORS = ["Teal", "Crimson", "Amber", "Violet", "Olive", "Indigo", "Coral", "Slate"]
LANGUAGES = ["German", "Spanish", "French", "Japanese", "English", "Portuguese", "Korean"]
TIMEZONES = ["Europe/Berlin", "America/Denver", "Asia/Tokyo", "UTC+2", "Australia/Sydney",
             "America/Lima", "Africa/Nairobi", "UTC-5"]
FIRST = ["ana", "bo", "cy", "dee", "eli", "fay", "gus", "hana", "ivo", "jun"]
LAST = ["ng", "ortiz", "keller", "sato", "okafor", "lund", "weiss", "rossi"]


@dataclass(frozen=True)
class AttrSpec:
    pool: list[str] | None
    statements: list[str]
    updates: list[str]
    hard: list[str]  # phrasings that simple rule-based extractors are expected to miss


SPECS: dict[str, AttrSpec] = {
    "city": AttrSpec(CITIES, ["I live in {v}.", "I'm based in {v}.", "My city is {v}."],
                     ["I just moved to {v}.", "Update: my city is now {v}."],
                     ["Home base these days: {v}."]),
    "employer": AttrSpec(EMPLOYERS, ["I work at {v}.", "My employer is {v}."],
                         ["I switched jobs and now work at {v}.", "Update: my employer is now {v}."],
                         ["Paychecks come from {v}."]),
    "plan": AttrSpec(PLANS, ["My plan is {v}.", "I signed up for {v}."],
                     ["I switched my plan to {v}.", "Update: my plan is now {v}."],
                     ["Billing-wise we're on {v}."]),
    "email": AttrSpec(None, ["My email is {v}.", "You can email me at {v}."],
                      ["My new email is {v}.", "Update: my email is now {v}."],
                      ["Best inbox for me: {v}."]),
    "favorite_color": AttrSpec(COLORS, ["My favorite color is {v}.", "I really like the color {v}."],
                               ["My new favorite color is {v}.", "Update: my favorite color is now {v}."],
                               ["These days I'm all about {v}."]),
    "language": AttrSpec(LANGUAGES, ["My preferred language is {v}.", "Please reply to me in {v}."],
                         ["Update: my preferred language is now {v}.", "Please switch to {v} from now on."],
                         ["{v} works best for me."]),
    "timezone": AttrSpec(TIMEZONES, ["My timezone is {v}.", "I'm in timezone {v}."],
                         ["Update: my timezone is now {v}.", "I relocated to timezone {v}."],
                         ["Schedule things around {v}."]),
    "phone": AttrSpec(None, ["My phone number is {v}.", "You can call me on {v}."],
                      ["Update: my phone number is now {v}.", "My new number is {v}."],
                      ["Ring me at {v} sometime."]),
}
assert set(SPECS) == set(ATTRIBUTE_LABELS)

CORRECTIONS = ["Sorry, I got that wrong: my {label} is {v}.", "Correction: my {label} is {v}."]
THIRD_PARTY = "My {rel}'s {label} is {v}."
RELATIVES = ["sister", "colleague", "manager", "neighbor"]
RECORDS_LINE = "Our records say your {label} is {v}. Is that still right?"
OPENERS = ["Hi, quick question about my account.", "Hello again!", "Hey, got a minute?"]
ACKS = ["Got it, thanks.", "Noted.", "Understood, anything else?"]
CLOSERS = ["Thanks, that's all.", "Great, bye for now."]
QUESTION = "What is my {label}?"


class _Builder:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.serial = 0

    def draw(self, attr: str, exclude: set[str]) -> str:
        spec = SPECS[attr]
        while True:
            if spec.pool is not None:
                v = self.rng.choice(spec.pool)
            elif attr == "email":
                self.serial += 1
                v = f"{self.rng.choice(FIRST)}.{self.rng.choice(LAST)}{self.serial}@example.com"
            else:
                self.serial += 1
                v = f"+1-555-{(self.serial * 37 + self.rng.randint(0, 36)) % 10000:04d}"
            if v not in exclude:
                return v

    def say(self, attr: str, value: str, kind: str = "statements") -> str:
        spec = SPECS[attr]
        if kind == "statements" and self.rng.random() < self.cfg.hard_prob:
            kind = "hard"
        if kind == "updates" and self.rng.random() < self.cfg.hard_prob:
            kind = "hard"
        return self.rng.choice(getattr(spec, kind)).format(v=value)

    def session(self, cid: str, n: int, day: int, lines: list[tuple[str, str]]) -> Conversation:
        turns = [Turn(speaker="user", text=self.rng.choice(OPENERS))]
        for speaker, text in lines:
            turns.append(Turn(speaker=speaker, text=text))
            if speaker == "user" and self.rng.random() < 0.5:
                turns.append(Turn(speaker="assistant", text=self.rng.choice(ACKS)))
        turns.append(Turn(speaker="user", text=self.rng.choice(CLOSERS)))
        return Conversation(id=f"{cid}-s{n}", customer_id=cid, timestamp=day, turns=turns)

    def customer(self, cid: str, distractor: bool) -> tuple[Customer, list[Conversation]]:
        cfg, rng = self.cfg, self.rng
        attrs = rng.sample(list(SPECS), k=rng.randint(cfg.min_facts, min(cfg.max_facts, len(SPECS))))
        used: dict[str, set[str]] = {a: set() for a in attrs}
        final: dict[str, str] = {}
        stale: list[StaleFact] = []
        blocks: list[list[tuple[str, str]]] = []  # atomic groups of lines

        for a in attrs:
            if rng.random() < cfg.contradiction_prob:
                wrong = self.draw(a, used[a])
                used[a].add(wrong)
                stale.append(StaleFact(attribute=a, value=wrong, reason="contradiction"))
                right = self.draw(a, used[a])
                used[a].add(right)
                label = ATTRIBUTE_LABELS[a]
                blocks.append([
                    ("user", self.say(a, wrong)),
                    ("assistant", "Just to double check, could you confirm that?"),
                    ("user", rng.choice(CORRECTIONS).format(label=label, v=right)),
                ])
                final[a] = right
            else:
                final[a] = self.draw(a, used[a])
                used[a].add(final[a])
                blocks.append([("user", self.say(a, final[a]))])
        if rng.random() < cfg.third_party_prob:
            blocks.append([("user", self.third_party(attrs, used))])

        day = rng.randint(0, 120)
        convs: list[Conversation] = []
        half = max(1, len(blocks) // 2) if rng.random() < 0.5 else len(blocks)
        chunks = [blocks[:half]] + ([blocks[half:]] if half < len(blocks) else [])
        for chunk in chunks:
            convs.append(self.session(cid, len(convs), day, [ln for blk in chunk for ln in blk]))
            day += rng.randint(1, 15)

        updated = [a for a in attrs if rng.random() < cfg.update_prob]
        rng.shuffle(updated)
        while updated:
            n = rng.randint(1, 2)
            group, updated = updated[:n], updated[n:]
            lines = []
            if rng.random() < 0.7:
                lines.append(("assistant", RECORDS_LINE.format(label=ATTRIBUTE_LABELS[group[0]], v=final[group[0]])))
            for a in group:
                new = self.draw(a, used[a])
                used[a].add(new)
                stale.append(StaleFact(attribute=a, value=final[a], reason="update"))
                final[a] = new
                lines.append(("user", self.say(a, new, "updates")))
            if rng.random() < cfg.third_party_prob / 2:
                lines.append(("user", self.third_party(attrs, used)))
            convs.append(self.session(cid, len(convs), day, lines))
            day += rng.randint(1, 30)

        return Customer(id=cid, is_distractor=distractor, facts=final, stale=stale), convs

    def third_party(self, attrs: list[str], used: dict[str, set[str]]) -> str:
        a = self.rng.choice(attrs)
        v = self.draw(a, used[a])
        return THIRD_PARTY.format(rel=self.rng.choice(RELATIVES), label=ATTRIBUTE_LABELS[a], v=v)


def generate(config: GeneratorConfig | None = None) -> Dataset:
    cfg = config or GeneratorConfig()
    b = _Builder(cfg)
    n_dis = round(cfg.n_customers * cfg.distractor_ratio)
    total = cfg.n_customers + n_dis
    flags = [True] * n_dis + [False] * cfg.n_customers
    b.rng.shuffle(flags)
    customers: list[Customer] = []
    conversations: list[Conversation] = []
    for i, is_dis in enumerate(flags):
        cust, convs = b.customer(f"cust-{i:04d}", is_dis)
        customers.append(cust)
        conversations.extend(convs)
    conversations.sort(key=lambda c: (c.timestamp, c.id))
    queries = [
        Query(customer_id=c.id, attribute=a, text=QUESTION.format(label=ATTRIBUTE_LABELS[a]), expected_value=v)
        for c in customers if not c.is_distractor for a, v in c.facts.items()
    ]
    assert len(customers) == total
    return Dataset(config=cfg, customers=customers, conversations=conversations, queries=queries)
