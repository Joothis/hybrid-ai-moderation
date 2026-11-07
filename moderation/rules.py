"""Layer 1: deterministic, high-precision rules.

Rules only fire on clear-cut cases. Each rule is matched against one of three views of the text:
  raw     - original text (needed for PII like emails, card numbers, URLs)
  norm    - lowercased, leetspeak undone, long letter runs collapsed ("Fuuuck" -> "fuck", "1d10t" -> "idiot")
  squash  - norm with every non-alphanumeric char removed ("f.u.c.k y.o.u" -> "fuckyou")
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_RUNS = re.compile(r"([a-z])\1{2,}")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    t = text.lower().translate(_LEET)
    return _RUNS.sub(r"\1", t)


def squash(text: str) -> str:
    return _NON_ALNUM.sub("", normalize(text))


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    labels: list[str]
    action: str = "block"  # "block" or "allow"
    view: str = "norm"  # "raw" | "norm" | "squash"
    reason: str = ""


@dataclass
class RuleHit:
    action: str
    labels: list[str]
    rule_names: list[str]
    reason: str


class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules

    @classmethod
    def from_config(cls, cfg: list[dict]) -> "RuleEngine":
        rules = []
        for r in cfg or []:
            flags = 0 if r.get("view", "norm") == "raw" and r.get("case_sensitive") else re.IGNORECASE
            rules.append(
                Rule(
                    name=r["name"],
                    pattern=re.compile(r["pattern"], flags),
                    labels=list(r.get("labels", [])),
                    action=r.get("action", "block"),
                    view=r.get("view", "norm"),
                    reason=r.get("reason", r["name"]),
                )
            )
        return cls(rules)

    def check(self, text: str) -> RuleHit | None:
        """Return a hit if any block rule fires (union of labels), else an allow hit, else None."""
        views = {"raw": text, "norm": normalize(text), "squash": squash(text)}
        blocks = [r for r in self.rules if r.action == "block" and r.pattern.search(views[r.view])]
        if blocks:
            labels = sorted({l for r in blocks for l in r.labels})
            return RuleHit("block", labels, [r.name for r in blocks], "; ".join(r.reason for r in blocks))
        for r in self.rules:
            if r.action == "allow" and r.pattern.search(views[r.view]):
                return RuleHit("allow", [], [r.name], r.reason)
        return None
