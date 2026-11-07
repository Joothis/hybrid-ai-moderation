from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ModerationResult:
    """One moderation decision, with enough detail to explain *which layer* decided and why."""

    text: str
    labels: list[str]
    layer: str  # "rules" | "ml" | "llm"
    confidence: float
    reason: str = ""
    latency_ms: float = 0.0
    llm_called: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    probs: dict[str, float] = field(default_factory=dict)
    rule_name: str | None = None
    llm_error: str | None = None

    @property
    def flagged(self) -> bool:
        return bool(self.labels)

    @property
    def action(self) -> str:
        return "REMOVE" if self.flagged else "ALLOW"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flagged"] = self.flagged
        d["action"] = self.action
        return d
