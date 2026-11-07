"""The moderation pipelines being compared. All share one interface: `moderate(text) -> ModerationResult`.

  ml_only   TF-IDF + Logistic Regression alone (per-label thresholds tuned on validation data)
  llm_only  every comment goes to the LLM
  rules_ml  rules, then ML (ablation: what the LLM layer adds on top)
  hybrid    rules -> ML -> LLM only for the ambiguous band  (the proposed system)
"""
from __future__ import annotations

import time

from .ml_model import MLModel
from .rules import RuleEngine
from .schema import ModerationResult


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


class Pipeline:
    name = "base"

    def moderate(self, text: str) -> ModerationResult:
        raise NotImplementedError

    def moderate_batch(self, texts, progress=None) -> list[ModerationResult]:
        out = []
        for i, t in enumerate(texts):
            out.append(self.moderate(t))
            if progress:
                progress(i + 1, len(texts))
        return out


def _ml_decision(ml: MLModel, text: str, t0: float) -> ModerationResult:
    probs = ml.proba_dict(text)
    labels = ml.predict_labels(probs)
    top = max(probs.values())
    conf = top if labels else 1 - top
    return ModerationResult(text, labels, "ml", conf, "Logistic Regression", _ms(t0), probs=probs)


def _rule_result(hit, text: str, t0: float) -> ModerationResult:
    return ModerationResult(text, hit.labels, "rules", 1.0, hit.reason, _ms(t0), rule_name=", ".join(hit.rule_names))


class MLOnlyPipeline(Pipeline):
    name = "ml_only"

    def __init__(self, ml: MLModel):
        self.ml = ml

    def moderate(self, text: str) -> ModerationResult:
        return _ml_decision(self.ml, text, time.perf_counter())


class RulesMLPipeline(Pipeline):
    name = "rules_ml"

    def __init__(self, rules: RuleEngine, ml: MLModel):
        self.rules, self.ml = rules, ml

    def moderate(self, text: str) -> ModerationResult:
        t0 = time.perf_counter()
        if hit := self.rules.check(text):
            return _rule_result(hit, text, t0)
        return _ml_decision(self.ml, text, t0)


class LLMOnlyPipeline(Pipeline):
    name = "llm_only"

    def __init__(self, llm, cfg: dict):
        self.llm, self.cfg = llm, cfg

    def moderate(self, text: str) -> ModerationResult:
        r = self.llm.classify(text, self.cfg)
        return ModerationResult(text, r.labels, "llm", r.confidence, r.reason, r.latency_ms, True,
                                r.prompt_tokens, r.completion_tokens, llm_error=r.error)


class HybridPipeline(Pipeline):
    name = "hybrid"

    def __init__(self, rules: RuleEngine, ml: MLModel, llm, cfg: dict, margin: float | None = None):
        self.rules, self.ml, self.llm, self.cfg = rules, ml, llm, cfg
        # Band half-width in log-odds around each label's decision threshold (default: calibrated at training
        # time from the domain's escalation budget).
        self.margin = ml.margin if margin is None else margin
        # Severity-aware routing: for high-risk labels (e.g. self_harm) any probability above this floor
        # gets a second opinion from the LLM instead of being auto-allowed.
        self.floors = (cfg.get("routing") or {}).get("lo_by_label") or {}

    def route(self, probs: dict[str, float]) -> str:
        """'flag' | 'clean' | 'escalate' based on the label closest to (or furthest past) its threshold."""
        top = max(self.ml.scores(probs).values())
        if top >= self.margin:
            return "flag"
        if top <= -self.margin and all(probs[l] <= f for l, f in self.floors.items() if l in probs):
            return "clean"
        return "escalate"

    def moderate(self, text: str) -> ModerationResult:
        t0 = time.perf_counter()
        # Layer 1: rules
        if hit := self.rules.check(text):
            return _rule_result(hit, text, t0)
        # Layer 2: ML
        probs = self.ml.proba_dict(text)
        route = self.route(probs)
        scores = self.ml.scores(probs)
        top_label = max(scores, key=scores.get)
        if route == "flag":
            labels = sorted(set(self.ml.predict_labels(probs)) | {top_label})
            return ModerationResult(text, labels, "ml", probs[top_label],
                                    f"ML confident: {top_label} p={probs[top_label]:.2f}", _ms(t0), probs=probs)
        if route == "clean":
            return ModerationResult(text, [], "ml", 1 - probs[top_label],
                                    f"ML confident clean (highest: {top_label} p={probs[top_label]:.2f})", _ms(t0), probs=probs)
        # Layer 3: LLM for the ambiguous band
        local_ms = _ms(t0)
        r = self.llm.classify(text, self.cfg)
        if r.error:  # graceful degradation: fall back to the ML decision
            res = _ml_decision(self.ml, text, t0)
            res.llm_called, res.llm_error, res.reason = True, r.error, f"LLM failed, ML fallback: {r.error}"
            res.prompt_tokens, res.completion_tokens = r.prompt_tokens, r.completion_tokens
            res.latency_ms = local_ms + r.latency_ms
            return res
        return ModerationResult(text, r.labels, "llm", r.confidence, r.reason, local_ms + r.latency_ms, True,
                                r.prompt_tokens, r.completion_tokens, probs=probs)
