import json

import pytest

from moderation.domains import load_domain
from moderation.llm import MockLLM, build_messages, parse_response
from moderation.ml_model import _logit
from moderation.pipeline import HybridPipeline, LLMOnlyPipeline, MLOnlyPipeline
from moderation.rules import RuleEngine

CFG = load_domain("social")


class FakeML:
    """ML stand-in whose probability is chosen per text."""

    labels = CFG["label_names"]

    margin = 1.0

    def __init__(self, p_by_text):
        self.p = p_by_text
        self.thresholds = {l: 0.5 for l in self.labels}

    def proba_dict(self, text):
        p = self.p.get(text, 0.0)
        return {l: (p if l == "toxic" else 0.0) for l in self.labels}

    def scores(self, probs):
        return {l: float(_logit(probs[l]) - _logit(self.thresholds[l])) for l in self.labels}

    def predict_labels(self, probs):
        return [l for l in self.labels if probs[l] >= self.thresholds[l]]


def hybrid(p_by_text, llm=None):
    # margin 1.0 in log-odds around 0.5 -> unsure band is roughly p in (0.27, 0.73)
    return HybridPipeline(RuleEngine.from_config(CFG["rules"]), FakeML(p_by_text), llm or MockLLM(lambda t: ["insult"]), CFG, margin=1.0)


def test_rules_decide_first_and_skip_llm():
    llm = MockLLM()
    r = hybrid({}, llm).moderate("kys")
    assert r.layer == "rules" and r.flagged and llm.calls == 0


@pytest.mark.parametrize("p,layer,flagged", [(0.95, "ml", True), (0.05, "ml", False), (0.5, "llm", True)])
def test_routing_by_confidence(p, layer, flagged):
    llm = MockLLM(lambda t: ["insult"])
    r = hybrid({"some text": p}, llm).moderate("some text")
    assert r.layer == layer and r.flagged == flagged
    assert llm.calls == (1 if layer == "llm" else 0)
    assert r.llm_called == (layer == "llm")


def test_severity_floor_escalates_faint_high_risk_signal():
    cfg = {**CFG, "routing": {"escalation_budget": 0.2, "lo_by_label": {"threat": 0.03}}}
    h = HybridPipeline(RuleEngine([]), FakeML({}), MockLLM(), cfg, margin=1.0)
    assert h.route({"toxic": 0.1, "insult": 0, "obscene": 0, "threat": 0.05, "identity_attack": 0}) == "escalate"
    assert h.route({"toxic": 0.1, "insult": 0, "obscene": 0, "threat": 0.01, "identity_attack": 0}) == "clean"


def test_margin_calibration_hits_budget():
    import numpy as np
    from moderation.ml_model import MLModel

    class Stub(MLModel):
        def predict_proba(self, texts):
            return np.linspace(0.01, 0.99, len(texts))[:, None]

    m = Stub(["x"], estimator=object())
    margin = m.calibrate_margin(list(range(1000)), budget=0.2)
    S = np.abs(m.score_matrix(m.predict_proba(range(1000))).max(1))
    assert abs((S < margin).mean() - 0.2) < 0.01


def test_llm_failure_falls_back_to_ml():
    class Broken(MockLLM):
        def classify(self, text, cfg):
            r = super().classify(text, cfg)
            r.error = "ollama unreachable"
            return r

    r = hybrid({"x": 0.6}, Broken()).moderate("x")
    assert r.layer == "ml" and r.llm_error and r.flagged  # 0.6 >= 0.5 tuned threshold


def test_ml_only_and_llm_only():
    assert MLOnlyPipeline(FakeML({"a": 0.9})).moderate("a").labels == ["toxic"]
    r = LLMOnlyPipeline(MockLLM(lambda t: ["threat", "not_a_label"]), CFG).moderate("a")
    assert r.labels == ["threat"] and r.llm_called


def test_parse_response_filters_and_normalises():
    labels, conf, reason = parse_response(json.dumps({"labels": ["Toxic", "bogus"], "confidence": 3, "reason": "x"}), CFG["label_names"])
    assert labels == ["toxic"] and conf == 1.0
    with pytest.raises(ValueError):
        parse_response("not json", CFG["label_names"])
    with pytest.raises(ValueError):
        parse_response("[]", CFG["label_names"])


def test_prompt_contains_policy_labels_and_fewshots():
    msgs = build_messages("hello", CFG)
    assert msgs[0]["role"] == "system" and "identity_attack" in msgs[0]["content"]
    assert msgs[-1]["content"].endswith("<<<hello>>>")
    assert len(msgs) == 2 + 2 * len(CFG["llm"]["examples"])
