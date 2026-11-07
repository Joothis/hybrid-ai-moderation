from __future__ import annotations

from .domains import load_domain
from .llm import OllamaLLM
from .ml_model import MLModel
from .pipeline import HybridPipeline, LLMOnlyPipeline, MLOnlyPipeline, RulesMLPipeline
from .rules import RuleEngine


def build_pipelines(domain: str, llm=None, margin: float | None = None) -> dict:
    cfg = load_domain(domain)
    rules = RuleEngine.from_config(cfg["rules"])
    ml = MLModel.load(domain)
    llm = llm or OllamaLLM()
    return {
        "ml_only": MLOnlyPipeline(ml),
        "rules_ml": RulesMLPipeline(rules, ml),
        "llm_only": LLMOnlyPipeline(llm, cfg),
        "hybrid": HybridPipeline(rules, ml, llm, cfg, margin),
    }
