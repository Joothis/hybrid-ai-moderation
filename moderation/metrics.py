from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from .costing import cost_summary
from .schema import ModerationResult


def label_matrix(results: list[ModerationResult], labels: list[str]) -> np.ndarray:
    return np.array([[int(l in r.labels) for l in labels] for r in results])


def flag_metrics(y_flag: np.ndarray, p_flag: np.ndarray) -> dict:
    tp = int((y_flag & p_flag).sum())
    fp = int((~y_flag & p_flag).sum())
    fn = int((y_flag & ~p_flag).sum())
    tn = int((~y_flag & ~p_flag).sum())
    return {
        "f1": round(f1_score(y_flag, p_flag, zero_division=0), 4),
        "precision": round(precision_score(y_flag, p_flag, zero_division=0), 4),
        "recall": round(recall_score(y_flag, p_flag, zero_division=0), 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def evaluate(results: list[ModerationResult], Y: np.ndarray, labels: list[str]) -> dict:
    P = label_matrix(results, labels)
    y_flag, p_flag = Y.any(1).astype(bool), P.any(1).astype(bool)
    lat = np.array([r.latency_ms for r in results])
    llm = [r for r in results if r.llm_called]
    p_tok = sum(r.prompt_tokens for r in llm)
    c_tok = sum(r.completion_tokens for r in llm)
    llm_seconds = sum(r.latency_ms for r in llm) / 1000
    layers = {k: sum(r.layer == k for r in results) for k in ("rules", "ml", "llm")}
    per_label = {l: round(f1_score(Y[:, j], P[:, j], zero_division=0), 4) for j, l in enumerate(labels)}
    return {
        "n": len(results),
        "flag": flag_metrics(y_flag, p_flag),
        "macro_label_f1": round(float(np.mean(list(per_label.values()))), 4),
        "per_label_f1": per_label,
        "latency_ms": {
            "mean": round(float(lat.mean()), 1), "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1), "total_s": round(float(lat.sum() / 1000), 1),
        },
        "llm_calls": len(llm),
        "escalation_rate": round(len(llm) / max(len(results), 1), 4),
        "llm_errors": sum(bool(r.llm_error) for r in results),
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "decided_by": layers,
        "cost": cost_summary(len(results), len(llm), p_tok, c_tok, llm_seconds),
    }
