"""Layer 2: TF-IDF (word + char n-grams) + one-vs-rest Logistic Regression.

Gives a calibrated-ish probability per label. The hybrid pipeline uses those probabilities to decide
whether the model is confident (done) or unsure (escalate to the LLM).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from .domains import ROOT

MODEL_DIR = ROOT / "models"


def build_estimator() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True)),
        ]
    )
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=2000, solver="liblinear"))
    return Pipeline([("features", features), ("clf", clf)])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class MLModel:
    def __init__(self, labels: list[str], estimator: Pipeline | None = None, thresholds: dict[str, float] | None = None,
                 margin: float = 1.0):
        self.labels = labels
        self.estimator = estimator or build_estimator()
        # Per-label decision thresholds used when the ML model decides alone (tuned on validation data).
        self.thresholds = thresholds or {l: 0.5 for l in labels}
        # Width of the "unsure" band around each threshold, in log-odds; set from the escalation budget.
        self.margin = margin

    def scores(self, probs: dict[str, float]) -> dict[str, float]:
        """Signed distance of each label from its decision threshold in log-odds (>0 means 'violation')."""
        return {l: float(_logit(probs[l]) - _logit(self.thresholds[l])) for l in self.labels}

    def score_matrix(self, P: np.ndarray) -> np.ndarray:
        return _logit(P) - _logit(np.array([self.thresholds[l] for l in self.labels]))

    def calibrate_margin(self, texts, budget: float, skip: np.ndarray | None = None) -> float:
        """Choose the band width so that ~`budget` of the comments reaching the ML layer are escalated.

        A comment is unsure when its strongest label score lies within (-margin, +margin).
        `skip` marks comments already decided by the rules layer.
        """
        S = self.score_matrix(self.predict_proba(texts)).max(1)
        if skip is not None:
            S = S[~skip]
        self.margin = float(np.quantile(np.abs(S), budget)) if len(S) else 1.0
        return self.margin

    def fit(self, texts, Y: np.ndarray) -> "MLModel":
        self.estimator.fit(list(texts), Y)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        return self.estimator.predict_proba(list(texts))

    def proba_dict(self, text: str) -> dict[str, float]:
        p = self.predict_proba([text])[0]
        return {l: float(v) for l, v in zip(self.labels, p)}

    def predict_labels(self, probs: dict[str, float]) -> list[str]:
        return [l for l in self.labels if probs[l] >= self.thresholds[l]]

    def tune_thresholds(self, texts, Y: np.ndarray) -> dict[str, float]:
        """Pick, per label, the probability cut-off that maximises F1 on held-out data."""
        P = self.predict_proba(texts)
        grid = np.concatenate([[0.01, 0.02, 0.03, 0.04], np.arange(0.05, 0.96, 0.025)])
        for j, label in enumerate(self.labels):
            if Y[:, j].sum() == 0:
                continue
            scores = [f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in grid]
            self.thresholds[label] = float(grid[int(np.argmax(scores))])
        return self.thresholds

    def save(self, domain: str) -> Path:
        MODEL_DIR.mkdir(exist_ok=True)
        path = MODEL_DIR / f"{domain}.joblib"
        joblib.dump({"labels": self.labels, "estimator": self.estimator, "thresholds": self.thresholds,
                     "margin": self.margin}, path)
        return path

    @classmethod
    def load(cls, domain: str) -> "MLModel":
        d = joblib.load(MODEL_DIR / f"{domain}.joblib")
        return cls(d["labels"], d["estimator"], d["thresholds"], d.get("margin", 1.0))
