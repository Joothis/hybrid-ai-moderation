"""Train the Logistic Regression layer for each domain -> models/<domain>.joblib"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, ROOT, load_domain  # noqa: E402
from moderation.ml_model import MLModel  # noqa: E402
from moderation.rules import RuleEngine  # noqa: E402


def load_split(domain: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "processed" / domain / f"{split}.parquet")


def train(domain: str) -> None:
    cfg = load_domain(domain)
    labels = cfg["label_names"]
    tr, va = load_split(domain, "train"), load_split(domain, "val")
    t0 = time.perf_counter()
    model = MLModel(labels).fit(tr["text"], tr[labels].values)
    fit_s = time.perf_counter() - t0
    model.tune_thresholds(va["text"], va[labels].values)
    pred = model.predict_proba(va["text"]) >= [model.thresholds[l] for l in labels]
    y = va[labels].values
    flag_f1 = f1_score(y.any(1), pred.any(1))
    macro = f1_score(y, pred, average="macro", zero_division=0)

    # Turn the LLM escalation budget into a band width, ignoring comments the rules layer already decides.
    rules = RuleEngine.from_config(cfg["rules"])
    skip = np.array([rules.check(t) is not None for t in va["text"]])
    budget = cfg["routing"]["escalation_budget"]
    margin = model.calibrate_margin(va["text"], budget, skip)
    path = model.save(domain)
    print(f"{domain}: trained in {fit_s:.1f}s | val flag-F1 {flag_f1:.3f} | macro label-F1 {macro:.3f} | "
          f"rules decide {skip.mean():.1%} of val | band margin {margin:.2f} (budget {budget:.0%}) | "
          f"thresholds { {k: round(v, 3) for k, v in model.thresholds.items()} } -> {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    for d in ap.parse_args().domains:
        train(d)


if __name__ == "__main__":
    main()
