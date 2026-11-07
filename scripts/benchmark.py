"""Compare ML-only vs LLM-only vs Hybrid (+ a Rules+ML ablation) on the same test sample.

    python scripts/benchmark.py --sample 300            # all domains
    python scripts/benchmark.py --domains social --sample 100

Writes results/<domain>.json (metrics + threshold trade-off curve) and results/<domain>_items.parquet
(per-comment decisions of every pipeline, used by the app). LLM responses are cached in cache/, so a
rerun only calls the LLM for comments it has not seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, ROOT, load_domain  # noqa: E402
from moderation.factory import build_pipelines  # noqa: E402
from moderation.metrics import evaluate, flag_metrics  # noqa: E402

RESULTS = ROOT / "results"
ORDER = ["ml_only", "llm_only", "rules_ml", "hybrid"]


def tradeoff_curve(items: pd.DataFrame, labels: list[str], cfg: dict, ml) -> list[dict]:
    """What-if sweep of the hybrid's band width (margin), re-using every pipeline's stored per-item outputs.

    margin = 0 means "never escalate" (= Rules+ML); a huge margin means "always escalate" (~ LLM-only).
    """
    P = np.vstack(items["probs"].map(lambda d: [d[l] for l in labels]).values)
    S = ml.score_matrix(P).max(1)
    y = items[labels].values.any(1)
    rule_hit, rule_flag = items["rule_hit"].values, items["rule_flag"].values
    llm_flag = items["llm_only_flag"].values
    floors = (cfg.get("routing") or {}).get("lo_by_label") or {}
    floor_ok = np.ones(len(items), dtype=bool)
    for l, f in floors.items():
        floor_ok &= P[:, labels.index(l)] <= f
    points = []
    for margin in np.round(np.arange(0.0, 4.01, 0.1), 2):
        flag_ml = S >= margin
        clean_ml = (S <= -margin) & floor_ok & ~flag_ml
        if margin == 0:  # degenerate: everything decided by ML threshold (floors ignored)
            clean_ml = ~flag_ml
        esc = ~rule_hit & ~flag_ml & ~clean_ml
        pred = np.where(rule_hit, rule_flag, np.where(flag_ml, True, np.where(clean_ml, False, llm_flag)))
        m = flag_metrics(y, pred.astype(bool))
        points.append({"margin": float(margin), "f1": m["f1"], "escalation_rate": round(float(esc.mean()), 4),
                       "fp": m["fp"], "fn": m["fn"]})
    return points


def run_domain(domain: str, sample: int, seed: int) -> dict:
    cfg = load_domain(domain)
    labels = cfg["label_names"]
    test = pd.read_parquet(ROOT / "data" / "processed" / domain / "test.parquet")
    df = test.sample(n=min(sample, len(test)), random_state=seed).reset_index(drop=True)
    texts, Y = df["text"].tolist(), df[labels].values
    pipes = build_pipelines(domain)
    if not pipes["llm_only"].llm.available():
        raise SystemExit(f"Ollama model '{pipes['llm_only'].llm.model}' not reachable - start Ollama / pull the model first.")

    results, metrics = {}, {}
    for name in ORDER:
        t0 = time.perf_counter()

        def progress(i, n, name=name):
            if i % 25 == 0 or i == n:
                print(f"  [{domain}] {name}: {i}/{n} ({time.perf_counter() - t0:.0f}s)", flush=True)

        results[name] = pipes[name].moderate_batch(texts, progress)
        metrics[name] = evaluate(results[name], Y, labels)
        m = metrics[name]
        print(f"  [{domain}] {name:8s} flag-F1 {m['flag']['f1']:.3f}  FP {m['flag']['fp']:3d}  FN {m['flag']['fn']:3d}  "
              f"LLM calls {m['llm_calls']:4d}  p50 {m['latency_ms']['p50']:7.1f}ms  "
              f"API $/day {m['cost']['api_usd_per_day']}", flush=True)

    items = df.copy()
    rules = pipes["hybrid"].rules
    hits = [rules.check(t) for t in texts]
    items["rule_hit"] = [h is not None for h in hits]
    items["rule_flag"] = [bool(h and h.labels) for h in hits]
    items["probs"] = [pipes["hybrid"].ml.proba_dict(t) for t in texts]
    for name in ORDER:
        items[f"{name}_labels"] = [r.labels for r in results[name]]
        items[f"{name}_flag"] = [r.flagged for r in results[name]]
    items["hybrid_layer"] = [r.layer for r in results["hybrid"]]
    items["hybrid_reason"] = [r.reason for r in results["hybrid"]]
    items["llm_reason"] = [r.reason for r in results["llm_only"]]

    out = {
        "domain": domain, "display_name": cfg["display_name"], "sample": len(df), "seed": seed,
        "llm_model": pipes["llm_only"].llm.model, "routing": cfg["routing"], "margin": pipes["hybrid"].margin,
        "positive_rate": round(float(Y.any(1).mean()), 4), "metrics": metrics,
        "tradeoff": tradeoff_curve(items, labels, cfg, pipes["hybrid"].ml),
        "generated_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{domain}.json").write_text(json.dumps(out, indent=2))
    items.drop(columns=["probs"]).assign(probs=items["probs"].map(json.dumps)).to_parquet(RESULTS / f"{domain}_items.parquet")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for d in args.domains:
        print(f"== {d} (sample {args.sample}) ==", flush=True)
        run_domain(d, args.sample, args.seed)
    print("done -> results/")


if __name__ == "__main__":
    main()
