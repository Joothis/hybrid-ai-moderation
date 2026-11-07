"""Print a markdown results table from results/*.json (pasted into the README)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, ROOT  # noqa: E402

NAMES = {"ml_only": "ML only", "llm_only": "LLM only", "rules_ml": "Rules + ML", "hybrid": "**Hybrid**"}


def main() -> None:
    print("| Use case | Pipeline | F1 | Precision | Recall | FP | FN | LLM calls | p50 latency | p95 latency | LLM calls/day @10k | API $/month | LLM compute h/day |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in DOMAINS:
        p = ROOT / "results" / f"{d}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for k in ("ml_only", "llm_only", "rules_ml", "hybrid"):
            m = r["metrics"][k]
            f, lat, c = m["flag"], m["latency_ms"], m["cost"]
            print(f"| {r['display_name']} (n={r['sample']}) | {NAMES[k]} | {f['f1']:.3f} | {f['precision']:.3f} | {f['recall']:.3f} | "
                  f"{f['fp']} | {f['fn']} | {m['llm_calls']} | {lat['p50']:,.0f} ms | {lat['p95']:,.0f} ms | "
                  f"{c['llm_calls_per_day']:,} | ${c['api_usd_per_month']:,.2f} | {c['llm_compute_hours_per_day']} |")


if __name__ == "__main__":
    main()
