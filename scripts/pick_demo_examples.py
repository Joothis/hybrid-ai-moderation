"""List real test comments that make good demo material, from results/<domain>_items.parquet.

  llm_saves   hybrid escalated to the LLM and was right where ML-only was wrong
  ml_wins     ML confidently right, LLM-only wrong (the LLM is not always better)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, ROOT, load_domain  # noqa: E402


def main() -> None:
    for d in DOMAINS:
        p = ROOT / "results" / f"{d}_items.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        labels = load_domain(d)["label_names"]
        df["truth"] = df[labels].any(axis=1)
        short = df["text"].str.len() < 220
        saves = df[short & (df["hybrid_layer"] == "llm") & (df["hybrid_flag"] == df["truth"]) & (df["ml_only_flag"] != df["truth"])]
        ml_wins = df[short & (df["hybrid_layer"] == "ml") & (df["hybrid_flag"] == df["truth"]) & (df["llm_only_flag"] != df["truth"])]
        print(f"\n=== {d}: {len(saves)} LLM saves, {len(ml_wins)} ML-right/LLM-wrong ===")
        for title, part in (("LLM saves", saves), ("ML right, LLM-only wrong", ml_wins)):
            print(f"-- {title}")
            for _, r in part.head(8).iterrows():
                print(f"  truth={'FLAG' if r['truth'] else 'ok  '} | {r['text'][:200]!r}\n      LLM: {r['llm_reason']}")


if __name__ == "__main__":
    main()
