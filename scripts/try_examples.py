"""Run each domain's demo examples through the hybrid pipeline and show which layer decided.

    python scripts/try_examples.py [--domains social ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moderation.domains import DOMAINS, load_domain  # noqa: E402
from moderation.factory import build_pipelines  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=DOMAINS)
    args = ap.parse_args()
    mismatches = 0
    for d in args.domains:
        hybrid = build_pipelines(d)["hybrid"]
        print(f"\n=== {d} ===")
        for ex in load_domain(d)["demo_examples"]:
            r = hybrid.moderate(ex["text"])
            ok = "OK " if r.layer == ex["expect"] else "!! "
            mismatches += r.layer != ex["expect"]
            top = max(r.probs.values()) if r.probs else None
            print(f"{ok}{r.layer:5s} (expect {ex['expect']:5s}) {r.action:6s} {r.labels} "
                  f"max_p={top if top is None else round(top, 2)} {r.latency_ms:7.0f}ms | {ex['text'][:60]}")
            if r.layer == "llm":
                print(f"      reason: {r.reason}")
    print(f"\n{mismatches} example(s) decided by a different layer than expected")


if __name__ == "__main__":
    main()
