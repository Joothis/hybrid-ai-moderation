from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DOMAINS = ["social", "ecommerce", "support"]


@lru_cache(maxsize=None)
def load_domain(name: str) -> dict:
    path = CONFIG_DIR / "domains" / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["label_names"] = list(cfg["labels"].keys())
    return cfg


@lru_cache(maxsize=None)
def load_costing() -> dict:
    with open(CONFIG_DIR / "costing.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
