"""Turn measured LLM usage into money and machine-time, extrapolated to the daily volume."""
from __future__ import annotations

from .domains import load_costing


def cost_summary(n: int, llm_calls: int, prompt_tokens: int, completion_tokens: int, llm_seconds: float) -> dict:
    c = load_costing()
    api, local, volume = c["hosted_api"], c["local_compute"], c["daily_volume"]
    n = max(n, 1)
    api_cost_sample = (prompt_tokens * api["input_per_million_usd"] + completion_tokens * api["output_per_million_usd"]) / 1e6
    api_per_comment = api_cost_sample / n
    llm_hours_per_day = llm_seconds / n * volume / 3600
    return {
        "daily_volume": volume,
        "llm_calls_per_day": round(llm_calls / n * volume),
        "tokens_per_day": round((prompt_tokens + completion_tokens) / n * volume),
        "api_usd_per_day": round(api_per_comment * volume, 4),
        "api_usd_per_month": round(api_per_comment * volume * 30, 2),
        "llm_compute_hours_per_day": round(llm_hours_per_day, 2),
        "local_usd_per_day": round(llm_hours_per_day * local["usd_per_hour"], 2),
        "api_price_note": f"{api['name']}: ${api['input_per_million_usd']}/M in, ${api['output_per_million_usd']}/M out",
    }
