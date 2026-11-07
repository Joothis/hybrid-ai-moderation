"""Layer 3: LLM judgement via a local Ollama model.

- Domain policy + label definitions + few-shot examples are built from the domain YAML.
- JSON-mode output: {"labels": [...], "confidence": 0-1, "reason": "..."}; invalid JSON -> one retry.
- Every response is cached on disk (keyed by model + prompt + text) together with its original latency
  and token counts, so benchmark reruns are free but still report real costs.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from .domains import ROOT

DEFAULT_MODEL = os.environ.get("MODERATION_LLM_MODEL", "qwen3:8b")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CACHE_DIR = ROOT / "cache"
MAX_CHARS = 1500


@dataclass
class LLMResult:
    labels: list[str]
    confidence: float
    reason: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    error: str | None = None


def build_system_prompt(cfg: dict) -> str:
    label_lines = "\n".join(f"- {name}: {desc}" for name, desc in cfg["labels"].items())
    return (
        f"{cfg['llm']['policy'].strip()}\n\n"
        f"Labels you may assign (zero, one or several):\n{label_lines}\n\n"
        'Respond with ONLY a JSON object: {"labels": [label names], "confidence": number 0-1, '
        '"reason": "max 15 words"}. Use an empty labels list when the text is acceptable.'
    )


def build_messages(text: str, cfg: dict) -> list[dict]:
    msgs = [{"role": "system", "content": build_system_prompt(cfg)}]
    for ex in cfg["llm"].get("examples", []):
        msgs.append({"role": "user", "content": f"Text to moderate:\n<<<{ex['text']}>>>"})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"labels": ex["labels"], "confidence": 0.9, "reason": ex["reason"]})})
    msgs.append({"role": "user", "content": f"Text to moderate:\n<<<{text[:MAX_CHARS]}>>>"})
    return msgs


def parse_response(content: str, allowed: list[str]) -> tuple[list[str], float, str]:
    """Raises ValueError on unusable output."""
    data = json.loads(content)
    if not isinstance(data, dict) or "labels" not in data:
        raise ValueError("missing 'labels'")
    raw = data["labels"] or []
    if isinstance(raw, str):
        raw = [raw]
    allowed_set = {a.lower(): a for a in allowed}
    labels = sorted({allowed_set[l.strip().lower()] for l in raw if isinstance(l, str) and l.strip().lower() in allowed_set})
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return labels, min(max(conf, 0.0), 1.0), str(data.get("reason", ""))[:300]


class LLMCache:
    def __init__(self, path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self.data[row["key"]] = row["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def get(self, key: str) -> dict | None:
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        with self.lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


class OllamaLLM:
    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST, use_cache: bool = True, timeout: float = 180):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.cache = LLMCache(CACHE_DIR / f"llm_{model.replace(':', '_').replace('/', '_')}.jsonl") if use_cache else None

    # ------------------------------------------------------------------ plumbing
    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.host}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read()).get("models", [])]
            return any(n == self.model or n.split(":")[0] == self.model for n in names)
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _key(self, messages: list[dict]) -> str:
        return hashlib.sha256((self.model + json.dumps(messages, sort_keys=True)).encode()).hexdigest()

    # ------------------------------------------------------------------ API
    def classify(self, text: str, cfg: dict) -> LLMResult:
        messages = build_messages(text, cfg)
        key = self._key(messages)
        if self.cache and (hit := self.cache.get(key)):
            return LLMResult(**{**hit, "cached": True})

        allowed = cfg["label_names"]
        t0 = time.perf_counter()
        p_tok = c_tok = 0
        error = None
        labels, conf, reason = [], 0.0, ""
        for _attempt in range(2):
            try:
                resp = self._post("/api/chat", {
                    "model": self.model, "messages": messages, "stream": False, "format": "json",
                    "think": False, "options": {"temperature": 0, "num_predict": 120},
                })
                p_tok += resp.get("prompt_eval_count", 0)
                c_tok += resp.get("eval_count", 0)
                labels, conf, reason = parse_response(resp["message"]["content"], allowed)
                error = None
                break
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                error = f"bad output: {e}"
            except (urllib.error.URLError, OSError) as e:
                error = f"ollama unreachable: {e}"
                break
        result = LLMResult(labels, conf, reason, (time.perf_counter() - t0) * 1000, p_tok, c_tok, False, error)
        if self.cache and error is None:
            self.cache.put(key, {k: v for k, v in asdict(result).items() if k != "cached"})
        return result


class MockLLM:
    """Deterministic stand-in (tests / no Ollama). `fn(text) -> list[str]`."""

    model = "mock"

    def __init__(self, fn=None, latency_ms: float = 1500.0, tokens: tuple[int, int] = (450, 30)):
        self.fn = fn or (lambda text: [])
        self.latency_ms = latency_ms
        self.tokens = tokens
        self.calls = 0

    def available(self) -> bool:
        return True

    def classify(self, text: str, cfg: dict) -> LLMResult:
        self.calls += 1
        labels = [l for l in self.fn(text) if l in cfg["label_names"]]
        return LLMResult(labels, 0.9, "mock decision", self.latency_ms, *self.tokens)
