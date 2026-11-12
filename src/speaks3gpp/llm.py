"""Small Ollama client. num_ctx is always sent (a default 131072 context once made a 3B model take
23 GB and hang an 8 GB Mac in an earlier project), temperature 0 and a fixed seed."""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

NUM_CTX = 8192


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    wall_s: float
    raw: dict[str, Any] = field(default_factory=dict)


class LLM(Protocol):
    model: str

    def chat(self, system: str, user: str) -> Reply: ...


@dataclass
class OllamaLLM:
    model: str = "qwen2.5:3b"
    url: str = "http://localhost:11434"
    num_ctx: int = NUM_CTX
    num_predict: int = 8
    temperature: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_ctx > NUM_CTX:
            raise ValueError("num_ctx above 8192 is not allowed on an 8 GB machine")

    def chat(self, system: str, user: str) -> Reply:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }
        req = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
        except Exception as e:  # noqa: BLE001 - any transport problem is an LLMError
            raise LLMError(f"ollama chat failed: {e}") from e
        return Reply(
            text=data["message"]["content"],
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            wall_s=time.time() - t0,
            raw={k: data.get(k) for k in ("total_duration", "prompt_eval_duration", "eval_count")},
        )


@dataclass
class FakeLLM:
    """Test double: answers with a fixed option, or with whatever `rule(user)` returns."""

    model: str = "fake"
    answer: str = "1"
    rule: Any = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def chat(self, system: str, user: str) -> Reply:
        self.calls.append((system, user))
        text = self.rule(user) if self.rule else self.answer
        return Reply(text=text, prompt_tokens=len(user) // 4, wall_s=0.0)
