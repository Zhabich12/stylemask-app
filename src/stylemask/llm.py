"""Minimal client for OpenAI-compatible servers (LM Studio, Ollama, or a remote API)."""
from urllib.parse import urlparse

import httpx

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class LLM:
    def __init__(self, model: str, base_url: str = "http://localhost:1234/v1",
                 api_key: str = "lm-studio", local_only: bool = True):
        # private data (chat exports) may only ever be sent to a local server
        if local_only and urlparse(base_url).hostname not in LOCAL_HOSTS:
            raise ValueError(f"local_only=True but {base_url} is not a local server")
        self.model = model
        self.http = httpx.Client(base_url=base_url, timeout=300,
                                 headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 2048,
             reasoning_effort: str | None = "none") -> str:
        # reasoning ("thinking") is useless for rewriting and makes it ~20x slower
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        r = self.http.post("/chat/completions", json=body)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        r = self.http.post("/embeddings", json={"model": model, "input": texts})
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
