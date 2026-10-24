"""User feedback on rewrites, kept per persona and fed back into the style prompt.

data/feedback/<persona>.jsonl, one event per line:
  {"kind": "good",    "result": ...}                   — "this sounds like them"
  {"kind": "bad",     "result": ...}                   — "doesn't sound like them"
  {"kind": "fixed",   "result": ..., "fixed": ...}      — user's own corrected version
  {"kind": "note",    "note": ...}                      — free-text remark about the style
Good and fixed results become extra example messages (fixed ones first: they're
exactly the target style); notes become rules with top priority.
"""
import json
import re
from pathlib import Path

DIR = Path("data/feedback")


class FeedbackStore:
    def __init__(self, persona: str):
        safe = re.sub(r"[^\w.-]+", "_", persona)
        self.path = DIR / f"{safe}.jsonl"

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def add(self, kind: str, **fields) -> None:
        DIR.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")

    def notes(self) -> list[str]:
        return [e["note"] for e in self._events() if e["kind"] == "note"]

    def examples(self) -> list[str]:
        """Approved messages in the target style, newest and hand-fixed first."""
        events = self._events()[::-1]
        fixed = [e["fixed"] for e in events if e["kind"] == "fixed"]
        good = [e["result"] for e in events if e["kind"] == "good"]
        return fixed + good

    def counts(self) -> dict[str, int]:
        out = {"good": 0, "bad": 0, "fixed": 0, "note": 0}
        for e in self._events():
            out[e["kind"]] += 1
        return out
