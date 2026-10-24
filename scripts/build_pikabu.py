"""Build data/pikabu_fragments.jsonl from a shard of the IlyaGusev/pikabu dataset.

Two streaming passes over the .jsonl.zst shard (it doesn't fit in RAM decompressed):
1. count comment characters per author;
2. collect comments of the N most prolific authors, capped per author.

Usage: py scripts/build_pikabu.py data/raw/pikabu_00.jsonl.zst [n_authors]
"""
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import zstandard

from stylemask.telegram import to_fragments

MIN_COMMENT_CHARS = 20
MAX_CHARS_PER_AUTHOR = 120_000  # ~300 fragments of 400 chars

_URL = re.compile(r"https?://\S+")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def clean(text: str) -> str:
    return _URL.sub("", _MD_IMAGE.sub("", text or "")).strip()


def comments(path: Path):
    """Yield (author, text) for every comment in the shard, in file order."""
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with path.open("rb") as raw, dctx.stream_reader(raw) as reader:
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            # comments are stored column-wise: {"username": [...], "text_markdown": [...], ...}
            c = json.loads(line).get("comments") or {}
            for author, raw_text in zip(c.get("username") or [], c.get("text_markdown") or []):
                text = clean(raw_text)
                if author and len(text) >= MIN_COMMENT_CHARS:
                    yield author, text


shard = Path(sys.argv[1])
n_authors = int(sys.argv[2]) if len(sys.argv) > 2 else 50

volume = Counter()
for author, text in comments(shard):
    volume[author] += len(text)
chosen = {a for a, _ in volume.most_common(n_authors)}
print(f"pass 1: {len(volume)} authors, keeping top {len(chosen)}")

collected, used = [], Counter()
for author, text in comments(shard):
    if author in chosen and used[author] < MAX_CHARS_PER_AUTHOR:
        collected.append((f"pikabu:{author}", text))
        used[author] += len(text)

frags = to_fragments(collected)
out = Path("data/pikabu_fragments.jsonl")
with out.open("w", encoding="utf-8") as f:
    for author, text in frags:
        f.write(json.dumps({"author": author, "text": text}, ensure_ascii=False) + "\n")

counts = sorted(Counter(a for a, _ in frags).values())
print(f"pass 2: {len(frags)} fragments, per author min {counts[0]}, median {counts[len(counts) // 2]}, max {counts[-1]}")
print(f"-> {out}")
