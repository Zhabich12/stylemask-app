"""Build data/fragments.jsonl from Telegram export folders.

Usage: py scripts/build_dataset.py <export_dir> [<export_dir> ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

from stylemask.telegram import parse_export, to_fragments

out_path = Path("data/fragments.jsonl")
out_path.parent.mkdir(exist_ok=True)

frags = []
for folder in sys.argv[1:]:
    frags += to_fragments(parse_export(Path(folder)))

with out_path.open("w", encoding="utf-8") as f:
    for author, text in frags:
        f.write(json.dumps({"author": author, "text": text}, ensure_ascii=False) + "\n")

for author, n in Counter(a for a, _ in frags).most_common():
    print(f"{n:6d}  {author}")
print(f"-> {out_path}")
