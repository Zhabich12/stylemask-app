"""Load data/fragments.jsonl and produce a fixed, reproducible train/test split."""
import json
import random
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

MIN_FRAGS, MAX_FRAGS = 30, 1500


CUSTOM_STYLES = Path("data/custom_styles.jsonl")  # styles added from the app
DEFAULT_SOURCES = (Path("data/fragments.jsonl"), Path("data/pikabu_fragments.jsonl"), CUSTOM_STYLES)


def load_split(paths=DEFAULT_SOURCES):
    """Return X_train, X_test, y_train, y_test from every existing file in `paths`.

    Authors with fewer than MIN_FRAGS fragments are dropped; the rest are
    capped at MAX_FRAGS so one chatty author doesn't dominate.
    """
    by_author = defaultdict(list)
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            by_author[row["author"]].append(row["text"])

    random.seed(0)
    texts, labels = [], []
    for author, items in by_author.items():
        if len(items) < MIN_FRAGS:
            continue
        for t in random.sample(items, min(len(items), MAX_FRAGS)):
            texts.append(t)
            labels.append(author)

    return train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=0)
