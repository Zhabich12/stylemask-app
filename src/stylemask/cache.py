"""Fitted attributors cached on disk — the 53-author n-gram model takes minutes to train."""
from pathlib import Path

import joblib

from stylemask.attribution import build_attributor

CACHE = Path("data/cache")


def cached_attacker(kind: str, setting: str, X_train, y_train):
    """Load the `kind` attributor for `setting` ("chat" or "all"), training it on first use.

    The cache key does not include the data: delete data/cache/ after rebuilding datasets.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"attacker_{setting}_{kind}.joblib"
    if not path.exists():
        joblib.dump(build_attributor(kind).fit(X_train, y_train), path)
    return joblib.load(path)
