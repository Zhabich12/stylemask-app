"""Measure how well each attributor identifies authors on original texts.

Usage: py scripts/baseline.py [--kind ngram|style|delta|morph|all] [--chat-only]
"""
import argparse
import time
from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, classification_report

from stylemask.attribution import KINDS, build_attributor
from stylemask.dataset import DEFAULT_SOURCES, load_split

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="all", choices=(*KINDS, "all"))
ap.add_argument("--chat-only", action="store_true")
args = ap.parse_args()

X_tr, X_te, y_tr, y_te = load_split([Path("data/fragments.jsonl")] if args.chat_only else DEFAULT_SOURCES)
n_authors = len(set(y_tr))
print(f"authors: {n_authors}, train: {len(X_tr)}, test: {len(X_te)}, random guess: {1 / n_authors:.3f}\n")

for kind in KINDS if args.kind == "all" else (args.kind,):
    t = time.time()
    pred = build_attributor(kind).fit(X_tr, y_tr).predict(X_te)
    report = classification_report(y_te, pred, zero_division=0, output_dict=True)
    own = [a for a in sorted(set(y_te)) if not a.startswith("pikabu:")]
    print(f"[{kind:5}] balanced accuracy {balanced_accuracy_score(y_te, pred):.3f}  "
          f"({time.time() - t:.0f}s)  " + "  ".join(f"{a}: {report[a]['recall']:.2f}" for a in own))
