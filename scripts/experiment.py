"""Rewrite test fragments of several authors with one method and measure
whether they can still be told apart.

Measuring only "is the user still recognised" is misleading: a rewritten
text that looks like nobody falls into whatever class is the default sink.
So we rewrite every evaluated author and report balanced accuracy:
- static:   attackers trained on original texts, tested on rewrites;
- adaptive: attacker trained *on rewrites* (5-fold CV) — an adversary who
            knows our tool and has samples of its output;
- meaning:  embedding similarity original vs. rewrite.
Chance level is 1 / number of rewritten authors (adaptive) or
1 / number of attacker classes (static).

Usage:
  py scripts/experiment.py --method neutral|persona|loop [--setting chat|all]
                           [--n 40] [--persona pikabu:<name>] [--model qwen/qwen3.5-4b]
Metrics (no texts) are appended to data/experiments.jsonl.
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from stylemask import config
from stylemask.attribution import build_attributor
from stylemask.cache import CACHE, cached_attacker
from stylemask.dataset import DEFAULT_SOURCES, load_split
from stylemask.llm import LLM
from stylemask.personas import build_rewriter
from stylemask.rewrite import HintMaker, Rewriter

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=("neutral", "persona", "loop"))
ap.add_argument("--setting", default="chat", choices=("chat", "all"))
ap.add_argument("--n", type=int, default=40, help="fragments per rewritten author")
ap.add_argument("--persona", default=None)
ap.add_argument("--shots", type=int, default=6)
ap.add_argument("--model", default="qwen/qwen3.5-4b")
ap.add_argument("--attackers", default="style,morph,delta,ngram")
ap.add_argument("--legacy", action="store_true",
                help="v1 rewriter: one pass, temperature 0.7, 6 random examples, no style card, no meaning check")
args = ap.parse_args()
cfg = config.load()

# --- data -----------------------------------------------------------------
sources = [Path("data/fragments.jsonl")] if args.setting == "chat" else list(DEFAULT_SOURCES)
X_tr, X_te, y_tr, y_te = load_split(sources)
P_tr, _, py_tr, _ = load_split(DEFAULT_SOURCES)  # persona examples always from the full pool

random.seed(0)
pikabu = sorted(a for a in set(py_tr) if a.startswith("pikabu:"))
persona = args.persona or random.choice(pikabu)
chat_authors = sorted(a for a in set(y_tr) if not a.startswith("pikabu:"))
authors = chat_authors if args.setting == "chat" else chat_authors + random.sample(
    [a for a in pikabu if a != persona], 5)

random.seed(1)
originals, labels = [], []
for a in authors:
    pool = [x for x, y in zip(X_te, y_te) if y == a]
    for x in random.sample(pool, min(args.n, len(pool))):
        originals.append(x)
        labels.append(a)
labels = np.array(labels)

# --- attackers (cached: the 53-author n-gram model takes minutes) ----------
attackers = {kind: cached_attacker(kind, args.setting, X_tr, y_tr) for kind in args.attackers.split(",")}

# --- rewriting (cached per method/persona/model/text) ----------------------
llm = LLM(args.model)
rewriters = {}
if args.legacy:
    shots = random.Random(2).sample([x for x, y in zip(P_tr, py_tr) if y == persona], args.shots)
    legacy = dict(check=False, temperature=0.7, n_examples=args.shots)
for a in authors:
    if args.legacy:
        hm = HintMaker(X_tr, y_tr, a, target=persona if persona in set(y_tr) else None) if args.method == "loop" else None
        rewriters[a] = Rewriter(llm, args.method, shots, guide=attackers.get("style"), author=a, hint_maker=hm, **legacy)
    else:
        # the evaluated author plays "me": the loop steers away from *their* style
        rewriters[a] = build_rewriter(llm, args.method, persona, P_tr, py_tr, {**cfg, "me": a},
                                      guide=attackers.get("style"))

cache_path = CACHE / "rewrites.jsonl"
cache = {}
if cache_path.exists():
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        cache[row["key"]] = row["rewritten"]

version = "" if args.legacy else "|v3"  # legacy keeps v1 cache keys, so old rewrites are reused as is
tag = f"{args.method}|{persona if args.method != 'neutral' else ''}|{args.model}|{args.shots}{version}"
rewritten, warned, t0 = [], 0, time.time()
with cache_path.open("a", encoding="utf-8") as f:
    for i, (text, a) in enumerate(zip(originals, labels), 1):
        key = hashlib.sha1(f"{tag}|{text}".encode()).hexdigest()
        if key not in cache:
            cache[key] = rewriters[a].rewrite(text)
            warned += bool(rewriters[a].warning)
            f.write(json.dumps({"key": key, "rewritten": cache[key]}, ensure_ascii=False) + "\n")
            f.flush()
        rewritten.append(cache[key])
        print(f"\rrewritten {i}/{len(originals)}  ({time.time() - t0:.0f}s)", end="", flush=True)
print()
rewrite_seconds = time.time() - t0

# meaning judge on every final text, same prompt for all methods (note: for v2 the
# same model already checked its own output, so v2 numbers are optimistic)
judge = Rewriter(llm)
flagged = 0
for i, (o, r) in enumerate(zip(originals, rewritten), 1):
    flagged += bool(judge._meaning_problems(o, r))
    print(f"\rjudged {i}/{len(originals)}", end="", flush=True)
print()

# --- metrics --------------------------------------------------------------
result = {"method": args.method, "setting": args.setting, "model": args.model, "persona": persona,
          "n_authors": len(authors), "n": len(originals), "static": {}, "adaptive": {}}
print(f"\nmethod={args.method} setting={args.setting} persona={persona if args.method != 'neutral' else '-'}")
print(f"rewritten authors: {len(authors)}, fragments: {len(originals)}")
print(f"{'attacker':8} {'orig':>6} {'rewr':>6}   (static, balanced accuracy over rewritten authors)")
for kind, m in attackers.items():
    before = balanced_accuracy_score(labels, m.predict(originals))
    after = balanced_accuracy_score(labels, m.predict(rewritten))
    result["static"][kind] = [round(before, 3), round(after, 3)]
    note = "  <- used as guide in loop" if args.method == "loop" and kind == "style" else ""
    print(f"{kind:8} {before:6.2f} {after:6.2f}{note}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(f"\nadaptive (trained on rewrites, 5-fold), chance {1 / len(authors):.2f}:")
for kind in ("style", "morph"):
    orig_acc = balanced_accuracy_score(labels, cross_val_predict(build_attributor(kind), originals, labels, cv=cv))
    rew_acc = balanced_accuracy_score(labels, cross_val_predict(build_attributor(kind), rewritten, labels, cv=cv))
    result["adaptive"][kind] = [round(orig_acc, 3), round(rew_acc, 3)]
    print(f"{kind:8} {orig_acc:6.2f} {rew_acc:6.2f}")

emb = np.array(llm.embed(originals + rewritten, EMBED_MODEL))
emb /= np.linalg.norm(emb, axis=1, keepdims=True)
sim = np.sum(emb[:len(originals)] * emb[len(originals):], axis=1)
result["meaning"] = [round(float(sim.mean()), 3), round(float(sim.min()), 3)]
result["meaning_flagged"] = round(flagged / len(originals), 3)
result["seconds_per_text"] = round(rewrite_seconds / len(originals), 2)
print(f"\nmeaning similarity: mean {sim.mean():.2f}, min {sim.min():.2f}")
print(f"meaning judge flagged: {flagged}/{len(originals)} ({flagged / len(originals):.0%})"
      + (f"; rewriter gave up after retries: {warned}" if warned else ""))

if args.method != "neutral":
    # does the result read like the persona? 53-author style attacker, persona is one of its classes
    full = cached_attacker("style", "all", P_tr, py_tr)
    k = list(full.classes_).index(persona)
    result["persona_hit"] = round(float(np.mean(full.predict(rewritten) == persona)), 3)
    print(f"looks like {persona}: {result['persona_hit']:.0%} of rewrites "
          f"(mean P before {full.predict_proba(originals)[:, k].mean():.2f}, after {full.predict_proba(rewritten)[:, k].mean():.2f}, "
          f"chance {1 / len(full.classes_):.2f})")
print(f"speed: {result['seconds_per_text']:.1f} s per text")

with Path("data/experiments.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False) + "\n")
