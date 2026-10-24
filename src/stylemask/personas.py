"""Target styles ("personas"): build a rewriter for one, add new ones from Telegram exports."""
import json
import random
from collections import Counter
from pathlib import Path

from stylemask.dataset import CUSTOM_STYLES
from stylemask.feedback import FeedbackStore
from stylemask.llm import LLM
from stylemask.rewrite import ExamplePicker, HintMaker, Rewriter, style_card
from stylemask.telegram import parse_export, to_fragments


def list_personas(labels, me: str = "") -> list[str]:
    """Everyone except the user, custom styles first, then chats, then Pikabu."""
    order = {"tg:": 0, "pikabu:": 2}
    return sorted((a for a in set(labels) if a != me),
                  key=lambda a: (next((v for k, v in order.items() if a.startswith(k)), 1), a))


def build_rewriter(llm: LLM, mode: str, persona: str, X_tr, y_tr, cfg: dict,
                   guide=None, **rewriter_kwargs) -> Rewriter:
    if mode == "neutral":
        return Rewriter(llm, "neutral", **rewriter_kwargs)
    pool = [x for x, y in zip(X_tr, y_tr) if y == persona]
    if not pool:
        raise ValueError(f"нет текстов для {persona!r}")
    others = [x for x, y in zip(X_tr, y_tr) if y != persona]
    others = random.Random(0).sample(others, min(2000, len(others)))
    card = style_card(pool, others)
    picker = ExamplePicker(llm, pool, cfg["embed_model"])
    fb = FeedbackStore(persona)
    rewriter_kwargs = {"notes": fb.notes(), "approved": fb.examples(), **rewriter_kwargs}
    if mode == "loop":
        me = cfg["me"]
        if not me:
            raise ValueError("для проверки укажи me = \"...\" в stylemask.toml")
        return Rewriter(llm, "loop", picker, card, guide=guide, author=me,
                        hint_maker=HintMaker(X_tr, y_tr, me, target=persona), **rewriter_kwargs)
    return Rewriter(llm, "persona", picker, card, **rewriter_kwargs)


def export_authors(folder: Path) -> Counter:
    """How many ~400-char fragments each participant of a Telegram export has."""
    return Counter(a for a, _ in to_fragments(parse_export(folder)))


def add_telegram_style(folder: Path, author: str, label: str | None = None) -> tuple[str, int]:
    """Append one participant's fragments to data/custom_styles.jsonl as persona `tg:<label>`."""
    label = "tg:" + (label or author).strip()
    frags = [t for a, t in to_fragments(parse_export(folder)) if a == author]
    existing = []
    if CUSTOM_STYLES.exists():
        existing = [l for l in CUSTOM_STYLES.read_text(encoding="utf-8").splitlines()
                    if json.loads(l)["author"] != label]  # re-adding replaces the old copy
    CUSTOM_STYLES.parent.mkdir(exist_ok=True)
    with CUSTOM_STYLES.open("w", encoding="utf-8") as f:
        for line in existing:
            f.write(line + "\n")
        for t in frags:
            f.write(json.dumps({"author": label, "text": t}, ensure_ascii=False) + "\n")
    return label, len(frags)
