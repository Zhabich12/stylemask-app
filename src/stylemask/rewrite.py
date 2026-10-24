"""The product itself: rewrite a text so its author can't be recognised.

Pipeline (each step can be switched off for experiments):
1. facts:  the model lists what the message says — facts, negations,
           questions, requests. Meaning is pinned down before style changes.
2. write:  a new message is written from the facts only (never seeing the
           original wording), in the target style: neutral, or a persona
           described by a style card + examples similar to this message.
3. check:  the model compares original and result; on a mismatch the result
           is rewritten with the list of discrepancies, up to `retries` times.
4. loop:   (mode="loop") a guide attributor checks whether the author still
           shows through and, if so, the model gets concrete hints.
"""
import hashlib
import random
import re
from collections import Counter

import numpy as np

from stylemask.llm import LLM

FACTS_SYSTEM = (
    "Перескажи сообщение коротким списком пунктов от первого лица, как будто это заметки автора "
    "для себя: что произошло, что я думаю и чувствую, о чём спрашиваю или прошу. "
    "Сохрани все отрицания и вопросы. Ничего не добавляй и не пропускай. "
    "Не пиши «автор», «в тексте», «сообщение» — только сами мысли. Ответь только списком."
)

NEUTRAL_STYLE = (
    "Пиши как нейтральный, ничем не примечательный человек в мессенджере: обычная пунктуация, "
    "без характерных словечек."
)

PERSONA_STYLE = (
    "Пиши так, как пишет этот человек.\n\nЕго манера:\n{card}\n{notes}\nПримеры его сообщений:\n\n{examples}"
)

NOTES_BLOCK = "\nЗамечания от того, кто хорошо знает его стиль (это важнее всего остального):\n{notes}\n"

WRITE_SYSTEM = (
    "Ты пишешь сообщение другу в мессенджер от первого лица. {style}\n\n"
    "Тебе дают заметки о том, что нужно сказать. Преврати их в живое сообщение (не список!). "
    "Передай все мысли, ничего не добавляй от себя, не меняй факты, отрицания и вопросы. "
    "Ответь только текстом сообщения, без пояснений."
)

WRITE_USER = "Заметки, что нужно сказать:\n{facts}\n\nНапиши сообщение."

CHECK_SYSTEM = (
    "Сравни два сообщения по смыслу. Стиль, слова и порядок не важны — важны факты, кто что сделал, "
    "отрицания, вопросы и просьбы. Если смысл совпадает, ответь одним словом: ОК. "
    "Если нет — перечисли расхождения коротким списком."
)

CHECK_USER = "ИСХОДНОЕ:\n{original}\n\nНОВОЕ:\n{draft}"

FIX_USER = (
    "Заметки, что нужно сказать:\n{facts}\n\nТвой прошлый вариант:\n{draft}\n\n"
    "В нём искажён смысл:\n{problems}\n\nНапиши заново, исправив это."
)

HINT_USER = (
    "Заметки, что нужно сказать:\n{facts}\n\nТвой прошлый вариант:\n{draft}\n\n"
    "По нему всё ещё узнаётся настоящий автор. Что его выдаёт:\n{hints}\n\n"
    "Напиши заново с учётом этого, смысл сохрани."
)

_WORD = re.compile(r"\w+")


def surface_stats(text: str) -> dict[str, float]:
    """Cheap, human-explainable style habits."""
    lines = [l for l in text.split("\n") if l.strip()] or [text]
    words = max(1, len(_WORD.findall(text)))
    return {
        "lower_start": float(np.mean([l.strip()[0].islower() for l in lines])),
        "line_len": float(np.mean([len(l) for l in lines])),
        "period": text.count(".") / words,
        "comma": text.count(",") / words,
        "question": text.count("?") / words,
        "exclam": text.count("!") / words,
        "paren": (text.count(")") + text.count("(")) / words,
        "ellipsis": text.count("...") / words,
    }


def mean_stats(texts) -> dict[str, float]:
    rows = [surface_stats(t) for t in texts]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def marker_words(person_texts, other_texts, min_count=10, ratio=3.0, top=15) -> list[str]:
    """Words this person uses far more often than other people, most typical first."""
    a = Counter(w.lower() for t in person_texts for w in _WORD.findall(t))
    o = Counter(w.lower() for t in other_texts for w in _WORD.findall(t))
    a_total, o_total = max(1, sum(a.values())), max(1, sum(o.values()))
    scored = [((c / a_total) / ((o[w] + 1) / o_total), w) for w, c in a.items()
              if c >= min_count and not w.isdigit()]
    return [w for r, w in sorted(scored, reverse=True) if r >= ratio][:top]


def style_card(person_texts, other_texts) -> str:
    """Explicit rules describing how a person writes, relative to everyone else."""
    p, o = mean_stats(person_texts), mean_stats(other_texts)
    rules = []

    def rel(key, more, less, factor=1.5):
        if p[key] > o[key] * factor and p[key] > 0:
            rules.append(more)
        elif p[key] * factor < o[key]:
            rules.append(less)

    if p["lower_start"] > 0.6:
        rules.append("почти всегда начинает с маленькой буквы")
    elif p["lower_start"] < 0.15:
        rules.append("начинает с заглавной буквы")
    lines_per_text = np.mean([len([l for l in t.split("\n") if l.strip()]) for t in person_texts])
    if lines_per_text >= 3:
        rules.append(f"пишет короткими отдельными сообщениями, по одной мысли на строку (в среднем {p['line_len']:.0f} символов)")
    else:
        rules.append(f"пишет связными сообщениями (в среднем {p['line_len']:.0f} символов на строку)")
    rel("period", "ставит точки в конце предложений", "почти не ставит точки")
    rel("comma", "любит запятые и длинные фразы", "мало запятых, простые фразы")
    rel("question", "часто задаёт вопросы", "редко задаёт вопросы")
    rel("exclam", "часто ставит восклицательные знаки", "почти не ставит восклицательных знаков")
    rel("paren", "ставит скобочки-смайлы ) и (", "не ставит скобочек-смайлов")
    rel("ellipsis", "часто использует многоточие", "не использует многоточие")
    words = marker_words(person_texts, other_texts)
    if words:
        rules.append("характерные слова: " + ", ".join(words))
    return "\n".join(f"- {r}" for r in rules)


class ExamplePicker:
    """Pick the person's messages most similar in topic/length to the text being rewritten.

    Embeddings of the pool are computed once per pool and cached in memory.
    """

    _cache: dict[str, np.ndarray] = {}

    def __init__(self, llm: LLM, pool: list[str], embed_model: str):
        self.llm, self.pool, self.embed_model = llm, pool, embed_model
        self.key = hashlib.sha1("\x00".join(pool).encode()).hexdigest()

    def _pool_emb(self) -> np.ndarray:
        if self.key not in self._cache:
            embs = []
            for i in range(0, len(self.pool), 64):
                embs += self.llm.embed(self.pool[i:i + 64], self.embed_model)
            e = np.array(embs)
            self._cache[self.key] = e / np.linalg.norm(e, axis=1, keepdims=True)
        return self._cache[self.key]

    def pick(self, text: str, k: int) -> list[str]:
        if len(self.pool) <= k:
            return list(self.pool)
        try:
            pool = self._pool_emb()
            q = np.array(self.llm.embed([text], self.embed_model)[0])
            order = np.argsort(-(pool @ (q / np.linalg.norm(q))))
            return [self.pool[i] for i in order[:k]]
        except Exception:  # embedding model unavailable: random examples still work
            return random.Random(0).sample(self.pool, k)


class HintMaker:
    """Explain in words why a text still looks like `author`."""

    _HINTS = {
        "lower_start": ("не начинай строки с маленькой буквы", "начинай некоторые строки с маленькой буквы"),
        "line_len": ("пиши более короткими отдельными сообщениями", "объединяй мысли в более длинные сообщения"),
        "period": ("ставь меньше точек", "ставь точки в конце предложений"),
        "comma": ("используй меньше запятых, проще фразы", "используй больше запятых и придаточных"),
        "question": ("меньше вопросительных знаков", "можно добавить вопросительную интонацию"),
        "exclam": ("меньше восклицательных знаков", "можно добавить восклицательные знаки"),
        "paren": ("не используй скобочки-смайлы ) и (", "можно добавить скобочку-смайл )"),
        "ellipsis": ("не используй многоточия", "можно использовать многоточие"),
    }

    def __init__(self, texts, labels, author, target=None):
        mine = [t for t, y in zip(texts, labels) if y == author]
        theirs = [t for t, y in zip(texts, labels) if (y == target if target else y != author)]
        self.author_stats, self.target_stats = mean_stats(mine), mean_stats(theirs)
        self.marker_words = set(marker_words(mine, theirs, min_count=15, top=200))

    def hints(self, draft: str) -> list[str]:
        out = []
        s = surface_stats(draft)
        for k, (more, less) in self._HINTS.items():
            a, t = self.author_stats[k], self.target_stats[k]
            if abs(a - t) > 1e-9 and abs(s[k] - a) < abs(s[k] - t):
                out.append(more if a > t else less)
        used = sorted({w.lower() for w in _WORD.findall(draft)} & self.marker_words)
        if used:
            out.append("не используй слова, типичные для автора: " + ", ".join(used[:12]))
        return out


class Rewriter:
    """mode: "neutral" | "persona" | "loop". For persona/loop pass `examples`
    (a list, or an ExamplePicker) and optionally a `card` from style_card()."""

    def __init__(self, llm: LLM, mode: str = "neutral", examples=None, card: str = "",
                 n_examples: int = 15, guide=None, author=None, hint_maker: HintMaker | None = None,
                 rounds: int = 2, retries: int = 2, temperature: float = 0.3,
                 two_step: bool = True, check: bool = True,
                 notes: list[str] | None = None, approved: list[str] | None = None):
        self.llm, self.mode, self.card, self.n_examples = llm, mode, card, n_examples
        self.examples = examples
        # user feedback: remarks about the style, and approved/hand-fixed messages
        self.notes, self.approved = notes or [], approved or []
        self.guide, self.author, self.hint_maker, self.rounds = guide, author, hint_maker, rounds
        self.retries, self.temperature, self.two_step, self.check = retries, temperature, two_step, check
        self.warning = ""  # set when the meaning check still fails after all retries

    def _style(self, text: str) -> str:
        if self.mode == "neutral":
            return NEUTRAL_STYLE
        approved = self.approved[:8]
        k = max(self.n_examples - len(approved), 5)
        shots = (self.examples.pick(text, k) if isinstance(self.examples, ExamplePicker)
                 else list(self.examples)[:k])
        notes = NOTES_BLOCK.format(notes="\n".join(f"- {n}" for n in self.notes)) if self.notes else ""
        return PERSONA_STYLE.format(card=self.card or "- (см. примеры)", notes=notes,
                                    examples="\n\n---\n\n".join(approved + shots))

    def _chat(self, system: str, user: str) -> str:
        return self.llm.chat(system, user, temperature=self.temperature)

    def _meaning_problems(self, original: str, draft: str) -> str:
        verdict = self.llm.chat(CHECK_SYSTEM, CHECK_USER.format(original=original, draft=draft),
                                temperature=0.0, max_tokens=300)
        return "" if re.match(r"^\W*(ок|ok)\b", verdict.strip(), re.I) else verdict

    def _recognised(self, text: str) -> bool:
        classes = list(self.guide.classes_)
        p = self.guide.predict_proba([text])[0]
        return classes[int(p.argmax())] == self.author or p[classes.index(self.author)] > 1.5 / len(classes)

    def planned_steps(self) -> int:
        """Steps on the happy path (no retries, loop not triggered) — for progress display."""
        return 1 + self.two_step + (self.mode != "neutral") + self.check

    def rewrite(self, text: str, on_step=None) -> str:
        """`on_step(label, step_no)` is called before each slow step, from the calling thread."""
        self.warning = ""
        n = 0

        def step(label):
            nonlocal n
            n += 1
            if on_step:
                on_step(label, n)

        if self.mode != "neutral":
            step("Подбираю примеры стиля")
        system = WRITE_SYSTEM.format(style=self._style(text))
        if self.two_step:
            step("Разбираю смысл")
            facts = self._chat(FACTS_SYSTEM, text)
            step("Пишу")
            draft = self._chat(system, WRITE_USER.format(facts=facts))
        else:
            facts = text
            step("Пишу")
            draft = self._chat(system, text)

        if self.mode == "loop":
            for r in range(self.rounds):
                if not self._recognised(draft):
                    break
                hints = self.hint_maker.hints(draft)
                if not hints:
                    break
                step(f"Убираю твои приметы (раунд {r + 1})")
                draft = self._chat(system, HINT_USER.format(
                    facts=facts, draft=draft, hints="\n".join(f"- {h}" for h in hints)))

        if self.check:
            for attempt in range(self.retries + 1):
                step("Проверяю смысл")
                problems = self._meaning_problems(text, draft)
                if not problems:
                    break
                if attempt == self.retries:
                    self.warning = "смысл мог измениться: " + problems
                    break
                step(f"Исправляю смысл (попытка {attempt + 1})")
                draft = self._chat(system, FIX_USER.format(facts=facts, draft=draft, problems=problems))
        return draft
