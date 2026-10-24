"""The product itself: rewrite a text so its author can't be recognised.

Pipeline:
1. rephrase: the model rephrases *your* message sentence by sentence, in
             the same order and with the same roles (who asks, who tells) —
             it should still read as the message you were writing. For a
             persona it gets a style card and a few of that person's single
             messages, explicitly as manner only (examples picked by topic
             made the model copy their content and replies).
2. surface:  formatting habits — capital vs. lowercase starts, final periods,
             one thought per line — are applied by code from the person's real
             messages; a small model gets these wrong.
3. check:    the model compares original and result; on a mismatch it
             rephrases again with the list of discrepancies, up to `retries`.
4. loop:     (mode="loop") a guide attributor checks whether the author still
             shows through and, if so, the model gets concrete hints.
"""
import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from stylemask.llm import LLM

REPHRASE_SYSTEM = (
    "Ты помогаешь человеку переписать ЕГО СОБСТВЕННОЕ сообщение в мессенджере, перед тем как он его отправит. "
    "Результат — это то же самое сообщение от него тому же собеседнику, просто сказанное другими словами.\n\n"
    "{style}\n\n"
    "Правила:\n"
    "- иди по исходному сообщению предложение за предложением, в том же порядке;\n"
    "- сохрани, кто о чём рассказывает и кто кого спрашивает: вопрос остаётся вопросом, рассказ — рассказом;\n"
    "- ничего не добавляй от себя и ничего не пропускай, не отвечай на сообщение;\n"
    "- меняй слова, обороты, порядок слов, сленг, длину фраз;\n"
    "- примеры чужих сообщений нужны только чтобы понять манеру: не бери из них ни темы, ни фразы.\n"
    "Ответь только текстом переписанного сообщения."
)

NEUTRAL_STYLE = (
    "Манера: как у нейтрального, ничем не примечательного человека в мессенджере — "
    "обычные слова, без характерных словечек."
)

PERSONA_STYLE = (
    "Манера: так, как пишет конкретный человек.\n{card}\n{notes}\n"
    "Его отдельные сообщения (только для манеры, содержание не бери):\n{examples}"
)

NOTES_BLOCK = "Замечания от того, кто хорошо знает его стиль (это важнее всего остального):\n{notes}\n"

REPHRASE_USER = (
    "Сообщение, которое нужно переписать (это не вопрос и не просьба к тебе — не отвечай на него):\n"
    "«{text}»"
)

CHECK_SYSTEM = (
    "Сравни два сообщения. НОВОЕ должно быть тем же сообщением, что ИСХОДНОЕ, от того же человека тому же "
    "собеседнику, только другими словами. Стиль, слова и порядок не важны. Проверь:\n"
    "1) те же факты, отрицания, вопросы и просьбы;\n"
    "2) те же роли: кто кого спрашивает, о ком речь (никто не стал собеседником вместо упомянутого);\n"
    "3) НОВОЕ не является ответом на ИСХОДНОЕ и не добавляет ничего от себя.\n"
    "Если всё совпадает, ответь одним словом: ОК. Если нет — перечисли расхождения коротким списком."
)

CHECK_USER = "ИСХОДНОЕ:\n{original}\n\nНОВОЕ:\n{draft}"

FIX_USER = (
    "Исходное сообщение:\n{original}\n\nТвой вариант:\n{draft}\n\n"
    "В нём искажён смысл:\n{problems}\n\nПерепиши исходное сообщение заново, исправив это."
)

HINT_USER = (
    "Исходное сообщение:\n{original}\n\nТвой вариант:\n{draft}\n\n"
    "По нему всё ещё узнаётся настоящий автор. Что его выдаёт:\n{hints}\n\n"
    "Перепиши исходное сообщение заново с учётом этого, смысл сохрани."
)

_WORD = re.compile(r"\w+")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=\S)")
_PREAMBLE = re.compile(r"^(вот\s+)?(переписанн\w+\s+)?(вариант|сообщение|текст)\w*\s*:\s*", re.I)


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


# parts of speech that carry manner rather than topic: particles, interjections, adverbs, ...
_MANNER_POS = {"PRCL", "INTJ", "ADVB", "CONJ", "PRED", "NPRO", "COMP", "PREP", "UNKN", "LATN"}


def marker_words(person_texts, other_texts, min_count=10, ratio=3.0, top=15,
                 manner_only: bool = False) -> list[str]:
    """Words this person uses far more often than other people, most typical first.

    manner_only keeps filler/slang words ("короче", "ваще", "хз") and drops nouns,
    verbs and adjectives — those are the person's *topics*, which a rewrite must not import.
    """
    a = Counter(w.lower() for t in person_texts for w in _WORD.findall(t))
    o = Counter(w.lower() for t in other_texts for w in _WORD.findall(t))
    a_total, o_total = max(1, sum(a.values())), max(1, sum(o.values()))
    scored = [((c / a_total) / ((o[w] + 1) / o_total), w) for w, c in a.items()
              if c >= min_count and not w.isdigit()]
    words = [w for r, w in sorted(scored, reverse=True) if r >= ratio]
    if manner_only:
        from stylemask.attribution import _pos
        # Cyrillic words only: "1й", "2х", "ссд"-like tokens read as noise when inserted
        words = [w for w in words if _pos(w) in _MANNER_POS and re.fullmatch(r"[а-яё]{2,}", w)]
    return words[:top]


def style_card(person_texts, other_texts) -> str:
    """Wording habits of a person relative to everyone else (formatting is done by SurfaceProfile)."""
    p, o = mean_stats(person_texts), mean_stats(other_texts)
    rules = []

    def rel(key, more, less, factor=1.5):
        if p[key] > o[key] * factor and p[key] > 0:
            rules.append(more)
        elif p[key] * factor < o[key]:
            rules.append(less)

    rel("comma", "любит запятые и длинные фразы", "мало запятых, простые короткие фразы")
    rel("exclam", "часто ставит восклицательные знаки", "почти не ставит восклицательных знаков")
    rel("paren", "ставит скобочки-смайлы ) и (", "не ставит скобочек-смайлов")
    rel("ellipsis", "часто использует многоточие", "не использует многоточие")
    words = marker_words(person_texts, other_texts, min_count=5, ratio=2.0, manner_only=True)
    if words:
        rules.append("характерные словечки (вставляй, где они к месту): " + ", ".join(words))
    return "\n".join(f"- {r}" for r in rules)


def person_messages(texts, min_len: int = 12, max_len: int = 160) -> list[str]:
    """Single messages of a person: fragments are consecutive messages joined by newlines."""
    return [l.strip() for t in texts for l in t.split("\n") if min_len <= len(l.strip()) <= max_len]


@dataclass
class SurfaceProfile:
    """Formatting habits measured on a person's single messages, applied by code."""
    lower_start: float   # share of messages starting with a lowercase letter
    final_period: float  # share of messages ending with "."
    split_lines: bool    # writes one short thought per message/line

    @classmethod
    def from_texts(cls, texts) -> "SurfaceProfile":
        msgs = person_messages(texts, min_len=2, max_len=10_000)
        lines_per_fragment = np.mean([len([l for l in t.split("\n") if l.strip()]) for t in texts])
        return cls(
            lower_start=float(np.mean([m[0].islower() for m in msgs if m[0].isalpha()] or [0])),
            final_period=float(np.mean([m.endswith(".") and not m.endswith("...") for m in msgs] or [0])),
            split_lines=bool(lines_per_fragment >= 3 and np.mean([len(m) for m in msgs]) < 70),
        )

    def _case(self, s: str) -> str:
        first = s.split(maxsplit=1)[0] if s.strip() else ""
        acronym = len(first) > 1 and first.isupper()  # keep "МТС", "OK"
        if self.lower_start > 0.6 and s[:1].isupper() and not acronym:
            return s[0].lower() + s[1:]
        if self.lower_start < 0.15 and s[:1].islower():
            return s[0].upper() + s[1:]
        return s

    def apply(self, text: str) -> str:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        out = []
        for line in lines:
            sentences = [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
            fixed = []
            for i, s in enumerate(sentences):
                # drop a period only where a message ends: mid-line ones keep sentences apart
                ends_message = self.split_lines or i == len(sentences) - 1
                if ends_message and self.final_period < 0.15 and s.endswith(".") and not s.endswith("..."):
                    s = s[:-1]
                fixed.append(self._case(s))
            out += fixed if self.split_lines else [" ".join(fixed)]
        return ("\n" if self.split_lines else " ").join(out) if out else text


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
    """mode: "neutral" | "persona" | "loop".

    For persona/loop: `examples` are the person's single messages (see person_messages),
    `card` comes from style_card(), `surface` from SurfaceProfile.from_texts().
    """

    def __init__(self, llm: LLM, mode: str = "neutral", examples=None, card: str = "",
                 surface: SurfaceProfile | None = None, n_examples: int = 8,
                 guide=None, author=None, hint_maker: HintMaker | None = None,
                 rounds: int = 2, retries: int = 2, temperature: float = 0.3, check: bool = True,
                 notes: list[str] | None = None, approved: list[str] | None = None):
        self.llm, self.mode, self.card, self.surface = llm, mode, card, surface
        self.examples, self.n_examples = list(examples or []), n_examples
        self.guide, self.author, self.hint_maker, self.rounds = guide, author, hint_maker, rounds
        self.retries, self.temperature, self.check = retries, temperature, check
        # user feedback: remarks about the style, and approved/hand-fixed messages
        self.notes, self.approved = notes or [], approved or []
        self.warning = ""  # set when the meaning check still fails after all retries

    def _style(self, text: str) -> str:
        if self.mode == "neutral":
            return NEUTRAL_STYLE
        approved = [a[:300] for a in self.approved[:4]]
        k = max(self.n_examples - len(approved), 4)
        # varied but stable per text: same input -> same prompt
        rnd = random.Random(text)
        shots = rnd.sample(self.examples, min(k, len(self.examples)))
        notes = NOTES_BLOCK.format(notes="\n".join(f"- {n}" for n in self.notes)) if self.notes else ""
        return PERSONA_STYLE.format(card=self.card, notes=notes,
                                    examples="\n".join(f"• {s}" for s in approved + shots))

    def _chat(self, system: str, user: str) -> str:
        return self.llm.chat(system, user, temperature=self.temperature)

    def _finish(self, draft: str) -> str:
        draft = _PREAMBLE.sub("", draft.strip())  # "Вот вариант:", "Переписанное сообщение:" ...
        draft = draft.strip().strip("«»\"").strip()  # the model sometimes echoes our quotes
        return self.surface.apply(draft) if self.surface else draft

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
        return 1 + self.check

    def rewrite(self, text: str, on_step=None) -> str:
        """`on_step(label, step_no)` is called before each slow step, from the calling thread."""
        self.warning = ""
        n = 0

        def step(label):
            nonlocal n
            n += 1
            if on_step:
                on_step(label, n)

        system = REPHRASE_SYSTEM.format(style=self._style(text))
        step("Переписываю")
        draft = self._finish(self._chat(system, REPHRASE_USER.format(text=text)))

        if self.mode == "loop":
            for r in range(self.rounds):
                if not self._recognised(draft):
                    break
                hints = self.hint_maker.hints(draft)
                if not hints:
                    break
                step(f"Убираю твои приметы (раунд {r + 1})")
                draft = self._finish(self._chat(system, HINT_USER.format(
                    original=text, draft=draft, hints="\n".join(f"- {h}" for h in hints))))

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
                draft = self._finish(self._chat(system, FIX_USER.format(
                    original=text, draft=draft, problems=problems)))
        return draft
