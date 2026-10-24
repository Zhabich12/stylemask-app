"""Parse Telegram Desktop HTML exports into per-author text fragments.

Chat messages are too short to carry style on their own, so consecutive
messages of one author are glued into fragments of roughly `target_chars`.
"""
import html
import re
from pathlib import Path

_MSG_SPLIT = re.compile(r'<div class="message default clearfix( joined)?" id="message\d+">')
# top-level from_name / text sit at fixed indentation; forwarded ones are deeper
_FROM = re.compile(r'\n       <div class="from_name">\n(.*?)\n       </div>', re.S)
_TEXT = re.compile(r'\n       <div class="text">\n(.*?)\n       </div>', re.S)
_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+")


def _clean(raw: str) -> str:
    text = html.unescape(_TAG.sub("", raw.replace("<br>", "\n")))
    return _URL.sub("", text).strip()


def parse_export(folder: Path) -> list[tuple[str, str]]:
    """Return (author, text) for every own (non-forwarded) text message, in order."""
    files = sorted(folder.glob("messages*.html"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem) or 1))
    out, author = [], None
    for f in files:
        parts = _MSG_SPLIT.split(f.read_text(encoding="utf-8"))
        # split() yields [prefix, joined_flag, body, joined_flag, body, ...]
        for joined, body in zip(parts[1::2], parts[2::2]):
            if not joined:
                m = _FROM.search(body)
                author = m.group(1).strip() if m else None
            if author is None or 'class="forwarded body"' in body:
                continue
            m = _TEXT.search(body)
            if m and (text := _clean(m.group(1))):
                out.append((author, text))
    return out


def to_fragments(messages: list[tuple[str, str]], target_chars: int = 400) -> list[tuple[str, str]]:
    # dialogue turns are short and interleaved, so each author gets their own buffer
    frags, bufs = [], {}
    for author, text in messages:
        buf = bufs.setdefault(author, [])
        buf.append(text)
        if sum(map(len, buf)) >= target_chars:
            frags.append((author, "\n".join(buf)))
            bufs[author] = []
    return frags
