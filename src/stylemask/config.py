"""Settings: defaults, overridden by stylemask.toml in the working directory."""
import json
import tomllib
from pathlib import Path

PATH = Path("stylemask.toml")

DEFAULTS = {
    "model": "qwen/qwen3.5-4b",
    "embed_model": "text-embedding-nomic-embed-text-v1.5",
    "base_url": "http://localhost:1234/v1",
    "me": "",          # your author label in data/fragments.jsonl
    "persona": "",     # default target author, e.g. "pikabu:<name>"
    "shots": 6,
    "mode": "neutral",           # neutral | persona | loop — used by the hotkey
    "scope": "all",              # all: whole input field; selection: only selected text
    "hotkey": "ctrl+alt+q",      # rewrite the field you're typing in
    "undo_hotkey": "ctrl+alt+z",  # put the original text back
}


def load(path: Path = PATH) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        # utf-8-sig: Windows editors and PowerShell often prepend a BOM
        cfg.update(tomllib.loads(path.read_text(encoding="utf-8-sig")))
    return cfg


def save(cfg: dict, path: Path = PATH) -> None:
    """Write flat str/int/bool settings back as TOML (JSON string escaping is valid TOML)."""
    lines = []
    for key, value in cfg.items():
        if isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(str(value), ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
