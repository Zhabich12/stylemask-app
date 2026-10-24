"""stylemask command line.

  stylemask gui                           open the desktop window
  stylemask rewrite "текст"               rewrite text (or pipe it via stdin)
  stylemask personas                      list available target authors
  stylemask add-style <export_dir>        add a person's style from a Telegram export
  stylemask build-telegram <dir> [...]    Telegram exports -> data/fragments.jsonl
  stylemask build-pikabu <shard.zst>      Pikabu shard -> data/pikabu_fragments.jsonl
  stylemask baseline [...]                how well attackers recognise authors
  stylemask experiment [...]              rewrite + measure (see scripts/experiment.py)
"""
import argparse
import runpy
import sys
from collections import Counter
from pathlib import Path

from stylemask import config

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _run_script(name: str, argv: list[str]) -> None:
    sys.argv = [name, *argv]
    runpy.run_path(str(SCRIPTS / name), run_name="__main__")


def cmd_personas(cfg, _args) -> None:
    from stylemask.dataset import load_split
    from stylemask.personas import list_personas
    _, _, y_tr, _ = load_split()
    counts = Counter(y_tr)
    for author in list_personas(y_tr, cfg["me"]):
        print(f"{counts[author]:5d}  {author}")


def cmd_rewrite(cfg, args) -> None:
    from stylemask.dataset import load_split
    from stylemask.llm import LLM
    from stylemask.personas import build_rewriter

    text = args.text or sys.stdin.read()
    llm = LLM(args.model or cfg["model"], base_url=cfg["base_url"])
    persona = args.persona or cfg["persona"]
    mode = "neutral" if args.neutral or not persona else ("loop" if args.loop else "persona")
    X_tr, _, y_tr, _ = load_split() if mode != "neutral" else ([], None, [], None)
    guide = None
    if mode == "loop":
        from stylemask.cache import cached_attacker
        guide = cached_attacker("style", "all", X_tr, y_tr)
    try:
        rw = build_rewriter(llm, mode, persona, X_tr, y_tr, cfg, guide=guide)
    except ValueError as e:
        sys.exit(str(e))
    print(rw.rewrite(text))
    if rw.warning:
        print(f"\n⚠ {rw.warning}", file=sys.stderr)


def cmd_add_style(cfg, args) -> None:
    from stylemask.personas import add_telegram_style, export_authors
    folder = Path(args.folder)
    if not args.author:
        for author, n in export_authors(folder).most_common():
            print(f"{n:5d}  {author}")
        print("\nвыбери автора: stylemask add-style <папка> --author \"<имя>\"")
        return
    label, n = add_telegram_style(folder, args.author, args.name)
    print(f"добавлен {label}: {n} фрагментов" + ("" if n >= 30 else " (мало: нужно от 30)"))


def main() -> None:
    cfg = config.load()
    ap = argparse.ArgumentParser(prog="stylemask", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rw = sub.add_parser("rewrite", help="rewrite text so its author can't be recognised")
    rw.add_argument("text", nargs="?")
    rw.add_argument("--persona", help="target author, e.g. pikabu:<name>")
    rw.add_argument("--neutral", action="store_true", help="no persona, just neutral style")
    rw.add_argument("--loop", action="store_true", help="re-check and rewrite again with hints")
    rw.add_argument("--model")
    sub.add_parser("personas", help="list target authors")
    sub.add_parser("gui", help="open the desktop window")
    add = sub.add_parser("add-style", help="add a person's style from a Telegram export")
    add.add_argument("folder")
    add.add_argument("--author", help="participant name as in the export (omit to list them)")
    add.add_argument("--name", help="label to show in the persona list")
    for name in ("build-telegram", "build-pikabu", "baseline", "experiment"):
        sub.add_parser(name, add_help=False)

    args, rest = ap.parse_known_args()
    scripts = {"build-telegram": "build_dataset.py", "build-pikabu": "build_pikabu.py",
               "baseline": "baseline.py", "experiment": "experiment.py"}
    if args.cmd in scripts:
        _run_script(scripts[args.cmd], rest)
    elif args.cmd == "personas":
        cmd_personas(cfg, args)
    elif args.cmd == "add-style":
        cmd_add_style(cfg, args)
    elif args.cmd == "gui":
        from stylemask.gui import main as gui_main
        gui_main()
    else:
        cmd_rewrite(cfg, args)


if __name__ == "__main__":
    main()
