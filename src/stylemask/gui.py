"""Desktop window: settings + a global hotkey that rewrites the field you're typing in.

Typical use: set mode/persona/hotkeys once on the "Настройки" tab, minimise
the window, then in any app (Telegram, browser) press the hotkey — the text
in the input field is replaced with its rewritten version. The undo hotkey
puts the original back. The "Вручную" tab rewrites pasted text.

Runs on the local LM Studio server; nothing leaves the machine.
Slow work (data loading, LLM calls) runs in worker threads. Tk isn't
thread-safe, so workers never touch widgets: they put callbacks on a queue
that the Tk thread drains every 50 ms.
"""
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
import winsound
from pathlib import Path
from tkinter import filedialog, ttk

import httpx

from stylemask import config, hotkey, log
from stylemask.feedback import FeedbackStore
from stylemask.llm import LLM
from stylemask.personas import add_telegram_style, build_rewriter, export_authors, list_personas
from stylemask.rewrite import Rewriter

LMS = Path(os.environ.get("USERPROFILE", "~")) / ".lmstudio" / "bin" / "lms.exe"

class Overlay:
    """Small always-on-top note in the screen corner, visible while you're in another app.

    It must never take focus: the hotkey pastes into whatever window is active.
    Tk activates a toplevel whenever it maps it, so the window is mapped once,
    off-screen, at startup and afterwards only moved — never withdrawn or
    deiconified again. Moves go through Tk's own geometry (Tk snaps the window
    back if it's moved behind its back); topmost is re-asserted with SWP_NOACTIVATE.
    """

    GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST = -20, 0x08000000, 0x80, 0x8
    HWND_TOPMOST, SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = -1, 0x2, 0x1, 0x10
    OFFSCREEN = (-32000, -32000)

    def __init__(self, root: tk.Tk):
        import ctypes
        self.user32 = ctypes.windll.user32
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"1x1+{self.OFFSCREEN[0]}+{self.OFFSCREEN[1]}")
        self.win.configure(bg="#1f2937")
        self.text = tk.StringVar()
        tk.Label(self.win, textvariable=self.text, bg="#1f2937", fg="#f9fafb",
                 font=("Segoe UI", 10), padx=14, pady=8).pack()
        self.win.update()
        self.hwnd = int(self.win.wm_frame(), 16)
        style = self.user32.GetWindowLongW(self.hwnd, self.GWL_EXSTYLE)
        self.user32.SetWindowLongW(self.hwnd, self.GWL_EXSTYLE,
                                   style | self.WS_EX_NOACTIVATE | self.WS_EX_TOOLWINDOW | self.WS_EX_TOPMOST)
        root.focus_force()  # mapping the overlay may have activated it; give focus back to the main window
        self.visible = False
        self._hide_job = None

    def _place(self, x, y, w, h):
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        self.win.update_idletasks()
        self.user32.SetWindowPos(self.hwnd, self.HWND_TOPMOST, 0, 0, 0, 0,
                                 self.SWP_NOMOVE | self.SWP_NOSIZE | self.SWP_NOACTIVATE)

    def _anchor_rect(self, hwnd: int):
        """Rect of the window the user is typing in — so the note shows up on *their* monitor."""
        import ctypes
        from ctypes import wintypes
        r = wintypes.RECT()
        if hwnd and self.user32.IsWindowVisible(hwnd) and self.user32.GetWindowRect(hwnd, ctypes.byref(r)):
            if r.right - r.left > 200 and r.bottom - r.top > 150:
                return r.left, r.top, r.right, r.bottom
        return None

    def show(self, text: str, hide_after_ms: int | None = None, anchor: int = 0):
        """`anchor`: window handle to sit inside (bottom-right, above its input area)."""
        if self._hide_job:
            self.win.after_cancel(self._hide_job)
            self._hide_job = None
        self.text.set(text)
        self.win.update_idletasks()
        w, h = self.win.winfo_reqwidth(), self.win.winfo_reqheight()
        rect = self._anchor_rect(anchor)
        if rect:
            left, top, right, bottom = rect
            x, y = right - w - 20, bottom - h - 90  # inside the window, above a chat input box
            x, y = max(x, left + 8), max(y, top + 8)
        else:
            x = self.win.winfo_screenwidth() - w - 24
            y = self.win.winfo_screenheight() - h - 72  # above the taskbar
        self._place(x, y, w, h)
        self.visible = True
        if hide_after_ms:
            self._hide_job = self.win.after(hide_after_ms, self.hide)

    def hide(self):
        self._hide_job = None
        self._place(*self.OFFSCREEN, 1, 1)
        self.visible = False


MODES = {
    "neutral": "Нейтрально",
    "persona": "Как другой автор",
    "loop": "Как другой автор + проверка",
}
SCOPES = {"all": "Всё поле ввода", "selection": "Только выделенный текст"}


class App:
    def __init__(self, root: tk.Tk):
        self.root, self.cfg = root, config.load()
        self.llm = LLM(self.cfg["model"], base_url=self.cfg["base_url"])
        self.data = None     # (X_train, y_train), loaded lazily in the background
        self.guide = None    # style attributor for loop mode, trained on first use
        self.busy = False
        self.listener = None
        self.last = None     # (window, original text) for undo
        self.last_result = None  # (persona or "", result) for feedback
        self.inbox = queue.Queue()  # callbacks from worker threads, run on the Tk thread

        root.title("stylemask")
        root.geometry("1000x600")
        root.minsize(700, 420)
        style = ttk.Style()
        style.configure(".", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Head.TLabel", font=("Segoe UI", 11, "bold"))

        tabs = ttk.Notebook(root)
        tabs.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        tabs.add(self._settings_tab(tabs), text="Настройки")
        tabs.add(self._manual_tab(tabs), text="Вручную")

        bottom = ttk.Frame(root, padding=(12, 6, 12, 10))
        bottom.pack(fill="x")
        self.server_btn = ttk.Button(bottom, text="Запустить LM Studio", command=self.start_server)
        self.progress = ttk.Progressbar(bottom, length=160, mode="determinate")
        self.status = tk.StringVar()
        ttk.Label(bottom, textvariable=self.status, foreground="#555").pack(side="left")

        self.overlay = Overlay(root)
        self.anchor = 0          # window the progress note sits in (the one being typed in)
        self.step = ("", 0, 1)   # (label, step number, planned steps) of the running rewrite
        self.started = 0.0
        self.rewriters = {}      # built rewriters by (mode, persona, feedback state): building one takes seconds
        self._drain()
        self.apply_hotkeys(save=False)
        self._work(self._check_server, self._show_server)
        self._work(self._load_data, self._show_personas)

    # --- layout ------------------------------------------------------------
    def _settings_tab(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=16)
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Как переписывать", style="Head.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.mode = tk.StringVar(value=self.cfg["mode"])
        modes = ttk.Frame(tab)
        modes.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        for key, label in MODES.items():
            ttk.Radiobutton(modes, text=label, value=key, variable=self.mode,
                            command=self._on_mode).pack(side="left", padx=(0, 14))

        ttk.Label(tab, text="Под кого писать:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.persona = tk.StringVar(value=self.cfg["persona"])
        persona_row = ttk.Frame(tab)
        persona_row.grid(row=2, column=1, sticky="w", pady=(10, 0))
        self.persona_box = ttk.Combobox(persona_row, textvariable=self.persona, width=32, state="disabled")
        self.persona_box.pack(side="left")
        ttk.Button(persona_row, text="Добавить стиль из Telegram…",
                   command=self.add_style).pack(side="left", padx=(8, 0))

        ttk.Label(tab, text="Что заменять:").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.scope = tk.StringVar(value=self.cfg["scope"])
        scopes = ttk.Frame(tab)
        scopes.grid(row=3, column=1, sticky="w", pady=(10, 0))
        for key, label in SCOPES.items():
            ttk.Radiobutton(scopes, text=label, value=key, variable=self.scope).pack(side="left", padx=(0, 14))

        ttk.Label(tab, text="Горячие клавиши", style="Head.TLabel").grid(row=4, column=0, columnspan=2, sticky="w", pady=(22, 0))
        ttk.Label(tab, text="Переписать:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.hk = tk.StringVar(value=self.cfg["hotkey"])
        ttk.Entry(tab, textvariable=self.hk, width=20).grid(row=5, column=1, sticky="w", pady=(6, 0))
        ttk.Label(tab, text="Вернуть как было:").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.undo_hk = tk.StringVar(value=self.cfg["undo_hotkey"])
        ttk.Entry(tab, textvariable=self.undo_hk, width=20).grid(row=6, column=1, sticky="w", pady=(6, 0))
        ttk.Label(tab, foreground="#666", text="Формат: ctrl+alt+r, ctrl+shift+f8 и т.п. "
                  "Модификаторы: ctrl, alt, shift, win.").grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

        save_row = ttk.Frame(tab)
        save_row.grid(row=8, column=0, sticky="w", pady=(18, 0))
        ttk.Button(save_row, text="Сохранить", style="Accent.TButton", command=self.apply_hotkeys).pack(side="left")
        ttk.Button(save_row, text="Показать индикатор", command=self.test_overlay).pack(side="left", padx=(8, 0))
        self.hk_status = tk.StringVar()
        ttk.Label(tab, textvariable=self.hk_status).grid(row=8, column=1, sticky="w", pady=(18, 0))

        ttk.Label(tab, foreground="#666", wraplength=620, justify="left", text=(
            "Как пользоваться: пишешь сообщение в любой программе, нажимаешь «Переписать» — "
            "текст в поле заменится через пару секунд. Не переключай окно, пока идёт переписывание. "
            "Перед отправкой перечитай: модель иногда искажает смысл. "
            "Программа не записывает нажатия клавиш — она реагирует только на свои сочетания."
        )).grid(row=9, column=0, columnspan=2, sticky="w", pady=(22, 0))
        self._on_mode()
        return tab

    def _manual_tab(self, parent) -> ttk.Frame:
        tab = ttk.Frame(parent, padding=10)
        tab.columnconfigure((0, 1), weight=1, uniform="col")
        tab.rowconfigure(1, weight=1)
        ttk.Label(tab, text="Исходный текст").grid(row=0, column=0, sticky="w")
        ttk.Label(tab, text="Результат").grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.src = self._text(tab, 0)
        self.dst = self._text(tab, 1, padx=(10, 0))
        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.go = ttk.Button(buttons, text="Переписать", style="Accent.TButton", command=self.rewrite)
        self.go.pack(side="left")
        ttk.Button(buttons, text="Копировать результат", command=self.copy).pack(side="left", padx=8)
        ttk.Button(buttons, text="Очистить", command=self.clear).pack(side="left")
        self.src.bind("<Control-Return>", lambda _e: (self.rewrite(), "break")[1])

        # feedback on the last result — goes into the persona's style profile
        fb = ttk.LabelFrame(tab, text="Отзыв о последнем результате (учится стиль выбранного человека)", padding=8)
        fb.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        fb.columnconfigure(1, weight=1)
        row1 = ttk.Frame(fb)
        row1.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(row1, text="👍 Похоже", command=lambda: self.feedback("good")).pack(side="left")
        ttk.Button(row1, text="👎 Не похоже", command=lambda: self.feedback("bad")).pack(side="left", padx=6)
        ttk.Button(row1, text="✏️ Сохранить мой вариант (поправь текст справа)",
                   command=lambda: self.feedback("fixed")).pack(side="left")
        ttk.Label(fb, text="Замечание о стиле:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.note = tk.StringVar()
        note_entry = ttk.Entry(fb, textvariable=self.note)
        note_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=(6, 0))
        note_entry.bind("<Return>", lambda _e: self.feedback("note"))
        ttk.Button(fb, text="Добавить", command=lambda: self.feedback("note")).grid(row=1, column=2, pady=(6, 0))
        ttk.Label(fb, foreground="#666", text="напр.: никогда не ставит запятые · вместо «очень» "
                  "говорит «оч» · не пишет «кстати»").grid(row=2, column=1, sticky="w")
        self.fb_status = tk.StringVar()
        ttk.Label(fb, textvariable=self.fb_status).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        return tab

    def _text(self, parent, col, padx=(0, 0)) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.grid(row=1, column=col, sticky="nsew", padx=padx, pady=(4, 0))
        text = tk.Text(frame, wrap="word", font=("Segoe UI", 11), undo=True, relief="solid", borderwidth=1)
        bar = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return text

    # --- background work ---------------------------------------------------
    def _drain(self):
        while not self.inbox.empty():
            self.inbox.get_nowait()()
        self.root.after(50, self._drain)

    def _work(self, fn, done=None):
        def run():
            try:
                result, error = fn(), None
            except Exception as e:  # shown to the user, not swallowed
                result, error = None, e
            if done:
                self.inbox.put(lambda: done(result, error))
        threading.Thread(target=run, daemon=True).start()

    def _check_server(self) -> bool:
        httpx.get(self.cfg["base_url"] + "/models", timeout=3).raise_for_status()
        return True

    def _show_server(self, ok, _error):
        if ok:
            self.server_btn.pack_forget()
            self.status.set("LM Studio подключён")
        else:
            self.server_btn.pack(side="right")
            self.status.set("LM Studio не запущен")

    def start_server(self):
        self.status.set("Запускаю LM Studio…")
        self._work(lambda: subprocess.run([str(LMS), "server", "start"], capture_output=True,
                                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                   lambda _r, _e: self._work(self._check_server, self._show_server))

    def _load_data(self):
        from stylemask.dataset import load_split
        X_tr, _, y_tr, _ = load_split()
        return X_tr, y_tr

    def _show_personas(self, data, error):
        if error:
            self.status.set(f"Не удалось загрузить данные: {error}")
            return
        self.data = data
        personas = list_personas(data[1], self.cfg["me"])
        self.persona_box["values"] = personas
        if not self.persona.get() and personas:
            self.persona.set(personas[0])

    # --- adding a style from a Telegram export -------------------------------
    def add_style(self):
        folder = filedialog.askdirectory(title="Папка экспорта Telegram (где лежит messages.html)")
        if not folder:
            return
        self.status.set("Читаю экспорт…")
        self._work(lambda: export_authors(Path(folder)),
                   lambda authors, e: self._pick_author(Path(folder), authors, e))

    def _pick_author(self, folder: Path, authors, error):
        if error or not authors:
            self.status.set(f"Не получилось прочитать экспорт: {error or 'в папке нет сообщений'}")
            return
        self.status.set("")
        win = tk.Toplevel(self.root)
        win.title("Чей стиль добавить?")
        win.transient(self.root)
        win.grab_set()
        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Участник (число фрагментов, нужно от 30):").pack(anchor="w")
        box = tk.Listbox(frame, height=min(12, len(authors)), width=44, font=("Segoe UI", 10))
        ranked = [a for a, _ in authors.most_common()]
        for a in ranked:
            box.insert("end", f"{authors[a]:5d}   {a}")
        box.pack(fill="both", expand=True, pady=(4, 8))
        ttk.Label(frame, text="Как назвать в списке:").pack(anchor="w")
        name = tk.StringVar()
        ttk.Entry(frame, textvariable=name, width=30).pack(anchor="w", pady=(2, 8))
        box.bind("<<ListboxSelect>>", lambda _e: box.curselection() and name.set(ranked[box.curselection()[0]]))

        def ok():
            if not box.curselection():
                return
            author = ranked[box.curselection()[0]]
            win.destroy()
            self.status.set("Добавляю стиль…")
            self._work(lambda: add_telegram_style(folder, author, name.get() or author), self._style_added)
        ttk.Button(frame, text="Добавить", style="Accent.TButton", command=ok).pack(anchor="w")

    def _style_added(self, result, error):
        if error:
            self.status.set(f"Ошибка: {error}")
            return
        label, n = result
        self.status.set(f"Добавлен {label}: {n} фрагментов" + ("" if n >= 30 else " — мало, нужно от 30"))
        self.persona.set(label)
        self._work(self._load_data, self._show_personas)

    # --- settings ----------------------------------------------------------
    def _on_mode(self):
        self.persona_box.configure(state="disabled" if self.mode.get() == "neutral" else "readonly")

    def apply_hotkeys(self, save: bool = True):
        if save:
            self.cfg.update(mode=self.mode.get(), persona=self.persona.get(), scope=self.scope.get(),
                            hotkey=self.hk.get().strip(), undo_hotkey=self.undo_hk.get().strip())
            config.save(self.cfg)
        if self.listener:
            self.listener.stop()
        self.listener = hotkey.HotkeyListener(
            {"rewrite": self.cfg["hotkey"], "undo": self.cfg["undo_hotkey"]},
            lambda name: self.inbox.put(lambda: self.on_hotkey(name)))
        log.get().info("hotkeys %s / %s registered, errors=%s",
                       self.cfg["hotkey"], self.cfg["undo_hotkey"], self.listener.errors)
        if self.listener.errors:
            self.hk_status.set("⚠ " + "; ".join(self.listener.errors))
        else:
            self.hk_status.set(("Сохранено. " if save else "") +
                               f"Работает: {self.cfg['hotkey']} — переписать, {self.cfg['undo_hotkey']} — вернуть")

    # --- rewriting ---------------------------------------------------------
    def _make_rewriter(self, mode: str, persona: str) -> Rewriter:
        # runs in a worker thread: must not read Tk variables, they're passed in
        if mode == "neutral":
            return build_rewriter(self.llm, "neutral", "", [], [], self.cfg)
        if self.data is None:
            raise RuntimeError("данные ещё загружаются, попробуй через пару секунд")
        if not persona:
            raise RuntimeError("выбери, под кого писать, на вкладке «Настройки»")
        X_tr, y_tr = self.data
        if mode == "loop" and self.guide is None:
            from stylemask.cache import cached_attacker
            self._set_step("Первый запуск: обучаю линейку, ~1 мин", 0, 1)
            self.guide = cached_attacker("style", "all", X_tr, y_tr)
        key = (mode, persona, tuple(FeedbackStore(persona).counts().values()))
        if key not in self.rewriters:  # style card + example embeddings take seconds to build
            self._set_step("Готовлю стиль (только в первый раз)", 0, 1)
            try:
                self.rewriters[key] = build_rewriter(self.llm, mode, persona, X_tr, y_tr, self.cfg, guide=self.guide)
            except ValueError as e:
                raise RuntimeError(str(e)) from e
            self.rewriters[key].examples.pick("прогрев", 1)  # embed the example pool now, not mid-rewrite
        return self.rewriters[key]

    # --- progress ----------------------------------------------------------
    def _set_step(self, label: str, n: int, total: int):
        """Thread-safe: called from workers via the inbox."""
        self.inbox.put(lambda: setattr(self, "step", (label, n, total)))

    def _rewrite_with_progress(self, rw: Rewriter, text: str) -> str:
        total = rw.planned_steps()
        return rw.rewrite(text, on_step=lambda label, n: self._set_step(label, n, total))

    def _tick(self):
        if not self.busy:
            return
        label, n, total = self.step
        total = max(total, n)
        elapsed = time.time() - self.started
        counter = f" ({n}/{total})" if n else ""
        text = f"stylemask · {label}{counter} · {elapsed:.0f} с"
        self.status.set(text)
        self.overlay.show(text, anchor=self.anchor)
        self.progress["value"] = 100 * max(n - 1, 0) / total + min(elapsed % 10, 9) / total  # creeps within a step
        self.root.after(250, self._tick)

    def _start(self, message: str) -> bool:
        if self.busy:
            return False
        self.busy = True
        self.started = time.time()
        self.step = (message, 0, 1)
        self.go.configure(state="disabled")
        self.progress["value"] = 0
        self.progress.pack(side="right")
        self._tick()
        return True

    def _finish(self, message: str, error=None):
        self.busy = False
        self.go.configure(state="normal")
        self.progress.pack_forget()
        elapsed = time.time() - self.started
        if error:
            log.get().error("failed: %s", error)
            self.status.set(f"Ошибка: {error}")
            self.overlay.show(f"stylemask · ✗ {str(error)[:80]}", hide_after_ms=6000, anchor=self.anchor)
            winsound.MessageBeep(winsound.MB_ICONHAND)
            self._work(self._check_server, lambda ok, e: None if ok else self._show_server(ok, e))
        else:
            self.status.set(f"{message} · {elapsed:.0f} с")
            warn = message.startswith("⚠")
            self.overlay.show(f"stylemask · {'⚠ перечитай текст' if warn else '✓ готово'} · {elapsed:.0f} с",
                              hide_after_ms=6000 if warn else 2500, anchor=self.anchor)

    def on_hotkey(self, name: str):
        log.get().info("hotkey pressed: %s (busy=%s)", name, self.busy)
        select_all = self.scope.get() == "all"
        if name == "undo":
            if not self.last or not self._start("Возвращаю исходный текст…"):
                return
            window, original = self.last

            def undo():
                if hotkey.foreground_window() != window:
                    raise RuntimeError("вернуть можно только в том же окне")
                saved = hotkey.get_clipboard()
                hotkey.paste_text(original, select_all)
                hotkey.set_clipboard(saved)
            self._work(undo, lambda _r, e: self._finish("Исходный текст возвращён", e))
            return

        if self.busy:
            return
        self.anchor = hotkey.foreground_window()
        self._start("Беру текст из поля")
        mode, persona = self.mode.get(), self.persona.get()

        def run():
            lg = log.get()
            window = hotkey.foreground_window()
            lg.info("hotkey: mode=%s scope=%s window=%s (%s)", mode, "all" if select_all else "selection",
                    window, hotkey.window_class(window))
            saved = hotkey.get_clipboard()
            original = hotkey.grab_text(select_all)
            lg.info("grabbed %s chars", len(original) if original else None)
            if not original or not original.strip():
                hotkey.set_clipboard(saved)
                raise RuntimeError("не удалось взять текст: курсор должен стоять в поле ввода")
            rw = self._make_rewriter(mode, persona)
            t = time.time()
            result = self._rewrite_with_progress(rw, original)
            lg.info("rewritten in %.1fs: %d chars, warning=%s", time.time() - t, len(result), bool(rw.warning))
            if not hotkey.wait_for_foreground(window):
                lg.info("foreground changed to %s: result left in clipboard", hotkey.foreground_window())
                hotkey.set_clipboard(result)
                return original, window, "Окно сменилось — результат скопирован в буфер обмена", result
            self._set_step("Вставляю", rw.planned_steps(), rw.planned_steps())
            hotkey.paste_text(result, select_all)
            hotkey.set_clipboard(saved)
            lg.info("pasted")
            if rw.warning:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                return original, window, f"⚠ Перечитай: {rw.warning[:200]}", result
            return original, window, f"Готово · {self.cfg['undo_hotkey']} — вернуть как было", result

        def done(res, error):
            if res:
                original, window, message, result = res
                self.last = (window, original)
                self._show_texts(original, result)
            self._finish(res[2] if res else "", error)
        self._work(run, done)

    def rewrite(self):
        text = self.src.get("1.0", "end").strip()
        if not text or self.busy:
            return
        self.anchor = int(self.root.wm_frame(), 16)
        self._start("Готовлюсь")
        mode, persona = self.mode.get(), self.persona.get()

        def run():
            rw = self._make_rewriter(mode, persona)
            return self._rewrite_with_progress(rw, text), rw.warning

        def done(result, error):
            if not error:
                self._show_texts(text, result[0])
            self._finish(f"⚠ Перечитай: {result[1][:200]}" if result and result[1] else "Готово", error)
        self._work(run, done)

    def _show_texts(self, original: str, result: str):
        """Mirror the last rewrite (manual or via hotkey) in the «Вручную» tab."""
        self.src.delete("1.0", "end")
        self.src.insert("1.0", original)
        self.dst.delete("1.0", "end")
        self.dst.insert("1.0", result)
        mode, persona = self.mode.get(), self.persona.get()
        self.last_result = (persona if mode != "neutral" else "", result)
        self._show_feedback_counts()

    def _show_feedback_counts(self):
        persona = self.last_result[0] if self.last_result else ""
        if not persona:
            self.fb_status.set("Отзывы работают в режимах «как другой автор».")
            return
        c = FeedbackStore(persona).counts()
        self.fb_status.set(f"{persona}: 👍 {c['good']} · 👎 {c['bad']} · ✏️ {c['fixed']} · замечаний {c['note']}")

    def feedback(self, kind: str):
        if not self.last_result or not self.last_result[0]:
            self.fb_status.set("Сначала перепиши текст в режиме «как другой автор».")
            return
        persona, result = self.last_result
        store = FeedbackStore(persona)
        if kind == "note":
            note = self.note.get().strip()
            if not note:
                return
            store.add("note", note=note)
            self.note.set("")
        elif kind == "fixed":
            fixed = self.dst.get("1.0", "end").strip()
            if fixed == result.strip():
                self.fb_status.set("Сначала поправь текст в поле «Результат» так, как написал бы этот человек.")
                return
            store.add("fixed", result=result, fixed=fixed)
        else:
            store.add(kind, result=result)
        log.get().info("feedback %s for %s", kind, persona)
        self._show_feedback_counts()

    def test_overlay(self):
        self.overlay.show("stylemask · так выглядит индикатор", hide_after_ms=4000,
                          anchor=int(self.root.wm_frame(), 16))
        log.get().info("overlay test: rect=%s", self.overlay.win.geometry())

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.dst.get("1.0", "end").strip())
        self.status.set("Скопировано")

    def clear(self):
        self.src.delete("1.0", "end")
        self.dst.delete("1.0", "end")


def _already_running() -> bool:
    """A named mutex lives as long as the first instance — a second one would lose the hotkeys silently."""
    import ctypes
    kernel32 = ctypes.windll.kernel32
    main._mutex = kernel32.CreateMutexW(None, False, "Local\\stylemask-gui")  # keep the handle alive
    return kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS


def main():
    if _already_running():
        from tkinter import messagebox
        tk.Tk().withdraw()
        messagebox.showwarning("stylemask", "stylemask уже запущен.\n\nЧтобы открыть новую версию, "
                               "закрой старое окно stylemask и запусти снова.")
        return
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.listener and app.listener.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
