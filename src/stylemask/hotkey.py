"""Global hotkeys and "replace the text being typed" on Windows, via plain Win32 (ctypes).

No keyboard hook and no keystroke logging: RegisterHotKey only reports our
own combination. When it fires we do what a person would do by hand —
select all, copy, (rewrite), paste — and restore the user's clipboard.
"""
import ctypes
import threading
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# 64-bit handles: without explicit restypes ctypes truncates them to int
user32.GetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.restype = wintypes.HANDLE
user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
user32.GetForegroundWindow.restype = wintypes.HWND
kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
kernel32.GlobalAlloc.argtypes = (wintypes.UINT, ctypes.c_size_t)
kernel32.GlobalLock.restype = wintypes.LPVOID
kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)

WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
MOD = {"alt": 0x1, "ctrl": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN = 0x11, 0x10, 0x12, 0x5B, 0x5C
KEYEVENTF_KEYUP = 0x2
CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x2
_SPECIAL_VK = {"space": 0x20, "enter": 0x0D, "tab": 0x09, "insert": 0x2D, "pause": 0x13,
               **{f"f{i}": 0x6F + i for i in range(1, 13)}}


def parse(combo: str) -> tuple[int, int]:
    """'ctrl+alt+r' -> (modifiers, virtual key code)."""
    mods, vk = 0, None
    for part in (p.strip().lower() for p in combo.split("+")):
        if part in MOD:
            mods |= MOD[part]
        elif part in _SPECIAL_VK:
            vk = _SPECIAL_VK[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        else:
            raise ValueError(f"unknown key {part!r} in {combo!r}")
    if vk is None or not mods:
        raise ValueError(f"{combo!r}: need at least one modifier and one key")
    return mods, vk


class HotkeyListener:
    """Registers hotkeys on its own thread (RegisterHotKey is per-thread) and
    calls `on_hotkey(name)` from that thread when one fires."""

    def __init__(self, hotkeys: dict[str, str], on_hotkey):
        self.hotkeys, self.on_hotkey = hotkeys, on_hotkey
        self.errors: list[str] = []
        self._thread_id = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(2)

    def _run(self):
        self._thread_id = kernel32.GetCurrentThreadId()
        names = {}
        for i, (name, combo) in enumerate(self.hotkeys.items(), 1):
            try:
                mods, vk = parse(combo)
            except ValueError as e:
                self.errors.append(str(e))
                continue
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                names[i] = name
            else:
                self.errors.append(f"{combo}: уже занято другой программой")
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in names:
                self.on_hotkey(names[msg.wParam])
        for i in names:
            user32.UnregisterHotKey(None, i)

    def stop(self):
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2)


# --- clipboard ---------------------------------------------------------------
def _open_clipboard():
    for _ in range(20):  # another app may hold it for a moment
        if user32.OpenClipboard(None):
            return
        time.sleep(0.02)
    raise RuntimeError("буфер обмена занят другой программой")


def get_clipboard() -> str | None:
    _open_clipboard()
    try:
        handle = user32.GetClipboardData(CF_UNICODETEXT)
        if not handle:
            return None
        ptr = kernel32.GlobalLock(handle)
        try:
            return ctypes.wstring_at(ptr)
        finally:
            kernel32.GlobalUnlock(handle)
    finally:
        user32.CloseClipboard()


def set_clipboard(text: str | None):
    _open_clipboard()
    try:
        user32.EmptyClipboard()
        if text is None:
            return
        data = ctypes.create_unicode_buffer(text)
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, ctypes.sizeof(data))
        ctypes.memmove(kernel32.GlobalLock(handle), data, ctypes.sizeof(data))
        kernel32.GlobalUnlock(handle)
        user32.SetClipboardData(CF_UNICODETEXT, handle)  # clipboard now owns the memory
    finally:
        user32.CloseClipboard()


# --- keyboard ----------------------------------------------------------------
def _wait_modifiers_released(timeout: float = 2.0):
    """The user is still holding ctrl/alt of the hotkey; our ctrl+a would become ctrl+alt+a."""
    end = time.time() + timeout
    while time.time() < end:
        if not any(user32.GetAsyncKeyState(vk) & 0x8000
                   for vk in (VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN)):
            return
        time.sleep(0.02)


def _ctrl(key: str):
    vk = ord(key.upper())
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def foreground_window() -> int:
    return user32.GetForegroundWindow() or 0


def window_class(hwnd: int) -> str:
    """Window class name (e.g. Telegram is Qt...), for the event log — titles may hold chat names."""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def grab_text(select_all: bool = True, timeout: float = 0.7) -> str | None:
    """Copy the text of the focused field (or the current selection)."""
    _wait_modifiers_released()
    before = user32.GetClipboardSequenceNumber()
    if select_all:
        _ctrl("a")
        time.sleep(0.05)
    _ctrl("c")
    end = time.time() + timeout
    while time.time() < end:
        if user32.GetClipboardSequenceNumber() != before:
            time.sleep(0.03)
            return get_clipboard()
        time.sleep(0.02)
    return None


def paste_text(text: str, select_all: bool = True):
    """Replace the focused field (or selection) with `text` via the clipboard."""
    _wait_modifiers_released()
    set_clipboard(text)
    if select_all:
        _ctrl("a")
        time.sleep(0.05)
    _ctrl("v")
    time.sleep(0.15)  # let the target app read the clipboard before we restore it
