"""Ejecución de acciones de ventana según el sistema operativo (pynput / ydotool / comando)."""
import os
import shlex
import shutil
import subprocess
import time

from .config import load_actions

DESTRUCTIVE = {"close_window"}

# Códigos de linux/input-event-codes.h (para ydotool)
YDO = {
    "esc": 1, "tab": 15, "enter": 28, "ctrl": 29, "shift": 42, "alt": 56, "space": 57,
    "meta": 125, "win": 125, "super": 125, "cmd": 125,
    "up": 103, "left": 105, "right": 106, "down": 108,
    "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63, "f6": 64, "f7": 65, "f8": 66,
    "f9": 67, "f10": 68, "f11": 87, "f12": 88,
}
for _c, _n in zip("qwertyuiop", range(16, 26)):
    YDO[_c] = _n
for _c, _n in zip("asdfghjkl", range(30, 39)):
    YDO[_c] = _n
for _c, _n in zip("zxcvbnm", range(44, 51)):
    YDO[_c] = _n


def resolve_backend(settings, info):
    choice = settings.get("system", {}).get("backend", "auto")
    if choice != "auto":
        return choice
    if info["system"] == "linux" and info["wayland"]:
        return "ydotool" if shutil.which("ydotool") else "pynput"
    return "pynput"


class PynputBackend:
    name = "pynput"
    _alias = {"win": "cmd", "meta": "cmd", "super": "cmd", "control": "ctrl",
              "escape": "esc", "return": "enter"}

    def __init__(self):
        from pynput.keyboard import Controller, Key
        self._kb = Controller()
        self._Key = Key

    def _k(self, name):
        n = self._alias.get(name.lower(), name.lower())
        if len(n) == 1:
            return n
        k = getattr(self._Key, n, None)
        if k is None:
            raise ValueError(f"Tecla desconocida: {name}")
        return k

    def press(self, name):
        self._kb.press(self._k(name))
        time.sleep(0.02)

    def release(self, name):
        self._kb.release(self._k(name))
        time.sleep(0.02)

    def tap(self, name, mods=()):
        mods = list(mods)
        for m in mods:
            self.press(m)
        self.press(name)
        self.release(name)
        for m in reversed(mods):
            self.release(m)


class YdotoolBackend:
    name = "ydotool"

    def __init__(self):
        if not shutil.which("ydotool"):
            raise RuntimeError("ydotool no está instalado (y ydotoold debe estar corriendo).")

    @staticmethod
    def _code(name):
        c = YDO.get(name.lower())
        if c is None:
            raise ValueError(f"Tecla desconocida para ydotool: {name}")
        return c

    @staticmethod
    def _run(seq):
        subprocess.run(["ydotool", "key", *seq], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def press(self, name):
        self._run([f"{self._code(name)}:1"])

    def release(self, name):
        self._run([f"{self._code(name)}:0"])

    def tap(self, name, mods=()):
        m = [self._code(x) for x in mods]
        c = self._code(name)
        self._run([f"{x}:1" for x in m] + [f"{c}:1", f"{c}:0"] + [f"{x}:0" for x in reversed(m)])


class ActionRunner:
    def __init__(self, settings, info):
        self.info = info
        self.cfg = load_actions()
        self.backend_name = resolve_backend(settings, info)
        self.alt_hold = float(settings["inference"].get("alt_hold_seconds", 1.2))
        self._backend = None
        self._alt_active = False
        self._alt_last = 0.0

    # ---- resolución ----
    @property
    def backend(self):
        if self._backend is None:
            self._backend = YdotoolBackend() if self.backend_name == "ydotool" else PynputBackend()
        return self._backend

    def action_for(self, gesture):
        return (self.cfg.get("gestures") or {}).get(gesture)

    def _spec(self, action):
        table = (self.cfg.get("actions") or {}).get(action)
        if not table:
            return None
        order = []
        if self.info["system"] == "linux":
            if self.info["wayland"]:
                order.append("linux_wayland")
            order.append("linux")
        else:
            order.append(self.info["system"])
        order.append("all")
        for key in order:
            if key in table:
                return table[key]
        return None

    @staticmethod
    def _text(spec):
        if "alt_tab" in spec:
            return "Alt+Tab" if spec["alt_tab"] != "backward" else "Alt+Shift+Tab"
        if "keys" in spec:
            return spec["keys"]
        return f"cmd: {spec.get('cmd')}"

    # ---- ejecución ----
    def run(self, action, dry_run=False):
        spec = self._spec(action)
        if spec is None:
            return f"{action}: sin definición para este sistema"
        text = f"{action} ({self._text(spec)})"
        if dry_run:
            return f"[dry-run] {text}"
        if "alt_tab" in spec:
            self._alt_tab(spec["alt_tab"] == "backward")
        else:
            self.release_alt()
            if "keys" in spec:
                parts = [p.strip() for p in spec["keys"].split("+")]
                self.backend.tap(parts[-1], parts[:-1])
            elif "cmd" in spec:
                subprocess.Popen(shlex.split(spec["cmd"], posix=(os.name != "nt")),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return text

    def _alt_tab(self, backward):
        if not self._alt_active:
            self.backend.press("alt")
            self._alt_active = True
        self.backend.tap("tab", ["shift"] if backward else [])
        self._alt_last = time.monotonic()

    def tick(self):
        """Suelta Alt cuando pasó el tiempo de espera sin nuevos swipes."""
        if self._alt_active and time.monotonic() - self._alt_last > self.alt_hold:
            self.release_alt()

    def release_alt(self):
        if self._alt_active:
            try:
                self.backend.release("alt")
            finally:
                self._alt_active = False
