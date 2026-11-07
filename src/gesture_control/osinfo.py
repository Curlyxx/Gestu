"""Detección del sistema operativo y del tipo de sesión gráfica."""
import os
import platform

NAMES = {"linux": "Linux", "windows": "Windows", "darwin": "macOS"}


def detect():
    system = platform.system().lower()
    wayland = system == "linux" and (
        os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    return {
        "system": system,
        "wayland": wayland,
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", ""),
        "python": platform.python_version(),
    }


def describe(info, backend):
    name = NAMES.get(info["system"], info["system"])
    extra = ""
    if info["system"] == "linux":
        extra = " (Wayland)" if info["wayland"] else " (X11)"
        if info["desktop"]:
            extra += f" · {info['desktop']}"
    text = f"{name}{extra} · Python {info['python']} · backend de acciones: {backend}"
    if info["system"] == "linux" and info["wayland"] and backend == "pynput":
        text += "\n⚠ Wayland: pynput no puede enviar teclas. Instala ydotool (ver README) o inicia sesión en X11."
    return text
