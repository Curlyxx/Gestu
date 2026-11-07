import re

import yaml

from .paths import ACTIONS_FILE, SETTINGS_FILE


def _load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings():
    return _load(SETTINGS_FILE)


def save_settings(settings):
    """Reescribe todo el archivo (pierde los comentarios). Preferir update_settings."""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(settings, f, sort_keys=False, allow_unicode=True)


def _fmt(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def update_settings(changes):
    """Cambia solo los valores indicados {(seccion, clave): valor} conservando comentarios."""
    lines = SETTINGS_FILE.read_text(encoding="utf-8").splitlines()
    pending = dict(changes)
    section = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\w+):\s*(#.*)?$", line)
        if m:
            section = m.group(1)
            continue
        m = re.match(r"^(\s+)(\w+):\s*([^#]*?)(\s*)(#.*)?$", line)
        if m and (section, m.group(2)) in pending:
            val = _fmt(pending.pop((section, m.group(2))))
            lines[i] = f"{m.group(1)}{m.group(2)}: {val}{m.group(4) or ''}{m.group(5) or ''}"
    SETTINGS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if pending:  # clave nueva que no estaba en el archivo: respaldo completo
        s = load_settings()
        for (a, b), v in pending.items():
            s.setdefault(a, {})[b] = v
        save_settings(s)


def load_actions():
    return _load(ACTIONS_FILE)
