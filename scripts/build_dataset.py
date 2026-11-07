"""Construye data/splits/splits.json (train/val/test por sesión de grabación)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture_control.config import load_settings  # noqa: E402
from gesture_control.dataset import build_splits  # noqa: E402
from gesture_control.paths import ensure_dirs  # noqa: E402

if __name__ == "__main__":
    s = load_settings()
    ensure_dirs(s["gestures"])
    report = build_splits(s)
    print(f"{'gesto':<18}{'clips':>6}{'train':>7}{'val':>6}{'test':>6}  modo")
    for g, r in report.items():
        print(f"{g:<18}{r['clips']:>6}{r['train']:>7}{r['val']:>6}{r['test']:>6}  {r['mode']}")
    low = [g for g, r in report.items() if min(r["train"], r["val"], r["test"]) < 1]
    if low:
        print(f"\nAviso: sin datos suficientes para train/val/test en: {', '.join(low)}")
