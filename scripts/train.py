"""Entrena el modelo con data/splits (split 'train', validación con 'val')."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture_control.config import load_settings  # noqa: E402
from gesture_control.training import train  # noqa: E402

if __name__ == "__main__":
    train(load_settings())
