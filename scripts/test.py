"""Evalúa el modelo sobre el set de test (accuracy, reporte, matriz de confusión)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture_control.config import load_settings  # noqa: E402
from gesture_control.evaluation import evaluate  # noqa: E402

if __name__ == "__main__":
    evaluate(load_settings())
