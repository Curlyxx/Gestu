"""Abre el panel de control. Uso:  python app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gesture_control.gui import main  # noqa: E402

if __name__ == "__main__":
    main()
