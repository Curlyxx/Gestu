"""Inicia/detiene el demonio de reconocimiento.

  python scripts/run.py                  # en primer plano
  python scripts/run.py --dry-run        # solo muestra las acciones, no las ejecuta
  python scripts/run.py --background     # desacoplado (sigue al cerrar la terminal)
  python scripts/run.py --status | --stop
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gesture_control.daemon import main  # noqa: E402

if __name__ == "__main__":
    main()
