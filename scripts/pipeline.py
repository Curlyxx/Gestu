"""Ejecuta en secuencia: importar videos -> construir dataset -> entrenar -> testear.

  python scripts/pipeline.py                   # todo
  python scripts/pipeline.py --skip-import     # sin importar videos
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-import", action="store_true")
    args = ap.parse_args()
    steps = ([] if args.skip_import else ["import_videos.py"]) + ["build_dataset.py", "train.py", "test.py"]
    for s in steps:
        print(f"\n===== {s} =====", flush=True)
        r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / s)], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"Falló {s} (código {r.returncode}). Pipeline detenido.")
    print("\nPipeline completo.")
