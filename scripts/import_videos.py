"""Extrae landmarks de videos en data/raw_videos/<gesto>/ y los guarda como clips.

Un video corto (<= ~2.4 s) es un clip. Uno largo se corta en segmentos de clip_seconds
(útil para 'ninguno': movimiento normal de la mano).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gesture_control.config import load_settings  # noqa: E402
from gesture_control.features import RAW_DIM, valid_mask  # noqa: E402
from gesture_control.hands import HandTracker  # noqa: E402
from gesture_control.paths import LANDMARKS, ensure_dirs  # noqa: E402
from gesture_control.status import VIDEO_EXT  # noqa: E402


def extract(video, mirror):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = HandTracker()  # uno por video: los timestamps deben empezar de cero
    lms, ts, i = [], [], 0
    zeros = np.zeros(RAW_DIM, np.float32)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            lm = tracker.process(frame, i / fps * 1000)
            lms.append(lm if lm is not None else zeros)
            ts.append(i / fps)
            i += 1
    finally:
        cap.release()
        tracker.close()
    return (np.stack(lms), np.array(ts)) if lms else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="reprocesar videos ya importados")
    args = ap.parse_args()
    s = load_settings()
    ensure_dirs(s["gestures"])
    clip_s = float(s["sequence"]["clip_seconds"])
    min_ratio = float(s["dataset"]["min_valid_ratio"])
    mirror = bool(s["camera"].get("mirror", True))
    total = 0
    for g in s["gestures"]:
        for video in sorted((Path(__file__).resolve().parents[1] / "data" / "raw_videos" / g).glob("*")):
            if video.suffix.lower() not in VIDEO_EXT:
                continue
            done = LANDMARKS / g / f"{video.stem}.done"
            if done.exists() and not args.force:
                continue
            lm, t = extract(video, mirror)
            if lm is None or t[-1] <= 0:
                print(f"[{g}] {video.name}: no se pudo leer")
                continue
            dur = float(t[-1])
            starts = [0.0] if dur <= clip_s * 1.6 else list(np.arange(0.0, dur - clip_s * 0.5, clip_s))
            n = 0
            for k, a in enumerate(starts):
                b = dur + 1e-6 if len(starts) == 1 else a + clip_s
                m = (t >= a) & (t < b)
                if m.sum() < 5 or valid_mask(lm[m]).mean() < min_ratio:
                    continue
                np.savez_compressed(LANDMARKS / g / f"vid_{video.stem}_s{k:02d}.npz",
                                    lm=lm[m].astype(np.float32), t=t[m] - t[m][0],
                                    session=np.array(f"vid_{video.stem}"))
                n += 1
            done.touch()
            total += n
            print(f"[{g}] {video.name}: {n} clip(s) válidos de {len(starts)} segmento(s)")
    print(f"\nImportados {total} clips. Siguiente paso: build_dataset.py")


if __name__ == "__main__":
    main()
