"""Graba clips de gestos con la cámara y guarda sus landmarks (y opcionalmente el video).

Teclas:  1-9 elegir gesto · ESPACIO grabar clip · U deshacer último · V guardar video on/off · Q/ESC salir
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from gesture_control.camera import open_camera  # noqa: E402
from gesture_control.config import load_settings  # noqa: E402
from gesture_control.dataset import clip_files  # noqa: E402
from gesture_control.features import RAW_DIM, valid_mask  # noqa: E402
from gesture_control.hands import HandTracker, draw  # noqa: E402
from gesture_control.paths import LANDMARKS, RAW_VIDEOS, ensure_dirs  # noqa: E402

WIN = "Grabar gestos"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-video", action="store_true", help="guardar también el .mp4 de cada clip")
    args = ap.parse_args()

    s = load_settings()
    gestures = s["gestures"]
    ensure_dirs(gestures)
    clip_s = float(s["sequence"]["clip_seconds"])
    min_ratio = float(s["dataset"]["min_valid_ratio"])
    mirror = bool(s["camera"].get("mirror", True))
    session = datetime.now().strftime("rec%Y%m%d-%H%M%S")
    save_video = args.save_video

    cap = open_camera(s)
    tracker = HandTracker()
    t0 = time.monotonic()
    zeros = np.zeros(RAW_DIM, np.float32)
    sel, counter, saved = 0, 0, []
    msg, msg_until = "", 0.0

    def grab():
        ok, raw = cap.read()
        if not ok:
            return None, None, None
        disp = cv2.flip(raw, 1) if mirror else raw
        lm = tracker.process(disp, (time.monotonic() - t0) * 1000)
        return raw, disp, lm

    def show(disp, lm, header, color=(255, 255, 255), border=None):
        img = draw(disp.copy(), lm)
        cv2.putText(img, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        counts = "  ".join(f"{i + 1}:{g[:8]}={len(clip_files(g))}" for i, g in enumerate(gestures))
        cv2.putText(img, counts, (10, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
        if border:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), border, 6)
        cv2.imshow(WIN, img)
        return cv2.waitKey(1) & 0xFF

    def record():
        nonlocal counter, msg, msg_until
        g = gestures[sel]
        for n in (3, 2, 1):
            end = time.monotonic() + 0.6
            while time.monotonic() < end:
                _, disp, lm = grab()
                if disp is not None:
                    show(disp, lm, f"Prepárate: {g}  ({n})", (0, 255, 255))
        lms, ts, frames = [], [], []
        start = time.monotonic()
        while time.monotonic() - start < clip_s:
            raw, disp, lm = grab()
            if disp is None:
                continue
            lms.append(lm if lm is not None else zeros)
            ts.append(time.monotonic() - start)
            if save_video:
                frames.append(raw)
            show(disp, lm, f"GRABANDO: {g}", (0, 0, 255), border=(0, 0, 255))
        arr, t = np.stack(lms), np.array(ts)
        ratio = float(valid_mask(arr).mean())
        if ratio < min_ratio:
            msg, msg_until = f"Descartado: mano detectada en {ratio:.0%} de los frames", time.time() + 3
            return
        counter += 1
        stem = f"{session}_{counter:03d}"
        path = LANDMARKS / g / f"{stem}.npz"
        np.savez_compressed(path, lm=arr.astype(np.float32), t=t, session=np.array(session))
        saved.append(path)
        if save_video and frames:
            fps = len(frames) / max(t[-1], 1e-3)
            h, w = frames[0].shape[:2]
            vw = cv2.VideoWriter(str(RAW_VIDEOS / g / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for fr in frames:
                vw.write(fr)
            vw.release()
            (LANDMARKS / g / f"{stem}.done").touch()  # ya está procesado: no re-importar
        msg, msg_until = f"Guardado {path.name} ({ratio:.0%} con mano)", time.time() + 2

    print(__doc__)
    try:
        while True:
            _, disp, lm = grab()
            if disp is None:
                continue
            head = f"[{sel + 1}] {gestures[sel]} · ESPACIO grabar · video {'ON' if save_video else 'off'}"
            key = show(disp, lm, msg if time.time() < msg_until else head)
            if key in (ord("q"), 27):
                break
            if key == 32:
                record()
            elif key == ord("v"):
                save_video = not save_video
            elif key == ord("u") and saved:
                p = saved.pop()
                p.unlink(missing_ok=True)
                msg, msg_until = f"Eliminado {p.name}", time.time() + 2
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(gestures):
                sel = key - ord("1")
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    print(f"Sesión {session}: {len(saved)} clips guardados. Siguiente paso: build_dataset.py")


if __name__ == "__main__":
    main()
