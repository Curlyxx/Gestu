"""Detección de landmarks de una mano con MediaPipe Tasks (HandLandmarker).

Usa solo la API Tasks: `mediapipe.solutions` fue eliminada en versiones recientes.
El modelo (.task) se descarga una vez a models/hand_landmarker.task.
"""
import urllib.request

import cv2
import numpy as np

from .paths import HAND_MODEL_FILE, MODELS

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_hand_model():
    if HAND_MODEL_FILE.exists() and HAND_MODEL_FILE.stat().st_size > 1000:
        return HAND_MODEL_FILE
    MODELS.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo de manos a {HAND_MODEL_FILE} ...", flush=True)
    try:
        tmp = HAND_MODEL_FILE.with_suffix(".tmp")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(HAND_MODEL_FILE)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"No se pudo descargar el modelo de manos ({e}). Descárgalo manualmente de\n{MODEL_URL}\n"
            f"y guárdalo como {HAND_MODEL_FILE}"
        ) from e
    return HAND_MODEL_FILE


class HandTracker:
    def __init__(self, min_detection=0.6, min_tracking=0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, frame_bgr, ts_ms):
        """Devuelve un vector (63,) con x,y,z de los 21 puntos, o None si no hay mano."""
        ts = max(int(ts_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(image, ts)
        if not res.hand_landmarks:
            return None
        pts = res.hand_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32).reshape(-1)

    def close(self):
        try:
            self._lm.close()
        except Exception:  # noqa: BLE001
            pass


def draw(frame, lm, color=(0, 255, 120)):
    """Dibuja el esqueleto de la mano sobre el frame."""
    if lm is None:
        return frame
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in lm.reshape(21, 3)[:, :2]]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
    return frame
