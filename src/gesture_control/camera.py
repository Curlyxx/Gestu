"""Apertura de cámara (misma lógica en Linux y Windows)."""
import os

import cv2


def open_camera(settings):
    cam = settings["camera"]
    idx = int(cam.get("index", 0))
    if os.name == "nt":
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {idx}. Revisa camera.index en config/settings.yaml.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
