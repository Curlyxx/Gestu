"""Estado del proyecto y control del demonio. Lo usan la interfaz y los scripts (sin TensorFlow)."""
import json
import os
import subprocess
import sys
import time

import psutil

from .dataset import clip_files, load_splits
from .paths import (COMMAND_FILE, LABELS_FILE, LANDMARKS, LOG_FILE, MODEL_FILE,
                    PID_FILE, RAW_VIDEOS, ROOT, RUNTIME, SCRIPTS, SPLITS_FILE, STATUS_FILE,
                    STOP_FLAG, TEST_RESULTS_FILE)

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


# ---------------- dataset ----------------
def pending_videos(gesture):
    d = RAW_VIDEOS / gesture
    if not d.exists():
        return []
    return [v for v in sorted(d.iterdir())
            if v.suffix.lower() in VIDEO_EXT and not (LANDMARKS / gesture / f"{v.stem}.done").exists()]


def dataset_status(settings):
    splits = load_splits()
    rows = []
    for g in settings["gestures"]:
        counts, unsplit = {"train": 0, "val": 0, "test": 0}, 0
        files = clip_files(g)
        for f in files:
            sp = splits.get(f"{g}/{f.name}")
            if sp in counts:
                counts[sp] += 1
            else:
                unsplit += 1
        rows.append({"gesture": g, "clips": len(files), **counts, "unsplit": unsplit,
                     "pending_videos": len(pending_videos(g))})
    return rows


def readiness(settings, cam=None):
    """Lista de (nombre, ok, detalle, requerido) y bandera global de 'listo'."""
    rows = dataset_status(settings)
    minc = settings["dataset"]["min_clips_per_gesture"]
    checks = []

    low = [f"{r['gesture']} {r['clips']}/{minc}" for r in rows if r["clips"] < minc]
    checks.append(("Datos de gestos", not low,
                   "suficientes" if not low else "faltan clips → " + ", ".join(low), True))

    vids = sum(r["pending_videos"] for r in rows)
    checks.append(("Videos por importar", vids == 0,
                   "ninguno pendiente" if vids == 0 else f"{vids} video(s) sin importar", False))

    bad = [r["gesture"] for r in rows
           if r["unsplit"] or min(r["train"], r["val"], r["test"]) < 1]
    ok_split = SPLITS_FILE.exists() and not bad
    checks.append(("Train / val / test", ok_split,
                   "dividido" if ok_split else "falta (re)construir el dataset → "
                   + (", ".join(bad) if bad else "sin splits"), True))

    labels = read_json(LABELS_FILE)
    ok_model = MODEL_FILE.exists() and bool(labels) and labels.get("gestures") == settings["gestures"]
    detail = "entrenado" if ok_model else "sin entrenar (o los gestos cambiaron)"
    if ok_model and SPLITS_FILE.exists() and SPLITS_FILE.stat().st_mtime > MODEL_FILE.stat().st_mtime:
        detail += " · el dataset es más nuevo que el modelo: conviene reentrenar"
    checks.append(("Modelo", ok_model, detail, True))

    res = read_json(TEST_RESULTS_FILE)
    checks.append(("Test", bool(res),
                   f"accuracy {res['accuracy']:.1%} en {res['n_test']} clips" if res else "sin evaluar",
                   False))

    if cam is None:
        checks.append(("Cámara", False, "sin verificar", False))
    else:
        checks.append(("Cámara", cam[0], cam[1], False))

    ready = all(ok for _, ok, _, req in checks if req)
    return checks, ready


def camera_check(settings):
    if daemon_pid():
        return True, "en uso por el demonio"
    try:
        from .camera import open_camera
        cap = open_camera(settings)
        ok, _ = cap.read()
        cap.release()
        return (True, f"cámara {settings['camera']['index']} disponible") if ok else (False, "no entrega imagen")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


# ---------------- demonio ----------------
def daemon_pid():
    try:
        pid = int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        p = psutil.Process(pid)
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE and any(
                "run.py" in part for part in p.cmdline()):
            return pid
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return None


def start_daemon(dry_run=False, preview=False):
    if daemon_pid():
        return daemon_pid()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    STOP_FLAG.unlink(missing_ok=True)
    write_json_atomic(COMMAND_FILE, {"dry_run": bool(dry_run)})
    cmd = [sys.executable, str(SCRIPTS / "run.py")]
    if dry_run:
        cmd.append("--dry-run")
    if preview:
        cmd.append("--preview")
    kw = {"stdin": subprocess.DEVNULL, "cwd": str(ROOT)}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kw)
    return proc.pid


def stop_daemon(timeout=6.0):
    pid = daemon_pid()
    if not pid:
        return False
    RUNTIME.mkdir(parents=True, exist_ok=True)
    STOP_FLAG.write_text("stop")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not daemon_pid():
            return True
        time.sleep(0.2)
    try:
        psutil.Process(pid).terminate()
    except psutil.Error:
        pass
    return True


def read_daemon_status():
    st = read_json(STATUS_FILE)
    if not st or not daemon_pid():
        return None
    return st


def write_command(**kw):
    write_json_atomic(COMMAND_FILE, kw)
