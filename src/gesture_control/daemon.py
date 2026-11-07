"""Demonio: mantiene la cámara abierta, reconoce gestos en vivo y ejecuta acciones."""
import argparse
import json
import os
import signal
import time
from collections import deque

import numpy as np

from . import osinfo, status
from .actions import DESTRUCTIVE, ActionRunner
from .config import load_settings
from .features import FEAT_DIM, RAW_DIM, make_features
from .paths import (COMMAND_FILE, LABELS_FILE, MODEL_FILE, PID_FILE, RUNTIME, STATUS_FILE,
                    STOP_FLAG, ensure_dirs)


def _write_status(**kw):
    status.write_json_atomic(STATUS_FILE, {**kw, "ts": time.time()})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Demonio de reconocimiento de gestos")
    ap.add_argument("--dry-run", action="store_true", help="detecta y muestra la acción, sin ejecutarla")
    ap.add_argument("--preview", action="store_true", help="ventana con la cámara y el esqueleto")
    ap.add_argument("--background", action="store_true", help="lanzar desacoplado y salir")
    ap.add_argument("--stop", action="store_true", help="detener el demonio en ejecución")
    ap.add_argument("--status", action="store_true", help="mostrar el estado del demonio")
    args = ap.parse_args(argv)

    if args.stop:
        print("Demonio detenido." if status.stop_daemon() else "El demonio no está corriendo.")
        return
    if args.status:
        pid = status.daemon_pid()
        print(f"Corriendo (pid {pid})\n{json.dumps(status.read_daemon_status(), indent=1)}" if pid
              else "Detenido.")
        return
    if args.background:
        if status.daemon_pid():
            print(f"Ya está corriendo (pid {status.daemon_pid()}).")
        else:
            print(f"Demonio lanzado en segundo plano (pid {status.start_daemon(args.dry_run, args.preview)}).")
        return
    if status.daemon_pid():
        raise SystemExit(f"Ya hay un demonio corriendo (pid {status.daemon_pid()}). Usa --stop.")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    settings = load_settings()
    ensure_dirs(settings["gestures"])
    RUNTIME.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))
    STOP_FLAG.unlink(missing_ok=True)
    dry_run = args.dry_run
    base = {"pid": os.getpid(), "dry_run": dry_run}
    _write_status(state="starting", message="cargando modelo", **base)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    runner = cap = tracker = None
    try:
        if not MODEL_FILE.exists():
            raise RuntimeError("No hay modelo entrenado (models/gesture_model.keras). Ejecuta train.py.")
        with open(LABELS_FILE, encoding="utf-8") as f:
            labels = json.load(f)["gestures"]
        import cv2
        from tensorflow import keras
        from .camera import open_camera
        from .hands import HandTracker, draw

        steps = int(settings["sequence"]["steps"])
        model = keras.models.load_model(MODEL_FILE)
        model(np.zeros((1, steps, FEAT_DIM), np.float32), training=False)  # calentamiento

        info = osinfo.detect()
        runner = ActionRunner(settings, info)
        tracker = HandTracker()
        cap = open_camera(settings)
        inf = settings["inference"]
        clip_s = float(settings["sequence"]["clip_seconds"])
        mirror = bool(settings["camera"].get("mirror", True))
        zeros = np.zeros(RAW_DIM, np.float32)

        buf = deque()
        t0 = time.monotonic()
        frame_i = fails = streak = 0
        streak_label = None
        cooldown_until = 0.0
        last = {"last_gesture": None, "last_conf": None, "last_action": None, "last_time": None}
        fps_t, fps_n, fps = t0, 0, 0.0
        next_status = cmd_mtime = 0.0
        hand = False
        preview = args.preview
        print(f"Demonio listo · {runner.backend_name} · dry_run={dry_run}", flush=True)

        while not stop["flag"] and not STOP_FLAG.exists():
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    raise RuntimeError("La cámara dejó de entregar imágenes.")
                time.sleep(0.02)
                continue
            fails = 0
            if mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic() - t0
            lm = tracker.process(frame, now * 1000)
            hand = lm is not None
            buf.append((now, lm if hand else zeros))
            while buf and now - buf[0][0] > clip_s:
                buf.popleft()
            frame_i += 1
            fps_n += 1
            runner.tick()

            if (frame_i % int(inf["predict_every"]) == 0 and now >= cooldown_until
                    and buf[-1][0] - buf[0][0] >= 0.9 * clip_s):
                arr = np.stack([b[1] for b in buf])
                ts = np.array([b[0] for b in buf])
                x = make_features(arr, ts, steps, min_ratio=float(inf["min_hand_ratio"]))
                label, conf = None, 0.0
                if x is not None:
                    probs = model(x[None], training=False).numpy()[0]
                    k = int(probs.argmax())
                    label, conf = labels[k], float(probs[k])
                action = runner.action_for(label) if label else None
                need = float(inf["destructive_threshold"] if action in DESTRUCTIVE else inf["threshold"])
                if label and label != "ninguno" and action and conf >= need:
                    streak = streak + 1 if label == streak_label else 1
                    streak_label = label
                    if streak >= int(inf["votes"]) + (1 if action in DESTRUCTIVE else 0):
                        try:
                            msg = runner.run(action, dry_run)
                        except Exception as e:  # noqa: BLE001
                            msg = f"ERROR en {action}: {e}"
                        print(f"[{time.strftime('%H:%M:%S')}] {label} ({conf:.2f}) -> {msg}", flush=True)
                        last = {"last_gesture": label, "last_conf": round(conf, 3),
                                "last_action": msg, "last_time": time.strftime("%H:%M:%S")}
                        cooldown_until = now + float(inf["cooldown"])
                        buf.clear()
                        streak, streak_label = 0, None
                else:
                    streak, streak_label = 0, None

            tm = time.monotonic()
            if tm - fps_t >= 1.0:
                fps, fps_n, fps_t = fps_n / (tm - fps_t), 0, tm
            if tm >= next_status:
                next_status = tm + 0.5
                if COMMAND_FILE.exists() and COMMAND_FILE.stat().st_mtime != cmd_mtime:
                    cmd_mtime = COMMAND_FILE.stat().st_mtime
                    cmd = status.read_json(COMMAND_FILE) or {}
                    if "dry_run" in cmd:
                        dry_run = bool(cmd["dry_run"])
                _write_status(state="running", pid=os.getpid(), dry_run=dry_run, fps=round(fps, 1),
                              hand=hand, backend=runner.backend_name, **last)
            if preview:
                try:
                    draw(frame, lm)
                    cv2.putText(frame, f"{'DRY-RUN ' if dry_run else ''}{last['last_gesture'] or '-'}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imshow("Gesture Control", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except Exception:  # noqa: BLE001
                    preview = False
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", flush=True)
        _write_status(state="error", message=str(e), **base)
        raise
    finally:
        if runner:
            try:
                runner.release_alt()
            except Exception:  # noqa: BLE001
                pass
        if cap:
            cap.release()
        if tracker:
            tracker.close()
        PID_FILE.unlink(missing_ok=True)
        STOP_FLAG.unlink(missing_ok=True)
        if not (status.read_json(STATUS_FILE) or {}).get("state") == "error":
            _write_status(state="stopped", **base)
        print("Demonio detenido.", flush=True)
