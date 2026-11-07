"""Entrenamiento del clasificador."""
import json
import os

import numpy as np

from .dataset import load_xy
from .features import FEAT_DIM
from .paths import HISTORY_FILE, LABELS_FILE, MODEL_FILE, MODELS


def train(settings):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow import keras

    from .model import build_model

    tr = settings["training"]
    gestures = settings["gestures"]
    steps = settings["sequence"]["steps"]
    np.random.seed(settings["dataset"].get("seed", 42))
    tf.random.set_seed(settings["dataset"].get("seed", 42))

    Xtr, ytr = load_xy(settings, "train", augment=True, seed=1)
    Xva, yva = load_xy(settings, "val", augment=False)
    if len(Xtr) == 0 or len(Xva) == 0:
        raise SystemExit("Faltan datos de train/val. Graba clips y ejecuta build_dataset.py primero.")
    missing = [g for i, g in enumerate(gestures) if not (ytr == i).any()]
    if missing:
        raise SystemExit(f"Sin datos de entrenamiento para: {', '.join(missing)}")
    print(f"train: {len(Xtr)} muestras (con aumento) · val: {len(Xva)} · clases: {len(gestures)}", flush=True)

    classes = np.arange(len(gestures))
    w = compute_class_weight("balanced", classes=classes, y=ytr)
    model = build_model(steps, FEAT_DIM, len(gestures))
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
    ]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=int(tr["epochs"]),
                     batch_size=int(tr["batch_size"]), class_weight=dict(zip(classes, w)),
                     callbacks=cbs, verbose=2)
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump({"gestures": gestures, "steps": steps, "feat_dim": FEAT_DIM}, f, indent=1)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()
                   if k in ("loss", "val_loss", "accuracy", "val_accuracy")}, f)
    best = max(hist.history["val_accuracy"])
    print(f"\nModelo guardado en {MODEL_FILE.name}. Mejor val_accuracy: {best:.1%}", flush=True)
