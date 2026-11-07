"""Evaluación sobre el set de test: accuracy, reporte por clase y matriz de confusión."""
import json
import os

from .dataset import load_xy
from .paths import CONFUSION_PNG, LABELS_FILE, MODEL_FILE, TEST_RESULTS_FILE


def evaluate(settings):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from sklearn.metrics import classification_report, confusion_matrix
    from tensorflow import keras

    if not MODEL_FILE.exists():
        raise SystemExit("No hay modelo entrenado. Ejecuta train.py primero.")
    with open(LABELS_FILE, encoding="utf-8") as f:
        gestures = json.load(f)["gestures"]
    X, y = load_xy(settings, "test", augment=False)
    if len(X) == 0:
        raise SystemExit("El set de test está vacío. Graba más clips y ejecuta build_dataset.py.")
    model = keras.models.load_model(MODEL_FILE)
    pred = model.predict(X, verbose=0).argmax(axis=1)
    labels = list(range(len(gestures)))
    acc = float((pred == y).mean())
    cm = confusion_matrix(y, pred, labels=labels)
    rep = classification_report(y, pred, labels=labels, target_names=gestures,
                                zero_division=0, output_dict=True)
    print(f"Test: {len(X)} clips · accuracy {acc:.1%}\n")
    print(classification_report(y, pred, labels=labels, target_names=gestures, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(cm)
    with open(TEST_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "n_test": int(len(X)), "labels": gestures,
                   "confusion": cm.tolist(), "per_class": rep}, f, indent=1)
    _plot(cm, gestures, acc)


def _plot(cm, labels, acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión · accuracy {acc:.1%}")
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=120)
    plt.close(fig)
