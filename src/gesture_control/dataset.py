"""Splits train/val/test y carga de datos (solo numpy, sin TensorFlow)."""
import json
import math
import random
import time
from collections import defaultdict

import numpy as np

from .features import FEAT_DIM, make_features
from .paths import LANDMARKS, SPLITS, SPLITS_FILE


def clip_files(gesture):
    d = LANDMARKS / gesture
    return sorted(d.glob("*.npz")) if d.exists() else []


def _session_of(path):
    with np.load(path) as d:
        return str(d["session"])


def build_splits(settings):
    """Divide por sesión de grabación (o por bloques cronológicos si hay pocas sesiones)."""
    ds = settings["dataset"]
    rng = random.Random(ds.get("seed", 42))
    val_r, test_r = ds["val_ratio"], ds["test_ratio"]
    files_map, report = {}, {}
    for g in settings["gestures"]:
        files = clip_files(g)
        n = len(files)
        if n == 0:
            report[g] = {"clips": 0, "mode": "-", "train": 0, "val": 0, "test": 0}
            continue
        by_sess = defaultdict(list)
        for f in files:
            by_sess[_session_of(f)].append(f)
        if len(by_sess) >= 3:
            mode = f"sesiones ({len(by_sess)})"
            sess = list(by_sess.values())
            rng.shuffle(sess)
            t_target, v_target = max(1, math.ceil(n * test_r)), max(1, math.ceil(n * val_r))
            train, val, test = [], [], []
            remaining = len(sess)
            for fs in sess:
                if len(test) < t_target and remaining > 2:
                    test += fs
                elif len(val) < v_target and remaining > 1:
                    val += fs
                else:
                    train += fs
                remaining -= 1
        else:
            mode = "bloques cronológicos"
            if n < 5:
                train, val, test = files, [], []
            else:
                n_test, n_val = max(1, round(n * test_r)), max(1, round(n * val_r))
                train = files[: n - n_val - n_test]
                val = files[n - n_val - n_test: n - n_test]
                test = files[n - n_test:]
        for sp, group in (("train", train), ("val", val), ("test", test)):
            for f in group:
                files_map[f"{g}/{f.name}"] = sp
        report[g] = {"clips": n, "mode": mode, "train": len(train), "val": len(val), "test": len(test)}
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_FILE, "w", encoding="utf-8") as f:
        json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files_map}, f, indent=1)
    return report


def load_splits():
    if not SPLITS_FILE.exists():
        return {}
    try:
        with open(SPLITS_FILE, encoding="utf-8") as f:
            return json.load(f).get("files", {})
    except (OSError, ValueError):
        return {}


def load_xy(settings, split, augment=False, seed=0):
    idx = {g: i for i, g in enumerate(settings["gestures"])}
    steps = settings["sequence"]["steps"]
    min_ratio = settings["dataset"]["min_valid_ratio"]
    k_aug = int(settings["training"]["aug_factor"]) if augment else 0
    rng = np.random.default_rng(seed)
    X, y = [], []
    for key, sp in load_splits().items():
        g = key.split("/")[0]
        path = LANDMARKS / key
        if sp != split or g not in idx or not path.exists():
            continue
        with np.load(path) as d:
            lm, t = d["lm"], d["t"]
        for k in range(1 + k_aug):
            x = make_features(lm, t, steps, augment=k > 0, rng=rng, min_ratio=min_ratio)
            if x is not None:
                X.append(x)
                y.append(idx[g])
    if not X:
        return np.zeros((0, steps, FEAT_DIM), np.float32), np.zeros((0,), np.int64)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.int64)
