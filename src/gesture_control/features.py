"""Preprocesado de secuencias de landmarks (solo numpy, sin TensorFlow)."""
import numpy as np

N_POINTS = 21
RAW_DIM = N_POINTS * 3          # 63 valores por frame
FEAT_DIM = RAW_DIM + 3          # + trayectoria de la muñeca


def valid_mask(lm):
    """Frames donde se detectó la mano (los frames sin mano se guardan en ceros)."""
    return np.abs(lm).sum(axis=1) > 0


def resample(lm, t, steps):
    """(T,63) con tiempos t -> (steps,63) en una malla uniforme. Devuelve (seq|None, ratio_válidos)."""
    lm = np.asarray(lm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return None, 0.0
    valid = valid_mask(lm)
    ratio = float(valid.mean())
    if valid.sum() < 2 or t[-1] <= t[0]:
        return None, ratio
    grid = np.linspace(t[0], t[-1], steps)
    tv = t[valid]
    out = np.empty((steps, lm.shape[1]))
    for j in range(lm.shape[1]):
        out[:, j] = np.interp(grid, tv, lm[valid, j])
    return out, ratio


def normalize(seq):
    """Forma de la mano relativa a la muñeca + trayectoria de la muñeca, ambas en unidades de palma."""
    s = seq.shape[0]
    p = seq.reshape(s, N_POINTS, 3)
    wrist = p[:, 0, :]
    scale = float(np.median(np.linalg.norm(p[:, 9, :2] - p[:, 0, :2], axis=1)))
    scale = max(scale, 0.02)
    shape = (p - wrist[:, None, :]) / scale
    traj = (wrist - wrist[0]) / scale
    return np.concatenate([shape.reshape(s, RAW_DIM), traj], axis=1).astype(np.float32)


def _crop_time(lm, t, rng):
    span = t[-1] - t[0]
    a = t[0] + rng.uniform(0, 0.15) * span
    b = t[-1] - rng.uniform(0, 0.15) * span
    keep = (t >= a) & (t <= b)
    if keep.sum() < 4:
        return lm, t
    return lm[keep], t[keep]


def _augment_seq(seq, rng):
    p = seq.reshape(-1, N_POINTS, 3).copy()
    origin = p[0, 0, :2].copy()
    ang = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([[c, -s], [s, c]])
    xy = (p[:, :, :2] - origin) @ rot.T
    xy *= rng.uniform(0.92, 1.08, size=2)
    p[:, :, :2] = xy + origin
    p += rng.normal(0, 0.004, p.shape)
    return p.reshape(seq.shape)


def make_features(lm, t, steps, augment=False, rng=None, min_ratio=0.0):
    """Clip crudo -> matriz (steps, FEAT_DIM) o None si el clip no sirve."""
    lm = np.asarray(lm, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if augment:
        rng = rng or np.random.default_rng()
        lm, t = _crop_time(lm, t, rng)
    seq, ratio = resample(lm, t, steps)
    if seq is None or ratio < min_ratio:
        return None
    if augment:
        seq = _augment_seq(seq, rng)
    return normalize(seq)
