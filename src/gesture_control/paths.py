"""Rutas del proyecto. Todo se calcula desde la raíz (la carpeta que contiene src/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = ROOT / "config"
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
ACTIONS_FILE = CONFIG_DIR / "actions.yaml"

DATA_DIR = ROOT / "data"
RAW_VIDEOS = DATA_DIR / "raw_videos"
LANDMARKS = DATA_DIR / "landmarks"
SPLITS = DATA_DIR / "splits"
SPLITS_FILE = SPLITS / "splits.json"

MODELS = ROOT / "models"
MODEL_FILE = MODELS / "gesture_model.keras"
LABELS_FILE = MODELS / "labels.json"
HISTORY_FILE = MODELS / "history.json"
TEST_RESULTS_FILE = MODELS / "test_results.json"
CONFUSION_PNG = MODELS / "confusion_matrix.png"
HAND_MODEL_FILE = MODELS / "hand_landmarker.task"

RUNTIME = ROOT / "runtime"
PID_FILE = RUNTIME / "daemon.pid"
STATUS_FILE = RUNTIME / "status.json"
COMMAND_FILE = RUNTIME / "command.json"
STOP_FLAG = RUNTIME / "stop.flag"
LOG_FILE = RUNTIME / "daemon.log"

SCRIPTS = ROOT / "scripts"


def ensure_dirs(gestures=()):
    for d in (RAW_VIDEOS, LANDMARKS, SPLITS, MODELS, RUNTIME):
        d.mkdir(parents=True, exist_ok=True)
    for g in gestures:
        (RAW_VIDEOS / g).mkdir(parents=True, exist_ok=True)
        (LANDMARKS / g).mkdir(parents=True, exist_ok=True)


def rel(path):
    """Ruta relativa a la raíz, con '/' (portable entre Linux y Windows)."""
    return Path(path).resolve().relative_to(ROOT).as_posix()
