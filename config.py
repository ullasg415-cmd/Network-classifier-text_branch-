"""All settings for the text branch, read from environment variables at call time.

Reading at call time (not import time) means tests and Docker can change them
without re-importing anything.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def model_dir() -> Path:
    return Path(os.getenv("TEXT_MODEL_DIR", str(PKG_DIR / "models")))


def model_file() -> Path:
    return model_dir() / "text_svm.joblib"


def meta_file() -> Path:
    return model_dir() / "model_meta.json"


# --- extraction -----------------------------------------------------------
def extractor_mode() -> str:
    """auto  = use Tika if TIKA_URL is set and reachable, else local libraries
    tika  = Tika only (error if unreachable)
    local = pypdf / python-docx / plain text only
    """
    mode = os.getenv("TEXT_EXTRACTOR", "auto").lower()
    return mode if mode in {"auto", "tika", "local"} else "auto"


def tika_url() -> str:
    return os.getenv("TIKA_URL", "").strip()


def tika_timeout() -> float:
    return float(os.getenv("TIKA_TIMEOUT", "30"))


def max_upload_bytes() -> int:
    return int(os.getenv("TEXT_MAX_BYTES", str(20 * 1024 * 1024)))


def min_chars() -> int:
    """Fewer extracted characters than this -> 'unknown' (scanned PDF, empty file)."""
    return int(os.getenv("TEXT_MIN_CHARS", "30"))


# --- classification ---------------------------------------------------------
def unknown_threshold() -> float | None:
    """Confidence below this -> category 'unknown'.

    None means 'use the value recommended by training (model_meta.json)'.
    Set TEXT_UNKNOWN_THRESHOLD=0 to switch rejection off.
    """
    raw = os.getenv("TEXT_UNKNOWN_THRESHOLD")
    return float(raw) if raw not in (None, "") else None


# --- MLflow -------------------------------------------------------------------
def model_source() -> str:
    """auto     = registry if MLFLOW_TRACKING_URI is set (fall back to local file)
    registry = registry only
    local    = the joblib file in models/ only
    """
    src = os.getenv("TEXT_MODEL_SOURCE", "auto").lower()
    return src if src in {"auto", "registry", "local"} else "auto"


def registered_model_name() -> str:
    return os.getenv("TEXT_MODEL_NAME", "text-classifier")


def model_alias() -> str:
    return os.getenv("TEXT_MODEL_ALIAS", "champion")


def experiment_name() -> str:
    return os.getenv("TEXT_MLFLOW_EXPERIMENT", "text-classifier")
