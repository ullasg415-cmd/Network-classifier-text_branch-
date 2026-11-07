"""Stages 2 and 3: TF-IDF vectoriser + Linear SVM, and loading the trained model.

Model source order (TEXT_MODEL_SOURCE=auto): the MLflow registry's `champion`
alias if MLFLOW_TRACKING_URI is set, otherwise (or if the registry is down) the
joblib file in models/. The loaded model is cached; call reload_model() after
promoting a new champion.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass

from . import config
from .registry import RegistryUnavailable

log = logging.getLogger(__name__)


def build_pipeline(max_features: int = 50_000, C: float = 1.0, calibration_cv: int = 3):
    """TF-IDF -> Linear SVM, wrapped in probability calibration.

    A plain LinearSVC only gives a distance from the decision boundary, not a
    probability. CalibratedClassifierCV turns that into a 0-1 confidence the
    API can report and threshold, which the Known/Unknown design needs.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, max_features=max_features,
        )),
        ("svm", CalibratedClassifierCV(
            LinearSVC(C=C, class_weight="balanced"), cv=calibration_cv,
        )),
    ])


@dataclass
class LoadedModel:
    pipeline: object
    meta: dict
    source: str  # "registry" | "local"

    @property
    def categories(self) -> list[str]:
        return [str(c) for c in self.pipeline.classes_]

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "unknown"))

    def threshold(self) -> float:
        env = config.unknown_threshold()
        if env is not None:
            return env
        return float(self.meta.get("unknown_threshold", 0.0))

    def predict(self, text: str, top_k: int = 3) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.categories, proba), key=lambda cp: cp[1], reverse=True)
        best, conf = ranked[0]
        return {
            "best_guess": best,
            "confidence": float(conf),
            "top_k": [{"category": c, "confidence": round(float(p), 4)} for c, p in ranked[:top_k]],
        }


_lock = threading.Lock()
_cached: LoadedModel | None = None


def _load_local() -> LoadedModel:
    import joblib
    path = config.model_file()
    if not path.exists():
        raise FileNotFoundError(
            f"No trained text model at {path}. Train one: python -m text.train --data-dir <folder>"
        )
    meta = {}
    if config.meta_file().exists():
        meta = json.loads(config.meta_file().read_text(encoding="utf-8"))
    check_library_compat(meta)
    pipeline = joblib.load(path)
    return LoadedModel(pipeline, meta, "local")


def check_library_compat(meta: dict) -> None:
    """The joblib file is a pickle: it must be loaded by the scikit-learn version that saved it.

    Mismatch -> warning by default (scikit-learn also emits InconsistentVersionWarning);
    with TEXT_STRICT_VERSIONS=1 it refuses to load, so a wrong environment fails loudly
    at start-up instead of predicting silently with a half-compatible model.
    """
    import os
    import sklearn
    trained = meta.get("sklearn_version")
    if trained and trained != sklearn.__version__:
        msg = (f"Model was trained with scikit-learn {trained} but {sklearn.__version__} is installed. "
               f"Install scikit-learn=={trained} or retrain: python -m text.train ...")
        if os.getenv("TEXT_STRICT_VERSIONS", "") in ("1", "true", "yes"):
            raise RuntimeError(msg)
        log.warning(msg)


def _load() -> LoadedModel:
    source = config.model_source()
    if source in ("auto", "registry"):
        from . import registry
        try:
            pipeline, meta = registry.load_champion()
            return LoadedModel(pipeline, meta, "registry")
        except RegistryUnavailable as e:
            if source == "registry":
                raise
            # MLFLOW_TRACKING_URI set = the operator expects the registry model, so a fallback must not be silent
            (log.warning if os.getenv("MLFLOW_TRACKING_URI") else log.info)(
                "Registry model NOT used (%s); serving the local model file instead", e)
    return _load_local()


def load_model() -> LoadedModel:
    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                _cached = _load()
                log.info("Text model loaded: v%s from %s", _cached.version, _cached.source)
    return _cached


def reload_model() -> LoadedModel:
    """Drop the cache and load again (use after promoting a new champion)."""
    global _cached
    with _lock:
        _cached = None
    return load_model()


def model_info() -> dict:
    m = load_model()
    return {
        "name": config.registered_model_name(),
        "version": m.version,
        "source": m.source,
        "categories": m.categories,
        "unknown_threshold": m.threshold(),
        "data_source": m.meta.get("data_source", "unknown"),
        "real_data_validated": bool(m.meta.get("real_data_validated", False)),  # False -> show a DEMO MODEL badge
        "evaluation_note": m.meta.get("evaluation_note", "No evaluation note recorded."),
        "trained_at": m.meta.get("trained_at"),
        "metrics": m.meta.get("metrics", {}),
    }
