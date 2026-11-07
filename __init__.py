"""Text branch of the pipeline (Person 2).

Public contract, agreed with the network branch and the app:

    classify_text(file_bytes: bytes, filename: str) -> {"category": str, "confidence": float}

`category` is one of the trained categories, or "unknown" when the document has
too little readable text or the model is not confident enough. Errors raise
TextExtractionError / UnsupportedFileType (the API layer turns them into 422/415).
"""
from __future__ import annotations

from . import config
from .extract import Extraction, TextExtractionError, UnsupportedFileType, extract_text
from .model import load_model, model_info, reload_model

__all__ = [
    "classify_text", "classify_text_detailed", "model_info", "reload_model",
    "TextExtractionError", "UnsupportedFileType",
]

UNKNOWN = "unknown"


def classify_text_detailed(file_bytes: bytes, filename: str | None = None) -> dict:
    """Like classify_text, plus the details the UI and logs want."""
    ex: Extraction = extract_text(file_bytes, filename)
    model = load_model()
    base = {"source": "svm", "extractor": ex.extractor, "file_type": ex.file_type,
            "text_chars": len(ex.text), "model_version": model.version}

    if len(ex.text) < config.min_chars():
        # scanned PDF with no text layer, empty file, ... nothing to classify
        return {"category": UNKNOWN, "confidence": 0.0, "best_guess": None, "top_k": [],
                "reason": "too_little_text", **base}

    pred = model.predict(ex.text)
    confident = pred["confidence"] >= model.threshold()
    return {
        "category": pred["best_guess"] if confident else UNKNOWN,
        "confidence": round(pred["confidence"], 4),
        "best_guess": pred["best_guess"],
        "top_k": pred["top_k"],
        "reason": "ok" if confident else "low_confidence",
        **base,
    }


def classify_text(file_bytes: bytes, filename: str) -> dict:
    """The agreed contract: exactly {"category", "confidence"}."""
    d = classify_text_detailed(file_bytes, filename)
    return {"category": d["category"], "confidence": d["confidence"]}
