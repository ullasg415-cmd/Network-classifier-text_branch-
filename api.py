"""FastAPI router for the text branch. Person 3 mounts it with two lines:

    from text.api import router as text_router
    app.include_router(text_router)

Routes:
    POST /classify/text          multipart upload (field name: file) -> classification
    GET  /classify/text/model    which model version is live
    POST /classify/text/model/reload   reload after promoting a new champion in MLflow
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import config, model_info, reload_model
from .service import ServiceError, classify_upload
from .service import health as service_health

router = APIRouter(prefix="/classify/text", tags=["text"])


class CategoryScore(BaseModel):
    category: str
    confidence: float


class TextClassification(BaseModel):
    # `model_version` starts with pydantic's reserved "model_" prefix: older pydantic 2.x releases warn about it.
    model_config = {"protected_namespaces": ()}

    filename: str
    category: str                      # trained category, or "unknown"
    confidence: float                  # 0-1
    source: str = "svm"                # same field the network branch uses ("rule" / "ml")
    best_guess: Optional[str] = None   # the model's top category even when 'unknown'
    top_k: List[CategoryScore] = []
    reason: str                        # ok | low_confidence | too_little_text
    extractor: str                     # tika | local-pdf | local-docx | local-text
    file_type: str
    text_chars: int
    model_version: str


@router.post("", response_model=TextClassification)
def classify(file: UploadFile = File(...)):
    # plain `def` (not async): FastAPI runs it in a worker thread, so the CPU-bound
    # model call does not block the event loop.
    data = file.file.read(config.max_upload_bytes() + 1)
    try:
        return classify_upload(file.filename, data)
    except ServiceError as e:
        raise HTTPException(status_code=e.status, detail=e.detail)


@router.get("/health")
def health_check():
    """200 when ok/degraded, 503 when classification is impossible (use as a container health check)."""
    report = service_health()
    if report["status"] == "unavailable":
        raise HTTPException(status_code=503, detail=report)
    return report


@router.get("/model")
def current_model():
    try:
        return model_info()
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/model/reload")
def reload():
    try:
        reload_model()
        return model_info()
    except (FileNotFoundError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e))
