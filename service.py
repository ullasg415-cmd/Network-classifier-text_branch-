"""Framework-free request handling for POST /classify/text.

Keeping this out of api.py means the validation and error mapping can be
unit-tested without FastAPI, and the app (Person 3) can call it from anywhere.
"""
from __future__ import annotations

from . import config, classify_text_detailed
from .extract import TextExtractionError, UnsupportedFileType


class ServiceError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def health() -> dict:
    """Readiness report for a container health check / the UI.

    status: "ok"          everything configured works
            "degraded"    works, but Tika is configured and down (local fallback is being used)
            "unavailable" cannot classify (no model, or TEXT_EXTRACTOR=tika with Tika down)
    """
    from . import model_info
    from .extract import tika_healthy

    mode, url = config.extractor_mode(), config.tika_url()
    reachable = tika_healthy(url)
    report: dict = {"extractor_mode": mode, "tika": {"configured": bool(url), "url": url or None, "reachable": reachable}}
    status = "ok"
    if reachable is False:
        if mode == "tika":
            status = "unavailable"
        else:
            status = "degraded"
            report["tika"]["note"] = "Tika unreachable: PDF/DOCX/TXT use local extractors; .doc/.rtf/.odt need Tika"
    try:
        info = model_info()
        report["model"] = {"loaded": True, "version": info["version"], "source": info["source"],
                           "real_data_validated": info["real_data_validated"]}
    except (FileNotFoundError, RuntimeError) as e:
        report["model"] = {"loaded": False, "error": str(e)}
        status = "unavailable"
    report["status"] = status
    return report


def classify_upload(filename: str | None, data: bytes) -> dict:
    """Validate an upload and classify it. Raises ServiceError(status, detail)."""
    if not data:
        raise ServiceError(400, "Empty file")
    if len(data) > config.max_upload_bytes():
        raise ServiceError(413, f"File larger than {config.max_upload_bytes() // (1024 * 1024)} MB")
    try:
        result = classify_text_detailed(data, filename)
    except UnsupportedFileType as e:
        raise ServiceError(415, str(e)) from e
    except TextExtractionError as e:
        raise ServiceError(422, str(e)) from e
    except (FileNotFoundError, RuntimeError) as e:
        # no trained model on disk / registry unavailable (RegistryUnavailable is a RuntimeError)
        # / scikit-learn version mismatch with TEXT_STRICT_VERSIONS=1
        raise ServiceError(503, f"Text model unavailable: {e}") from e
    return {"filename": filename or "", **result}
