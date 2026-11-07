"""Health check, encrypted / large files, version compatibility, service error paths. All local."""
import io
import json
import logging
import random
import time

import pytest

from text import classify_text_detailed, config, model_info
from text.data import generate_sample_dataset as gen
from text.extract import TextExtractionError, extract_text, tika_healthy
from text.service import ServiceError, classify_upload, health

DEAD = "http://127.0.0.1:9"   # nothing listens on port 9


# ------------------------------------------------------------------ encrypted / large
def test_encrypted_pdf_is_rejected_clearly(make_pdf):
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    for p in PdfReader(io.BytesIO(make_pdf("secret invoice text"))).pages:
        w.add_page(p)
    w.encrypt("pw")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(TextExtractionError, match="password"):
        extract_text(buf.getvalue(), "locked.pdf")
    with pytest.raises(ServiceError) as e:
        classify_upload("locked.pdf", buf.getvalue())
    assert e.value.status == 422


def test_large_text_file_is_fast_and_capped():
    data = ("the quarterly invoice payment terms net thirty days " * 100_000).encode()   # ~5 MB
    t0 = time.perf_counter()
    d = classify_text_detailed(data, "big.txt")
    assert time.perf_counter() - t0 < 10
    assert d["text_chars"] <= 200_000


def test_large_pdf_extraction_time(make_pdf):
    text = "\n".join(f"Line {i}: technical manual installation step torque bolts supply" for i in range(4000))  # ~70 pages
    pdf = make_pdf(text)
    t0 = time.perf_counter()
    r = extract_text(pdf, "big.pdf")
    assert time.perf_counter() - t0 < 30
    assert "technical manual" in r.text


def test_large_docx_extraction_time(make_docx):
    docx = make_docx("\n".join(f"Paragraph {i} about contract indemnity clause and arbitration" for i in range(20_000)))
    t0 = time.perf_counter()
    r = extract_text(docx, "big.docx")
    assert time.perf_counter() - t0 < 30
    assert "indemnity" in r.text


# ------------------------------------------------------------------ service error paths
def test_registry_unavailable_gives_503(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_SOURCE", "registry")        # registry required, none configured
    with pytest.raises(ServiceError) as e:
        classify_upload("a.txt", b"enough readable words to pass the minimum character check easily")
    assert e.value.status == 503 and "unavailable" in e.value.detail.lower()


def test_low_confidence_document_through_the_service(monkeypatch):
    monkeypatch.setenv("TEXT_UNKNOWN_THRESHOLD", "0.99")        # nothing is that confident
    out = classify_upload("a.txt", gen.make_document("invoice", random.Random(1)).encode())
    assert out["category"] == "unknown" and out["reason"] == "low_confidence" and out["best_guess"] == "invoice"


def test_unknown_document_through_the_service():
    out = classify_upload("tiny.txt", b"too short")           # 9 chars < TEXT_MIN_CHARS (30): e.g. a scanned PDF's empty text layer
    assert out["category"] == "unknown" and out["reason"] == "too_little_text" and out["confidence"] == 0.0


# ------------------------------------------------------------------ health check
def test_health_ok_without_tika():
    h = health()
    assert h["status"] == "ok" and h["tika"] == {"configured": False, "url": None, "reachable": None}
    assert h["model"]["loaded"] is True


def test_health_ok_with_tika(monkeypatch, tika_stub):
    monkeypatch.setenv("TIKA_URL", tika_stub["url"])
    h = health()
    assert h["status"] == "ok" and h["tika"]["reachable"] is True


def test_health_degraded_when_tika_down_but_local_fallback(monkeypatch):
    monkeypatch.setenv("TIKA_URL", DEAD)
    h = health()
    assert h["status"] == "degraded" and h["tika"]["reachable"] is False and "local extractors" in h["tika"]["note"]


def test_health_unavailable_when_tika_required_and_down(monkeypatch):
    monkeypatch.setenv("TIKA_URL", DEAD)
    monkeypatch.setenv("TEXT_EXTRACTOR", "tika")
    assert health()["status"] == "unavailable"


def test_health_unavailable_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path))
    h = health()
    assert h["status"] == "unavailable" and h["model"]["loaded"] is False


def test_tika_healthy_helper(monkeypatch, tika_stub):
    assert tika_healthy("") is None
    assert tika_healthy(tika_stub["url"]) is True
    assert tika_healthy(DEAD) is False
    tika_stub["health_status"] = 500
    assert tika_healthy(tika_stub["url"]) is False


# ------------------------------------------------------------------ version compatibility
def _copy_model(tmp_path, sklearn_version):
    import shutil
    shutil.copy(config.model_file(), tmp_path / "text_svm.joblib")
    meta = json.loads(config.meta_file().read_text())
    meta["sklearn_version"] = sklearn_version
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))


def test_version_mismatch_warns_by_default(monkeypatch, tmp_path, caplog):
    _copy_model(tmp_path, "0.0.1")
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="text.model"):
        model_info()
    assert any("0.0.1" in r.message and "retrain" in r.message for r in caplog.records)


def test_version_mismatch_refuses_in_strict_mode(monkeypatch, tmp_path):
    _copy_model(tmp_path, "0.0.1")
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("TEXT_STRICT_VERSIONS", "1")
    with pytest.raises(RuntimeError, match="scikit-learn"):
        model_info()
    with pytest.raises(ServiceError) as e:        # and the API reports 503 rather than predicting with a risky model
        classify_upload("a.txt", b"enough readable words to pass the minimum character check easily")
    assert e.value.status == 503


def test_shipped_model_matches_installed_sklearn_and_loads_without_warnings():
    import warnings
    import sklearn
    meta = json.loads(config.meta_file().read_text())
    assert meta["sklearn_version"] == sklearn.__version__, "shipped model was trained with a different scikit-learn: retrain"
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # any sklearn InconsistentVersionWarning would fail here
        model_info()


# ------------------------------------------------------------------ compose override (text check only; NOT a Docker test)
def test_backend_override_declares_the_documented_environment_variables():
    from pathlib import Path
    compose = (Path(config.PKG_DIR) / "docker-compose.text-backend.yml").read_text()
    for var in ("TIKA_URL: http://tika:9998", "TEXT_EXTRACTOR", "MLFLOW_TRACKING_URI", "TEXT_MODEL_SOURCE"):
        assert var in compose
