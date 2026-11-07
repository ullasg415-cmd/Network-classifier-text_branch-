"""HTTP-level tests through a real FastAPI app. Skipped automatically where FastAPI / httpx are missing.

NOT RUN in the environment this file was written in (FastAPI could not be installed there).
Run on your machine:  pip install fastapi httpx python-multipart && pytest text/tests/test_api.py -v
The logic behind every route is covered by the local tests (test_robustness.py, test_train_and_service.py);
this file checks the HTTP layer: routing, multipart upload, status codes, JSON shape.
"""
import io
import random

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from text.api import router  # noqa: E402
from text.data import generate_sample_dataset as gen  # noqa: E402

pytestmark = pytest.mark.fastapi
READABLE = b"enough readable words to pass the minimum character check easily"


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def post(client, name, data, ctype="application/octet-stream"):
    return client.post("/classify/text", files={"file": (name, data, ctype)})


def test_successful_upload_and_response_shape(client, fresh_docs):
    r = post(client, "inv.txt", fresh_docs["invoice"][0].encode(), "text/plain")
    assert r.status_code == 200
    b = r.json()
    assert b["category"] == "invoice" and b["source"] == "svm" and 0 <= b["confidence"] <= 1
    for k in ("filename", "best_guess", "top_k", "reason", "extractor", "file_type", "text_chars", "model_version"):
        assert k in b


def test_pdf_and_docx_upload(client, make_pdf, make_docx):
    doc = gen.make_document("resume", random.Random(6))
    assert post(client, "cv.docx", make_docx(doc)).json()["category"] == "resume"
    assert post(client, "cv.pdf", make_pdf(doc[:1500])).status_code == 200


def test_unsupported_type_415(client):
    assert post(client, "a.exe", b"MZ\x00\x01").status_code == 415


def test_empty_file_400(client):
    assert post(client, "a.txt", b"").status_code == 400


def test_oversized_file_413(client, monkeypatch):
    monkeypatch.setenv("TEXT_MAX_BYTES", "100")
    assert post(client, "a.txt", b"x" * 500).status_code == 413


def test_corrupted_document_422(client):
    assert post(client, "bad.pdf", b"%PDF-1.4 broken").status_code == 422


def test_low_confidence_document(client, monkeypatch):
    monkeypatch.setenv("TEXT_UNKNOWN_THRESHOLD", "0.99")
    b = post(client, "a.txt", gen.make_document("invoice", random.Random(1)).encode()).json()
    assert b["category"] == "unknown" and b["reason"] == "low_confidence" and b["best_guess"] == "invoice"


def test_unknown_document_too_little_text(client):
    b = post(client, "a.txt", b"too short").json()
    assert b["category"] == "unknown" and b["reason"] == "too_little_text"


def test_missing_file_field_422(client):
    assert client.post("/classify/text").status_code == 422


def test_model_unavailable_503(client, monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path))
    assert post(client, "a.txt", READABLE).status_code == 503
    assert client.get("/classify/text/model").status_code == 503
    assert client.get("/classify/text/health").status_code == 503


def test_registry_unavailable_503(client, monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_SOURCE", "registry")       # registry required but not configured
    assert post(client, "a.txt", READABLE).status_code == 503
    assert client.post("/classify/text/model/reload").status_code == 503


def test_model_info_reload_and_health(client):
    info = client.get("/classify/text/model").json()
    assert info["source"] == "local" and "real_data_validated" in info and "evaluation_note" in info
    assert client.post("/classify/text/model/reload").status_code == 200
    h = client.get("/classify/text/health")
    assert h.status_code == 200 and h.json()["status"] in ("ok", "degraded")
