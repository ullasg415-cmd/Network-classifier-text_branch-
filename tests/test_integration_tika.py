"""Tests against a REAL Apache Tika server. Skipped automatically if none is reachable.

Start one:   docker run -d --name tika -p 9998:9998 apache/tika:latest     (or: docker compose -f text/docker-compose.tika.yml up -d tika)
Run:         pytest text/tests/test_integration_tika.py -v -s        (TIKA_TEST_URL overrides http://localhost:9998)
Stop:        docker rm -f tika

NOT RUN in the environment this file was written in (no Tika, no Docker daemon). Expectations about
Tika's responses to bad files (HTTP 422) come from Tika's documentation; if one of them is wrong on
your Tika version, that is a real finding to fix in text/extract.py.
"""
import os
import random
import time

import pytest

from text import classify_text_detailed
from text.data import generate_sample_dataset as gen
from text.extract import TextExtractionError, extract_text, tika_healthy

pytestmark = pytest.mark.tika
URL = os.getenv("TIKA_TEST_URL", "http://localhost:9998")


@pytest.fixture(autouse=True)
def tika(monkeypatch):
    if not tika_healthy(URL, timeout=3):
        pytest.skip(f"no Tika server at {URL}")
    monkeypatch.setenv("TIKA_URL", URL)
    monkeypatch.setenv("TEXT_EXTRACTOR", "tika")


def test_pdf_via_tika(make_pdf):
    r = extract_text(make_pdf("Invoice total amount due payment terms"), "x.pdf")
    assert r.extractor == "tika" and "amount due" in r.text


def test_docx_via_tika(make_docx):
    r = extract_text(make_docx("Service agreement indemnity clause"), "x.docx")
    assert r.extractor == "tika" and "indemnity" in r.text


def test_plain_text_stays_local_by_design():
    assert extract_text(b"plain text document with enough words", "x.txt").extractor == "local-text"


def test_rtf_needs_tika_and_works():
    r = extract_text(rb"{\rtf1\ansi Quarterly invoice total amount due\par}", "x.rtf")
    assert r.extractor == "tika" and "invoice" in r.text


def test_corrupt_pdf_is_an_error_or_empty():
    try:
        r = extract_text(b"%PDF-1.4 this is not really a pdf", "bad.pdf")
        assert r.text == ""
    except TextExtractionError:
        pass


def test_encrypted_pdf_is_rejected(make_pdf):
    import io
    from pypdf import PdfReader, PdfWriter
    w = PdfWriter()
    for p in PdfReader(io.BytesIO(make_pdf("secret"))).pages:
        w.add_page(p)
    w.encrypt("pw")
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(TextExtractionError):
        extract_text(buf.getvalue(), "locked.pdf")


def test_end_to_end_classification_through_tika(make_docx):
    doc = gen.make_document("legal_contract", random.Random(8))
    d = classify_text_detailed(make_docx(doc), "contract.docx")
    assert d["extractor"] == "tika" and d["category"] == "legal_contract"


def test_large_document_extraction_time(make_pdf, make_docx, capsys):
    pdf = make_pdf("\n".join(f"Line {i}: technical manual installation step torque bolts" for i in range(4000)))
    docx = make_docx("\n".join(f"Paragraph {i} contract indemnity arbitration clause" for i in range(20_000)))
    for name, data in (("pdf", pdf), ("docx", docx)):
        t0 = time.perf_counter()
        r = extract_text(data, f"big.{name}")
        dt = time.perf_counter() - t0
        print(f"\nTika {name}: {len(data) / 1024:.0f} KB -> {len(r.text)} chars in {dt:.2f}s")
        assert r.extractor == "tika" and dt < 60


def test_falls_back_to_local_when_tika_goes_away(monkeypatch, make_pdf):
    monkeypatch.setenv("TEXT_EXTRACTOR", "auto")
    monkeypatch.setenv("TIKA_URL", "http://127.0.0.1:9")
    assert extract_text(make_pdf("Fallback works"), "x.pdf").extractor == "local-pdf"
