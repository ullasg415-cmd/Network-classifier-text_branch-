"""(1) Readable HTTP payloads from the network branch (Person 1); encrypted traffic is never decoded.
(2) The same document as TXT / DOCX / PDF gives the same text and the same classification.
"""
import gzip
import random

import pytest

from text import classify_text, classify_text_detailed
from text.data import generate_sample_dataset as gen
from text.extract import TextExtractionError, extract_text
from text.service import ServiceError, classify_upload

INVOICE = gen.make_document("invoice", random.Random(77))
CONTRACT = gen.make_document("legal_contract", random.Random(78))


# ------------------------------------------------------------------ readable HTTP payloads
def test_http_response_with_text_body():
    raw = ("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nServer: nginx\r\n\r\n" + INVOICE).encode()
    out = classify_text(raw, "payload")
    assert out["category"] == "invoice" and out["confidence"] > 0.5


def test_http_post_request_with_text_body():
    raw = ("POST /upload HTTP/1.1\r\nHost: files.example.com\r\nContent-Type: text/plain\r\n\r\n" + CONTRACT).encode()
    assert classify_text(raw, "payload")["category"] == "legal_contract"


def test_body_only_payload_without_extension():
    d = classify_text_detailed(INVOICE.encode(), "payload")
    assert d["category"] == "invoice" and d["extractor"] == "local-text" and d["file_type"] == "text"


def test_http_headers_only_is_unknown_not_a_guess():
    raw = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: curl/8\r\nAccept: */*\r\n\r\n"
    assert classify_text_detailed(raw, "payload")["category"] == "unknown"


# ------------------------------------------------------------------ encrypted / opaque traffic is refused, never decoded
def _tls_record(kind: int, size: int, seed: int) -> bytes:
    """A TLS record header (0x16 handshake / 0x17 application data, TLS 1.2) + pseudo-random ciphertext."""
    return bytes([kind, 0x03, 0x03, size >> 8, size & 0xFF]) + random.Random(seed).randbytes(size)


@pytest.mark.parametrize("kind,size", [(0x17, 600), (0x17, 4000), (0x16, 300)])
def test_tls_records_are_rejected_not_decoded(kind, size):
    with pytest.raises(TextExtractionError, match="binary"):
        classify_text(_tls_record(kind, size, seed=size), "payload")


def test_tls_record_through_the_service_is_422():
    with pytest.raises(ServiceError) as e:
        classify_upload("payload", _tls_record(0x17, 800, 5))
    assert e.value.status == 422


def test_gzip_compressed_body_is_refused_with_a_clear_reason():
    # compressed, not encrypted: the network branch should decompress first. We do not guess.
    with pytest.raises(TextExtractionError, match="binary"):
        classify_text(gzip.compress(INVOICE.encode()), "payload")


def test_unknown_extension_on_payload_name_is_unsupported():
    # documented behaviour: name the payload 'payload' or 'payload.txt', not 'payload.bin'
    from text.extract import UnsupportedFileType
    with pytest.raises(UnsupportedFileType):
        classify_text(INVOICE.encode(), "flow_12.bin")


# ------------------------------------------------------------------ same document, three formats
SHORT_DOC = "\n".join([
    "INVOICE No. INV-4821",
    "Bill To: Apex Systems, Pune",
    "Description Qty Unit price Amount",
    "steel brackets 10 x Rs 450 = Rs 4500",
    "network switches 3 x Rs 8200 = Rs 24600",
    "Subtotal: Rs 29100",
    "GST (18%): Rs 5238",
    "Total amount due: Rs 34338",
    "Payment terms: net 30 days from the invoice date.",
    "Please remit the balance by bank transfer quoting the invoice number.",
])


def test_extracted_text_is_identical_across_formats(make_docx, make_pdf):
    txt = extract_text(SHORT_DOC.encode(), "a.txt").text
    docx = extract_text(make_docx(SHORT_DOC), "a.docx").text
    pdf = extract_text(make_pdf(SHORT_DOC), "a.pdf").text
    assert txt == docx == pdf


def test_same_document_same_category_and_confidence_in_every_format(make_docx, make_pdf):
    results = [classify_text(data, name) for data, name in
               ((SHORT_DOC.encode(), "a.txt"), (make_docx(SHORT_DOC), "a.docx"), (make_pdf(SHORT_DOC), "a.pdf"))]
    assert {r["category"] for r in results} == {"invoice"}
    assert len({r["confidence"] for r in results}) == 1          # identical text -> identical confidence


def test_classification_is_deterministic_not_hardcoded():
    a = classify_text(INVOICE.encode(), "a.txt")
    assert a == classify_text(INVOICE.encode(), "a.txt")                                   # repeatable
    b = classify_text(CONTRACT.encode(), "b.txt")
    assert a["category"] != b["category"] and a["confidence"] != b["confidence"]           # depends on the content
