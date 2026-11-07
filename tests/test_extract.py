import pytest

from text import extract
from text.extract import TextExtractionError, UnsupportedFileType, extract_text

SAMPLE = "Quarterly report on network throughput.\nSecond line with more words."


def test_plain_text_utf8():
    r = extract_text(SAMPLE.encode(), "a.txt")
    assert r.text == "Quarterly report on network throughput. Second line with more words."
    assert r.extractor == "local-text" and r.file_type == "text"


def test_plain_text_latin1_fallback():
    r = extract_text("café résumé naïve".encode("latin-1"), "a.txt")
    assert "caf" in r.text


def test_no_extension_text_payload_is_accepted():
    # e.g. a readable HTTP body handed over by the network branch
    assert extract_text(b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\nhello", "payload").file_type == "text"


def test_binary_content_is_rejected():
    with pytest.raises(TextExtractionError, match="binary"):
        extract_text(bytes(range(256)) * 20, "payload.bin.txt")


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileType):
        extract_text(b"MZ\x90\x00\x03\x00\x00\x00", "program.exe")


def test_docx(make_docx):
    r = extract_text(make_docx(SAMPLE), "report.docx")
    assert "network throughput" in r.text and r.extractor == "local-docx"


def test_pdf(make_pdf):
    r = extract_text(make_pdf("Invoice total amount due"), "x.pdf")
    assert "Invoice total amount due" in r.text and r.extractor == "local-pdf"


def test_type_comes_from_bytes_not_filename(make_pdf):
    # a PDF uploaded with the wrong extension is still read as a PDF
    assert extract_text(make_pdf("Hidden pdf text here"), "mislabelled.txt").file_type == "pdf"


def test_corrupt_pdf():
    with pytest.raises(TextExtractionError):
        extract_text(b"%PDF-1.4 this is not really a pdf at all", "bad.pdf")


def test_fake_docx_extension():
    with pytest.raises(TextExtractionError):
        extract_text(b"just some text pretending", "fake.docx")


def test_legacy_doc_needs_tika():
    ole = bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 100
    with pytest.raises(UnsupportedFileType, match="Tika"):
        extract_text(ole, "old.doc")


def test_empty_pdf_text_layer_gives_empty_text(make_pdf):
    r = extract_text(make_pdf(""), "scan.pdf")
    assert r.text == ""  # classify_text turns this into 'unknown'


# ---------------------------------------------------------------- Tika paths
def test_tika_used_for_pdf_when_configured(monkeypatch, tika_stub, make_pdf):
    monkeypatch.setenv("TIKA_URL", tika_stub["url"])
    r = extract_text(make_pdf("ignored local text"), "x.pdf")
    assert r.extractor == "tika" and "Invoice number 42" in r.text
    seen = tika_stub["seen"][0]
    assert seen["path"] == "/tika" and seen["accept"] == "text/plain" and "x.pdf" in seen["disposition"]
    assert seen["bytes"].startswith(b"%PDF")


def test_tika_handles_legacy_doc(monkeypatch, tika_stub):
    monkeypatch.setenv("TIKA_URL", tika_stub["url"])
    ole = bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 100
    assert extract_text(ole, "old.doc").extractor == "tika"


def test_plain_text_skips_tika(monkeypatch, tika_stub):
    monkeypatch.setenv("TIKA_URL", tika_stub["url"])
    assert extract_text(SAMPLE.encode(), "a.txt").extractor == "local-text"
    assert tika_stub["seen"] == []


def test_auto_falls_back_to_local_when_tika_down(monkeypatch, make_pdf):
    monkeypatch.setenv("TIKA_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    r = extract_text(make_pdf("Fallback works fine"), "x.pdf")
    assert r.extractor == "local-pdf" and "Fallback works fine" in r.text


def test_tika_mode_errors_when_tika_down(monkeypatch, make_pdf):
    monkeypatch.setenv("TIKA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TEXT_EXTRACTOR", "tika")
    with pytest.raises(TextExtractionError, match="unreachable"):
        extract_text(make_pdf("x"), "x.pdf")


def test_tika_mode_without_url(monkeypatch, make_pdf):
    monkeypatch.setenv("TEXT_EXTRACTOR", "tika")
    with pytest.raises(TextExtractionError, match="TIKA_URL"):
        extract_text(make_pdf("x"), "x.pdf")


@pytest.mark.parametrize("status,exc", [(422, TextExtractionError), (415, UnsupportedFileType), (500, TextExtractionError)])
def test_tika_error_statuses(monkeypatch, tika_stub, make_docx, status, exc):
    monkeypatch.setenv("TIKA_URL", tika_stub["url"])
    tika_stub["status"] = status
    with pytest.raises(exc):
        extract_text(make_docx("hello world"), "x.docx")


def test_tika_204_means_empty(monkeypatch, tika_stub, make_docx):
    monkeypatch.setenv("TIKA_URL", tika_stub["url"])
    tika_stub["status"] = 204
    assert extract_text(make_docx("hello"), "x.docx").text == ""


def test_text_is_capped():
    big = ("word " * 100_000).encode()
    assert len(extract_text(big, "big.txt").text) == extract.MAX_CHARS
