"""How the Tika client behaves when the server misbehaves (all against local fake servers, not real Tika).

Covers: timeout, garbled HTTP reply, connection dropped mid-reply, binary/garbage 200 body,
an invalid TIKA_URL, and malformed documents. In 'auto' mode every transport failure must fall
back to the local extractors; in 'tika' mode it must be a clear TextExtractionError, never a crash.
"""
import socket
import threading
import time

import pytest

from text.extract import TextExtractionError, extract_text, tika_healthy


@pytest.fixture
def raw_server():
    """A TCP server that sends exactly `payload` bytes (or nothing) and closes. Returns (url, set_payload)."""
    state = {"payload": b"", "delay": 0.0}
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    stop = threading.Event()

    def serve():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            try:
                conn.settimeout(2)
                try:
                    conn.recv(65536)
                except OSError:
                    pass
                time.sleep(state["delay"])
                if state["payload"]:
                    conn.sendall(state["payload"])
            except OSError:
                pass
            finally:
                conn.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.getsockname()[1]}", state
    stop.set()
    srv.close()


def _use(monkeypatch, url, mode="auto", timeout="2"):
    monkeypatch.setenv("TIKA_URL", url)
    monkeypatch.setenv("TEXT_EXTRACTOR", mode)
    monkeypatch.setenv("TIKA_TIMEOUT", timeout)


# ------------------------------------------------------------------ timeout
def test_timeout_falls_back_in_auto_mode(monkeypatch, raw_server, make_pdf):
    url, st = raw_server
    st["delay"] = 3.0                                    # server accepts, then stays silent past the timeout
    _use(monkeypatch, url, timeout="0.3")
    t0 = time.perf_counter()
    r = extract_text(make_pdf("Timeout fallback works"), "x.pdf")
    assert r.extractor == "local-pdf" and time.perf_counter() - t0 < 2.5   # gave up at ~0.3 s, not 3 s


def test_timeout_is_a_clear_error_in_tika_mode(monkeypatch, raw_server, make_pdf):
    url, st = raw_server
    st["delay"] = 3.0
    _use(monkeypatch, url, mode="tika", timeout="0.3")
    with pytest.raises(TextExtractionError, match="unreachable"):
        extract_text(make_pdf("x"), "x.pdf")


# ------------------------------------------------------------------ garbled / dropped replies
@pytest.mark.parametrize("payload", [
    b"this is not http at all\r\n\r\n",                          # garbage status line
    b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\nshort",      # connection closes before the full body
    b"",                                                         # connection closed with no reply
])
def test_garbled_reply_falls_back_not_crashes(monkeypatch, raw_server, make_pdf, payload):
    url, st = raw_server
    st["payload"] = payload
    _use(monkeypatch, url)
    assert extract_text(make_pdf("Garbled reply fallback"), "x.pdf").extractor == "local-pdf"


def test_garbled_reply_in_tika_mode_is_text_extraction_error(monkeypatch, raw_server, make_pdf):
    url, st = raw_server
    st["payload"] = b"nonsense\r\n\r\n"
    _use(monkeypatch, url, mode="tika")
    with pytest.raises(TextExtractionError):
        extract_text(make_pdf("x"), "x.pdf")


def test_health_check_survives_garbled_reply(raw_server):
    url, st = raw_server
    st["payload"] = b"nonsense\r\n\r\n"
    assert tika_healthy(url) is False


# ------------------------------------------------------------------ odd but valid replies
def test_binary_garbage_in_a_200_body_does_not_crash(monkeypatch, tika_stub, make_docx):
    tika_stub["body"] = bytes(range(256)) * 10
    _use(monkeypatch, tika_stub["url"])
    r = extract_text(make_docx("hello"), "x.docx")
    assert r.extractor == "tika" and isinstance(r.text, str)


# ------------------------------------------------------------------ configuration mistakes
@pytest.mark.parametrize("bad", ["tika:9998", "localhost:9998", "htp://x", "not a url", "://"])
def test_invalid_tika_url_falls_back_in_auto_mode(monkeypatch, make_pdf, bad):
    _use(monkeypatch, bad)
    assert extract_text(make_pdf("Bad url fallback"), "x.pdf").extractor == "local-pdf"
    assert tika_healthy(bad) is False


def test_invalid_tika_url_in_tika_mode_is_a_clear_error(monkeypatch, make_pdf):
    _use(monkeypatch, "tika:9998", mode="tika")
    with pytest.raises(TextExtractionError, match="unreachable"):
        extract_text(make_pdf("x"), "x.pdf")


# ------------------------------------------------------------------ malformed documents never reach a bad state
@pytest.mark.parametrize("name,data", [
    ("empty.pdf", b"%PDF-"),
    ("trunc.pdf", b"%PDF-1.7\n1 0 obj\n<<"),
    ("bad.docx", b"PK\x03\x04" + b"\x00" * 30),
])
def test_malformed_documents_raise_text_extraction_error(name, data):
    with pytest.raises(TextExtractionError):
        extract_text(data, name)
