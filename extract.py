"""Stage 1 of the text branch: bytes of a document -> plain text.

Apache Tika is the primary extractor (Tika server over HTTP, no Python Tika
package needed). If Tika is not configured or not reachable, local libraries
cover the three formats on the slide: PDF (pypdf), DOCX (python-docx), TXT.

The same function is used for training and for inference, so the model never
sees text that was extracted differently from what it was trained on.
"""
from __future__ import annotations

import http.client
import io
import logging
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import PurePath

from . import config

log = logging.getLogger(__name__)

MAX_CHARS = 200_000  # more than enough for TF-IDF; keeps latency bounded

# Extensions we accept. Tika can read all of them; local fallback reads pdf/docx/text.
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".rtf", ".odt",
    ".txt", ".text", ".md", ".log", ".csv", ".html", ".htm",
}
_TEXT_EXTENSIONS = {".txt", ".text", ".md", ".log", ".csv"}


class TextExtractionError(Exception):
    """The file is a supported kind but could not be read (corrupt, encrypted...)."""


class UnsupportedFileType(TextExtractionError):
    """The file kind is not supported (or needs Tika, which is not available)."""


class _TikaUnavailable(Exception):
    pass


# Everything that means "could not talk to Tika" (as opposed to Tika answering with an error status).
# HTTPError is a URLError subclass but is handled separately, before this tuple, in _extract_tika.
_TRANSPORT_ERRORS = (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError,
                     http.client.HTTPException, ValueError, OSError)


@dataclass
class Extraction:
    text: str
    extractor: str   # e.g. "tika", "local-pdf", "local-docx", "local-text"
    file_type: str   # pdf | docx | doc | rtf | odt | html | text


# --------------------------------------------------------------------------
# type detection: trust the bytes first, the file name second
# --------------------------------------------------------------------------
def _looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    control = sum(1 for b in sample if b < 32 and b not in (9, 10, 12, 13))
    return control / len(sample) < 0.05


def sniff_type(data: bytes, filename: str | None) -> str:
    """Return pdf|docx|doc|rtf|odt|html|text, or raise UnsupportedFileType."""
    head = data[:1024]
    if b"%PDF-" in head:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except zipfile.BadZipFile:
            raise TextExtractionError("File looks like a zip/Office file but is corrupt")
        if "word/document.xml" in names:
            return "docx"
        if "content.xml" in names:
            return "odt"
        raise UnsupportedFileType("Zip/Office file is not a Word (.docx) or OpenDocument text file")
    if data[:8] == bytes.fromhex("D0CF11E0A1B11AE1"):
        return "doc"  # legacy OLE2 container (Word .doc; also xls/ppt - Tika will decide)
    if data[:5] == b"{\\rtf":
        return "rtf"

    ext = PurePath(filename or "").suffix.lower()
    if ext in (".html", ".htm"):
        return "html"
    if ext in _TEXT_EXTENSIONS or ext == "":
        if not _looks_like_text(data):
            raise TextExtractionError("Content is binary, not readable text (encrypted or compressed payload?)")
        return "text"
    if ext in ALLOWED_EXTENSIONS:
        # e.g. ".pdf" / ".docx" whose bytes are not that format
        raise TextExtractionError(f"File named '{ext}' does not contain a valid {ext[1:]} document")
    raise UnsupportedFileType(f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")


# --------------------------------------------------------------------------
# local extractors
# --------------------------------------------------------------------------
def _local_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("latin-1")


def _local_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise TextExtractionError("PDF is password-protected")
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except TextExtractionError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError, AssertionError, RecursionError) as e:
        raise TextExtractionError(f"Could not read PDF: {e}") from e


def _local_docx(data: bytes) -> str:
    from docx import Document

    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:  # python-docx raises many different types on bad input
        raise TextExtractionError(f"Could not read DOCX: {e}") from e
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:
        for row in table.rows:
            parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def _extract_local(data: bytes, file_type: str) -> Extraction:
    if file_type == "pdf":
        return Extraction(_local_pdf(data), "local-pdf", file_type)
    if file_type == "docx":
        return Extraction(_local_docx(data), "local-docx", file_type)
    if file_type == "text":
        return Extraction(_local_text(data), "local-text", file_type)
    raise UnsupportedFileType(
        f"'{file_type}' files need an Apache Tika server. Set TIKA_URL (e.g. http://tika:9998) "
        "or convert the file to PDF/DOCX/TXT."
    )


# --------------------------------------------------------------------------
# Apache Tika (REST server)
# --------------------------------------------------------------------------
def _extract_tika(data: bytes, filename: str | None, url: str, timeout: float) -> str:
    headers = {"Accept": "text/plain", "Content-Type": "application/octet-stream"}
    if filename:
        safe = urllib.parse.quote(PurePath(filename).name)
        headers["Content-Disposition"] = f"attachment; filename={safe}"
    try:
        req = urllib.request.Request(url.rstrip("/") + "/tika", data=data, method="PUT", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        if e.code == 204:
            return ""
        if e.code == 415:
            raise UnsupportedFileType("Tika does not support this file type") from e
        if e.code == 422:
            raise TextExtractionError("Tika could not parse the file (corrupt or encrypted)") from e
        raise TextExtractionError(f"Tika server error: HTTP {e.code}") from e
    except _TRANSPORT_ERRORS as e:
        # includes: refused / reset / timeout, a garbled or truncated HTTP reply (http.client.HTTPException),
        # and an unusable TIKA_URL (ValueError). All mean "Tika is not usable right now".
        raise _TikaUnavailable(str(e) or type(e).__name__) from e


def tika_healthy(url: str | None = None, timeout: float = 3.0) -> bool | None:
    """Is the Tika server answering? None = Tika is not configured (no TIKA_URL)."""
    url = config.tika_url() if url is None else url
    if not url:
        return None
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/tika", timeout=timeout) as resp:  # GET /tika -> greeting
            return resp.status == 200
    except _TRANSPORT_ERRORS:
        return False


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def extract_text(data: bytes, filename: str | None = None) -> Extraction:
    """Extract plain text from a document. Raises TextExtractionError subclasses."""
    file_type = sniff_type(data, filename)
    mode = config.extractor_mode()
    url = config.tika_url()

    # Plain text never needs Tika.
    use_tika = file_type != "text" and (mode == "tika" or (mode == "auto" and bool(url)))
    if use_tika:
        if not url:
            raise TextExtractionError("TEXT_EXTRACTOR=tika but TIKA_URL is not set")
        try:
            text = _extract_tika(data, filename, url, config.tika_timeout())
            return Extraction(clean_text(text), "tika", file_type)
        except _TikaUnavailable as e:
            if mode == "tika":
                raise TextExtractionError(f"Tika server unreachable at {url}: {e}") from e
            log.warning("Tika unreachable (%s); falling back to local extractors", e)

    result = _extract_local(data, file_type)
    result.text = clean_text(result.text)
    return result
