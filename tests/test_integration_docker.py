"""Tests that start the REAL Tika container from docker-compose.tika.yml. Skipped unless a Docker daemon is running.

Run:  pytest text/tests/test_integration_docker.py -v -s
Works with Podman too if `docker` is aliased to podman and `podman compose` is installed.

NOT RUN in the environment this file was written in (docker CLI present, but no daemon). Whether the
compose file and the apache/tika image behave as described is therefore UNVERIFIED until you run this.
"""
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from text.extract import extract_text, tika_healthy

pytestmark = pytest.mark.docker
COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.tika.yml"
URL = "http://localhost:9998"


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0


@pytest.fixture(scope="module")
def tika_container():
    if not _docker_ok():
        pytest.skip("no running Docker daemon")
    base = ["docker", "compose", "-f", str(COMPOSE)]
    subprocess.run(base + ["up", "-d", "tika"], check=True, capture_output=True, timeout=600)
    try:
        deadline = time.time() + 120                    # first start pulls the image and boots a JVM
        while time.time() < deadline and not tika_healthy(URL):
            time.sleep(2)
        assert tika_healthy(URL), "Tika container did not become healthy within 120s"
        yield base
    finally:
        subprocess.run(base + ["down"], capture_output=True, timeout=120)


def test_container_starts_and_answers_health_check(tika_container):
    assert tika_healthy(URL) is True


def test_backend_extracts_through_the_container(tika_container, monkeypatch, make_pdf):
    monkeypatch.setenv("TIKA_URL", URL)
    monkeypatch.setenv("TEXT_EXTRACTOR", "tika")
    r = extract_text(make_pdf("Invoice total amount due via container"), "x.pdf")
    assert r.extractor == "tika" and "amount due" in r.text


def test_tika_downtime_is_handled_with_local_fallback(tika_container, monkeypatch, make_pdf):
    monkeypatch.setenv("TIKA_URL", URL)
    monkeypatch.setenv("TEXT_EXTRACTOR", "auto")
    subprocess.run(tika_container + ["stop", "tika"], check=True, capture_output=True, timeout=120)
    try:
        assert tika_healthy(URL) is False
        assert extract_text(make_pdf("Still works without Tika"), "x.pdf").extractor == "local-pdf"
        from text.service import health
        assert health()["status"] == "degraded"
    finally:
        subprocess.run(tika_container + ["start", "tika"], capture_output=True, timeout=120)
