"""Static validation of the compose files with `docker compose config` (needs the docker CLI + compose plugin,
NOT a running daemon). This proves the files are well-formed and merge correctly. It does NOT prove the
containers start or that Tika works inside them; that is test_integration_docker.py.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
TIKA = PKG / "docker-compose.tika.yml"
BACKEND = PKG / "docker-compose.text-backend.yml"


def _compose_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _compose_available(), reason="docker CLI with the compose plugin not installed")


def config(*files, env=None):
    cmd = ["docker", "compose"]
    for f in files:
        cmd += ["-f", str(f)]
    r = subprocess.run(cmd + ["config", "--format", "json"], capture_output=True, text=True, env=env, timeout=60)
    return r.returncode, (json.loads(r.stdout) if r.returncode == 0 else r.stderr)


@pytest.fixture
def main_compose(tmp_path):
    """Stand-in for Person 3's main compose file."""
    f = tmp_path / "docker-compose.yml"
    f.write_text("services:\n  backend:\n    image: example/backend:dev\n  mlflow:\n    image: example/mlflow:dev\n")
    return f


def test_tika_file_is_valid_on_its_own():
    rc, cfg = config(TIKA)
    assert rc == 0, cfg                       # regression: it used to declare a `backend` service with no image
    assert set(cfg["services"]) == {"tika"}
    assert cfg["services"]["tika"]["image"] == "apache/tika:latest"
    assert any(p["published"] == "9998" and p["target"] == 9998 for p in cfg["services"]["tika"]["ports"])


def test_tika_image_can_be_pinned_without_editing_the_file(monkeypatch):
    import os
    rc, cfg = config(TIKA, env={**os.environ, "TIKA_IMAGE": "apache/tika:9.9.9-test"})
    assert rc == 0 and cfg["services"]["tika"]["image"] == "apache/tika:9.9.9-test"


def test_backend_override_alone_is_not_valid():
    rc, err = config(BACKEND)
    assert rc != 0                            # documented: it only works merged with a file that defines `backend`


def test_merged_with_main_compose_sets_the_documented_environment(main_compose):
    rc, cfg = config(main_compose, TIKA, BACKEND)
    assert rc == 0, cfg
    backend = cfg["services"]["backend"]
    env = backend["environment"]
    assert env["TIKA_URL"] == "http://tika:9998"
    assert env["TEXT_EXTRACTOR"] == "auto" and env["TEXT_MODEL_SOURCE"] == "auto"
    assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000"
    assert "tika" in backend["depends_on"]
    assert backend["image"] == "example/backend:dev"           # main file's settings survive the merge


def test_mlflow_uri_can_be_overridden_from_the_environment(main_compose):
    import os
    rc, cfg = config(main_compose, TIKA, BACKEND, env={**os.environ, "MLFLOW_TRACKING_URI": "http://elsewhere:5000"})
    assert rc == 0 and cfg["services"]["backend"]["environment"]["MLFLOW_TRACKING_URI"] == "http://elsewhere:5000"
