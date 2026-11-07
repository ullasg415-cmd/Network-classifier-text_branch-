"""Tests against REAL MLflow (the installed package, with a temporary SQLite tracking/registry store;
no server needed). Skipped automatically if `mlflow` is not installed.

Run:  pip install mlflow && pytest text/tests/test_integration_mlflow.py -v

NOT RUN in the environment this file was written in (mlflow could not be installed there). Until
it has been run, text/registry.py is unverified against the real MLflow API. For a test against
a real tracking SERVER, start `mlflow server` and set MLFLOW_TRACKING_URI to its address before running.
"""
import random

import pytest

mlflow = pytest.importorskip("mlflow")
pytestmark = pytest.mark.mlflow

from text import classify_text, config, model_info, registry, reload_model  # noqa: E402
from text.data import generate_sample_dataset as gen  # noqa: E402
from text.train import train  # noqa: E402

NAME = "text-classifier"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlflowdata")
    gen.generate(d / "real", gen.CATEGORIES[:3], per_class=25, seed=31)
    return d / "real"


@pytest.fixture
def store(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("TEXT_MLFLOW_EXPERIMENT", f"text-test-{tmp_path.name}")
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(f"text-test-{tmp_path.name}", artifact_location=str(tmp_path / "artifacts"))
    return mlflow.MlflowClient()


def run_train(dataset, tmp_path, policy="auto", seed=1):
    return train(dataset, tmp_path / f"out{seed}", cv=3, data_source="synthetic-test", register=policy, seed=seed)


def test_train_registers_and_first_version_is_champion(store, dataset, tmp_path):
    meta = run_train(dataset, tmp_path)
    assert meta["registry"]["version"] == "1" and meta["registry"]["promoted"] is True
    assert str(store.get_model_version_by_alias(NAME, "champion").version) == "1"
    run = store.get_run(meta["registry"]["run_id"])
    assert run.data.metrics["f1_macro"] == meta["metrics"]["f1_macro"] and run.data.params["data_source"] == "synthetic-test"


def test_second_version_and_version_comparison(store, dataset, tmp_path):
    run_train(dataset, tmp_path, seed=1)
    run_train(dataset, tmp_path, policy="never", seed=2)
    versions = store.search_model_versions(f"name='{NAME}'")
    assert sorted(int(v.version) for v in versions) == [1, 2]
    f1 = {str(v.version): store.get_run(v.run_id).data.metrics["f1_macro"] for v in versions}
    assert set(f1) == {"1", "2"}                                    # metrics available for side-by-side comparison
    assert str(store.get_model_version_by_alias(NAME, "champion").version) == "1"   # 'never' did not promote


def test_inference_loads_champion_then_rollback_and_reload(store, dataset, tmp_path):
    run_train(dataset, tmp_path, seed=1)                            # v1 champion
    run_train(dataset, tmp_path, policy="force", seed=2)            # v2 champion
    assert model_info()["source"] == "registry" and model_info()["version"] == "2"
    doc = gen.make_document("invoice", random.Random(4)).encode()
    assert classify_text(doc, "i.txt")["category"] == "invoice"

    store.set_registered_model_alias(NAME, "champion", "1")         # rollback
    assert reload_model().version == "1"
    store.set_registered_model_alias(NAME, "champion", "2")         # promote again
    assert reload_model().version == "2"


def test_falls_back_to_local_model_when_registry_is_unreachable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")      # do not wait through MLflow's retry/backoff
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    assert model_info()["source"] == "local"


def test_registry_only_mode_errors_when_unreachable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    monkeypatch.setenv("TEXT_MODEL_SOURCE", "registry")
    with pytest.raises(registry.RegistryUnavailable):
        model_info()
