"""registry.py logic against an in-memory FAKE MLflow (see fake_mlflow.py for what that does and does not prove).

Covers: first model becomes champion, promotion gate, force/never policies, MLflow 2.x/3.x log_model
fallback, what gets logged, loading the champion, rollback + backend reload, fallback to the local
model when the registry has nothing / is unreachable.
"""
import random
import types

import pytest

import text.model as model_mod
from text import classify_text, config, model_info, registry, reload_model
from text.data import generate_sample_dataset as gen
from text.model import build_pipeline

import fake_mlflow  # tests/ is on sys.path (rootdir conftest)

NAME, ALIAS = "text-classifier", "champion"


@pytest.fixture(scope="module")
def pipeline():
    rng = random.Random(21)
    cats = gen.CATEGORIES[:3]
    X = [gen.make_document(c, rng) for c in cats for _ in range(20)]
    y = [c for c in cats for _ in range(20)]
    return build_pipeline().fit(X, y)


def meta(f1):
    return {"metrics": {"f1_macro": f1, "accuracy": f1, "cv_f1_macro_mean": f1}, "categories": ["a"], "data_source": "unit-test"}


@pytest.fixture
def fake(monkeypatch):
    return fake_mlflow.install(monkeypatch)


def register(pipeline, f1, promote="auto"):
    return registry.log_and_register(pipeline, meta(f1), {"seed": 1}, promote=promote)


def champion(store):
    return store["aliases"].get((NAME, ALIAS))


# ---------------------------------------------------------------- registration and promotion
def test_first_model_becomes_champion(fake, pipeline):
    r = register(pipeline, 0.80)
    assert r == {"version": "1", "run_id": "run1", "promoted": True} and champion(fake) == 1


def test_better_model_is_promoted_worse_is_not(fake, pipeline):
    register(pipeline, 0.80)
    assert register(pipeline, 0.90)["promoted"] is True and champion(fake) == 2
    r = register(pipeline, 0.70)                              # registered as v3 but not promoted
    assert r["version"] == "3" and r["promoted"] is False and champion(fake) == 2
    assert register(pipeline, 0.90)["promoted"] is True       # tie -> promoted
    assert champion(fake) == 4


def test_promotion_policies(fake, pipeline):
    register(pipeline, 0.90)
    assert register(pipeline, 0.10, promote="force")["promoted"] is True and champion(fake) == 2
    assert register(pipeline, 0.99, promote="never")["promoted"] is False and champion(fake) == 2


def test_mlflow3_uses_name_and_exact_skops_trusted_types(fake, pipeline):
    """Regression: real MLflow refused to LOAD the model because skops was not told the two trusted types."""
    register(pipeline, 0.8)
    call = fake["log_model_calls"][0]
    assert call["name"] == "model"
    assert call["skops_trusted_types"] == [
        "sklearn.calibration._CalibratedClassifier",
        "sklearn.calibration._SigmoidCalibration",
    ]
    assert all(t.startswith("sklearn.") for t in registry.SKOPS_TRUSTED_TYPES)


def test_mlflow_uses_signature_guard_before_passing_skops_trusted_types(monkeypatch, pipeline):
    seen = {}

    def log_model(model, name=None, skops_trusted_types=None, **kwargs):
        seen["name"] = name
        seen["skops_trusted_types"] = skops_trusted_types
        return model

    mlflow = types.SimpleNamespace(sklearn=types.SimpleNamespace(log_model=log_model))
    assert registry._log_sklearn_model(mlflow, pipeline) is pipeline
    assert seen["name"] == "model"
    assert seen["skops_trusted_types"] == registry.SKOPS_TRUSTED_TYPES


def test_mlflow2_uses_artifact_path_and_exact_skops_trusted_types(monkeypatch, pipeline):
    seen = {}

    def log_model(model, artifact_path=None, skops_trusted_types=None, **kwargs):
        seen["artifact_path"] = artifact_path
        seen["skops_trusted_types"] = skops_trusted_types
        return types.SimpleNamespace(model_uri="runs:/run1/model")

    mlflow = types.SimpleNamespace(sklearn=types.SimpleNamespace(log_model=log_model))
    result = registry._log_sklearn_model(mlflow, pipeline)
    assert result.model_uri == "runs:/run1/model"
    assert seen["artifact_path"] == "model"
    assert seen["skops_trusted_types"] == registry.SKOPS_TRUSTED_TYPES


def test_mlflow_without_skops_argument_raises_clear_error(monkeypatch, pipeline):
    def log_model(model, artifact_path=None, **kwargs):
        raise AssertionError("this function should not be called when skops_trusted_types is unsupported")

    mlflow = types.SimpleNamespace(sklearn=types.SimpleNamespace(log_model=log_model))
    with pytest.raises(TypeError, match="skops_trusted_types"):
        registry._log_sklearn_model(mlflow, pipeline)


def test_registry_fallback_is_a_visible_warning_when_registry_is_configured(fake, pipeline, monkeypatch, caplog):
    import logging
    import sys
    register(pipeline, 0.8)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")
    monkeypatch.setattr(sys.modules["mlflow"].sklearn, "load_model", lambda uri: (_ for _ in ()).throw(fake_mlflow.FakeMlflowError("untrusted types")))
    with caplog.at_level(logging.INFO, logger="text.model"):
        assert model_info()["source"] == "local"
    rec = [r for r in caplog.records if "Registry model NOT used" in r.message]
    assert rec and rec[0].levelno == logging.WARNING and "untrusted types" in rec[0].message


def test_run_logs_params_metrics_meta_and_model(fake, pipeline):
    register(pipeline, 0.85)
    run = fake["runs"]["run1"]
    assert fake["experiment"] == config.experiment_name()
    assert run["params"] == {"seed": 1}
    assert run["metrics"] == {"f1_macro": 0.85, "accuracy": 0.85, "cv_f1_macro_mean": 0.85}
    assert run["artifacts"]["model_meta.json"]["data_source"] == "unit-test"
    assert run["model"] is pipeline


# ---------------------------------------------------------------- loading, reload, rollback
def test_load_champion_returns_registered_model_and_meta(fake, pipeline, monkeypatch):
    register(pipeline, 0.8)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")
    p, m = registry.load_champion()
    assert p is pipeline and m["version"] == "1" and m["name"] == NAME and m["data_source"] == "unit-test"


def test_load_champion_without_tracking_uri_is_unavailable(fake, pipeline):
    register(pipeline, 0.8)
    with pytest.raises(registry.RegistryUnavailable, match="MLFLOW_TRACKING_URI"):
        registry.load_champion()


def test_inference_uses_registry_champion_and_reload_follows_alias(fake, pipeline, monkeypatch):
    register(pipeline, 0.80)                                   # v1
    register(pipeline, 0.90)                                   # v2 -> champion
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")
    assert model_info()["source"] == "registry" and model_info()["version"] == "2"
    doc = gen.make_document("invoice", random.Random(2)).encode()
    assert classify_text(doc, "i.txt")["category"] == "invoice"          # prediction served by the registry model

    fake["aliases"][(NAME, ALIAS)] = 1                          # ROLLBACK: move the alias back to v1
    assert model_info()["version"] == "2"                       # cached: unchanged until reload (documented behaviour)
    reload_model()
    assert model_info()["version"] == "1"                       # backend now serves v1

    fake["aliases"][(NAME, ALIAS)] = 2                          # PROMOTE again
    assert reload_model().version == "2"


# ---------------------------------------------------------------- unavailable registry
def test_auto_falls_back_to_local_when_registry_has_no_champion(fake, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")       # reachable but nothing registered
    info = model_info()
    assert info["source"] == "local"


def test_auto_falls_back_to_local_when_registry_load_fails(fake, pipeline, monkeypatch):
    register(pipeline, 0.8)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")

    def boom(uri):
        raise fake_mlflow.FakeMlflowError("connection refused")
    import sys
    monkeypatch.setattr(sys.modules["mlflow"].sklearn, "load_model", boom)
    assert model_info()["source"] == "local"


def test_registry_only_mode_fails_loudly(fake, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")
    monkeypatch.setenv("TEXT_MODEL_SOURCE", "registry")
    with pytest.raises(registry.RegistryUnavailable, match="champion"):
        model_info()
