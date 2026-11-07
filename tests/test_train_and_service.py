import json

import pytest

from text import config, registry
from text.data import generate_sample_dataset as gen
from text.service import ServiceError, classify_upload
from text.train import train


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    gen.generate(d / "train", gen.CATEGORIES[:3], per_class=20, seed=1)
    gen.generate(d / "ood", gen.OOD_CATEGORIES, per_class=8, seed=2)
    return d


def test_training_writes_model_and_meta(tiny_dataset, tmp_path):
    meta = train(tiny_dataset / "train", tmp_path / "out", cv=3, ood_dir=tiny_dataset / "ood", data_source="unit-test")
    assert (tmp_path / "out" / "text_svm.joblib").exists()
    saved = json.loads((tmp_path / "out" / "model_meta.json").read_text())
    assert saved["categories"] == sorted(gen.CATEGORIES[:3])
    assert saved["data_source"] == "unit-test" and saved["version"].startswith("local-")
    assert 0.0 <= saved["metrics"]["f1_macro"] <= 1.0
    assert saved["unknown_threshold_tuned"] is False          # tuning is opt-in
    assert saved["threshold_table_validation"] and saved["threshold_table_test"]
    assert meta["n_train"] + meta["n_val"] + meta["n_test"] == 60


def test_trained_model_is_loadable_and_used(tiny_dataset, tmp_path, monkeypatch):
    train(tiny_dataset / "train", tmp_path / "out2", cv=3, data_source="unit-test")
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path / "out2"))
    from text import classify_text, model_info
    assert model_info()["categories"] == sorted(gen.CATEGORIES[:3])
    import random
    doc = gen.make_document("invoice", random.Random(9))
    assert classify_text(doc.encode(), "i.txt")["category"] == "invoice"


def test_duplicates_are_dropped_before_splitting(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    same = "this is exactly the same document text repeated across files " * 3
    for i in range(3):
        (tmp_path / "a" / f"{i}.txt").write_text(same)
    (tmp_path / "b" / "0.txt").write_text("a different document about something else entirely " * 3)
    from text.train import load_folder
    texts, labels, stats = load_folder(tmp_path)
    assert len(texts) == 2 and stats["duplicates"] == 2


# ------------------------------------------------------------ promotion gate
def test_promotion_gate():
    assert registry.should_promote(0.5, None) is True      # first model becomes champion
    assert registry.should_promote(0.91, 0.90) is True
    assert registry.should_promote(0.90, 0.90) is True     # ties promote (newer data)
    assert registry.should_promote(0.89, 0.90) is False    # worse model is registered but not promoted


def test_registry_missing_mlflow_is_a_clear_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def no_mlflow(name, *a, **k):
        if name == "mlflow":
            raise ImportError("no mlflow")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mlflow)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    with pytest.raises(registry.RegistryUnavailable, match="not installed"):
        registry.load_champion()


def test_auto_mode_falls_back_to_local_when_registry_unavailable(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # set, but mlflow absent or server down
    from text import model_info
    assert model_info()["source"] == "local"


def test_registry_only_mode_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setenv("TEXT_MODEL_SOURCE", "registry")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from text import model_info
    with pytest.raises(registry.RegistryUnavailable):
        model_info()


# ---------------------------------------------------------------- service layer
def test_service_rejects_empty():
    with pytest.raises(ServiceError) as e:
        classify_upload("a.txt", b"")
    assert e.value.status == 400


def test_service_rejects_too_large(monkeypatch):
    monkeypatch.setenv("TEXT_MAX_BYTES", "100")
    with pytest.raises(ServiceError) as e:
        classify_upload("a.txt", b"x" * 101)
    assert e.value.status == 413


def test_service_unsupported_is_415():
    with pytest.raises(ServiceError) as e:
        classify_upload("a.exe", b"MZ\x00\x01")
    assert e.value.status == 415


def test_service_corrupt_is_422():
    with pytest.raises(ServiceError) as e:
        classify_upload("a.pdf", b"%PDF-1.4 broken")
    assert e.value.status == 422


def test_service_no_model_is_503(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT_MODEL_DIR", str(tmp_path))
    with pytest.raises(ServiceError) as e:
        classify_upload("a.txt", b"enough readable words to pass the minimum character check easily")
    assert e.value.status == 503


def test_service_success_shape(fresh_docs):
    out = classify_upload("inv.txt", fresh_docs["invoice"][0].encode())
    assert out["filename"] == "inv.txt" and out["category"] == "invoice" and out["source"] == "svm"
