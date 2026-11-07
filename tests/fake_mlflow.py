"""A tiny in-memory stand-in for the parts of MLflow that text/registry.py uses.

WHAT THIS IS: a test double written from my understanding of the MLflow API. It lets us test
registry.py's own logic (when to promote, how the alias moves, what is logged, how reload and the
fallback behave) without MLflow installed.

WHAT THIS IS NOT: proof that registry.py works against real MLflow. Only tests marked `mlflow`
(test_integration_mlflow.py, which use the real package) can show that.
"""
from __future__ import annotations

import json
import os
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace


class FakeMlflowError(Exception):
    pass


def install(monkeypatch, legacy_log_model: bool = False, no_skops_arg: bool = False) -> dict:
    """Put a fake `mlflow` into sys.modules. legacy_log_model=True mimics MLflow 2.x (log_model uses
    `artifact_path=` instead of `name=`); no_skops_arg=True mimics an older MLflow without
    `skops_trusted_types`. Returns the shared store."""
    store = {"runs": {}, "versions": {}, "aliases": {}, "active": None, "n": 0, "experiment": None, "log_model_calls": []}

    mlflow = types.ModuleType("mlflow")
    sk = types.ModuleType("mlflow.sklearn")

    def set_experiment(name):
        store["experiment"] = name

    @contextmanager
    def start_run():
        store["n"] += 1
        rid = f"run{store['n']}"
        store["runs"][rid] = {"params": {}, "metrics": {}, "artifacts": {}, "model": None}
        store["active"] = rid
        try:
            yield SimpleNamespace(info=SimpleNamespace(run_id=rid))
        finally:
            store["active"] = None

    store["tracking_uri"] = "sqlite:///fake"
    mlflow.set_experiment = set_experiment
    mlflow.set_tracking_uri = lambda uri: store.__setitem__("tracking_uri", uri)
    mlflow.get_tracking_uri = lambda: store["tracking_uri"]
    mlflow.start_run = start_run
    mlflow.log_params = lambda p: store["runs"][store["active"]]["params"].update(p)
    mlflow.log_metrics = lambda m: store["runs"][store["active"]]["metrics"].update(m)
    mlflow.log_dict = lambda d, path: store["runs"][store["active"]]["artifacts"].__setitem__(path, d)

    if legacy_log_model and no_skops_arg:
        def log_model(model, artifact_path=None, **kw):
            raise TypeError("log_model() got an unexpected keyword argument 'skops_trusted_types'")
    elif legacy_log_model:
        def log_model(model, artifact_path=None, skops_trusted_types=None, **kw):
            store["log_model_calls"].append({"name": None, "artifact_path": artifact_path, "skops_trusted_types": skops_trusted_types, **kw})
            rid = store["active"]
            store["runs"][rid]["model"] = model
            return SimpleNamespace(model_uri=f"runs:/{rid}/model")
    elif no_skops_arg:
        def log_model(model, name=None, artifact_path=None, **kw):
            raise TypeError("log_model() got an unexpected keyword argument 'skops_trusted_types'")
    else:
        def log_model(model, name=None, artifact_path=None, skops_trusted_types=None, **kw):
            store["log_model_calls"].append({"name": name, "artifact_path": artifact_path, "skops_trusted_types": skops_trusted_types, **kw})
            rid = store["active"]
            store["runs"][rid]["model"] = model
            return SimpleNamespace(model_uri=f"runs:/{rid}/model")

    def load_model(uri):
        assert uri.startswith("models:/") and "@" in uri, uri
        name, alias = uri[len("models:/"):].split("@")
        version = store["aliases"].get((name, alias))
        if version is None:
            raise FakeMlflowError(f"alias {alias} not found")
        return store["runs"][store["versions"][name][version - 1]]["model"]

    sk.log_model, sk.load_model = log_model, load_model

    def register_model(model_uri, name):
        rid = model_uri.split("/")[1]
        store["versions"].setdefault(name, []).append(rid)
        return SimpleNamespace(version=len(store["versions"][name]))

    class MlflowClient:
        def get_model_version_by_alias(self, name, alias):
            v = store["aliases"].get((name, alias))
            if v is None:
                raise FakeMlflowError(f"Registered model alias {alias} not found.")
            return SimpleNamespace(version=v, run_id=store["versions"][name][v - 1])

        def get_run(self, run_id):
            return SimpleNamespace(data=SimpleNamespace(metrics=store["runs"][run_id]["metrics"]))

        def set_registered_model_alias(self, name, alias, version):
            store["aliases"][(name, alias)] = int(version)

        def download_artifacts(self, run_id, path, dst_path=None):
            out = os.path.join(dst_path, path)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(store["runs"][run_id]["artifacts"][path], f)
            return out

    mlflow.sklearn = sk
    mlflow.register_model = register_model
    mlflow.MlflowClient = MlflowClient
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", sk)
    return store
